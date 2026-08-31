//! The training recipe on Burn's `Learner` harness.
//!
//! Ports the pieces of `notebooks/hierarchical_train.py` the campaign froze, each unit-tested
//! against values computed by the PyTorch/sklearn implementation it replaces, and plugs them into
//! `SupervisedTraining` at its trait seams: [`WarmupCosine`] as the `LrScheduler`, and
//! [`RankGatedAdamW`] as the single optimizer, with [`losses`] inside the model's train step.
//! Metrics and early stopping are Burn's built-ins; [`roc_auc`] is the sklearn-checked oracle for
//! evaluation code and tests.

use std::sync::Arc;

use burn::config::Config;
use burn::lr_scheduler::LrScheduler;
use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, AdamWState, SimpleOptimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, Transaction};
use burn::train::metric::{Adaptor, AurocInput, LossInput};
use burn::train::{InferenceStep, ItemLazy, LearningResult, TrainOutput, TrainStep};

use crate::model::{HierarchicalSlopClassifier, ModelOutput};

/// The frozen training recipe (EXPERIMENTS.md, "Recommended configurations").
///
/// This is a tuple, not independent knobs: with a cosine schedule, `epochs` shapes the LR curve
/// rather than merely capping it, so re-running with a different `epochs` is a different recipe.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Peak learning rate, reached after one epoch of linear warmup.
    #[config(default = 1e-3)]
    pub lr: f64,
    /// `AdamW` decay, applied to matrices only — decaying biases and `LayerNorm` parameters is a known
    /// way to make `AdamW` look spuriously bad.
    #[config(default = 0.01)]
    pub weight_decay: f32,
    /// Epoch budget; the cosine horizon. The 45-epoch arm confirmed 30 is not undertrained.
    #[config(default = 30)]
    pub epochs: usize,
    /// Epochs without a validation-AUC improvement before stopping.
    #[config(default = 6)]
    pub patience: usize,
    /// Documents per step.
    #[config(default = 128)]
    pub batch_size: usize,
    /// Lambda in `L_doc + lambda * L_chunk`. Chunk labels are the document label copied down — a
    /// noisy auxiliary signal the ablation showed still earns its keep.
    #[config(default = 0.3)]
    pub chunk_loss_weight: f64,
    /// Varies init, shuffling and dropout draws.
    #[config(default = 42)]
    pub seed: u64,
}

// --------------------------------------------------------------------------------------------
// Learning rate schedule
// --------------------------------------------------------------------------------------------

/// One epoch of linear warmup, then cosine decay to zero at `total_steps`.
///
/// Exact port of the `LambdaLR` lambda in `hierarchical_train.py`, including `PyTorch`'s stepping
/// convention: the LR used by optimizer step `k` (0-based) is `base_lr * lambda(k)`, so warmup
/// starts at `base/warmup`, and the last warmup step and the first cosine step both produce the
/// full base LR.
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    step: usize,
}

impl WarmupCosine {
    /// A schedule planned against `total_steps` with `warmup_steps` of linear warmup.
    #[must_use]
    pub const fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            step: 0,
        }
    }
}

impl LrScheduler for WarmupCosine {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> f64 {
        let s = self.step;
        self.step += 1;
        let lambda = if s < self.warmup_steps {
            (s + 1) as f64 / self.warmup_steps as f64
        } else {
            let progress = (s - self.warmup_steps) as f64
                / (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64;
            0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        };
        self.base_lr * lambda
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step = record;
        self
    }
}

// --------------------------------------------------------------------------------------------
// Optimizer
// --------------------------------------------------------------------------------------------

/// `AdamW` that applies weight decay to matrices only.
///
/// Stands in for `PyTorch`'s parameter groups: the tensor rank decides the group, so one optimizer
/// serves the whole module and fits `Learner::new`. Rank >= 2 (embedding and linear weights)
/// decays; biases and `LayerNorm` parameters do not.
#[derive(Clone)]
pub struct RankGatedAdamW {
    decay: AdamW,
    plain: AdamW,
}

impl RankGatedAdamW {
    /// Build from the recipe's decay setting. Epsilon is `PyTorch`'s default; Burn's differs.
    #[must_use]
    pub fn new(weight_decay: f32) -> Self {
        Self {
            decay: AdamWConfig::new()
                .with_weight_decay(weight_decay)
                .with_epsilon(1e-8)
                .build(),
            plain: AdamWConfig::new()
                .with_weight_decay(0.0)
                .with_epsilon(1e-8)
                .build(),
        }
    }

    /// Wrap into an [`Optimizer`](burn::optim::Optimizer) usable by the `Learner`.
    #[must_use]
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        weight_decay: f32,
    ) -> OptimizerAdaptor<Self, M, B> {
        Self::new(weight_decay).into()
    }
}

impl<B: Backend> SimpleOptimizer<B> for RankGatedAdamW {
    type State<const D: usize> = AdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        if D >= 2 {
            SimpleOptimizer::<B>::step(&self.decay, lr, tensor, grad, state)
        } else {
            SimpleOptimizer::<B>::step(&self.plain, lr, tensor, grad, state)
        }
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        <AdamW as SimpleOptimizer<B>>::to_device(state, device)
    }
}

// --------------------------------------------------------------------------------------------
// Loss
// --------------------------------------------------------------------------------------------

/// Elementwise binary cross-entropy on logits: `max(x,0) - x*y + ln(1 + exp(-|x|))`.
///
/// The standard numerically stable form, matching
/// `torch.nn.functional.binary_cross_entropy_with_logits`.
fn bce_with_logits<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> Tensor<B, D> {
    logits.clone().clamp_min(0.0) - logits.clone() * targets
        + (logits.abs().neg().exp() + 1.0).log()
}

/// Document loss plus a masked, down-weighted chunk loss over real chunks only.
///
/// Chunk targets are the document label copied down (`y_chunk` in Python exists for `CutMix`'s soft
/// targets, which are not ported).
pub fn losses<B: Backend>(
    output: &ModelOutput<B>,
    labels: Tensor<B, 1>,
    chunk_loss_weight: f64,
) -> Tensor<B, 1> {
    let [b, c] = output.chunk_logits.dims();
    let doc_loss = bce_with_logits(output.doc_logits.clone(), labels.clone()).mean();

    let y_chunk = labels.reshape([b, 1]).expand([b, c]);
    let per_chunk = bce_with_logits(output.chunk_logits.clone(), y_chunk);
    let real = output.chunk_mask.clone().float();
    let chunk_loss = (per_chunk * real.clone()).sum() / real.sum().clamp_min(1.0);

    doc_loss + chunk_loss * chunk_loss_weight
}

// --------------------------------------------------------------------------------------------
// Train / validation steps
// --------------------------------------------------------------------------------------------

/// One training or validation batch. Building these from encoded documents is the data layer's
/// job (see [`crate::data`]); the steps only consume them.
#[derive(Debug, Clone)]
pub struct Batch<B: Backend> {
    /// Compact token ids, `[batch, chunks, chunk_size]`.
    pub ids: Tensor<B, 3, Int>,
    /// Document labels in `{0.0, 1.0}`, `[batch]`.
    pub labels: Tensor<B, 1>,
}

/// What a step hands the metrics: the loss it backpropagated, plus the document logits and
/// targets the AUROC metric consumes.
#[derive(Debug, Clone)]
pub struct SlopOutput<B: Backend> {
    /// The scalar batch loss.
    pub loss: Tensor<B, 1>,
    /// Raw document logits, `[batch]`; `sigmoid` gives the AI probability.
    pub logits: Tensor<B, 1>,
    /// Document labels, `[batch]`.
    pub targets: Tensor<B, 1>,
}

/// The classifier plus the loss weighting its steps need; this is the module the `Learner`
/// trains and checkpoints.
#[derive(Module, Debug)]
pub struct TrainableClassifier<B: Backend> {
    /// The network itself.
    pub model: HierarchicalSlopClassifier<B>,
    /// Lambda in `L_doc + lambda * L_chunk`.
    pub chunk_loss_weight: f64,
}

impl<B: Backend> TrainableClassifier<B> {
    /// Forward pass plus the recipe's loss.
    pub fn forward_step(&self, batch: Batch<B>) -> SlopOutput<B> {
        let output = self.model.forward(batch.ids);
        let loss = losses(&output, batch.labels.clone(), self.chunk_loss_weight);
        SlopOutput {
            loss,
            logits: output.doc_logits,
            targets: batch.labels,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for TrainableClassifier<B> {
    type Input = Batch<B>;
    type Output = SlopOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<SlopOutput<B>> {
        let output = self.forward_step(batch);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for TrainableClassifier<B> {
    type Input = Batch<B>;
    type Output = SlopOutput<B>;

    fn step(&self, batch: Batch<B>) -> SlopOutput<B> {
        self.forward_step(batch)
    }
}

impl<B: Backend> ItemLazy for SlopOutput<B> {
    type ItemSync = SlopOutput<burn::backend::Flex>;

    fn sync(self) -> Self::ItemSync {
        let [loss, logits, targets] = Transaction::default()
            .register(self.loss)
            .register(self.logits)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");
        let device = &burn::backend::flex::FlexDevice;
        SlopOutput {
            loss: Tensor::from_data(loss, device),
            logits: Tensor::from_data(logits, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for SlopOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<AurocInput<B>> for SlopOutput<B> {
    /// Burn's AUROC metric softmaxes a `[batch, 2]` output and reads class 1, so `[0, logit]`
    /// hands it exactly `sigmoid(logit)`.
    fn adapt(&self) -> AurocInput<B> {
        let [n] = self.logits.dims();
        let outputs = Tensor::stack::<2>(vec![self.logits.zeros_like(), self.logits.clone()], 1);
        AurocInput::new(outputs, self.targets.clone().int().reshape([n]))
    }
}

// --------------------------------------------------------------------------------------------
// Evaluation metric
// --------------------------------------------------------------------------------------------

/// Area under the ROC curve via the Mann-Whitney U statistic, with average ranks for ties.
///
/// Matches `sklearn.metrics.roc_auc_score`. Panics if either class is absent — an AUC over one
/// class is meaningless and a corrupted validation split should fail loudly.
#[must_use]
pub fn roc_auc(labels: &[f32], scores: &[f32]) -> f64 {
    assert_eq!(labels.len(), scores.len());
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));

    // Average ranks over tied score runs (1-based ranks, as in the U statistic). The comparison
    // is deliberately exact: only bitwise-equal scores are a tie, matching sklearn.
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    #[allow(clippy::float_cmp)]
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg;
        }
        i = j + 1;
    }

    let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
    let n_neg = labels.len() - n_pos;
    assert!(n_pos > 0 && n_neg > 0, "AUC needs both classes present");

    let pos_rank_sum: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|&(&l, _)| l > 0.5)
        .map(|(_, &r)| r)
        .sum();
    (pos_rank_sum - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}

// --------------------------------------------------------------------------------------------
// Training entry point
// --------------------------------------------------------------------------------------------

/// Run the frozen recipe on Burn's `Learner` harness.
///
/// `renderer` overrides the default metrics dashboard (pass
/// `None::<burn::train::renderer::CliMetricsRenderer>` to keep it); checkpoints, metric logs and
/// the summary land under `artifact_dir`.
///
/// Two knowing deviations from the Python loop, both from leaning on Burn's built-ins: early
/// stopping acts on the store's mean of per-batch validation AUCs rather than the full-split AUC,
/// and the returned model carries the final epoch's weights rather than the best epoch's. The
/// parity gate for the Rust port (a real run landing in the campaign's combo band) is the check
/// that neither matters in practice.
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    model: HierarchicalSlopClassifier<B>,
    config: &TrainingConfig,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<B, Batch<B>>>,
    dataloader_valid: Arc<
        dyn burn::data::dataloader::DataLoader<B::InnerBackend, Batch<B::InnerBackend>>,
    >,
    device: &B::Device,
    renderer: Option<impl burn::train::renderer::MetricsRenderer + 'static>,
) -> LearningResult<TrainableClassifier<B::InnerBackend>> {
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use burn::train::metric::store::{Aggregate, Direction, Split};
    use burn::train::metric::{AurocMetric, IterationSpeedMetric, LearningRateMetric, LossMetric};
    use burn::train::{
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
    };

    B::seed(device, config.seed);

    let steps_per_epoch = dataloader_train.num_items().div_ceil(config.batch_size);
    let schedule = WarmupCosine::new(config.lr, steps_per_epoch, config.epochs * steps_per_epoch);
    let optim = RankGatedAdamW::init(config.weight_decay);
    let trainable = TrainableClassifier {
        model,
        chunk_loss_weight: config.chunk_loss_weight,
    };

    let auroc = AurocMetric::new();
    let early_stopping = MetricEarlyStoppingStrategy::new(
        &auroc,
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
        StoppingCondition::NoImprovementSince {
            n_epochs: config.patience,
        },
    );

    // Full precision checkpoints: a resumed or reloaded model must carry the trained weights,
    // not an f16 rounding of them.
    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(auroc)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .with_file_checkpointer(NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .early_stopping(early_stopping)
        .num_epochs(config.epochs)
        .summary();
    if let Some(renderer) = renderer {
        training = training.renderer(renderer);
    }

    training.launch(Learner::new(trainable, optim, schedule))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type B = NdArray;

    #[test]
    fn warmup_cosine_matches_pytorch_lambda_lr() {
        // torch.optim.lr_scheduler.LambdaLR with base 1e-3, warmup 3, total 12: the lr consumed by
        // each of the 12 optimizer steps.
        let expected = [
            3.333333333333e-04,
            6.666666666667e-04,
            1.000000000000e-03,
            1.000000000000e-03,
            9.698463103930e-04,
            8.830222215595e-04,
            7.500000000000e-04,
            5.868240888335e-04,
            4.131759111665e-04,
            2.500000000000e-04,
            1.169777784405e-04,
            3.015368960705e-05,
        ];
        let mut sched = WarmupCosine::new(1e-3, 3, 12);
        for (i, want) in expected.iter().enumerate() {
            let got = LrScheduler::step(&mut sched);
            assert!(
                (got - want).abs() < 1e-15,
                "step {i}: got {got:e}, torch says {want:e}"
            );
        }
    }

    #[test]
    fn losses_match_pytorch() {
        // Values computed with torch.nn.functional.binary_cross_entropy_with_logits and the
        // Python losses(): doc_loss 0.3936693668, chunk_loss 1.0513024330, total 0.7090600729
        // at chunk_loss_weight 0.3 with one masked-out chunk.
        let device = NdArrayDevice::default();
        let output = ModelOutput {
            doc_logits: Tensor::<B, 1>::from_floats([0.5, -1.0], &device),
            chunk_logits: Tensor::<B, 2>::from_floats([[0.2, -0.3], [1.5, 0.7]], &device),
            chunk_mask: Tensor::<B, 2>::from_floats([[1.0, 1.0], [1.0, 0.0]], &device)
                .greater_elem(0.5),
        };
        let labels = Tensor::<B, 1>::from_floats([1.0, 0.0], &device);

        let total = losses(&output, labels, 0.3).into_scalar();
        assert!(
            (f64::from(total) - 0.709_060_072_9).abs() < 1e-6,
            "got {total}, torch says 0.7090600729"
        );
    }

    #[test]
    fn auc_matches_sklearn() {
        // sklearn.metrics.roc_auc_score on the same inputs, including tied scores within and
        // across classes.
        let cases: [(&[f32], &[f32], f64); 3] = [
            (&[0., 0., 1., 1.], &[0.1, 0.4, 0.35, 0.8], 0.75),
            (&[0., 1., 0., 1.], &[0.5, 0.5, 0.2, 0.9], 0.875),
            (
                &[1., 0., 1., 1., 0., 0.],
                &[0.9, 0.9, 0.8, 0.3, 0.3, 0.1],
                0.666_666_666_7,
            ),
        ];
        for (labels, scores, want) in cases {
            let got = roc_auc(labels, scores);
            assert!((got - want).abs() < 1e-9, "got {got}, sklearn says {want}");
        }
    }

    #[test]
    fn rank_gated_adamw_decays_matrices_only() {
        // One step with zero gradients isolates the decay term: the matrix must shrink by
        // lr * weight_decay, the vector must not move.
        let device = NdArrayDevice::default();
        let optim = RankGatedAdamW::new(0.5);

        let matrix = Tensor::<B, 2>::ones([2, 2], &device);
        let vector = Tensor::<B, 1>::ones([2], &device);
        let (matrix_after, _) =
            SimpleOptimizer::<B>::step(&optim, 0.1, matrix, Tensor::zeros([2, 2], &device), None);
        let (vector_after, _) =
            SimpleOptimizer::<B>::step(&optim, 0.1, vector, Tensor::zeros([2], &device), None);

        let m: Vec<f32> = matrix_after.into_data().to_vec().unwrap();
        let v: Vec<f32> = vector_after.into_data().to_vec().unwrap();
        assert!(
            m.iter().all(|x| (x - 0.95).abs() < 1e-6),
            "matrix should decay to 1 - lr*wd = 0.95, got {m:?}"
        );
        assert!(
            v.iter().all(|x| (x - 1.0).abs() < 1e-6),
            "vector must not decay, got {v:?}"
        );
    }

    #[test]
    fn learner_pipeline_learns_a_toy_rule() {
        use burn::backend::Autodiff;
        use burn::data::dataloader::DataLoaderBuilder;
        use burn::data::dataset::InMemDataset;
        use burn::train::renderer::CliMetricsRenderer;

        use crate::data::{EncodedDocument, SlopBatcher};

        type AB = Autodiff<NdArray>;
        let device = NdArrayDevice::default();
        // Reduce sensitivity to parallel tests sharing the global RNG; `train` reseeds again but
        // model init happens before it.
        AB::seed(&device, 7);

        // Tiny model, tiny corpus: label 1 iff the document's tokens come from the upper half of
        // the vocabulary. The full harness (batcher, learner, schedule, optimizer, AUROC early
        // stopping) must recover the rule.
        let model_cfg = crate::model::HierarchicalModelConfig::new()
            .with_chunk_size(8)
            .with_vocab_size(64)
            .with_embed_dim(8)
            .with_hidden(16)
            .with_dropout(0.0)
            .with_spatial_dropout(0.0);
        let pad = model_cfg.pad_id() as u16;

        // The trunk consumes tokens in groups (pairs at group=2), so the generator must draw
        // train and validation docs from the same finite pair set or the toy demands
        // memorization: token_t depends only on k, giving 32 distinct grids per class, all of
        // which appear in a large enough training split. `k = i/2 + seed` keeps k decoupled from
        // the label (`i % 2`) — an affine map of `i` would couple grid parity to the label and
        // make each class's validation grids unseen at training time.
        let make_docs = |n: usize, seed: i64| -> Vec<EncodedDocument> {
            (0..n)
                .map(|i| {
                    let label = (i % 2) as i64;
                    let base = if label == 1 { 32 } else { 0 };
                    let k = (i as i64 / 2 + seed) % 32;
                    // Two chunk slots: one real, one padded, so masking is exercised end to end.
                    let mut ids: Vec<u16> = (0..8)
                        .map(|t| (base + ((k + t * 17) % 32)) as u16)
                        .collect();
                    ids.extend([pad; 8]);
                    EncodedDocument {
                        ids,
                        label: label as f32,
                    }
                })
                .collect()
        };

        let batcher = SlopBatcher {
            max_chunks: 2,
            chunk_size: 8,
        };
        // Enough training documents that memorizing beats nothing — the token-range rule is the
        // only way to fit — and a validation set drawn from the same generator.
        let dataloader_train = DataLoaderBuilder::new(batcher.clone())
            .batch_size(16)
            .shuffle(42)
            .build(InMemDataset::new(make_docs(256, 0)));
        let dataloader_valid = DataLoaderBuilder::new(batcher)
            .batch_size(16)
            .build(InMemDataset::new(make_docs(64, 1)));

        let config = TrainingConfig::new()
            .with_lr(1e-2)
            .with_epochs(20)
            .with_patience(20)
            .with_batch_size(16);
        let artifacts = tempfile::tempdir().unwrap();
        let result = train::<AB>(
            artifacts.path().to_str().unwrap(),
            model_cfg.init::<AB>(&device),
            &config,
            dataloader_train,
            dataloader_valid,
            &device,
            Some(CliMetricsRenderer::new()),
        );

        // Score the returned model on the validation set with the oracle-tested AUC.
        let valid_docs = make_docs(64, 1);
        let batch: Batch<NdArray> = burn::data::dataloader::batcher::Batcher::batch(
            &SlopBatcher {
                max_chunks: 2,
                chunk_size: 8,
            },
            valid_docs,
            &device,
        );
        let output = result.model.forward_step(batch);
        let logits: Vec<f32> = output.logits.into_data().to_vec().unwrap();
        let labels: Vec<f32> = output.targets.into_data().to_vec().unwrap();
        let auc = roc_auc(&labels, &logits);

        // Diagnostics: train-split AUC of the returned model.
        let train_batch: Batch<NdArray> = burn::data::dataloader::batcher::Batcher::batch(
            &SlopBatcher {
                max_chunks: 2,
                chunk_size: 8,
            },
            make_docs(256, 0),
            &device,
        );
        let train_out = result.model.forward_step(train_batch);
        let train_auc = roc_auc(
            &train_out.targets.into_data().to_vec::<f32>().unwrap(),
            &train_out.logits.into_data().to_vec::<f32>().unwrap(),
        );
        eprintln!("DIAG returned-model AUC: train={train_auc:.4} valid={auc:.4}");

        assert!(
            auc > 0.9,
            "harness failed to learn a linearly separable rule: AUC {auc}"
        );
    }
}
