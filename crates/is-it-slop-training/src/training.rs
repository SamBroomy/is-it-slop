//! The training recipe on Burn's `Learner` harness.
//!
//! Ports the pieces of `notebooks/hierarchical_train.py` the campaign froze, each unit-tested
//! against values computed by the PyTorch/sklearn implementation it replaces, and plugs them into
//! `SupervisedTraining` at its trait seams: [`WarmupCosine`] as the `LrScheduler`, and
//! [`RankGatedAdamW`] as the single optimizer, with [`losses`] inside the model's train step.
//! Metrics and early stopping are Burn's built-ins; the sklearn-checked evaluation oracle lives
//! in [`crate::evaluation`].

use std::sync::Arc;

use burn::{
    config::Config,
    lr_scheduler::{LrScheduler, LrSchedulerRecord},
    module::Module,
    optim::{AdamW, AdamWConfig, AdamWState, ModuleOptimizer, Optimizer},
    tensor::{Device, Int, Tensor},
    train::{
        ClassificationOutput, InferenceStep, LearningResult, TrainOutput, TrainStep,
        logger::{EvaluationProgressLogger, TrainingProgressLogger},
        metric::MetricDefinition,
        renderer::{
            EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
            MetricsRendererTraining,
        },
    },
};

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
    /// `AdamW` decay, applied to matrices only — decaying biases and `LayerNorm` parameters is a
    /// known way to make `AdamW` look spuriously bad.
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
    fn step(&mut self) -> f64 {
        let s = self.step;
        self.step += 1;
        let lambda = if s < self.warmup_steps {
            (s + 1) as f64 / self.warmup_steps as f64
        } else {
            let progress = (s - self.warmup_steps) as f64
                / (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64;
            f64::midpoint(1.0, (std::f64::consts::PI * progress).cos())
        };
        self.base_lr * lambda
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new().with_scalar("step", self.step as u64)
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(step) = record.scalar::<u64>("step") {
            self.step = step as usize;
        }
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

    /// Wrap into a [`ModuleOptimizer`] usable by the `Learner`.
    #[must_use]
    pub fn init(weight_decay: f32) -> ModuleOptimizer {
        ModuleOptimizer::from(Self::new(weight_decay))
    }
}

impl Optimizer for RankGatedAdamW {
    type State<const D: usize> = AdamWState<D>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        if D >= 2 {
            Optimizer::step(&self.decay, lr, tensor, grad, state)
        } else {
            Optimizer::step(&self.plain, lr, tensor, grad, state)
        }
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        <AdamW as Optimizer>::to_device(state, device)
    }
}

// --------------------------------------------------------------------------------------------
// Loss
// --------------------------------------------------------------------------------------------

/// Elementwise binary cross-entropy on logits: `max(x,0) - x*y + ln(1 + exp(-|x|))`.
///
/// The standard numerically stable form, matching
/// `torch.nn.functional.binary_cross_entropy_with_logits`.
fn bce_with_logits<const D: usize>(logits: Tensor<D>, targets: Tensor<D>) -> Tensor<D> {
    logits.clone().clamp_min(0.0) - logits.clone() * targets
        + (logits.abs().neg().exp() + 1.0).log()
}

/// Document loss plus a masked, down-weighted chunk loss over real chunks only.
///
/// Chunk targets are the document label copied down (`y_chunk` in Python exists for `CutMix`'s soft
/// targets, which are not ported).
#[must_use]
pub fn losses(output: &ModelOutput, labels: Tensor<1>, chunk_loss_weight: f64) -> Tensor<1> {
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
pub struct Batch {
    /// Compact token ids, `[batch, chunks, chunk_size]`.
    pub ids: Tensor<3, Int>,
    /// Document labels in `{0.0, 1.0}`, `[batch]`.
    pub labels: Tensor<1>,
}

/// The classifier plus the loss weighting its steps need; this is the module the `Learner`
/// trains.
#[derive(Module, Debug)]
pub struct TrainableClassifier {
    /// The network itself.
    pub model: HierarchicalSlopClassifier,
    /// Lambda in `L_doc + lambda * L_chunk`.
    pub chunk_loss_weight: f64,
}

impl TrainableClassifier {
    /// Forward pass plus the recipe's loss, packaged as Burn's [`ClassificationOutput`] so the
    /// built-in metrics (and their `ItemLazy` sync) apply without custom adaptors.
    ///
    /// The single document logit `x` becomes the two-class row `[-x, x]`. Accuracy's argmax is then
    /// the 0.5-threshold decision (class 1 wins iff `x > 0`). Burn 0.22's AUROC scores every
    /// column one-vs-rest and macro-averages them: with `[-x, x]` both columns rank documents
    /// identically, so the average is the document AUC. A constant column (the old `[0, x]`)
    /// would score 0.5 and drag the reported value to `(0.5 + AUC) / 2`.
    #[must_use]
    pub fn forward_step(&self, batch: Batch) -> ClassificationOutput {
        let output = self.model.forward(batch.ids);
        let loss = losses(&output, batch.labels.clone(), self.chunk_loss_weight);
        let logits =
            Tensor::stack::<2>(vec![output.doc_logits.clone().neg(), output.doc_logits], 1);
        ClassificationOutput::new(loss, logits, batch.labels.int())
    }
}

impl TrainStep for TrainableClassifier {
    type Input = Batch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Batch) -> TrainOutput<ClassificationOutput> {
        let output = self.forward_step(batch);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl InferenceStep for TrainableClassifier {
    type Input = Batch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Batch) -> ClassificationOutput {
        self.forward_step(batch)
    }
}

/// Extract `(labels, document logits)` from a step output, for the evaluation side
/// ([`crate::evaluation`]).
///
/// Column 1 of the `[batch, 2]` output is the raw document logit (column 0 is its negation, which
/// [`TrainableClassifier::forward_step`] adds for the metric contract).
#[must_use]
pub fn doc_scores(output: &ClassificationOutput) -> (Vec<f32>, Vec<f32>) {
    let [n, _] = output.output.dims();
    let logits = output.output.clone().slice([0..n, 1..2]).reshape([n]);
    // convert() first: on an f16/flex32 element backend the data arrives in that type.
    (
        output
            .targets
            .clone()
            .float()
            .into_data()
            .convert::<f32>()
            .try_to_vec()
            .unwrap(),
        logits.into_data().convert::<f32>().try_to_vec().unwrap(),
    )
}

// --------------------------------------------------------------------------------------------
// Progress rendering
// --------------------------------------------------------------------------------------------

/// A quiet progress renderer for non-interactive runs.
///
/// Burn's fallback `CliMetricsRenderer` (chosen whenever stdout is not a terminal) reports every
/// iteration; this prints one compact line per `INTERVAL` instead. Per-epoch metrics still land
/// in the artifact dir's metric logs and the end-of-run summary.
pub struct ThrottledRenderer {
    last_print: Option<std::time::Instant>,
    last_items: usize,
    split: String,
    split_total: usize,
    epoch: usize,
    total_epochs: usize,
}

impl ThrottledRenderer {
    /// Seconds between progress lines.
    const INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

    /// Create a renderer that prints its first line immediately, then every `INTERVAL`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_print: None,
            last_items: 0,
            split: String::new(),
            split_total: 0,
            epoch: 0,
            total_epochs: 0,
        }
    }
}

impl Default for ThrottledRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingProgressLogger for ThrottledRenderer {
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        _total_items: Option<usize>,
        _label: Option<&str>,
    ) {
        self.total_epochs = total_epochs;
        self.epoch = starting_epoch;
    }

    fn update_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    fn start_split(&mut self, split: &str, total_items: usize) {
        split.clone_into(&mut self.split);
        self.split_total = total_items;
        self.last_items = 0;
    }

    fn update_split(&mut self, items_processed: usize) {
        use std::fmt::Write;

        let now = std::time::Instant::now();
        let elapsed = self.last_print.map(|t| now.duration_since(t));
        if elapsed.is_some_and(|e| e < Self::INTERVAL) {
            return;
        }
        let mut line = format!(
            "{} epoch {}/{} | {items_processed}/{} items",
            self.split, self.epoch, self.total_epochs, self.split_total
        );
        // Items/s over the printing interval; skip across epoch wraps and split switches.
        if let (Some(elapsed), true) = (elapsed, items_processed >= self.last_items) {
            let rate = (items_processed - self.last_items) as f64 / elapsed.as_secs_f64().max(1e-9);
            if rate > 0.0 {
                let _ = write!(line, " | {rate:.0} items/s");
            }
        }
        self.last_items = items_processed;
        println!("{line}");
        self.last_print = Some(now);
    }

    fn end_split(&mut self) {}

    fn end(&mut self) {}

    fn log_event_training(&mut self, _event: String) {}
}

impl EvaluationProgressLogger for ThrottledRenderer {
    fn start_global_progress(&mut self, _total_tests: usize) {}

    fn start_test(&mut self, name: &str, total_items: usize) {
        name.clone_into(&mut self.split);
        self.split_total = total_items;
        self.last_items = 0;
    }

    fn update_test_progress(&mut self, items_processed: usize) {
        let now = std::time::Instant::now();
        if self
            .last_print
            .is_some_and(|t| now.duration_since(t) < Self::INTERVAL)
        {
            return;
        }
        println!("test {items_processed}/{} items", self.split_total);
        self.last_print = Some(now);
    }

    fn end_test(&mut self) {}

    fn end_global_progress(&mut self) {}

    fn log_event_evaluation(&mut self, _event: String) {}
}

impl MetricsRendererTraining for ThrottledRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}
}

impl MetricsRendererEvaluation for ThrottledRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for ThrottledRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, _definition: MetricDefinition) {}
}

// --------------------------------------------------------------------------------------------
// Training entry point
// --------------------------------------------------------------------------------------------

/// Run the frozen recipe on Burn's `Learner` harness.
///
/// `renderer` overrides the default metrics dashboard (pass
/// `None::<burn::train::renderer::CliMetricsRenderer>` to keep it); metric logs and
/// the summary land under `artifact_dir`.
///
/// Two knowing deviations from the original Python loop, both from leaning on Burn's built-ins:
/// the returned model carries the final epoch's weights rather than the best epoch's; and the
/// loader trains on the final partial batch each epoch (Python used `drop_last=True`; the cosine
/// horizon below counts steps the Burn way, so schedule and loader agree). Early stopping acts
/// on Burn's validation AUROC, which Burn 0.22 accumulates over the whole epoch, as Python did.
/// (The parity numbers below were measured on Burn 0.21, where it was a mean of per-batch AUCs.)
///
/// Check [`LearningResult::error`] and `interrupted` on the result: Burn 0.22 stops and reports a
/// failed metric read, dataloader, checkpointer or worker there instead of panicking. The parity
/// run verified none of it matters: val AUC 0.9960 / test 0.9967 against Python's 0.9961 / 0.9968,
/// inside seed noise (EXPERIMENTS.md, "Burn training port"). This implementation is the reference
/// now; the Python numbers stay only as the quality baseline.
pub fn train(
    artifact_dir: &str,
    model: HierarchicalSlopClassifier,
    config: &TrainingConfig,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<Batch>>,
    dataloader_valid: Arc<dyn burn::data::dataloader::DataLoader<Batch>>,
    device: &Device,
    renderer: Option<impl MetricsRenderer + 'static>,
) -> LearningResult<TrainableClassifier> {
    use burn::train::{
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
        metric::{
            AccuracyMetric, AurocMetric, IterationSpeedMetric, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    };

    device.seed(config.seed);

    let steps_per_epoch = dataloader_train.num_items().div_ceil(config.batch_size);
    let schedule = WarmupCosine::new(config.lr, steps_per_epoch, config.epochs * steps_per_epoch);
    let optim = RankGatedAdamW::init(config.weight_decay);
    let trainable = TrainableClassifier {
        model,
        chunk_loss_weight: config.chunk_loss_weight,
    };

    let auroc = AurocMetric::binary();
    let early_stopping = MetricEarlyStoppingStrategy::new(
        &auroc,
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
        StoppingCondition::NoImprovementSince {
            n_epochs: config.patience,
        },
    );

    // No epoch checkpointer yet: nothing resumes from checkpoints (NEXT_RELEASE, "Resume a
    // training run"), and the returned model carries the final weights. Burn 0.22's
    // `with_default_checkpointers()` writes burnpack checkpoints and is the place to start when
    // resume support lands.
    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(auroc)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .early_stopping(early_stopping)
        .num_epochs(config.epochs)
        .summary();
    if let Some(renderer) = renderer {
        training = training.renderer(Box::new(renderer));
    }

    training.launch(Learner::new(trainable, optim, schedule))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let device = Device::flex();
        let output = ModelOutput {
            doc_logits: Tensor::<1>::from_floats([0.5, -1.0], &device),
            chunk_logits: Tensor::<2>::from_floats([[0.2, -0.3], [1.5, 0.7]], &device),
            chunk_mask: Tensor::<2>::from_floats([[1.0, 1.0], [1.0, 0.0]], &device)
                .greater_elem(0.5),
        };
        let labels = Tensor::<1>::from_floats([1.0, 0.0], &device);

        let total = losses(&output, labels, 0.3).into_scalar::<f32>();
        assert!(
            (f64::from(total) - 0.709_060_072_9).abs() < 1e-6,
            "got {total}, torch says 0.7090600729"
        );
    }

    #[test]
    fn rank_gated_adamw_decays_matrices_only() {
        // One step with zero gradients isolates the decay term: the matrix must shrink by
        // lr * weight_decay, the vector must not move.
        let device = Device::flex();
        let optim = RankGatedAdamW::new(0.5);

        let matrix = Tensor::<2>::ones([2, 2], &device);
        let vector = Tensor::<1>::ones([2], &device);
        let (matrix_after, _) =
            Optimizer::step(&optim, 0.1, matrix, Tensor::zeros([2, 2], &device), None);
        let (vector_after, _) =
            Optimizer::step(&optim, 0.1, vector, Tensor::zeros([2], &device), None);

        let m: Vec<f32> = matrix_after.into_data().try_to_vec().unwrap();
        let v: Vec<f32> = vector_after.into_data().try_to_vec().unwrap();
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
        use burn::{
            data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
            train::renderer::CliMetricsRenderer,
        };

        use crate::{
            data::{EncodedDocument, SlopBatcher},
            evaluation::roc_auc,
        };

        let inner_device = Device::flex();
        let device = inner_device.clone().autodiff();
        // Reduce sensitivity to parallel tests sharing the global RNG; `train` reseeds again but
        // model init happens before it.
        device.seed(7);

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
        let result = train(
            artifacts.path().to_str().unwrap(),
            model_cfg.init(&device),
            &config,
            dataloader_train,
            dataloader_valid,
            &device,
            Some(CliMetricsRenderer::new()),
        );

        // Score the returned model on the validation set with the oracle-tested AUC.
        let valid_docs = make_docs(64, 1);
        let batch: Batch = burn::data::dataloader::batcher::Batcher::batch(
            &SlopBatcher {
                max_chunks: 2,
                chunk_size: 8,
            },
            valid_docs,
            &inner_device,
        );
        let output = result.model.forward_step(batch);
        let (labels, logits) = doc_scores(&output);
        let auc = roc_auc(&labels, &logits);

        // Diagnostics: train-split AUC of the returned model.
        let train_batch: Batch = burn::data::dataloader::batcher::Batcher::batch(
            &SlopBatcher {
                max_chunks: 2,
                chunk_size: 8,
            },
            make_docs(256, 0),
            &inner_device,
        );
        let (train_labels, train_logits) = doc_scores(&result.model.forward_step(train_batch));
        let train_auc = roc_auc(&train_labels, &train_logits);
        eprintln!("DIAG returned-model AUC: train={train_auc:.4} valid={auc:.4}");

        assert!(
            auc > 0.9,
            "harness failed to learn a linearly separable rule: AUC {auc}"
        );
    }
}
