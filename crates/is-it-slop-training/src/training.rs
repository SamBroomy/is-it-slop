//! The training recipe: schedule, losses, metric, and optimizer split.
//!
//! Ports the pieces of `notebooks/hierarchical_train.py` the campaign froze. Each numeric piece is
//! unit-tested against values computed by the PyTorch/sklearn implementation it replaces.

use burn::config::Config;
use burn::module::{ModuleVisitor, ParamId};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::model::ModelOutput;

/// The frozen training recipe (EXPERIMENTS.md, "Recommended configurations").
///
/// This is a tuple, not independent knobs: with a cosine schedule, `epochs` shapes the LR curve
/// rather than merely capping it, so re-running with a different `epochs` is a different recipe.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Peak learning rate, reached after one epoch of linear warmup.
    #[config(default = 1e-3)]
    pub lr: f64,
    /// AdamW decay, applied to matrices only — decaying biases and LayerNorm parameters is a known
    /// way to make AdamW look spuriously bad.
    #[config(default = 0.01)]
    pub weight_decay: f64,
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

/// One epoch of linear warmup, then cosine decay to zero at `total_steps`.
///
/// Exact port of the `LambdaLR` lambda in `hierarchical_train.py`, including PyTorch's stepping
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

    /// The LR for the next optimizer step; advances the schedule.
    pub fn step(&mut self) -> f64 {
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
}

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
/// Chunk targets are the document label copied down (`y_chunk` in Python exists for CutMix's soft
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

/// Area under the ROC curve via the Mann-Whitney U statistic, with average ranks for ties.
///
/// Matches `sklearn.metrics.roc_auc_score`. Panics if either class is absent — an AUC over one
/// class is meaningless and a corrupted validation split should fail loudly.
#[must_use]
pub fn roc_auc(labels: &[f32], scores: &[f32]) -> f64 {
    assert_eq!(labels.len(), scores.len());
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));

    // Average ranks over tied score runs (1-based ranks, as in the U statistic).
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
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

/// Parameter ids split by tensor rank: matrices (rank >= 2) get weight decay, biases and
/// LayerNorm parameters (rank < 2) do not — the same split `hierarchical_train.py` makes.
#[derive(Debug, Default)]
pub struct DecaySplit {
    /// Parameters AdamW should decay.
    pub decay: Vec<ParamId>,
    /// Parameters that must not be decayed.
    pub no_decay: Vec<ParamId>,
}

impl<B: Backend> ModuleVisitor<B> for DecaySplit {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
        if D >= 2 {
            self.decay.push(param.id);
        } else {
            self.no_decay.push(param.id);
        }
    }
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
            let got = sched.step();
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
    fn decay_split_partitions_by_rank() {
        use burn::module::Module;

        let model =
            crate::model::HierarchicalModelConfig::new().init::<B>(&NdArrayDevice::default());
        let mut split = DecaySplit::default();
        model.visit(&mut split);

        // Matrices: embed + 5 linear weights. Vectors: 5 linear biases + 3 norms x (gamma, beta).
        assert_eq!(split.decay.len(), 6, "expected 6 rank>=2 parameters");
        assert_eq!(split.no_decay.len(), 11, "expected 11 rank<2 parameters");
    }
}
