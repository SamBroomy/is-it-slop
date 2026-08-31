//! The hierarchical slop classifier: a shared per-chunk trunk, a chunk head, and an
//! attention-pooled document head.
//!
//! Port of `HierarchicalSlopClassifier` in `notebooks/hierarchical_train.py`. The trunk weights
//! are shared across chunks, so adding chunks costs compute but no parameters. Padding is masked
//! out of both pooling steps: a global max over mostly-pad positions would otherwise let padding
//! decide the chunk representation.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

/// Everything that changes the trained network's shape.
///
/// Defaults are the configuration the hyperparameter campaign froze (EXPERIMENTS.md,
/// "Recommended configurations"): `group=2` was the campaign's largest single win, the rest are
/// the accuracy pick. `vocab_size`/`embed_dim` select between the accuracy (16384/64), balanced
/// (16384/48) and minimal (8192/48) artifacts.
#[derive(Config, Debug)]
pub struct HierarchicalModelConfig {
    /// Tokens per chunk. Must be divisible by `group`.
    #[config(default = 192)]
    pub chunk_size: usize,
    /// Distinct token rows kept, ranked by corpus frequency; rarer ids collapse onto the OOV row.
    #[config(default = 16384)]
    pub vocab_size: usize,
    /// Embedding width. The embedding is ~90% of the artifact, so size lives here and in
    /// `vocab_size`.
    #[config(default = 64)]
    pub embed_dim: usize,
    /// Positions reshaped into one trunk entity. The campaign measured 2 as the sweet spot:
    /// 4 (Magika's) destroys resolution, 1 doubles compute for no test gain.
    #[config(default = 2)]
    pub group: usize,
    /// Trunk width.
    #[config(default = 256)]
    pub hidden: usize,
    /// Elementwise dropout inside the trunk and document head.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Channel dropout on the embedding, shared across positions.
    #[config(default = 0.1)]
    pub spatial_dropout: f64,
}

impl HierarchicalModelConfig {
    /// Row index for tokens outside the vocabulary.
    #[must_use]
    pub const fn oov_id(&self) -> usize {
        self.vocab_size
    }

    /// Row index for padding. Masked out of every pooling step.
    #[must_use]
    pub const fn pad_id(&self) -> usize {
        self.vocab_size + 1
    }

    /// Embedding rows: the vocabulary plus the OOV and padding rows.
    #[must_use]
    pub const fn n_embeddings(&self) -> usize {
        self.vocab_size + 2
    }

    /// Trunk positions per chunk after grouping.
    #[must_use]
    pub const fn groups_per_chunk(&self) -> usize {
        self.chunk_size / self.group
    }

    /// Initialize the model on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HierarchicalSlopClassifier<B> {
        assert!(
            self.chunk_size.is_multiple_of(self.group),
            "chunk_size {} not divisible by group {}",
            self.chunk_size,
            self.group
        );
        HierarchicalSlopClassifier {
            embed: EmbeddingConfig::new(self.n_embeddings(), self.embed_dim).init(device),
            spatial_dropout: SpatialDropout1d {
                prob: self.spatial_dropout,
            },
            trunk_in: LinearConfig::new(self.group * self.embed_dim, self.hidden).init(device),
            trunk_norm: LayerNormConfig::new(self.hidden).init(device),
            trunk_dropout: DropoutConfig::new(self.dropout).init(),
            trunk_out: LinearConfig::new(self.hidden, self.hidden).init(device),
            gelu: Gelu::new(),
            chunk_norm: LayerNormConfig::new(self.hidden).init(device),
            chunk_head: LinearConfig::new(self.hidden, 1).init(device),
            attend: LinearConfig::new(self.hidden, 1).init(device),
            doc_norm: LayerNormConfig::new(2 * self.hidden).init(device),
            doc_dropout: DropoutConfig::new(self.dropout).init(),
            doc_head: LinearConfig::new(2 * self.hidden, 1).init(device),
            chunk_size: self.chunk_size,
            group: self.group,
            pad_id: self.pad_id(),
        }
    }
}

/// Drop whole embedding channels across all positions, as in Magika's spatial dropout.
///
/// Mirrors `torch.nn.Dropout2d` applied channelwise: one Bernoulli draw per (sample, channel),
/// shared across sequence positions. Gated on `B::ad_enabled()` exactly like Burn's own
/// [`Dropout`], so validation and inference run without it.
#[derive(Module, Debug, Clone)]
pub struct SpatialDropout1d {
    /// Probability of dropping a channel.
    pub prob: f64,
}

impl SpatialDropout1d {
    /// Apply channel dropout to `[batch, positions, channels]`.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        if !B::ad_enabled(&input.device()) || self.prob == 0.0 {
            return input;
        }
        let prob_keep = 1.0 - self.prob;
        let [n, _, channels] = input.dims();
        let mask = Tensor::<B, 3>::random(
            [n, 1, channels],
            Distribution::Bernoulli(prob_keep),
            &input.device(),
        );
        input * mask * (1.0 / prob_keep)
    }
}

/// See the [module documentation](self).
#[derive(Module, Debug)]
pub struct HierarchicalSlopClassifier<B: Backend> {
    embed: Embedding<B>,
    spatial_dropout: SpatialDropout1d,
    trunk_in: Linear<B>,
    trunk_norm: LayerNorm<B>,
    trunk_dropout: Dropout,
    trunk_out: Linear<B>,
    gelu: Gelu,
    chunk_norm: LayerNorm<B>,
    chunk_head: Linear<B>,
    attend: Linear<B>,
    doc_norm: LayerNorm<B>,
    doc_dropout: Dropout,
    doc_head: Linear<B>,
    chunk_size: usize,
    group: usize,
    pad_id: usize,
}

/// The model's raw outputs for one batch.
pub struct ModelOutput<B: Backend> {
    /// Document logits, `[batch]`.
    pub doc_logits: Tensor<B, 1>,
    /// Per-chunk logits, `[batch, chunks]`.
    pub chunk_logits: Tensor<B, 2>,
    /// Which chunk slots are real (slot 0 always counts), `[batch, chunks]`.
    pub chunk_mask: Tensor<B, 2, burn::tensor::Bool>,
}

impl<B: Backend> HierarchicalSlopClassifier<B> {
    /// Encode ids `[batch, chunks, chunk_size]` into chunk vectors `[batch, chunks, hidden]` plus
    /// a real-chunk mask `[batch, chunks]`.
    fn encode_chunks(
        &self,
        ids: Tensor<B, 3, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, burn::tensor::Bool>) {
        let [b, c, length] = ids.dims();
        let groups = length / self.group;
        let hidden = self.trunk_out.weight.dims()[1];
        let neg_inf = f64::NEG_INFINITY;
        let device = ids.device();

        // Every mask is derived up front so the trunk below stays one uninterrupted compute
        // region: interleaving these view/bool chains with the matmuls splits the fuser's
        // regions and keeps `token_real` alive across the whole block (Burn performance guide —
        // group view ops, don't hold tensors across compute ops).
        let token_real = ids.clone().not_equal_elem(self.pad_id as i64);
        // Burn's Embedding has no padding_idx; zeroing pad positions after lookup is equivalent —
        // the forward sees zeros and the pad row receives no gradient (its contribution is
        // multiplied by zero).
        let pad_zero = token_real.clone().reshape([b * c, length, 1]).float();
        // A group is real if any token in it is; -inf elsewhere so padding cannot win the max.
        let group_fill = token_real
            .clone()
            .reshape([b * c, groups, self.group])
            .any_dim(2)
            .bool_not()
            .expand([b * c, groups, hidden]);
        let chunk_real = token_real.any_dim(2).reshape([b, c]);
        let chunk_fill = chunk_real
            .clone()
            .bool_not()
            .reshape([b * c, 1])
            .expand([b * c, hidden]);
        // Slot 0 always counts as real: a document that tokenizes to nothing still needs one
        // finite slot, or the document pools reduce over an empty set and produce NaN.
        let slot0 = Tensor::<B, 1, Int>::arange(0..c as i64, &device)
            .equal_elem(0)
            .reshape([1, c])
            .expand([b, c]);
        let chunk_mask = chunk_real.bool_or(slot0);

        let h = self.embed.forward(ids.reshape([b * c, length]));
        let h = h * pad_zero;
        let h = self.spatial_dropout.forward(h);
        let h = h.reshape([b * c, groups, self.group * self.embed_dim()]);
        let h = self.trunk_in.forward(h);
        let h = self.gelu.forward(h);
        let h = self.trunk_norm.forward(h);
        let h = self.trunk_dropout.forward(h);
        let h = self.trunk_out.forward(h);
        let h = self.gelu.forward(h);

        let h = h.mask_fill(group_fill, neg_inf);
        let h = h.max_dim(1).reshape([b * c, hidden]);
        // A chunk with no real tokens is all -inf; zero it so it cannot pollute the document pool.
        let h = h.mask_fill(chunk_fill, 0.0);

        let h = self.chunk_norm.forward(h.reshape([b, c, hidden]));
        (h, chunk_mask)
    }

    /// Produce document logits `[batch]`, chunk logits `[batch, chunks]`, and the real-chunk mask.
    pub fn forward(&self, ids: Tensor<B, 3, Int>) -> ModelOutput<B> {
        let (h, chunk_mask) = self.encode_chunks(ids);
        let [b, c, hidden] = h.dims();
        let neg_inf = f64::NEG_INFINITY;
        let not_real = chunk_mask.clone().bool_not();
        let max_fill = not_real.clone().reshape([b, c, 1]).expand([b, c, hidden]);

        let chunk_logits = self.chunk_head.forward(h.clone()).reshape([b, c]);

        // Attention-pooled and max-pooled document vectors are concatenated: the first asks which
        // chunks matter, the second preserves "one chunk was decisive".
        let weights = self
            .attend
            .forward(h.clone())
            .reshape([b, c])
            .mask_fill(not_real, neg_inf);
        let weights = softmax(weights, 1);
        let attended = (h.clone() * weights.reshape([b, c, 1]))
            .sum_dim(1)
            .reshape([b, hidden]);

        let pooled_max = h
            .mask_fill(max_fill, neg_inf)
            .max_dim(1)
            .reshape([b, hidden]);

        let doc = Tensor::cat(vec![attended, pooled_max], 1);
        let doc = self.doc_norm.forward(doc);
        let doc = self.doc_dropout.forward(doc);
        let doc_logits = self.doc_head.forward(doc).reshape([b]);

        ModelOutput {
            doc_logits,
            chunk_logits,
            chunk_mask,
        }
    }

    fn embed_dim(&self) -> usize {
        self.embed.weight.dims()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type B = NdArray;

    #[test]
    fn forward_shapes_and_finiteness() {
        let device = NdArrayDevice::default();
        let cfg = HierarchicalModelConfig::new();
        let model = cfg.init::<B>(&device);
        let pad = cfg.pad_id() as i64;

        // Two documents: one with 1.5 real chunks, one fully padded (the NaN-guard case).
        let mut data = vec![pad; 2 * 3 * cfg.chunk_size];
        for (i, v) in data.iter_mut().enumerate().take(cfg.chunk_size * 3 / 2) {
            *v = (i % 100) as i64;
        }
        let ids = Tensor::<B, 1, Int>::from_ints(data.as_slice(), &device).reshape([
            2,
            3,
            cfg.chunk_size,
        ]);

        let out = model.forward(ids);
        assert_eq!(out.doc_logits.dims(), [2]);
        assert_eq!(out.chunk_logits.dims(), [2, 3]);
        assert_eq!(out.chunk_mask.dims(), [2, 3]);

        let logits: Vec<f32> = out.doc_logits.into_data().to_vec().unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "document logits must be finite even for an all-pad document: {logits:?}"
        );

        // Doc 0: chunks 0 and 1 real, chunk 2 padded. Doc 1: only the slot-0 clamp.
        let mask: Vec<bool> = out.chunk_mask.into_data().to_vec().unwrap();
        assert_eq!(mask, vec![true, true, false, true, false, false]);
    }

    #[test]
    fn parameter_count_matches_python() {
        // Python reported 1,183,363 parameters for v16384-d64 at group=4. At group=2 the first
        // trunk layer halves its input: 1,183,363 - (4-2) * 64 * 256 = 1,150,595.
        let model = HierarchicalModelConfig::new().init::<B>(&NdArrayDevice::default());
        assert_eq!(model.num_params(), 1_150_595);
    }
}
