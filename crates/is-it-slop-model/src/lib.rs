//! The hierarchical slop classifier: a shared per-chunk trunk, a chunk head, and an
//! attention-pooled document head.
//!
//! Port of `HierarchicalSlopClassifier` in `notebooks/hierarchical_train.py`. The grouped trunk
//! and masked max pooling are Magika's design; the hierarchy on top — a chunk level and a
//! document level trained jointly — is ours.
//!
//! Input is `[batch, chunks, chunk_size]` compact token ids from the data layer (the vocabulary
//! already mapped rare tokens to the OOV row and padding to the pad row). The forward pass, with
//! the code split between [`HierarchicalSlopClassifier::encode_chunks`] (stages 1-4) and
//! [`HierarchicalSlopClassifier::forward`] (stages 5-6):
//!
//! 1. **Embed**: look up each token's row, `[B*C, chunk_size, embed_dim]`. Pad positions are
//!    zeroed and spatial dropout drops whole embedding channels (training only).
//! 2. **Group**: reshape so every `group` adjacent tokens become one position carrying their
//!    concatenated embeddings, `[B*C, chunk_size/group, group*embed_dim]` — the trunk's first
//!    linear layer therefore sees local token n-grams, which is why an explicit n-gram
//!    vocabulary is not needed.
//! 3. **Trunk**: a two-layer MLP over every group position, shared across all chunks and
//!    positions — more chunks cost compute, never parameters.
//! 4. **Chunk pooling**: masked max over the group positions gives one vector per chunk
//!    (padding is filled with `-inf` first so it can never win the max), then `LayerNorm`.
//! 5. **Chunk head**: one logit per chunk — the training-time auxiliary signal; each chunk
//!    inherits its document's label, noisy but measurably useful.
//! 6. **Document head**: two pooled views of the chunk vectors concatenated — attention
//!    (which chunks matter) and masked max (one decisive chunk suffices) — then a linear head.
//!    This *is* the chunk aggregation, learned end to end; nothing outside the network combines
//!    chunk scores into the document verdict.
//!
//! Outputs are raw logits: `sigmoid` gives the AI probability, and the decision threshold is a
//! bundle artifact fitted after training, never part of the network. The chunk axis is dynamic —
//! `max_chunks` is a training-time budget, not a model limit — because the trunk is per-chunk
//! and both document pools are masked reductions. Padding is masked out of every pooling step: a
//! global max over mostly-pad positions would otherwise let padding decide the representation.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

/// A configuration the network cannot be built from.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelConfigError {
    /// The trunk groups adjacent tokens, so a chunk must hold a whole number of groups.
    #[error("chunk_size {chunk_size} is not a positive multiple of group {group}")]
    ChunkSizeNotGrouped {
        /// Tokens per chunk.
        chunk_size: usize,
        /// Tokens per trunk position.
        group: usize,
    },
}

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

    /// Check the shape constraints a configuration read from a bundle may violate.
    pub fn validate(&self) -> Result<(), ModelConfigError> {
        if self.group == 0 || self.chunk_size == 0 || !self.chunk_size.is_multiple_of(self.group) {
            return Err(ModelConfigError::ChunkSizeNotGrouped {
                chunk_size: self.chunk_size,
                group: self.group,
            });
        }
        Ok(())
    }

    /// Initialize the model on the given device.
    ///
    /// # Panics
    ///
    /// Panics if the configuration fails [`Self::validate`]; validate first when the
    /// configuration comes from a file.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HierarchicalSlopClassifier<B> {
        if let Err(e) = self.validate() {
            panic!("{e}");
        }
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

/// See the [module documentation](self) for the stage-by-stage dataflow.
#[derive(Module, Debug)]
pub struct HierarchicalSlopClassifier<B: Backend> {
    /// Stage 1: token rows, `[vocab_size + 2, embed_dim]` — ~90% of the parameters.
    embed: Embedding<B>,
    /// Stage 1: channel dropout on the embedding, shared across positions.
    spatial_dropout: SpatialDropout1d,
    /// Stage 3: first trunk layer, `group * embed_dim -> hidden` — reads grouped token n-grams.
    trunk_in: Linear<B>,
    /// Stage 3: trunk normalization.
    trunk_norm: LayerNorm<B>,
    /// Stage 3: elementwise dropout inside the trunk.
    trunk_dropout: Dropout,
    /// Stage 3: second trunk layer, `hidden -> hidden`.
    trunk_out: Linear<B>,
    /// Stage 3: activation after each trunk layer.
    gelu: Gelu,
    /// Stage 4: normalizes the max-pooled chunk vectors.
    chunk_norm: LayerNorm<B>,
    /// Stage 5: per-chunk logit, `hidden -> 1` — the auxiliary loss head.
    chunk_head: Linear<B>,
    /// Stage 6: attention scores over chunks, `hidden -> 1`.
    attend: Linear<B>,
    /// Stage 6: normalizes the concatenated attention + max document vector.
    doc_norm: LayerNorm<B>,
    /// Stage 6: dropout before the document head.
    doc_dropout: Dropout,
    /// Stage 6: document logit, `2 * hidden -> 1`.
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
    /// Stages 1-4: encode ids `[batch, chunks, chunk_size]` into chunk vectors
    /// `[batch, chunks, hidden]` plus a real-chunk mask `[batch, chunks]` — embed, group, run
    /// the shared trunk, and max-pool each chunk.
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

    /// Stages 5-6: produce document logits `[batch]`, chunk logits `[batch, chunks]`, and the
    /// real-chunk mask — the chunk head plus the attention+max document pooling.
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
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;

    type B = Flex;

    #[test]
    fn forward_shapes_and_finiteness() {
        let device = FlexDevice;
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
        let model = HierarchicalModelConfig::new().init::<B>(&FlexDevice);
        assert_eq!(model.num_params(), 1_150_595);
    }
}
