//! Batching for training.
//!
//! Only the batching seam lives here so far: [`EncodedDocument`] is what the future data layer
//! (parquet reading, vocabulary application, chunk encoding via the preprocessing crate) will
//! produce, and [`SlopBatcher`] turns a list of them into the tensors the steps consume.

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::training::Batch;

/// One document, already encoded to the rectangular `[max_chunks * chunk_size]` grid of compact
/// token ids (padding included), plus its label.
#[derive(Debug, Clone)]
pub struct EncodedDocument {
    /// Compact token ids, row-major `[max_chunks, chunk_size]`.
    pub ids: Vec<i64>,
    /// 1.0 for AI, 0.0 for human.
    pub label: f32,
}

/// Stacks encoded documents into `[batch, max_chunks, chunk_size]` id tensors.
#[derive(Debug, Clone)]
pub struct SlopBatcher {
    /// Training-time chunk budget; every document carries this many rows.
    pub max_chunks: usize,
    /// Tokens per chunk.
    pub chunk_size: usize,
}

impl<B: Backend> Batcher<B, EncodedDocument, Batch<B>> for SlopBatcher {
    fn batch(&self, items: Vec<EncodedDocument>, device: &B::Device) -> Batch<B> {
        let n = items.len();
        let mut ids = Vec::with_capacity(n * self.max_chunks * self.chunk_size);
        let mut labels = Vec::with_capacity(n);
        for item in items {
            assert_eq!(
                item.ids.len(),
                self.max_chunks * self.chunk_size,
                "document grid does not match the batcher's shape"
            );
            ids.extend_from_slice(&item.ids);
            labels.push(item.label);
        }
        Batch {
            ids: Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([
                n,
                self.max_chunks,
                self.chunk_size,
            ]),
            labels: Tensor::<B, 1>::from_floats(labels.as_slice(), device),
        }
    }
}
