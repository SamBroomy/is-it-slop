//! The data layer: split reading, vocabulary, encoding, and batching.
//!
//! Mirrors the Python pipeline's data path (`load_split` → `build_vocab` → `encode_corpus` →
//! `DataLoader`): read a released parquet split, rank a vocabulary on the training split, encode
//! every document once up front into a compact grid, and let Burn's `DataLoader` shuffle and
//! batch the encoded documents each epoch. Python cached encodings on disk because re-encoding
//! there was slow; here encoding is a one-off parallel pass at startup.

mod splits;

use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{Int, Tensor, backend::Backend},
};
pub use is_it_slop_preprocessing::encoder::{
    DocumentEncoder, EncodedDocument, EncoderBundleConfig, EncoderError, O200K_CEILING, Vocabulary,
};
pub use splits::{SplitData, read_split};

use crate::training::Batch;

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
            ids.extend(item.ids.iter().map(|&id| i64::from(id)));
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
