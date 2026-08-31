//! The data layer: split reading, vocabulary, encoding, and batching.
//!
//! Mirrors the Python pipeline's data path (`load_split` → `build_vocab` → `encode_corpus` →
//! `DataLoader`): read a released parquet split, rank a vocabulary on the training split, encode
//! every document once up front into a compact grid, and let Burn's `DataLoader` shuffle and
//! batch the encoded documents each epoch. Python cached encodings on disk because re-encoding
//! there was slow; here encoding is a one-off parallel pass at startup.

mod encoder;
mod splits;
mod vocabulary;

pub use encoder::DocumentEncoder;
pub use splits::{SplitData, read_split};
pub use vocabulary::{O200K_CEILING, Vocabulary};

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::training::Batch;

/// Failures loading or preparing data.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// Filesystem failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Parquet reading failure.
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
    /// Vocabulary (de)serialization failure.
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// The data does not match the artifact contract.
    #[error("{0}")]
    Contract(String),
}

/// One document, encoded to the rectangular `[max_chunks * chunk_size]` grid of compact token
/// ids (padding included), plus its label.
///
/// Ids are `u16`: the vocabulary asserts `vocab_size + 2` fits, and the compact width keeps a
/// full encoded split ~4x smaller in memory than wide ids would.
#[derive(Debug, Clone)]
pub struct EncodedDocument {
    /// Compact token ids, row-major `[max_chunks, chunk_size]`.
    pub ids: Vec<u16>,
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
