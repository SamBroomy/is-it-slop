//! How a batch of documents is spread over forward passes and threads.
//!
//! The two knobs trade against each other differently per backend, which is why the defaults live
//! in the backend module (`cpu` or `accelerated`) and the bench sweeps both.

use rayon::prelude::*;

use super::{MAX_CHUNKS_PER_DOCUMENT, UnifiedPrediction, forward::Forward};
use crate::model::ENCODER;

/// Whether forward passes run one after another or concurrently on rayon's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concurrency {
    /// One pass at a time.
    Sequential,
    /// Passes run on rayon's global pool.
    Concurrent,
}

/// How many documents share a forward pass, and how those passes are scheduled.
///
/// Built with [`Strategy::new`], which enforces at least one document per pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Strategy {
    /// Documents per forward pass; a pass is padded to its longest document.
    pub docs_per_forward: usize,
    /// How the passes are scheduled.
    pub concurrency: Concurrency,
}

impl Strategy {
    /// One document per pass, passes one after another.
    pub const SEQUENTIAL: Self = Self::new(1, Concurrency::Sequential);

    /// Build a strategy; `docs_per_forward` must be at least one.
    ///
    /// # Panics
    ///
    /// Panics if `docs_per_forward` is zero.
    #[must_use]
    pub const fn new(docs_per_forward: usize, concurrency: Concurrency) -> Self {
        assert!(docs_per_forward > 0, "a forward pass needs a document");
        Self {
            docs_per_forward,
            concurrency,
        }
    }
}

/// Score non-blank texts under `strategy`, one `forward` per group, results in input order.
pub(crate) fn run(texts: &[&str], strategy: Strategy, forward: Forward) -> Vec<UnifiedPrediction> {
    let chunk_size = ENCODER.chunk_size();
    let mut docs = ENCODER.encode_batch_chunks(texts);
    for doc in &mut docs {
        doc.truncate(MAX_CHUNKS_PER_DOCUMENT * chunk_size);
    }

    // Neighbours in length share a pass, so each pass pads as little as possible.
    let mut order: Vec<usize> = (0..docs.len()).collect();
    if strategy.docs_per_forward > 1 {
        order.sort_by_key(|&i| docs[i].len());
    }
    let groups: Vec<&[usize]> = order.chunks(strategy.docs_per_forward).collect();
    let score = |group: &[usize]| {
        let batch: Vec<&[u16]> = group.iter().map(|&i| docs[i].as_slice()).collect();
        forward(&batch)
    };
    let scored: Vec<Vec<UnifiedPrediction>> =
        if matches!(strategy.concurrency, Concurrency::Concurrent) {
            groups.par_iter().map(|group| score(group)).collect()
        } else {
            groups.iter().map(|group| score(group)).collect()
        };

    let mut out: Vec<Option<UnifiedPrediction>> = vec![None; texts.len()];
    for (group, predictions) in groups.iter().zip(scored) {
        for (&i, prediction) in group.iter().zip(predictions) {
            out[i] = Some(prediction);
        }
    }
    out.into_iter()
        .map(|prediction| prediction.expect("every document scored exactly once"))
        .collect()
}
