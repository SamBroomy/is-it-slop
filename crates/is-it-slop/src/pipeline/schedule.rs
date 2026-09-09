//! How a batch of documents is spread over forward passes and threads.
//!
//! The two knobs trade against each other differently per backend, which is why the default lives
//! in the backend module (`cpu` or `accelerated`) and the bench sweeps both.

use rayon::prelude::*;

use super::{MAX_CHUNKS_PER_DOCUMENT, UnifiedPrediction, forward};
use crate::model::ENCODER;

/// How many documents share a forward pass, and whether passes run concurrently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strategy {
    /// Documents per forward pass; a pass is padded to its longest document.
    pub docs_per_forward: usize,
    /// Run the passes on rayon's global pool instead of one after another.
    pub concurrent: bool,
}

impl Strategy {
    /// One document per pass, passes one after another.
    pub const SEQUENTIAL: Self = Self::new(1, false);

    /// Build a strategy; `docs_per_forward` must be at least one.
    #[must_use]
    pub const fn new(docs_per_forward: usize, concurrent: bool) -> Self {
        assert!(docs_per_forward > 0, "a forward pass needs a document");
        Self {
            docs_per_forward,
            concurrent,
        }
    }
}

impl Default for Strategy {
    /// The compiled backend's policy.
    fn default() -> Self {
        super::backend::STRATEGY
    }
}

/// Score non-blank texts under `strategy`, results in input order.
pub fn run(texts: &[&str], strategy: Strategy) -> Vec<UnifiedPrediction> {
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
        forward::run(&batch)
    };
    let scored: Vec<Vec<UnifiedPrediction>> = if strategy.concurrent {
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
