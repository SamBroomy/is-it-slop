//! Scheduling on a CPU backend (Flex, ndarray or burn-cpu).
//!
//! A CPU forward runs on one thread and costs linear in chunk slots, so putting documents together
//! in a pass only adds padding, while independent passes on rayon's pool scale with cores. On the
//! real-corpus bench (`just bench-inference`, 512 documents) one document per pass with concurrency
//! was the best setting on every CPU backend and batching degraded monotonically: Flex 3424 docs/s
//! against 441 at 64 per pass.

use super::Strategy;
#[cfg(not(accelerated))]
use super::{UnifiedPrediction, forward, schedule};

/// One document per forward pass, passes concurrent.
pub const STRATEGY: Strategy = Strategy::new(1, true);

/// Score non-blank texts, results in input order. An accelerated build routes its small calls
/// through [`STRATEGY`] on the CPU copy of the model instead (see `accelerated`).
#[cfg(not(accelerated))]
pub fn infer_each(texts: &[&str]) -> Vec<UnifiedPrediction> {
    schedule::run(texts, STRATEGY, forward::run)
}
