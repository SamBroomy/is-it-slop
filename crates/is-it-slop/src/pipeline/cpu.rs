//! Scheduling on a CPU backend (Flex, burn-cpu or ndarray).
//!
//! A Flex forward runs on one thread and costs linear in chunk slots, so putting documents
//! together in a pass only adds padding, while independent passes on rayon's pool scale with
//! cores. Measured on the real-corpus bench (`just bench-inference`): one document per pass with
//! concurrency beat every batched setting, and batching 32 documents per pass ran at a third of
//! its throughput.

use super::Strategy;

/// One document per forward pass, passes concurrent.
pub const STRATEGY: Strategy = Strategy::new(1, true);
