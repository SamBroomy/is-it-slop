//! Scheduling on an accelerator backend (`cfg(accelerated)`: cuda, rocm, metal, vulkan, webgpu, wgpu).
//!
//! A GPU pass costs a near-constant launch and sync overhead (~6 ms on Apple silicon via Metal,
//! whether the document has 1 chunk or 12) plus very little per chunk, and passes serialise on
//! the device queue, so concurrency across passes buys almost nothing while many documents per
//! pass amortise the overhead. Length-sorted grouping (see `schedule`) keeps the padding that
//! batching adds small. The setting below is the knee of the `strategy` sweep in
//! `just bench-inference-gpu` on an Apple M-series GPU over the real corpus: 1, 16, 32, 64 and
//! 128 documents per pass gave 158, 2170, 3590, 3820 and 3100 docs/s sequentially, and running
//! passes concurrently added 3% at best. Past 64 the padding to the longest document in the pass
//! outweighs the saved launches.

use super::Strategy;

/// Many documents per forward pass, passes one after another.
pub const STRATEGY: Strategy = Strategy::new(64, false);
