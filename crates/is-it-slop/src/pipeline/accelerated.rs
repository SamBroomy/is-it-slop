//! Scheduling on an accelerator backend (`cfg(accelerated)`: cuda, rocm, metal, vulkan, webgpu,
//! wgpu).
//!
//! A GPU pass costs a near-constant launch and sync overhead (6-8 ms on Apple silicon, whether the
//! document has 1 chunk or 16) plus very little per chunk, and passes serialise on the device
//! queue. So many documents per pass amortise the overhead, concurrency across passes only contends
//! (25-40% slower at 32-64 per pass on 512 documents), and length-sorted grouping (see `schedule`)
//! keeps the padding batching adds small. The `strategy` sweep in `just bench-inference-gpu` over
//! the real corpus put the knee at 64 per pass: 1 / 16 / 32 / 64 / 128 gave 172 / 2465 / 4294 /
//! 4882 / 4639 docs/s on Metal, with the same shape on wgpu and webgpu.
//!
//! The launch cost also means a small call is faster on the CPU, so calls below [`CPU_UP_TO`] run
//! on the CPU copy of the model under the CPU policy. Where the crossover sits depends on which CPU
//! backend is compiled in. Latency of one call in ms on the same machine (`input_size` group):
//!
//! | documents | Metal | Flex  | ndarray | ndarray + Accelerate |
//! | --------- | ----- | ----- | ------- | -------------------- |
//! | 1         | 6.1   | 3.6   | 2.7     | 2.0                  |
//! | 4         | 7.5   | 5.0   | 3.9     | 3.1                  |
//! | 8         | 8.9   | 9.0   | 7.1     | 5.6                  |
//! | 16        | 11.5  | 12.5  | 9.8     | 7.5                  |
//! | 128       | 45.6  | 52.3  | 42.9    | 34.7                 |
//! | 256       | 65.3  | 85.2  | 67.3    | 55.9                 |
//! | 512       | 103.7 | 165.7 | 130.1   | 106.8                |
//!
//! Flex ties Metal at 8 documents, ndarray at 256, and ndarray + Accelerate only at 512. The
//! numbers are hardware-specific (an Apple M-series GPU and CPU); a discrete GPU will cross lower.

use super::{Concurrency, Strategy, UnifiedPrediction, cpu, forward, schedule};

/// Many documents per forward pass, passes one after another.
pub const STRATEGY: Strategy = Strategy::new(64, Concurrency::Sequential);

/// Calls with fewer documents than this run on the CPU instead: the tie point of the compiled CPU
/// backend against the GPU in the table above.
pub const CPU_UP_TO: usize = if cfg!(feature = "accelerate") {
    512
} else if cfg!(feature = "ndarray") {
    256
} else {
    8
};

/// Score non-blank texts, results in input order: small calls on the CPU, large ones on the GPU.
pub fn infer_each(texts: &[&str]) -> Vec<UnifiedPrediction> {
    if texts.len() < CPU_UP_TO {
        schedule::run(texts, cpu::STRATEGY, forward::run_on_cpu)
    } else {
        schedule::run(texts, STRATEGY, forward::run)
    }
}
