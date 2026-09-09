//! Scheduling on an accelerator backend (`cfg(accelerated)`: cuda, rocm, metal, vulkan, webgpu, wgpu).
//!
//! A GPU pass costs a near-constant launch and sync overhead (6-8 ms on Apple silicon, whether the
//! document has 1 chunk or 16) plus very little per chunk, and passes serialise on the device queue.
//! So many documents per pass amortise the overhead, concurrency across passes only contends (25-40%
//! slower at 32-64 per pass on 512 documents), and length-sorted grouping (see `schedule`) keeps the
//! padding batching adds small. The `strategy` sweep in `just bench-inference-gpu` over the real
//! corpus put the knee at 64 per pass: 1 / 16 / 32 / 64 / 128 gave 172 / 2465 / 4294 / 4882 / 4639
//! docs/s on Metal, with the same shape on wgpu and webgpu.
//!
//! The launch cost also means a small call is faster on the CPU: on the same machine one document
//! took 6.1 ms on Metal and 2.0-3.6 ms on the CPU backends, and the CPU stayed ahead or level up to
//! about 256 documents per call. Calls below [`CPU_UP_TO`] therefore run on the CPU copy of the
//! model under the CPU policy. The crossover is hardware-specific; this one is from an Apple
//! M-series machine.

use super::{Strategy, UnifiedPrediction, cpu, forward, schedule};

/// Many documents per forward pass, passes one after another.
pub const STRATEGY: Strategy = Strategy::new(64, false);

/// Calls with fewer documents than this run on the CPU instead.
pub const CPU_UP_TO: usize = 256;

/// Score non-blank texts, results in input order: small calls on the CPU, large ones on the GPU.
pub fn infer_each(texts: &[&str]) -> Vec<UnifiedPrediction> {
    if texts.len() < CPU_UP_TO {
        schedule::run(texts, cpu::STRATEGY, forward::run_on_cpu)
    } else {
        schedule::run(texts, STRATEGY, forward::run)
    }
}
