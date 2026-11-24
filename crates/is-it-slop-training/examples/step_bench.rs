//! Per-iteration cost decomposition for the training loop, on the selected backend.
//!
//! Times a raw loop over one resident synthetic batch — no dataloader, no Learner, no metrics —
//! so the full pipeline's iteration time can be apportioned between model compute and harness.
//! `notebooks/step_bench.py` is the torch-MPS mirror; results from both live in EXPERIMENTS.md
//! ("Training throughput"). Variants, each 50 warmup + 300 timed iterations
//! (`STEP_BENCH_WARMUP`/`STEP_BENCH_ITERS` override, `STEP_BENCH_ONLY` picks one):
//!
//! - `fwd`: train-mode forward + loss, synced per iteration via `loss.into_scalar::<f32>()`. Broken
//!   in Burn 0.21: a forward-only loop under `Autodiff` leaks until macOS SIGKILLs the process
//!   (~130 iterations), so run this one alone and short if at all.
//! - `fwd_inner`: inference-mode forward on the inner backend (no autodiff, dropout off) — compare
//!   against a torch `eval()` + `no_grad()` forward.
//! - `fwd_bwd`: forward + backward, no optimizer step.
//! - `step`: forward + backward + optimizer step, same scalar-loss sync — the Python train loop's
//!   shape exactly (`loss.item()` every step).
//! - `step+pull`: `step` plus pulling logits and targets to host each iteration — an upper bound on
//!   the `ItemLazy` sync the Learner's metric pipeline performs per step.
//! - `step+h2d`: `step` but rebuilding the ids tensor from host data each iteration — adds the
//!   host-to-device transfer the dataloader path pays (not the CPU-side batch assembly, which
//!   workers overlap).
//!
//! Run with the backend under test, never two benches concurrently (GPU contention), e.g.:
//!
//! ```sh
//! STEP_BENCH_BACKEND=metal cargo run --release -p is-it-slop-training --example step_bench --features metal
//! ```
//!
//! `STEP_BENCH_BACKEND` names one of the backends compiled in (default: the best one).

#[path = "../src/backend.rs"]
#[allow(dead_code)]
mod backend;

use std::{hint::black_box, time::Instant};

use burn::{
    module::Module,
    optim::GradientsParams,
    tensor::{Device, Int, Tensor, TensorData},
};
use is_it_slop_training::{
    model::HierarchicalModelConfig,
    training::{Batch, RankGatedAdamW, TrainableClassifier, TrainingConfig},
};

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| {
        v.parse().expect("iteration counts are integers")
    })
}

fn synthetic_batch(
    config: &HierarchicalModelConfig,
    batch_size: usize,
    max_chunks: usize,
    device: &Device,
) -> (Vec<i32>, Batch) {
    // Deterministic LCG so runs are comparable; ids uniform over the vocabulary, no padding.
    let mut state = 0x853c_49e6_748f_ea9bu64;
    let n = batch_size * max_chunks * config.chunk_size;
    let ids: Vec<i32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % config.vocab_size as u64) as i32
        })
        .collect();
    let labels: Vec<f32> = (0..batch_size).map(|i| (i % 2) as f32).collect();
    let batch = Batch {
        ids: Tensor::from_data(
            TensorData::new(ids.clone(), [batch_size, max_chunks, config.chunk_size]),
            device,
        ),
        labels: Tensor::from_data(TensorData::new(labels, [batch_size]), device),
    };
    (ids, batch)
}

fn bench(label: &str, mut iter: impl FnMut()) {
    // STEP_BENCH_ONLY runs a single variant: a long forward-only loop under Autodiff dies on
    // Metal (see EXPERIMENTS.md), so isolating variants keeps the others measurable.
    if let Ok(only) = std::env::var("STEP_BENCH_ONLY")
        && only != label
    {
        return;
    }
    let warmup = env_count("STEP_BENCH_WARMUP", 50);
    let iters = env_count("STEP_BENCH_ITERS", 300);
    for _ in 0..warmup {
        iter();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        iter();
    }
    let per_iter = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    // ms per iteration is numerically the same as seconds per 1000 iterations.
    println!("{label:<10} {per_iter:>8.2} ms/iter  ({per_iter:.1} s/1000)");
}

fn main() {
    let backend = std::env::var("STEP_BENCH_BACKEND").map_or_else(
        |_| backend::Backend::best(),
        |name| {
            backend::Backend::from_name(&name)
                .expect("STEP_BENCH_BACKEND names a compiled-in backend")
        },
    );
    // Autodiff is a property of the device now: tensors and models created on it record a graph.
    let device = backend.device().autodiff();
    let model_config = HierarchicalModelConfig::new();
    let config = TrainingConfig::new();
    println!(
        "backend={} v{}-d{} batch={} [b, 8, {}]",
        backend.name(),
        model_config.vocab_size,
        model_config.embed_dim,
        config.batch_size,
        model_config.chunk_size,
    );
    device.seed(config.seed);

    let (host_ids, batch) = synthetic_batch(&model_config, config.batch_size, 8, &device);
    let shape = [config.batch_size, 8, model_config.chunk_size];

    let trainable = TrainableClassifier {
        model: model_config.init(&device),
        chunk_loss_weight: config.chunk_loss_weight,
    };

    {
        let trainable = trainable.clone();
        let batch = batch.clone();
        bench("fwd", move || {
            let output = trainable.forward_step(batch.clone());
            black_box(output.loss.into_scalar::<f32>());
        });
    }

    {
        // Inference-mode forward on the inner backend: no autodiff bookkeeping, dropout off.
        // Compare against a torch `eval()` + `no_grad()` forward, not the train-mode one.
        let model = trainable.model.valid();
        let ids = batch.ids.clone().without_autodiff();
        bench("fwd_inner", move || {
            let output = model.forward(ids.clone());
            black_box(output.doc_logits.mean().into_scalar::<f32>());
        });
    }

    {
        let trainable = trainable.clone();
        let batch = batch.clone();
        // No weight update, so iteration k+1 never depends on backward k — but the device queue
        // is in-order, so the per-iteration loss read still forces the previous backward to
        // finish. Off by one backward over the whole window, which amortizes to nothing.
        bench("fwd_bwd", move || {
            let output = trainable.forward_step(batch.clone());
            let grads = GradientsParams::from_grads(output.loss.backward(), &trainable);
            black_box(&grads);
            black_box(output.loss.into_scalar::<f32>());
        });
    }

    {
        let mut trainable = trainable.clone();
        let mut optim = RankGatedAdamW::init(config.weight_decay);
        let batch = batch.clone();
        bench("step", move || {
            let output = trainable.forward_step(batch.clone());
            let grads = GradientsParams::from_grads(output.loss.backward(), &trainable);
            trainable = optim.step(config.lr, trainable.clone(), grads);
            black_box(output.loss.into_scalar::<f32>());
        });
    }

    {
        let mut trainable = trainable.clone();
        let mut optim = RankGatedAdamW::init(config.weight_decay);
        let batch = batch.clone();
        bench("step+pull", move || {
            let output = trainable.forward_step(batch.clone());
            let grads = GradientsParams::from_grads(output.loss.backward(), &trainable);
            trainable = optim.step(config.lr, trainable.clone(), grads);
            black_box(output.output.into_data());
            black_box(output.targets.into_data());
            black_box(output.loss.into_scalar::<f32>());
        });
    }

    {
        let mut trainable = trainable;
        let mut optim = RankGatedAdamW::init(config.weight_decay);
        let labels = batch.labels;
        bench("step+h2d", move || {
            let fresh = Batch {
                ids: Tensor::<3, Int>::from_data(TensorData::new(host_ids.clone(), shape), &device),
                labels: labels.clone(),
            };
            let output = trainable.forward_step(fresh);
            let grads = GradientsParams::from_grads(output.loss.backward(), &trainable);
            trainable = optim.step(config.lr, trainable.clone(), grads);
            black_box(output.loss.into_scalar::<f32>());
        });
    }
}
