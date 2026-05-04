//! Inference pipeline profiling example
//!
//! Validates chunking and profiles batch inference across sizes to find the
//! optimal batch size for the ONNX session's dense matrix allocation.
//!
//! Outputs `onnx_rows_per_call` alongside throughput — this is the actual
//! number of rows sent to ONNX each call (texts × `mean_chunks_per_text`) and
//! directly explains why large batches regress: `to_dense()` has to allocate
//! an enormous contiguous matrix at once.
//!
//! Batch sweep: 1, 5, 10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400,
//!              500, 750, 1000, 2000, 5000, full
//!
//! Run with samply:
//!   samply record `target/profiling/examples/inference_profile`
//!
//! Run with flamegraph:
//!   cargo flamegraph --profile profiling --example `inference_profile` --features cli
//!
//! Run for timing only:
//!   cargo run --profile profiling --example `inference_profile` --features cli
//!
//! Control sample count (default 2000):
//!   `PROFILE_SAMPLES=5000` cargo run --profile profiling --example `inference_profile` --features
//! cli

use std::{path::PathBuf, time::Instant};

use is_it_slop::Predictor;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// ── Text-length buckets ───────────────────────────────────────────────────────
// Char-count thresholds — cheap proxy for token count, fine for bucketing.
const SHORT_MAX_CHARS: usize = 500;
const MEDIUM_MAX_CHARS: usize = 2_000;
// > MEDIUM_MAX_CHARS → long
const BUCKET_BATCH: usize = 100; // adjust after reading optimal from sweep above
struct Bucket<'a> {
    label: &'static str,
    texts: &'a [String],
}

// ── Profiling scenarios ───────────────────────────────────────────────────────
// `#[inline(never)]` preserves each function as a named frame in the profiler.

/// One text at a time, fully sequential.
#[inline(never)]
fn profile_single_sequential(predictor: &Predictor, texts: &[String]) -> u32 {
    let mut sink = 0u32;
    for text in texts {
        let p = predictor.predict(text).expect("predict failed");
        sink ^= p.prediction.ai_probability().to_bits();
    }
    sink
}

/// Fixed-size batches — covers the middle ground between 1 and full.
#[inline(never)]
fn profile_batch(predictor: &Predictor, texts: &[String], batch_size: usize) -> u32 {
    let mut sink = 0u32;
    for chunk in texts.chunks(batch_size) {
        for p in predictor
            .predict_batch(chunk)
            .expect("predict_batch failed")
        {
            sink ^= p.prediction.ai_probability().to_bits();
        }
    }
    sink
}

/// Entire dataset in one call — maximum batch parallelism.
#[inline(never)]
fn profile_batch_full(predictor: &Predictor, texts: &[String]) -> u32 {
    let mut sink = 0u32;
    for p in predictor
        .predict_batch(texts)
        .expect("predict_batch failed")
    {
        sink ^= p.prediction.ai_probability().to_bits();
    }
    sink
}

// ── Chunk validation ─────────────────────────────────────────────────────────

/// Run a sample of texts through `predict`, report chunk counts, assert chunking
/// is happening for long texts, and return mean chunks per text for use in
/// ONNX-row estimates during the batch sweep.
fn validate_chunking(predictor: &Predictor, texts: &[String]) -> f64 {
    const SAMPLE: usize = 10;

    info!("=== chunk validation ({SAMPLE} sample texts) ===");

    // Use a capped sample for per-text logging; full set for aggregate stats.
    let sample_texts = texts.iter().take(SAMPLE.min(texts.len()));
    for (i, text) in sample_texts.enumerate() {
        let n_chunks = predictor
            .predict(text)
            .expect("predict failed")
            .chunk_predictions
            .len();
        info!(sample = i, chars = text.len(), n_chunks, "chunked");
    }

    // Aggregate stats over the first 200 texts (cheap, representative).
    let stat_texts = &texts[..texts.len().min(200)];
    let chunk_counts: Vec<usize> = stat_texts
        .iter()
        .map(|t| {
            predictor
                .predict(t)
                .expect("predict failed")
                .chunk_predictions
                .len()
        })
        .collect();

    let min_chunks = *chunk_counts.iter().min().unwrap();
    let max_chunks = *chunk_counts.iter().max().unwrap();
    let mean_chunks = chunk_counts.iter().sum::<usize>() as f64 / stat_texts.len() as f64;
    let multi_chunk = chunk_counts.iter().filter(|&&c| c > 1).count();

    info!(
        min_chunks,
        max_chunks,
        mean_chunks = format!("{mean_chunks:.2}"),
        multi_chunk_texts = multi_chunk,
        sampled = stat_texts.len(),
        "chunk stats"
    );

    let long_enough = stat_texts.iter().filter(|t| t.len() > 1_000).count();
    if long_enough > 0 {
        assert!(
            multi_chunk > 0,
            "Expected multi-chunk predictions for texts >1000 chars but all have 1 chunk — \
             chunking may be broken."
        );
        info!("chunking confirmed: {multi_chunk}/{long_enough} eligible texts produce >1 chunk");
    } else {
        info!("all sampled texts are short — single-chunk results expected");
    }

    mean_chunks
}

// ── Timing helper ─────────────────────────────────────────────────────────────

struct TextStats {
    #[allow(dead_code)]
    total: usize,
    min: usize,
    max: usize,
    mean: f64,
}

fn text_stats(texts: &[String]) -> TextStats {
    let chars: Vec<usize> = texts.iter().map(String::len).collect();
    let total_chars = chars.iter().sum();
    TextStats {
        total: total_chars,
        min: *chars.iter().min().unwrap_or(&0),
        max: *chars.iter().max().unwrap_or(&0),
        mean: total_chars as f64 / texts.len() as f64,
    }
}

/// `batch_size` is how many texts per `predict_batch` call (None = single/full).
/// `mean_chunks` is used to estimate how many rows each ONNX call receives.
fn time_scenario<F>(
    label: &str,
    texts: &[String],
    batch_size: Option<usize>,
    mean_chunks: f64,
    f: F,
) where
    F: FnOnce() -> u32,
{
    let stats = text_stats(texts);
    let n = texts.len();
    let onnx_rows_per_call = batch_size.unwrap_or(n) as f64 * mean_chunks;
    let start = Instant::now();
    let _sink = f();
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    info!(
        scenario = label,
        n,
        total_ms = elapsed.as_millis(),
        avg_ms = format!("{:.2}", elapsed.as_millis() as f64 / n as f64),
        texts_per_sec = format!("{:.0}", n as f64 / secs),
        onnx_rows_per_call = format!("{:.0}", onnx_rows_per_call),
        mean_chars = format!("{:.0}", stats.mean),
        "done"
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_level(true)
                .with_timer(tracing_subscriber::fmt::time::uptime()),
        )
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "inference_profile=info,is_it_slop=info".into()),
        )
        .init();

    info!("=== Inference Pipeline Profiler ===");

    let predictor = {
        let t = Instant::now();
        let p = Predictor::new();
        info!(elapsed_ms = t.elapsed().as_millis(), "predictor ready");
        p
    };

    let max_samples: usize = std::env::var("PROFILE_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let all_texts = load_data(max_samples);

    // ── Text length stats ────────────────────────────────────────────────────
    let short: Vec<String> = all_texts
        .iter()
        .filter(|t| t.len() <= SHORT_MAX_CHARS)
        .cloned()
        .collect();
    let medium: Vec<String> = all_texts
        .iter()
        .filter(|t| t.len() > SHORT_MAX_CHARS && t.len() <= MEDIUM_MAX_CHARS)
        .cloned()
        .collect();
    let long: Vec<String> = all_texts
        .iter()
        .filter(|t| t.len() > MEDIUM_MAX_CHARS)
        .cloned()
        .collect();

    let ts = text_stats(&all_texts);
    info!(
        total = all_texts.len(),
        short = short.len(),
        medium = medium.len(),
        long = long.len(),
        mean_chars = format!("{:.0}", ts.mean),
        min_chars = ts.min,
        max_chars = ts.max,
        "data loaded"
    );

    // ── Warm-up ──────────────────────────────────────────────────────────────
    // Not timed — just ensures the ONNX session and any lazy state are hot.
    info!("warming up...");
    let _ = predictor.predict_batch(all_texts.iter().take(10).collect::<Vec<_>>().as_slice());

    // ── Chunk validation ─────────────────────────────────────────────────────
    let mean_chunks = validate_chunking(&predictor, &all_texts);

    // ── Batch-size sweep ─────────────────────────────────────────────────────
    // Goal: find the onnx_rows_per_call sweet spot before to_dense() allocation
    // cost outweighs ONNX batching gains. Each text produces ~mean_chunks rows.
    info!("=== starting profiling ===");

    time_scenario("single/sequential", &all_texts, None, mean_chunks, || {
        profile_single_sequential(&predictor, &all_texts)
    });

    // Coarse sweep 1–50 to anchor the low end.
    for &bs in &[1usize, 5, 10, 25, 50] {
        if all_texts.len() >= bs {
            let label = format!("batch/{bs}");
            time_scenario(&label, &all_texts, Some(bs), mean_chunks, || {
                profile_batch(&predictor, &all_texts, bs)
            });
        }
    }

    // Fine sweep 75–300 — expected optimal zone.
    for &bs in &[75usize, 100, 125, 150, 175, 200, 250, 300] {
        if all_texts.len() >= bs {
            let label = format!("batch/{bs}");
            time_scenario(&label, &all_texts, Some(bs), mean_chunks, || {
                profile_batch(&predictor, &all_texts, bs)
            });
        }
    }

    // Coarse sweep 400–5000 to show the decline.
    for &bs in &[400usize, 500, 750, 1000, 2000, 5000] {
        if all_texts.len() >= bs {
            let label = format!("batch/{bs}");
            time_scenario(&label, &all_texts, Some(bs), mean_chunks, || {
                profile_batch(&predictor, &all_texts, bs)
            });
        }
    }

    // Full batch — expected to regress badly for large datasets.
    time_scenario("batch/full", &all_texts, None, mean_chunks, || {
        profile_batch_full(&predictor, &all_texts)
    });

    // ── Per-length bucket with optimal-ish batch size ─────────────────────────
    // Shows whether the optimal batch size differs per text length / chunk count.

    for bucket in [
        Bucket {
            label: "bucket/short",
            texts: &short,
        },
        Bucket {
            label: "bucket/medium",
            texts: &medium,
        },
        Bucket {
            label: "bucket/long",
            texts: &long,
        },
    ] {
        if bucket.texts.is_empty() {
            info!(bucket = bucket.label, "skipped (no samples)");
            continue;
        }
        let label_seq = format!("{}/sequential", bucket.label);
        time_scenario(&label_seq, bucket.texts, None, mean_chunks, || {
            profile_single_sequential(&predictor, bucket.texts)
        });
        if bucket.texts.len() >= BUCKET_BATCH {
            let label_batch = format!("{}/batch{BUCKET_BATCH}", bucket.label);
            time_scenario(
                &label_batch,
                bucket.texts,
                Some(BUCKET_BATCH),
                mean_chunks,
                || profile_batch(&predictor, bucket.texts, BUCKET_BATCH),
            );
        }
    }

    info!("=== profiling complete ===");
}

// ── Data loading ──────────────────────────────────────────────────────────────

fn load_data(max_samples: usize) -> Vec<String> {
    let path = PathBuf::from("profile/test.csv");

    assert!(
        path.exists(),
        "profile/test.csv not found — place a CSV with a text/generation/content/body column there"
    );

    let mut reader = csv::Reader::from_path(&path).expect("valid CSV");
    let headers = reader.headers().expect("CSV has headers").clone();

    let text_col_idx = headers
        .iter()
        .position(|name| {
            matches!(
                name.to_lowercase().as_str(),
                "text" | "generation" | "content" | "body"
            )
        })
        .unwrap_or(1);

    info!(
        text_column = headers.get(text_col_idx).unwrap_or("?"),
        "using text column"
    );

    let texts: Vec<String> = reader
        .records()
        .filter_map(Result::ok)
        .map(|r| r.get(text_col_idx).unwrap_or("").trim().to_string())
        .filter(|s| !s.is_empty())
        .take(max_samples)
        .collect();

    info!(num_texts = texts.len(), "loaded texts");
    texts
}
