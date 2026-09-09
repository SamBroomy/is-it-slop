//! Inference throughput on the compiled backend; group names carry the backend so logs compare.
//!
//! - `window`: documents per second when `predict_each` (the backend's default `Strategy`) is
//!   handed the corpus in windows of N documents, the way the CLI streams. Windows above the
//!   corpus size are skipped, so the 256 and 512 points need a larger corpus.
//! - `strategy`: documents per second over the whole corpus for each `Strategy`, sweeping documents
//!   per forward pass against concurrent passes. This is where a backend's default comes from.
//! - `input_size`: wall time of one `predict_each` call over the first N documents (the latency a
//!   caller with N documents sees, and where CPU and GPU builds cross over).
//! - `document_length`: one forward's cost against the document's chunk count.
//!
//! The corpus is a fixed random sample of the released test split (`corpus.jsonl`, 128 documents,
//! both labels, lengths from a sentence to about fifteen chunks). Point `IS_IT_SLOP_BENCH_CORPUS`
//! at another JSONL file with a `text` field per line to measure different text.
//!
//! Under `cargo test` (no `--bench` flag) the corpus shrinks so every group runs once as a smoke
//! test.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use is_it_slop::Predictor;
use is_it_slop::model::BACKEND;
use is_it_slop::pipeline::Strategy;

const WINDOWS: [usize; 9] = [1, 4, 8, 16, 32, 64, 128, 256, 512];
const DOCS_PER_FORWARD: [usize; 8] = [1, 4, 16, 32, 64, 128, 256, 512];
/// Call sizes for the `input_size` group.
const INPUT_SIZES: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Chunk counts the `document_length` group looks for; each is served by the corpus document
/// whose real chunk count is nearest.
const TARGET_CHUNKS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(serde::Deserialize)]
struct Record {
    text: String,
}

fn parse_corpus(jsonl: &str, what: &str) -> Vec<String> {
    let docs: Vec<String> = jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Record>(line)
                .unwrap_or_else(|e| panic!("{what}: bad record {line:.60}: {e}"))
                .text
        })
        .collect();
    assert!(!docs.is_empty(), "{what} has no records");
    docs
}

fn corpus(full: bool) -> Vec<String> {
    let mut docs = std::env::var("IS_IT_SLOP_BENCH_CORPUS").ok().map_or_else(
        || parse_corpus(include_str!("corpus.jsonl"), "benches/corpus.jsonl"),
        |path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading IS_IT_SLOP_BENCH_CORPUS={path}: {e}"));
            parse_corpus(&text, &format!("IS_IT_SLOP_BENCH_CORPUS={path}"))
        },
    );
    if !full {
        docs.truncate(8);
    }
    docs
}

fn full_run() -> bool {
    std::env::args().any(|a| a == "--bench")
}

/// The model loaded and every corpus shape already seen once: GPU backends autotune per tensor
/// shape, so a single warm-up prediction would leave the first sample of each shape paying for it.
fn warmed(corpus: &[String]) -> Predictor {
    let predictor = Predictor::new();
    // Small and large calls, so an accelerated build loads both its models.
    black_box(predictor.predict_each(&corpus[..1]));
    black_box(predictor.predict_each(corpus));
    black_box(predictor.predict_each_with(corpus, Strategy::new(usize::MAX, false)));
    predictor
}

fn window(c: &mut Criterion) {
    let corpus = corpus(full_run());
    let predictor = warmed(&corpus);

    let mut group = c.benchmark_group(format!("{BACKEND}/window"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .throughput(Throughput::Elements(corpus.len() as u64));
    for &size in WINDOWS.iter().filter(|&&size| size <= corpus.len()) {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                for batch in corpus.chunks(size) {
                    black_box(predictor.predict_each(batch));
                }
            });
        });
    }
    group.finish();
}

/// Wall time of one `predict_each` call over the first N documents under the backend's default
/// `Strategy`: the latency curve a caller with N documents in hand sees, and where a CPU and a GPU
/// build cross over.
fn input_size(c: &mut Criterion) {
    let corpus = corpus(full_run());
    let predictor = warmed(&corpus);

    let mut group = c.benchmark_group(format!("{BACKEND}/input_size"));
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for &n in INPUT_SIZES.iter().filter(|&&n| n <= corpus.len()) {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(predictor.predict_each(&corpus[..n])));
        });
    }
    group.finish();
}

fn strategy(c: &mut Criterion) {
    let corpus = corpus(full_run());
    let predictor = warmed(&corpus);

    let mut group = c.benchmark_group(format!("{BACKEND}/strategy"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .throughput(Throughput::Elements(corpus.len() as u64));
    for &docs in DOCS_PER_FORWARD
        .iter()
        .filter(|&&docs| docs <= corpus.len())
    {
        for concurrent in [false, true] {
            let strategy = Strategy::new(docs, concurrent);
            let id = format!("{docs}x{}", if concurrent { "par" } else { "seq" });
            group.bench_with_input(
                BenchmarkId::from_parameter(id),
                &strategy,
                |b, &strategy| {
                    b.iter(|| black_box(predictor.predict_each_with(&corpus, strategy)));
                },
            );
        }
    }
    group.finish();
}

fn document_length(c: &mut Criterion) {
    let corpus = corpus(full_run());
    let predictor = warmed(&corpus);
    let chunk_counts: Vec<usize> = corpus
        .iter()
        .map(|text| predictor.chunk_texts(text).len())
        .collect();

    // The document nearest each target; the id carries its real chunk count.
    let mut picked: Vec<(usize, &String)> = TARGET_CHUNKS
        .iter()
        .map(|&target| {
            let (i, &chunks) = chunk_counts
                .iter()
                .enumerate()
                .min_by_key(|&(_, &chunks)| chunks.abs_diff(target))
                .expect("non-empty corpus");
            (chunks, &corpus[i])
        })
        .collect();
    picked.sort_by_key(|&(chunks, _)| chunks);
    picked.dedup_by_key(|&mut (chunks, _)| chunks);

    let mut group = c.benchmark_group(format!("{BACKEND}/document_length"));
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for (chunks, text) in picked {
        group.bench_with_input(BenchmarkId::new("chunks", chunks), text, |b, text| {
            b.iter(|| black_box(predictor.predict(text)));
        });
    }
    group.finish();
}

criterion_group!(benches, window, input_size, strategy, document_length);
criterion_main!(benches);
