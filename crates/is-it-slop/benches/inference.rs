//! Inference throughput against batch size and document length.
//!
//! Batch size is how many documents `predict_each` scores concurrently, one forward each, so the
//! `batch_size` group measures how throughput scales with the cores available; `document_length`
//! measures one forward's cost against its chunk count. The corpus is a fixed random sample of the
//! released test split (`corpus.jsonl`, 128 documents, both labels, lengths from a sentence to
//! about fifteen chunks; window sizes above the corpus size are skipped, so the 256 and 512 points
//! need a larger corpus). Point `IS_IT_SLOP_BENCH_CORPUS` at another JSONL file with a `text`
//! field per line to measure different text.
//!
//! Under `cargo test` (no `--bench` flag) the corpus shrinks so every group runs once as a smoke
//! test.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use is_it_slop::Predictor;

const BATCH_SIZES: [usize; 9] = [1, 4, 8, 16, 32, 64, 128, 256, 512];

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

fn warmed_predictor() -> Predictor {
    let predictor = Predictor::new();
    // Force the lazily loaded model and encoder out of the first sample.
    predictor.predict("warm up").expect("warm-up prediction");
    predictor
}

fn batch_size(c: &mut Criterion) {
    let full = std::env::args().any(|a| a == "--bench");
    let predictor = warmed_predictor();
    let corpus = corpus(full);

    let mut group = c.benchmark_group("batch_size");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .throughput(Throughput::Elements(corpus.len() as u64));
    for &size in BATCH_SIZES.iter().filter(|&&size| size <= corpus.len()) {
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

fn document_length(c: &mut Criterion) {
    let full = std::env::args().any(|a| a == "--bench");
    let predictor = warmed_predictor();
    let corpus = corpus(full);
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

    let mut group = c.benchmark_group("document_length");
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for (chunks, text) in picked {
        group.bench_with_input(BenchmarkId::new("chunks", chunks), text, |b, text| {
            b.iter(|| black_box(predictor.predict(text)));
        });
    }
    group.finish();
}

criterion_group!(benches, batch_size, document_length);
criterion_main!(benches);
