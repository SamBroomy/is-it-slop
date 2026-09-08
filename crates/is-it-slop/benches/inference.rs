//! Inference throughput against batch size and document length.
//!
//! Batching trades forward-pass count against padding: every document in a batch is padded to the
//! longest one, so the sweet spot depends on how mixed the lengths are. The synthetic corpus mixes
//! lengths from a few sentences to several windows. Point `IS_IT_SLOP_BENCH_CORPUS` at a file with
//! one document per line to measure real text instead.
//!
//! Under `cargo test` (no `--bench` flag) the corpus shrinks so every group runs once as a smoke
//! test.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use is_it_slop::Predictor;

const BATCH_SIZES: [usize; 7] = [1, 4, 8, 16, 32, 64, 128];

/// Words per synthetic document, chosen uniformly so lengths are log-uniform from about one chunk
/// to a couple of windows.
const WORD_COUNTS: [usize; 8] = [20, 40, 80, 160, 320, 640, 1280, 2560];

#[rustfmt::skip]
const WORDS: &[&str] = &[
    "the", "model", "reads", "every", "document", "as", "a", "grid", "of", "chunks", "and", "pools",
    "them", "into", "one", "score", "while", "short", "texts", "cost", "little", "but", "long",
    "ones", "set", "the", "padding", "for", "their", "whole", "batch", "so", "throughput", "moves",
    "with", "how", "mixed", "the", "lengths", "are", "in", "practice", "which", "is", "what", "this",
    "corpus", "tries", "to", "imitate", "without", "shipping", "any", "real", "data.",
];

/// Deterministic generator so runs are comparable across machines and commits.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next() % items.len()]
    }
}

fn synthetic_document(rng: &mut Lcg, words: usize) -> String {
    let mut text = String::with_capacity(words * 6);
    for i in 0..words {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(rng.pick(WORDS));
    }
    text
}

fn synthetic_corpus(docs: usize) -> Vec<String> {
    let mut rng = Lcg(0x5eed);
    (0..docs)
        .map(|_| {
            let words = *rng.pick(&WORD_COUNTS);
            synthetic_document(&mut rng, words)
        })
        .collect()
}

fn corpus(full: bool) -> Vec<String> {
    if let Ok(path) = std::env::var("IS_IT_SLOP_BENCH_CORPUS") {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading IS_IT_SLOP_BENCH_CORPUS={path}: {e}"));
        let docs: Vec<String> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_owned)
            .collect();
        assert!(
            !docs.is_empty(),
            "IS_IT_SLOP_BENCH_CORPUS={path} has no non-blank lines"
        );
        return docs;
    }
    synthetic_corpus(if full { 128 } else { 8 })
}

fn batch_size(c: &mut Criterion) {
    let full = std::env::args().any(|a| a == "--bench");
    let predictor = Predictor::new();
    // Force the lazily loaded model and encoder out of the first sample.
    predictor.predict("warm up").expect("warm-up prediction");
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
    let predictor = Predictor::new();
    predictor.predict("warm up").expect("warm-up prediction");
    let mut rng = Lcg(0x5eed);

    let mut group = c.benchmark_group("document_length");
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for &words in &WORD_COUNTS {
        let text = synthetic_document(&mut rng, words);
        group.bench_with_input(BenchmarkId::from_parameter(words), &text, |b, text| {
            b.iter(|| black_box(predictor.predict(text)));
        });
    }
    group.finish();
}

criterion_group!(benches, batch_size, document_length);
criterion_main!(benches);
