//! Golden outputs of the inference pipeline: the shipped model's document and chunk
//! probabilities for a fixed set of real documents, recorded in `fixtures/inference_golden.json`.
//!
//! Any change to encoding, batching or padding must reproduce these to within `TOLERANCE`.
//! Regenerate deliberately, and only when the model artifacts change:
//!
//! ```text
//! IS_IT_SLOP_UPDATE_GOLDEN=1 cargo test -p is-it-slop --test inference_golden
//! ```

use std::path::PathBuf;

use is_it_slop::{Prediction, Predictor, UnifiedPrediction};
use is_it_slop_preprocessing::pre_processor::tokenize;

const TOLERANCE: f32 = 1e-5;

/// Chunk counts the corpus is sampled at; each is served by the nearest real document whose
/// scores are least saturated.
const TARGET_CHUNKS: [usize; 7] = [1, 2, 4, 7, 8, 9, 12];

/// A probability this close to 0 or 1 cannot register a regression at [`TOLERANCE`]: a shift of
/// several logits moves it by less than the tolerance, so such cases are avoided when selecting.
const SATURATED: f32 = 1e-3;

#[derive(serde::Serialize, serde::Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Case {
    name: String,
    text: String,
    doc_ai_probability: f32,
    chunk_ai_probabilities: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Record {
    text: String,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inference_golden.json")
}

/// The documents the golden covers: real corpus documents at a spread of chunk counts, two of
/// them concatenated for a many-window document, one whose token count leaves a runt tail for
/// the `min_fill` rule to drop, and the degenerate inputs.
fn select_texts(predictor: &Predictor) -> Vec<(String, String)> {
    let corpus: Vec<String> = include_str!("../benches/corpus.jsonl")
        .lines()
        .map(|l| {
            serde_json::from_str::<Record>(l)
                .expect("corpus record")
                .text
        })
        .collect();
    let chunks_of = |text: &str| predictor.chunk_texts(text).len();
    // How many chunk scores of `text` sit away from 0 and 1, i.e. could register a regression.
    let live_chunks = |text: &str| {
        predictor
            .predict(text)
            .expect("predict")
            .chunk_predictions
            .iter()
            .filter(|p| {
                let p = p.ai_probability();
                p > SATURATED && p < 1.0 - SATURATED
            })
            .count()
    };
    let docs: Vec<(usize, usize)> = corpus
        .iter()
        .map(|t| (chunks_of(t), live_chunks(t)))
        .collect();

    let mut cases = Vec::new();
    for target in TARGET_CHUNKS {
        // The liveliest single document at exactly this count; if every one is saturated (or
        // there is none), join two live documents whose combined chunking lands on the count.
        let mut best: Option<(usize, String, String)> = (0..corpus.len())
            .filter(|&i| docs[i].0 == target)
            .max_by_key(|&i| docs[i].1)
            .map(|i| (docs[i].1, format!("chunks_{target}"), corpus[i].clone()));
        if best.as_ref().is_none_or(|(live, _, _)| *live == 0) {
            let live: Vec<usize> = (0..corpus.len()).filter(|&i| docs[i].1 > 0).collect();
            let mut joins: Vec<(usize, String)> = Vec::new();
            for (n, &a) in live.iter().enumerate() {
                for &b in &live[n + 1..] {
                    if docs[a].0 + docs[b].0 > target + 1 {
                        continue;
                    }
                    let text = format!("{}\n\n{}", corpus[a], corpus[b]);
                    if chunks_of(&text) == target {
                        joins.push((docs[a].1 + docs[b].1, text));
                    }
                }
            }
            joins.sort_by_key(|(live, _)| std::cmp::Reverse(*live));
            for (_, text) in joins.into_iter().take(4) {
                let live = live_chunks(&text);
                if best
                    .as_ref()
                    .is_none_or(|(best_live, _, _)| live > *best_live)
                {
                    best = Some((live, format!("chunks_{target}_joined"), text));
                }
            }
        }
        let (live, name, text) = best.expect("a document or join at this chunk count");
        assert!(live > 0, "{name}: every chunk score is saturated");
        cases.push((name, text));
    }

    let mut by_len: Vec<&String> = corpus.iter().collect();
    by_len.sort_by_key(|t| std::cmp::Reverse(t.len()));
    let joined = format!("{}\n\n{}\n\n{}", by_len[0], by_len[1], by_len[2]);
    cases.push((format!("chunks_{}_joined", chunks_of(&joined)), joined));

    // A token count just past a chunk boundary: the tail is below min_fill (10% of 192) and
    // dropped, so the last real chunk ends before the text does. Liveliest such document.
    let chunk_size = 192;
    let runt = (0..corpus.len())
        .filter(|&i| {
            let n = tokenize(&[corpus[i].as_str()])[0].len();
            n > chunk_size && (1..19).contains(&(n % chunk_size))
        })
        .max_by_key(|&i| docs[i].1)
        .expect("a corpus document with a runt tail");
    cases.push(("runt_tail".to_owned(), corpus[runt].clone()));

    cases.push(("short".to_owned(), "Short text.".to_owned()));
    cases.push(("padded".to_owned(), "  Short text.  ".to_owned()));
    // Non-blank, but the cleaner strips it to nothing: the all-pad, slot-0-only grid.
    cases.push(("cleans_to_nothing".to_owned(), "\"\"".to_owned()));
    cases
}

fn record(predictor: &Predictor) -> Golden {
    let cases = select_texts(predictor)
        .into_iter()
        .map(|(name, text)| {
            let prediction = predictor.predict(&text).expect("predict");
            Case {
                name,
                doc_ai_probability: prediction.prediction.ai_probability(),
                chunk_ai_probabilities: prediction
                    .chunk_predictions
                    .iter()
                    .map(Prediction::ai_probability)
                    .collect(),
                text,
            }
        })
        .collect();
    Golden { cases }
}

fn assert_matches(case: &Case, got: &UnifiedPrediction, how: &str) {
    let doc = got.prediction.ai_probability();
    assert!(
        (doc - case.doc_ai_probability).abs() <= TOLERANCE,
        "{} via {how}: document {doc} vs golden {}",
        case.name,
        case.doc_ai_probability
    );
    let chunks: Vec<f32> = got
        .chunk_predictions
        .iter()
        .map(Prediction::ai_probability)
        .collect();
    assert_eq!(
        chunks.len(),
        case.chunk_ai_probabilities.len(),
        "{} via {how}: chunk count",
        case.name
    );
    for (i, (got, want)) in chunks.iter().zip(&case.chunk_ai_probabilities).enumerate() {
        assert!(
            (got - want).abs() <= TOLERANCE,
            "{} via {how}: chunk {i} {got} vs golden {want}",
            case.name
        );
    }
}

#[test]
fn shipped_model_reproduces_the_golden_outputs() {
    let predictor = Predictor::new();
    let path = fixture_path();

    if std::env::var_os("IS_IT_SLOP_UPDATE_GOLDEN").is_some() {
        // The CPU backend is the reference numerics; a GPU build must not re-record the truth.
        assert_eq!(
            is_it_slop::model::BACKEND,
            "flex",
            "regenerate the golden on the Flex backend only"
        );
        let golden = record(&predictor);
        std::fs::write(&path, serde_json::to_string_pretty(&golden).unwrap()).unwrap();
        eprintln!("wrote {} cases to {}", golden.cases.len(), path.display());
        return;
    }

    let golden: Golden = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .expect("fixture present; regenerate with IS_IT_SLOP_UPDATE_GOLDEN=1"),
    )
    .expect("fixture parses");
    assert!(golden.cases.len() >= 10, "fixture looks truncated");

    // One at a time.
    for case in &golden.cases {
        let got = predictor.predict(&case.text).expect("predict");
        assert_matches(case, &got, "predict");
    }

    // All together, both batch entry points, in input order.
    let texts: Vec<&str> = golden.cases.iter().map(|c| c.text.as_str()).collect();
    let batch = predictor.predict_batch(&texts).expect("predict_batch");
    for (case, got) in golden.cases.iter().zip(&batch) {
        assert_matches(case, got, "predict_batch");
    }
    let each = predictor.predict_each(&texts);
    for (case, got) in golden.cases.iter().zip(&each) {
        assert_matches(case, got.as_ref().expect("predict_each"), "predict_each");
    }
}
