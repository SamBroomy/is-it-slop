//! Inference pipeline for AI text detection.
//!
//! This module implements the complete prediction pipeline for classifying text as
//! human-written or AI-generated.
//!
//! # Pipeline Flow
//!
//! 1. **Encoding**: clean (universal rules), tokenize with tiktoken `o200k_base`, map tokens
//!    through the bundle vocabulary, and tile into 192-token chunks — the exact encoding the
//!    model was trained with, reconstructed from the shipped bundle
//! 2. **Forward pass**: Burn forwards over `[docs, chunks, 192]`, spread over passes and threads
//!    by the compiled backend's [`Strategy`] (one document per pass, concurrent, on the CPU;
//!    many per pass, sequential, on a GPU); the network's document head aggregates the chunks
//!    (learned end to end — nothing outside the model recombines chunk scores)
//! 3. **Calibration**: document probability = `sigmoid(doc_logit / T)`; the shipped
//!    classification threshold was fitted on these calibrated probabilities
//!
//! Long documents are *not* truncated: the chunk axis is dynamic, so every chunk of the
//! document goes through the same forward pass (up to [`MAX_CHUNKS_PER_DOCUMENT`]), and a
//! document is padded only to its own last chunk.
//!
//! # Public API
//!
//! - [`predict`]: single text prediction
//! - [`predict_batch`]: batch prediction for multiple texts, all or nothing
//! - [`predict_each`]: batch prediction with a result per text
//! - [`predict_each_with`]: the same under an explicit scheduling [`Strategy`]
//! - [`chunk_texts`]: the text behind each chunk prediction
//!
//! # Example
//!
//! ```rust
//! use is_it_slop::Predictor;
//!
//! let predictor = Predictor::new();
//! let result = predictor.predict("Example text").unwrap();
//! println!(
//!     "AI probability: {:.2}%",
//!     result.prediction.ai_probability() * 100.0
//! );
//! println!(
//!     "Classification: {}",
//!     result.classification(predictor.threshold())
//! );
//! ```

mod classification;
mod forward;
mod prediction;
mod schedule;

// `accelerated` is set by build.rs when any GPU backend feature is on.
#[cfg(accelerated)]
mod accelerated;
#[cfg(accelerated)]
use accelerated as backend;
#[cfg(not(accelerated))]
mod cpu;
#[cfg(not(accelerated))]
use cpu as backend;

pub use classification::Classification;
pub use error::{PipelineError, Result};
pub use prediction::{ChunkInfo, ConfidenceMetrics, Prediction, UnifiedPrediction};
pub use schedule::Strategy;

use crate::model::ENCODER;

/// Cap on chunks per document (~49k tokens at the shipped encoding).
///
/// The chunk axis is dynamic but memory grows with document length, so extremely long inputs
/// are judged on their first ~49k tokens.
pub const MAX_CHUNKS_PER_DOCUMENT: usize = 256;

/// Predict classification for a single text.
///
/// Runs the complete pipeline: encode → forward → calibrate.
///
/// # Example
///
/// ```rust
/// use is_it_slop::pipeline::{PipelineError, predict};
///
/// let result = predict("Some text")?;
/// println!(
///     "AI probability: {:.2}%",
///     result.prediction.ai_probability() * 100.0
/// );
/// # Ok::<(), PipelineError>(())
/// ```
pub fn predict(input: &str) -> Result<UnifiedPrediction> {
    if input.trim().is_empty() {
        return Err(PipelineError::EmptyInput);
    }
    Ok(schedule::run(&[input], Strategy::SEQUENTIAL).remove(0))
}

/// Predict each text under the compiled backend's [`Strategy`].
///
/// Results come back in input order. A blank text fails on its own with
/// [`PipelineError::EmptyInput`] rather than failing the batch, so callers processing a stream
/// can report it in place. An empty slice yields an empty vector.
///
/// # Example
///
/// ```rust
/// use is_it_slop::pipeline::predict_each;
///
/// let results = predict_each(&["Text 1", "   ", "Text 3"]);
/// assert!(results[0].is_ok());
/// assert!(results[1].is_err());
/// assert!(results[2].is_ok());
/// ```
#[must_use]
pub fn predict_each(inputs: &[&str]) -> Vec<Result<UnifiedPrediction>> {
    predict_each_with(inputs, Strategy::default())
}

/// [`predict_each`] with an explicit [`Strategy`], for tuning to particular hardware.
///
/// Every strategy yields the same predictions; only the time taken differs.
#[must_use]
pub fn predict_each_with(inputs: &[&str], strategy: Strategy) -> Vec<Result<UnifiedPrediction>> {
    let is_blank = |text: &&str| text.trim().is_empty();
    let valid: Vec<&str> = inputs.iter().copied().filter(|t| !is_blank(t)).collect();
    let mut predictions = if valid.is_empty() {
        Vec::new()
    } else {
        schedule::run(&valid, strategy)
    }
    .into_iter();
    inputs
        .iter()
        .map(|text| {
            if is_blank(text) {
                Err(PipelineError::EmptyInput)
            } else {
                Ok(predictions
                    .next()
                    .expect("one prediction per non-blank input"))
            }
        })
        .collect()
}

/// Predict classifications for multiple texts, all or nothing.
///
/// Fails with [`PipelineError::EmptyInput`] if the slice is empty or any text is blank; use
/// [`predict_each`] for per-text results.
///
/// # Example
///
/// ```rust
/// use is_it_slop::pipeline::{PipelineError, predict_batch};
///
/// let texts = vec!["Text 1", "Text 2", "Text 3"];
/// let results = predict_batch(&texts)?;
///
/// for (i, result) in results.iter().enumerate() {
///     println!(
///         "Text {}: {:.2}% AI",
///         i + 1,
///         result.prediction.ai_probability() * 100.0
///     );
/// }
/// # Ok::<(), PipelineError>(())
/// ```
pub fn predict_batch(inputs: &[&str]) -> Result<Vec<UnifiedPrediction>> {
    if inputs.is_empty() {
        return Err(PipelineError::EmptyInput);
    }
    predict_each(inputs).into_iter().collect()
}

/// The text of each chunk behind [`predict`]'s `chunk_predictions` for `input`, in order.
///
/// This is the cleaned text the model saw, cut at the same chunk boundaries and capped at
/// [`MAX_CHUNKS_PER_DOCUMENT`], so it lines up index for index with the chunk predictions.
///
/// # Example
///
/// ```rust
/// use is_it_slop::pipeline::{PipelineError, chunk_texts, predict};
///
/// let text = "Some text";
/// let result = predict(text)?;
/// let chunks = chunk_texts(text);
/// assert_eq!(chunks.len(), result.chunk_predictions.len());
/// # Ok::<(), PipelineError>(())
/// ```
#[must_use]
pub fn chunk_texts(input: &str) -> Vec<String> {
    let mut texts = ENCODER.chunk_texts(input);
    texts.truncate(MAX_CHUNKS_PER_DOCUMENT);
    texts
}

mod error {
    use thiserror::Error;

    /// Result type for the prediction pipeline.
    pub type Result<T> = std::result::Result<T, PipelineError>;

    /// Errors that can occur during the prediction pipeline.
    #[derive(Debug, Error)]
    pub enum PipelineError {
        /// The input text is empty or reduces to nothing after cleaning.
        ///
        /// This covers both a literally empty/whitespace-only string and text
        /// that contains only cleaning artifacts (HTML entities, encoding noise,
        /// etc.) with no real content left after the universal cleaning step.
        #[error("Input text must be non-empty")]
        EmptyInput,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_end_to_end() {
        let result =
            predict("This is a test sentence for the full pipeline.").expect("predict succeeds");

        assert!(result.prediction.human_probability() >= 0.0);
        assert!(result.prediction.human_probability() <= 1.0);
        assert!(result.prediction.ai_probability() >= 0.0);
        assert!(result.prediction.ai_probability() <= 1.0);
        assert!(
            (result.prediction.human_probability() + result.prediction.ai_probability() - 1.0)
                .abs()
                < 1e-5
        );
        assert!(!result.chunk_predictions.is_empty());
    }

    #[test]
    fn test_predict_batch_end_to_end() {
        let texts = [
            "First test sentence.",
            "Second test sentence here.",
            "Third example of text.",
        ];
        let results = predict_batch(&texts).expect("batch predict succeeds");

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.prediction.human_probability() >= 0.0);
            assert!(result.prediction.ai_probability() <= 1.0);
        }
    }

    #[test]
    fn test_batch_matches_single() {
        // The concurrent batch path and the single path must agree exactly.
        let long = "This sentence repeats to span multiple chunks of the encoding. ".repeat(80);
        let texts = ["Short text.", long.as_str()];

        let batched = predict_batch(&texts).expect("batch");
        for (text, from_batch) in texts.iter().zip(&batched) {
            let alone = predict(text).expect("single");
            assert!(
                (alone.prediction.ai_probability() - from_batch.prediction.ai_probability()).abs()
                    < 1e-5,
                "batched and single predictions disagree for {text:?}"
            );
            assert_eq!(
                alone.chunk_predictions.len(),
                from_batch.chunk_predictions.len()
            );
        }
    }

    #[test]
    fn every_strategy_gives_the_single_document_answer() {
        // Grouped passes pad documents to the longest in their group and may reorder them by
        // length; the results must still be the single-document ones, in input order.
        let long =
            "Grouping puts this long document next to short ones in a padded pass. ".repeat(60);
        let texts = [
            "Short one.",
            long.as_str(),
            "   ",
            "Another short document of a different length entirely.",
            "Mid-length text that spans a little more than the shortest ones do here.",
        ];
        let alone: Vec<_> = texts.iter().map(|t| predict(t).ok()).collect();
        for strategy in [
            Strategy::SEQUENTIAL,
            Strategy::new(1, true),
            Strategy::new(2, false),
            Strategy::new(3, true),
            Strategy::new(usize::MAX, false),
        ] {
            let grouped = predict_each_with(&texts, strategy);
            for (i, (want, got)) in alone.iter().zip(&grouped).enumerate() {
                match (want, got) {
                    (None, Err(PipelineError::EmptyInput)) => {}
                    (Some(want), Ok(got)) => {
                        assert!(
                            (want.prediction.ai_probability() - got.prediction.ai_probability())
                                .abs()
                                < 1e-5,
                            "{strategy:?} text {i}"
                        );
                        assert_eq!(
                            want.chunk_predictions.len(),
                            got.chunk_predictions.len(),
                            "{strategy:?} text {i}"
                        );
                    }
                    other => panic!("{strategy:?} text {i}: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn chunk_texts_line_up_with_chunk_predictions() {
        let long =
            "Each sentence here adds a handful of tokens to the running document. ".repeat(120);
        // `""` (two quote characters) is non-blank but cleans to nothing: the model's slot-0
        // rule still yields one chunk prediction, so one (empty) chunk text must match it.
        for text in ["Short text.", "  padded  ", "\"\"", long.as_str()] {
            let chunks = chunk_texts(text);
            let prediction = predict(text).expect("predict");
            assert_eq!(
                chunks.len(),
                prediction.chunk_predictions.len(),
                "{text:.40?}"
            );
        }
        assert!(
            chunk_texts(long.as_str()).len() > ENCODER.max_chunks(),
            "spans windows"
        );
    }

    #[test]
    fn test_long_document_uses_all_chunks() {
        // ~24 tokens * 150 ≈ 3600 tokens, past one 8-chunk (1536-token) window: the dynamic
        // chunk axis must grow instead of truncating.
        let long = "The quick brown fox jumps over the lazy dog while contemplating \
                    the nature of statistical language models and their outputs. "
            .repeat(150);
        let result = predict(&long).expect("long predict succeeds");
        assert!(
            result.chunk_predictions.len() > 8,
            "expected more than one window of chunks, got {}",
            result.chunk_predictions.len()
        );
    }

    #[test]
    fn test_predict_rejects_empty() {
        assert!(predict("").is_err());
        assert!(predict("   \n\t  ").is_err());
    }

    #[test]
    fn test_predict_each_fails_blank_texts_in_place() {
        let results = predict_each(&["Real text here.", "   ", "", "More real text."]);
        assert_eq!(results.len(), 4);
        assert!(results[0].is_ok());
        assert!(matches!(results[1], Err(PipelineError::EmptyInput)));
        assert!(matches!(results[2], Err(PipelineError::EmptyInput)));
        let alone = predict("More real text.").unwrap();
        let batched = results[3].as_ref().unwrap();
        assert!(
            (alone.prediction.ai_probability() - batched.prediction.ai_probability()).abs() < 1e-5
        );

        assert!(predict_each(&[]).is_empty());
        assert!(predict_each(&["  "]).iter().all(Result::is_err));
    }

    #[test]
    fn test_predict_batch_rejects_any_blank_text() {
        assert!(predict_batch(&["Real text here.", "   "]).is_err());
    }

    #[test]
    fn test_predict_batch_rejects_empty() {
        let empty: &[&str] = &[];
        assert!(predict_batch(empty).is_err());

        let all_whitespace = ["   ", "\t\t", "\n\n"];
        assert!(predict_batch(&all_whitespace).is_err());
    }

    #[test]
    fn test_dispersion_present_for_multichunk() {
        let long = "Different sentences with varied content and structure appear here. ".repeat(60);
        let result = predict(&long).expect("predict succeeds");
        assert!(result.chunk_predictions.len() > 1);
        assert!(result.chunk_dispersion().is_some());
    }
}
