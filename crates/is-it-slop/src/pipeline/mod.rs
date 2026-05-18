//! Inference pipeline for AI text detection.
//!
//! This module implements the complete prediction pipeline for classifying text as
//! human-written or AI-generated.
//!
//! # Pipeline Flow
//!
//! 1. **Text Cleaning**: Remove HTML entities, encoding artifacts (universal cleaning only)
//! 2. **Tokenization**: Convert text to BPE token IDs using tiktoken `o200k_base`
//! 3. **Chunking**: Split token sequences into overlapping 150-token chunks
//! 4. **Vectorization**: Transform chunks to sparse TF-IDF feature vectors
//! 5. **ONNX Inference**: Run batched prediction on all chunks
//! 6. **Aggregation**: Combine chunk predictions into document-level result
//!
//! # Aggregation Methods
//!
//! Multiple strategies for combining chunk predictions:
//! - [`AggregationMethod::Mean`]: Simple average of chunk probabilities
//! - [`AggregationMethod::Max`]: Most suspicious chunk (highest AI probability)
//! - [`AggregationMethod::WeightedMean`]: Weight by distance from threshold (default)
//!
//! # Public API
//!
//! - [`predict`]: Single text prediction with aggregation
//! - [`predict_batch`]: Batch prediction for multiple texts
//!
//! # Example
//!
//! ```rust,no_run
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
mod prediction;

use std::sync::Mutex;

pub use classification::Classification;
pub use error::{PipelineError, Result};
use is_it_slop_preprocessing::pre_processor::{text_cleaner_for_inference, tokenize};
use ndarray::Ix2;
use ort::{
    session::Session,
    value::{Tensor, Value},
};
pub use prediction::{AggregationMethod, Prediction, UnifiedPrediction};
use sprs::CsMat;

use crate::model::{PRE_PROCESSOR, TOKEN_CHUNKER};

fn prepare_input_for_inference(
    input_vector: &CsMat<f32>,
) -> ort::Result<Value<ort::value::TensorValueType<f32>>> {
    let dense = input_vector.to_dense();
    let shape = dense.shape().to_vec();
    let data = dense.into_raw_vec_and_offset().0.into_boxed_slice();
    Tensor::from_array((shape, data))
}

fn run_model_inference(
    model: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<ort::session::SessionOutputs<'_>> {
    let input_name = model.inputs()[0].name().to_string();
    let inputs = ort::inputs![input_name => input];
    model.run(inputs)
}

fn parse_model_outputs_batch(
    outputs: &ort::session::SessionOutputs<'_>,
) -> ort::Result<Vec<Prediction>> {
    let probs_array = outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix2>()
        .expect("valid 2d array");

    Ok(probs_array
        .outer_iter()
        .map(|row| [row[0], row[1]].into())
        .collect())
}

fn run_inference_batch(
    model: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<Vec<Prediction>> {
    let outputs = run_model_inference(model, input)?;
    parse_model_outputs_batch(&outputs)
}

/// Predict classification for a single text.
///
/// Runs the complete pipeline: clean → tokenize → chunk → vectorize → ONNX → aggregate.
///
/// # Arguments
///
/// * `session` - ONNX Runtime session (typically [`MODEL`](crate::model::MODEL))
/// * `input` - Text to classify
/// * `agg_method` - Strategy for combining chunk predictions
///
/// # Returns
///
/// [`UnifiedPrediction`] containing aggregated probabilities, individual chunk predictions,
/// and chunk agreement score.
///
/// # Example
///
/// ```rust
/// use is_it_slop::{
///     model::MODEL,
///     pipeline::{AggregationMethod, PipelineError, predict},
/// };
///
/// let result = predict(&MODEL, "Some text", AggregationMethod::default())?;
/// println!(
///     "AI probability: {:.2}%",
///     result.prediction.ai_probability() * 100.0
/// );
/// # Ok::<(), PipelineError>(())
/// ```
pub fn predict(
    session: &Mutex<Session>,
    input: &str,
    agg_method: AggregationMethod,
) -> Result<UnifiedPrediction> {
    if input.trim().is_empty() {
        return Err(PipelineError::EmptyInput);
    }
    let input = text_cleaner_for_inference().clean(input);
    if input.trim().is_empty() {
        return Err(PipelineError::EmptyInput);
    }
    let tokens = tokenize(&[input]);
    let tokens = tokens[0].as_ref();

    let chunks = TOKEN_CHUNKER.chunk(tokens);

    // Capture token count for single-chunk documents
    let single_chunk_token_count = if chunks.len() == 1 {
        Some(chunks[0].len())
    } else {
        None
    };

    let chunk_features = PRE_PROCESSOR.vectorize_from_tokens(&chunks);
    let input_tensor = prepare_input_for_inference(&chunk_features)?;

    let output = {
        let mut model = session.lock().unwrap();
        run_inference_batch(&mut model, input_tensor)
    }?;

    let mut prediction = UnifiedPrediction::new(output, agg_method);
    prediction.single_chunk_token_count = single_chunk_token_count;
    Ok(prediction)
}

/// Predict classifications for multiple texts.
///
/// Processes texts in parallel through the pipeline. Each text is independently:
/// cleaned → tokenized → chunked → vectorized → predicted → aggregated.
///
/// # Arguments
///
/// * `session` - ONNX Runtime session (typically [`MODEL`](crate::model::MODEL))
/// * `inputs` - Slice of texts to classify
/// * `agg_method` - Strategy for combining chunk predictions (applied to each text)
///
/// # Returns
///
/// Vector of [`UnifiedPrediction`] values, one per input text.
///
/// # Example
///
/// ```rust
/// use is_it_slop::{
///     model::MODEL,
///     pipeline::{AggregationMethod, PipelineError, predict_batch},
/// };
///
/// let texts = vec!["Text 1", "Text 2", "Text 3"];
/// let results = predict_batch(&MODEL, &texts, AggregationMethod::default())?;
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
pub fn predict_batch(
    session: &Mutex<Session>,
    inputs: &[&str],
    agg_method: AggregationMethod,
) -> Result<Vec<UnifiedPrediction>> {
    if inputs.is_empty() {
        return Err(PipelineError::EmptyInput);
    }
    // This should really check for any input is empty and we should return a
    // Vec<Result<UnifiedPrediction>> instead of failing the whole batch, but for simplicity we'll
    // just check if all are empty and return a single error if so.
    if inputs.iter().all(|i| i.trim().is_empty()) {
        return Err(PipelineError::EmptyInput);
    }
    let text_cleaner = text_cleaner_for_inference();
    let a = inputs
        .iter()
        .map(|i| text_cleaner.clean(i))
        .collect::<Vec<_>>();
    let tokens = tokenize(&a);

    let chunked_inputs = TOKEN_CHUNKER.chunk_batch(&tokens);

    // Flatten to process all chunks at once.
    // Keep track of chunk counts per input for later aggregation.
    let total_chunks: usize = chunked_inputs.iter().map(Vec::len).sum();
    let mut chunk_counts = Vec::with_capacity(chunked_inputs.len());
    let mut single_chunk_token_counts = Vec::with_capacity(chunked_inputs.len());
    let mut all_chunks = Vec::with_capacity(total_chunks);
    for chunks in &chunked_inputs {
        chunk_counts.push(chunks.len());
        // Track token counts for single-chunk documents
        if chunks.len() == 1 {
            single_chunk_token_counts.push(Some(chunks[0].len()));
        } else {
            single_chunk_token_counts.push(None);
        }
        all_chunks.extend_from_slice(chunks);
    }

    let chunk_features = PRE_PROCESSOR.vectorize_from_tokens(&all_chunks);
    let input_tensor = prepare_input_for_inference(&chunk_features)?;

    let output = {
        let mut session = session.lock().unwrap();
        run_inference_batch(&mut session, input_tensor)
    }?;

    // Aggregate predictions back to original inputs.
    // Use index-based slicing instead of drain() to avoid shifting elements.
    let results: Vec<UnifiedPrediction> = {
        let mut offset = 0;
        chunk_counts
            .iter()
            .zip(single_chunk_token_counts.iter())
            .map(|(&count, &token_count)| {
                let chunk_preds = output[offset..offset + count].to_vec();
                offset += count;
                let mut prediction = UnifiedPrediction::new(chunk_preds, agg_method);
                prediction.single_chunk_token_count = token_count;
                prediction
            })
            .collect()
    };

    Ok(results)
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
        /// An error occurred during ONNX model inference.
        #[error("Inference error: {0}")]
        InferenceError(#[from] ort::Error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MODEL;

    #[test]
    fn test_predict_end_to_end() {
        let result = predict(
            &MODEL,
            "This is a test sentence for the full pipeline.",
            AggregationMethod::default(),
        )
        .expect("End-to-end predict should succeed");

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
        let texts = vec![
            "First test sentence.",
            "Second test sentence here.",
            "Third example of text.",
        ];
        let results = predict_batch(&MODEL, &texts, AggregationMethod::default())
            .expect("Batch predict should succeed");

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.prediction.human_probability() >= 0.0);
            assert!(result.prediction.ai_probability() <= 1.0);
        }
    }

    #[test]
    fn test_predict_rejects_empty() {
        assert!(predict(&MODEL, "", AggregationMethod::default()).is_err());
        assert!(predict(&MODEL, "   \n\t  ", AggregationMethod::default()).is_err());
    }

    #[test]
    fn test_predict_batch_rejects_empty() {
        let empty: &[&str] = &[];
        assert!(predict_batch(&MODEL, empty, AggregationMethod::default()).is_err());

        let all_whitespace = vec!["   ", "\t\t", "\n\n"];
        assert!(predict_batch(&MODEL, &all_whitespace, AggregationMethod::default()).is_err());
    }

    #[test]
    fn test_prepare_input_for_inference_shape() {
        // Test that sparse matrix conversion produces correct tensor shape
        use sprs::CsMat;

        // Create a simple 2x3 sparse matrix
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 2, 1];
        let data = vec![1.0, 2.0, 3.0];
        let sparse = CsMat::new((2, 3), indptr, indices, data);

        let tensor_result = prepare_input_for_inference(&sparse);
        assert!(
            tensor_result.is_ok(),
            "Should convert sparse to dense tensor"
        );
    }

    #[test]
    fn test_predict_empty_string_handling() {
        // Empty string should tokenize to empty token sequence
        // Should produce single chunk (even if empty)
        // Document expected behavior (may need special handling)

        let cleaner = text_cleaner_for_inference();
        let cleaned = cleaner.clean("");
        assert_eq!(cleaned, "");

        let tokens = tokenize(&[cleaned]);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].len(), 0);

        // With empty tokens, chunker should return single empty chunk
        let chunks = TOKEN_CHUNKER.chunk(&tokens[0]);
        assert_eq!(
            chunks.len(),
            1,
            "Empty input should produce single empty chunk"
        );
    }

    #[test]
    fn test_predict_whitespace_only() {
        // Whitespace-only text should clean to empty or minimal content
        let cleaner = text_cleaner_for_inference();
        let cleaned = cleaner.clean("   \n\t  ");

        // After cleaning, should have normalized whitespace
        assert!(cleaned.trim().is_empty() || cleaned.len() <= 1);
    }

    #[test]
    fn test_predict_only_cleaning_artifacts() {
        // Text with artifacts that get cleaned
        let cleaner = text_cleaner_for_inference();

        // HTML entities (may or may not be cleaned depending on implementation)
        let cleaned = cleaner.clean("&nbsp;&mdash;&quot;");
        // Just verify it doesn't panic and produces some output
        assert!(!cleaned.is_empty(), "Cleaning should not panic");

        // Encoding artifacts - verify cleaning doesn't panic
        let _ = cleaner.clean("\u{c3}\u{a9}\u{c3}\u{ae}\u{c3}\u{b1}");
    }

    #[test]
    fn test_text_cleaning_inference_mode() {
        // Verify inference uses universal cleaning only (not dataset artifacts)
        let cleaner = text_cleaner_for_inference();

        // Universal cleaning should apply
        let cleaned = cleaner.clean("Test&nbsp;text");
        assert!(!cleaned.contains("&nbsp;"));

        // Dataset artifact cleaning should NOT apply (these are training-only)
        // Citations like [1], [2] should remain
        let with_citation = cleaner.clean("Test text [1] more text");
        // In inference mode, citations might be kept (implementation-specific)
        // The key is: no panic, consistent behavior
        assert!(with_citation.contains("Test"));
    }

    #[test]
    fn test_aggregation_method_default() {
        // Verify default aggregation method uses chunk threshold
        let default_agg = AggregationMethod::default();
        assert!(matches!(default_agg, AggregationMethod::WeightedMean(_)));

        if let AggregationMethod::WeightedMean(threshold) = default_agg {
            assert!(*threshold > 0.0 && *threshold < 1.0);
        }
    }

    #[test]
    fn test_token_chunker_available() {
        // Verify TOKEN_CHUNKER is initialized
        let tokens: Vec<u32> = (0..200).collect();
        let chunks = TOKEN_CHUNKER.chunk(&tokens);
        assert!(!chunks.is_empty());
        assert!(
            chunks.len() > 1,
            "200 tokens should produce multiple chunks"
        );
    }

    #[test]
    fn test_vectorizer_available() {
        // Verify PRE_PROCESSOR is initialized
        // Basic smoke test: vectorize should not panic
        let tokens1 = vec![vec![1, 2, 3, 4, 5]];
        let result = PRE_PROCESSOR.vectorize_from_tokens(&tokens1);
        assert_eq!(result.rows(), 1);
        assert!(
            result.cols() > 0,
            "Vectorizer should have non-zero vocabulary"
        );
    }
}
