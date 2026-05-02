//! Statistical features for AI text detection.
//!
//! This module provides feature extraction for writing style analysis, capturing
//! patterns orthogonal to content-based TF-IDF features.
//!
//! # The 9 Features
//!
//! ## Document-Level (6 features)
//! Computed once per document and replicated across all chunks:
//! 1. **`bigram_repetition_rate`** - Proportion of repeating word bigrams (d=-0.419)
//! 2. **`punctuation_entropy`** - Shannon entropy of punctuation distribution (d=-0.365)
//! 3. **`lexical_diversity`** - Unique words / total words (d=+0.165)
//! 4. **`vocab_richness`** - sqrt(unique words) / total words (d=+0.154)
//! 5. **`word_repetition_rate`** - Proportion of repeating words (d=-0.084)
//! 6. **`sentence_length_cv`** - Coefficient of variation for sentence lengths (d=-0.115)
//!
//! ## Chunk-Level (3 features)
//! Computed per chunk:
//! 7. **`chunk_avg_word_length`** - Mean character length per word (coef=+8.23)
//! 8. **`chunk_punctuation_entropy`** - Local punctuation entropy (d=-0.080)
//! 9. **`chunk_word_frequency_entropy`** - Shannon entropy of word frequencies (d=-0.098)
//!
//! # Usage
//!
//! ```rust
//! use is_it_slop_preprocessing::pre_processor::features::extract_combined_features;
//!
//! let text = "Sample document with multiple sentences. More text here!";
//! let chunk_texts = vec![text.to_string()];
//!
//! // Extract all 9 features (6 doc + 3 chunk) for each chunk
//! let features = extract_combined_features(text, &chunk_texts);
//! assert_eq!(features.shape(), &[1, 9]);
//! ```

mod statistical;
mod utils;

use ndarray::Array2;
pub use statistical::*;
use tracing::{debug, instrument};
pub use utils::*;

use super::attempt_reverse_tokenize;

/// Extract document-level features (6 features).
///
/// These features are computed once per document and should be replicated
/// across all chunks from the same document during training.
///
/// # Arguments
/// * `text` - Full document text
///
/// # Returns
/// Array of 6 features: [`bigram_rep`, `punct_ent`, `lex_div`, `vocab_rich`, `word_rep`, `sent_cv`]
#[instrument(level = "debug", skip(text), fields(text_len = text.len()))]
pub fn extract_document_features(text: &str) -> [f32; 6] {
    debug!("Extracting document-level features");

    [
        compute_bigram_repetition_rate(text),
        compute_punctuation_entropy(text),
        compute_lexical_diversity(text),
        compute_vocab_richness(text),
        compute_word_repetition_rate(text),
        compute_sentence_length_cv(text),
    ]
}

/// Extract chunk-level features (3 features) for multiple chunks.
///
/// # Arguments
/// * `chunk_texts` - Vector of chunk text strings
///
/// # Returns
/// Array2 of shape (`n_chunks`, 3): [`avg_word_len`, `punct_ent`, `word_freq_ent`]
#[instrument(level = "debug", skip(chunk_texts), fields(n_chunks = chunk_texts.len()))]
pub fn extract_chunk_features(chunk_texts: &[String]) -> Array2<f32> {
    debug!(n_chunks = chunk_texts.len(), "Extracting chunk features");

    let n_chunks = chunk_texts.len();
    let mut features = Array2::<f32>::zeros((n_chunks, 3));

    for (i, chunk_text) in chunk_texts.iter().enumerate() {
        features[[i, 0]] = compute_avg_word_length(chunk_text);
        features[[i, 1]] = compute_chunk_punctuation_entropy(chunk_text);
        features[[i, 2]] = compute_word_frequency_entropy(chunk_text);
    }

    features
}

/// Extract combined features (document + chunk) for all chunks.
///
/// Document features are replicated across all chunks.
///
/// # Arguments
/// * `full_text` - Full document text
/// * `chunk_texts` - Vector of chunk text strings
///
/// # Returns
/// Array2 of shape (`n_chunks`, 9):
/// - Columns 0-5: Document features (replicated)
/// - Columns 6-8: Chunk features
#[instrument(level = "debug", skip(full_text, chunk_texts), fields(text_len = full_text.len(), n_chunks = chunk_texts.len()))]
pub fn extract_combined_features(full_text: &str, chunk_texts: &[String]) -> Array2<f32> {
    debug!("Extracting combined features");

    let n_chunks = chunk_texts.len();

    // Extract document-level features (once)
    let doc_features = extract_document_features(full_text);

    // Extract chunk-level features (per chunk)
    let chunk_features = extract_chunk_features(chunk_texts);

    // Combine: replicate doc features for each chunk
    let mut combined = Array2::<f32>::zeros((n_chunks, 9));

    for i in 0..n_chunks {
        // Copy document features (columns 0-5)
        for j in 0..6 {
            combined[[i, j]] = doc_features[j];
        }

        // Copy chunk features (columns 6-8)
        for j in 0..3 {
            combined[[i, 6 + j]] = chunk_features[[i, j]];
        }
    }

    debug!(combined_shape = ?combined.shape(), "Combined features extracted");
    combined
}

/// Extract combined features for a batch of documents.
///
/// Processes multiple documents in parallel.
///
/// # Arguments
/// * `full_texts` - Vector of full document texts
/// * `chunked_tokens_batch` - Vector of chunked token sequences
///
/// # Returns
/// Array2 of shape (`total_chunks`, 9) where `total_chunks` is the sum of
/// chunks across all documents.
#[instrument(level = "debug", skip(full_texts, chunked_tokens_batch), fields(n_docs = full_texts.len()))]
pub fn extract_combined_batch(
    full_texts: &[String],
    chunked_tokens_batch: &[Vec<Vec<u32>>],
) -> Array2<f32> {
    debug!(n_docs = full_texts.len(), "Extracting batch features");

    assert_eq!(full_texts.len(), chunked_tokens_batch.len());

    let mut all_features: Vec<Array2<f32>> = Vec::new();

    for (text, chunks) in itertools::izip!(full_texts.iter(), chunked_tokens_batch.iter()) {
        // Decode chunk tokens to text with graceful failure handling
        let chunk_texts: Vec<String> = chunks
            .iter()
            .map(|chunk_tokens| {
                attempt_reverse_tokenize(chunk_tokens).unwrap_or_else(|| {
                    tracing::warn!(
                        num_tokens = chunk_tokens.len(),
                        "Failed to decode chunk tokens for feature extraction, using empty string fallback"
                    );
                    String::new() // Empty string → features compute as 0 or edge-case values
                })
            })
            .collect();

        // Extract features for this document
        let features = extract_combined_features(text, &chunk_texts);
        all_features.push(features);
    }

    // Concatenate all document features vertically
    if all_features.is_empty() {
        return Array2::<f32>::zeros((0, 9));
    }

    let total_chunks: usize = all_features.iter().map(ndarray::ArrayBase::nrows).sum();
    let mut combined = Array2::<f32>::zeros((total_chunks, 9));

    let mut row_offset = 0;
    for features in all_features {
        let n_rows = features.nrows();
        combined
            .slice_mut(ndarray::s![row_offset..row_offset + n_rows, ..])
            .assign(&features);
        row_offset += n_rows;
    }

    debug!(total_chunks, "Batch extraction complete");
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_processor::tokenize;

    #[test]
    fn test_extract_document_features() {
        let text = "Sample text with multiple sentences. More text here!";

        let features = extract_document_features(text);

        assert_eq!(features.len(), 6);
        // All features should be valid (not NaN, not infinite)
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_extract_chunk_features() {
        let chunks = vec![
            "First chunk text.".to_string(),
            "Second chunk text.".to_string(),
        ];

        let features = extract_chunk_features(&chunks);

        assert_eq!(features.shape(), &[2, 3]);
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_extract_combined_features() {
        let text = "Sample document with multiple sentences. More text here!";
        let chunk_texts = vec![text.to_string()];

        let features = extract_combined_features(text, &chunk_texts);

        assert_eq!(features.shape(), &[1, 9]);
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_extract_combined_batch() {
        let texts = vec![
            "First document.".to_string(),
            "Second document.".to_string(),
        ];
        let tokens_batch = tokenize(&texts);
        let chunked_batch = vec![vec![tokens_batch[0].clone()], vec![tokens_batch[1].clone()]];

        let features = extract_combined_batch(&texts, &chunked_batch);

        assert_eq!(features.shape(), &[2, 9]); // 2 docs, 1 chunk each
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_empty_batch() {
        let features = extract_combined_batch(&[], &[]);

        assert_eq!(features.shape(), &[0, 9]);
    }

    #[test]
    fn test_feature_extraction_with_multiple_chunks() {
        // Test that extract_combined_batch handles multiple chunks correctly
        use crate::pre_processor::tokenize;

        let texts = vec!["Valid text with multiple words.".to_string()];
        let tokens_batch = tokenize(&texts);

        // Create multiple chunks from the same document
        let chunk1 = tokens_batch[0][..3].to_vec();
        let chunk2 = tokens_batch[0][3..].to_vec();
        let chunked_batch = vec![vec![chunk1, chunk2]];

        // Should produce features for both chunks
        let features = extract_combined_batch(&texts, &chunked_batch);

        // Should have 2 rows (one per chunk), 9 columns
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 9);
        assert!(
            features.iter().all(|&f| f.is_finite()),
            "All features should be finite"
        );
    }

    #[test]
    fn test_chunk_boundaries_statistical_features() {
        use crate::pre_processor::{TokenChunker, tokenize};

        let text = "The quick brown fox jumps over the lazy dog. \
                    Photosynthesis utilizes chlorophyll molecules embedded \
                    within thylakoid membranes.";

        let tokens = tokenize(&[text])[0].clone();
        let chunker = TokenChunker::default();
        let chunks = chunker.chunk(&tokens);

        // Decode chunks
        let chunk_texts: Vec<String> = chunks
            .iter()
            .map(|chunk| attempt_reverse_tokenize(chunk).unwrap_or_default())
            .collect();

        let features = extract_combined_features(text, &chunk_texts);

        // Verify: all features finite (no NaN, no Inf)
        assert!(
            features.iter().all(|&f| f.is_finite()),
            "Features contain NaN or Inf values"
        );

        // Verify: reasonable value ranges
        for chunk_idx in 0..features.nrows() {
            // Lexical diversity should be 0-1
            let lex_div = features[[chunk_idx, 2]];
            assert!(
                (0.0..=1.0).contains(&lex_div),
                "Lexical diversity {lex_div} out of range [0, 1]"
            );
        }
    }

    #[test]
    fn test_round_trip_with_chunking_full_pipeline() {
        use crate::pre_processor::{TokenChunker, tokenize};

        // Full integration test: text → tokens → chunks → decode → features
        let text = "This is a comprehensive test of the full pipeline.";
        let tokens = tokenize(&[text])[0].clone();
        let chunker = TokenChunker::default();
        let chunks = chunker.chunk(&tokens);

        // Should not panic
        let chunk_texts: Vec<String> = chunks
            .iter()
            .map(|chunk| attempt_reverse_tokenize(chunk).unwrap_or_default())
            .collect();

        let features = extract_combined_features(text, &chunk_texts);

        assert_eq!(features.shape(), &[chunks.len(), 9]);
    }
}
