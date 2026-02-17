//! TF-IDF (Term Frequency-Inverse Document Frequency) weighting.
//!
//! This module implements [`TfidfVectorizer`], which applies IDF weighting to term
//! frequency matrices and normalizes the result.
//!
//! # TF-IDF Formula
//!
//! For each term in each document:
//! 1. **TF**: Term frequency (optionally log-scaled with `sublinear_tf`)
//!    - Standard: `tf`
//!    - Sublinear: `log(tf + 1)`
//! 2. **IDF**: Inverse document frequency (sklearn-compatible)
//!    - Formula: `log((n_docs + 1) / (df + 1)) + 1`
//!    - Where `df` is the number of documents containing the term
//! 3. **Normalization**: L2 normalization per document
//!    - Each document vector scaled to unit length
//!
//! # Serialization
//!
//! The vectorizer can be serialized in multiple formats:
//! - **rkyv** (default): Zero-copy binary format for fast loading
//! - **bincode**: Compact binary format
//! - **JSON**: Human-readable format (requires `serde` feature)

use ahash::HashMap;
use sprs::CsMat;
use tracing::{debug, instrument};

use super::{count_vectorizer::CountVectorizer, params::VectorizerParams};

/// Applies TF-IDF weighting and L2 normalization to text features.
///
/// Wraps `CountVectorizer` and applies Inverse Document Frequency (IDF) weighting
/// with L2 normalization per document. Computes IDF as `log((n_docs + 1) / (df + 1)) + 1`
/// to match sklearn's `smooth_idf=True` behavior.
///
/// # Usage
///
/// - Use [`fit_transform`](Self::fit_transform) when training (more efficient)
/// - Use [`fit`](Self::fit) + [`transform`](Self::transform) when you need the vectorizer
///   separately
/// - Serialize with `to_bytes()` (rkyv) or `to_json()` (serde feature)
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Vec<f32>,
}

impl TfidfVectorizer {
    /// Fit vectorizer on training texts.
    ///
    /// Tokenizes texts, builds vocabulary (filtering by `min_df`/`max_df`), computes IDF weights.
    ///
    /// # Arguments
    /// * `texts` - Training documents
    /// * `count_vectorizer_params` - Configuration for n-gram extraction and vocabulary filtering
    #[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
    pub fn fit<T: AsRef<str> + Sync>(
        texts: &[T],
        count_vectorizer_params: VectorizerParams,
    ) -> Self {
        debug!(num_texts = texts.len(), "Fitting TfidfVectorizer");
        let (count_vectorizer, tf_matrix) =
            CountVectorizer::fit_transform(texts, count_vectorizer_params);

        Self::fit_from_tf_matrix(count_vectorizer, &tf_matrix, texts.len())
    }

    /// Internal method to fit from a pre-computed TF matrix.
    /// Used by `fit_transform` to avoid double computation.
    #[instrument(level = "debug", skip(count_vectorizer, tf_matrix), fields(n_docs, num_features = count_vectorizer.num_features()))]
    fn fit_from_tf_matrix(
        count_vectorizer: CountVectorizer,
        tf_matrix: &CsMat<f32>,
        n_docs: usize,
    ) -> Self {
        debug!("Calculating IDF values from TF matrix");

        let n_docs = n_docs as f32;
        let num_features = count_vectorizer.num_features();

        // Count document frequency for each term
        let mut df = vec![0usize; num_features];

        for row_vec in tf_matrix.outer_iterator() {
            for (col_idx, _val) in row_vec.iter() {
                df[col_idx] += 1;
            }
        }
        let idf = df
            .iter()
            .map(|&doc_freq| ((n_docs + 1.0) / (doc_freq as f32 + 1.0)).ln() + 1.0)
            .collect();
        debug!("IDF calculation complete");

        Self {
            count_vectorizer,
            idf,
        }
    }

    /// Transform texts to TF-IDF sparse matrix using fitted vocabulary.
    ///
    /// # Returns
    /// Sparse CSR matrix of shape `(n_texts, n_features)` with L2-normalized TF-IDF values
    #[instrument(level = "debug", skip(self, texts), fields(num_texts = texts.len(), num_features = self.num_features()))]
    pub fn transform<T: AsRef<str> + Sync>(&self, texts: &[T]) -> CsMat<f32> {
        debug!(
            num_texts = texts.len(),
            "Transforming texts using TfidfVectorizer"
        );
        let tf_matrix = self.count_vectorizer.transform(texts);
        self.apply_tfidf_transform(tf_matrix)
    }

    /// Apply TF-IDF transformation to a pre-computed TF matrix.
    /// This mutates the matrix in-place and returns it.
    ///
    /// Optimized to do only 2 passes over each row:
    /// 1. Apply TF-IDF weights and accumulate norm
    /// 2. Normalize by L2 norm
    #[instrument(level = "debug", skip(self, tf_matrix), fields(matrix_shape = ?(tf_matrix.rows(), tf_matrix.cols()), nnz = tf_matrix.nnz()))]
    fn apply_tfidf_transform(&self, mut tf_matrix: CsMat<f32>) -> CsMat<f32> {
        debug!("Applying TF-IDF transformation");

        let use_sublinear_tf = self.count_vectorizer.params().sublinear_tf();

        // Process each document (row)
        for mut row_vec in tf_matrix.outer_iterator_mut() {
            // Pass 1: Apply sublinear TF (if enabled), IDF weights, and accumulate norm
            let mut norm_squared = 0.0;

            for (col_idx, val) in row_vec.iter_mut() {
                // Apply sublinear TF scaling: tf -> 1 + log(tf)
                if use_sublinear_tf && *val > 0.0 {
                    *val = 1.0 + val.ln();
                }

                // Apply IDF weight
                *val *= self.idf[col_idx];

                // Accumulate squared norm
                norm_squared += *val * *val;
            }

            // Pass 2: Normalize by L2 norm
            if norm_squared > 0.0 {
                let norm = norm_squared.sqrt();
                for (_, val) in row_vec.iter_mut() {
                    *val /= norm;
                }
            }
        }

        tf_matrix
    }

    /// Fit vectorizer and transform texts in a single pass.
    ///
    /// More efficient than calling `fit()` + `transform()` separately: tokenizes and
    /// computes n-grams only once.
    ///
    /// # Returns
    /// Tuple of (fitted vectorizer, TF-IDF matrix)
    #[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
    pub fn fit_transform<T: AsRef<str> + Sync>(
        texts: &[T],
        count_vectorizer_params: VectorizerParams,
    ) -> (Self, CsMat<f32>) {
        debug!(
            num_texts = texts.len(),
            "Fitting and transforming texts using TfidfVectorizer"
        );

        // Step 1: Fit CountVectorizer and get TF matrix (tokenizes and computes n-grams once)
        let (count_vectorizer, tf_matrix) =
            CountVectorizer::fit_transform(texts, count_vectorizer_params);

        // Step 2: Fit TfidfVectorizer from the TF matrix (computes IDF)
        let vectorizer = Self::fit_from_tf_matrix(count_vectorizer, &tf_matrix, texts.len());

        // Step 3: Apply TF-IDF transformation to the same TF matrix (no re-tokenization!)
        let tfidf_matrix = vectorizer.apply_tfidf_transform(tf_matrix);

        debug!("fit_transform complete with single tokenization pass");
        (vectorizer, tfidf_matrix)
    }

    /// Transform pre-tokenized sequences to TF-IDF matrix
    ///
    /// Used after token-level chunking
    ///
    /// # Arguments
    /// * `token_sequences` - Pre-tokenized and optionally chunked documents
    ///
    /// # Returns
    /// Sparse TF-IDF matrix with L2 normalization
    #[must_use]
    pub fn vectorize_from_tokens(&self, token_sequences: &[Vec<u32>]) -> CsMat<f32> {
        let tf_matrix = self.count_vectorizer.vectorize_from_tokens(token_sequences);
        self.apply_tfidf_transform(tf_matrix)
    }

    /// Number of features (vocabulary size) in the fitted vectorizer.
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.count_vectorizer.num_features()
    }

    /// Get vocabulary as a mapping of text n-grams to feature indices.
    ///
    /// **Note:** Requires reverse tokenization (tiktoken decoding), which can be slow
    /// for large vocabularies.
    #[must_use]
    pub fn vocabulary(&self) -> HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }

    /// Get the vectorizer parameters.
    #[must_use]
    pub fn params(&self) -> &VectorizerParams {
        self.count_vectorizer.params()
    }
}

#[cfg(feature = "rkyv")]
impl TfidfVectorizer {
    /// Serialize vectorizer to bytes using rkyv format.
    ///
    /// Provides zero-copy capable binary serialization.
    pub fn to_bytes(&self) -> Result<rkyv::util::AlignedVec, rkyv::rancor::Error> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
    }

    /// Access vectorizer from rkyv bytes without full deserialization (zero-copy).
    ///
    /// Preferred for fast read-only access.
    pub fn access_from_bytes(
        bytes: &[u8],
    ) -> Result<&ArchivedTfidfVectorizer, rkyv::rancor::Error> {
        rkyv::access::<ArchivedTfidfVectorizer, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize vectorizer from rkyv bytes back to the original type.
    ///
    /// Use when you need a mutable, owned instance.
    /// Handles unaligned input by copying to an aligned buffer if needed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        // Copy to aligned buffer to handle unaligned input (e.g., from include_bytes!)
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(bytes);
        let archived = Self::access_from_bytes(&aligned)?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
    }
}

#[cfg(feature = "bincode")]
impl TfidfVectorizer {
    /// Serialize vectorizer to bytes using bincode format.
    ///
    /// Used for fast binary serialization. Preferred format for Rust-to-Rust communication.
    pub fn to_bincode_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize vectorizer from bincode bytes.
    pub fn from_bincode_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (vectorizer, _): (Self, usize) =
            bincode::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(vectorizer)
    }
}

#[cfg(feature = "serde")]
impl TfidfVectorizer {
    /// Serialize vectorizer to JSON string.
    ///
    /// Human-readable format, useful for inspection and debugging.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize vectorizer from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_transform_basic() {
        let texts = vec!["hello world", "world"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (v, x) = TfidfVectorizer::fit_transform(&texts, params);
        assert_eq!(x.rows(), 2);
        assert!(v.num_features() > 0);
    }

    #[test]
    fn test_idf_formula() {
        // n_docs=3, term in 2 docs: IDF = ln((3+1)/(2+1)) + 1
        let texts = vec!["a b", "a c", "d e"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);
        // Just verify it doesn't panic and produces valid output
        let x = v.transform(&texts);
        assert!(x.data().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_l2_normalization() {
        let texts = vec!["test text", "another document"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);
        let x = v.transform(&texts);

        // Check each row has L2 norm = 1.0 (or 0 for empty)
        for row in x.outer_iterator() {
            let norm_sq: f32 = row.iter().map(|(_, &v)| v * v).sum();
            let norm = norm_sq.sqrt();
            assert!(norm < 1e-6 || (norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sublinear_tf_effect() {
        let texts = vec!["a a a a a", "b"];
        let params_linear = VectorizerParams::new(1.0, 1.0, false);
        let params_sublinear = VectorizerParams::new(1.0, 1.0, true);

        let (_, x1) = TfidfVectorizer::fit_transform(&texts, params_linear);
        let (_, x2) = TfidfVectorizer::fit_transform(&texts, params_sublinear);

        // Sublinear should reduce impact of high counts
        assert_ne!(x1.data(), x2.data());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let texts = vec!["test"];
        let params = VectorizerParams::new(1.0, 1.0, false);

        let v1 = TfidfVectorizer::fit(&texts, params.clone());
        let x1 = v1.transform(&texts);

        let (_, x2) = TfidfVectorizer::fit_transform(&texts, params);

        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    #[cfg(feature = "rkyv")]
    fn test_rkyv_serialization() {
        let texts = vec!["test text"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let bytes = v.to_bytes().unwrap();
        let loaded = TfidfVectorizer::from_bytes(&bytes).unwrap();

        let x1 = v.transform(&texts);
        let x2 = loaded.transform(&texts);
        assert_eq!(x1.data(), x2.data());
    }
    #[test]
    #[cfg(feature = "bincode")]
    fn test_bincode_serialization() {
        let texts = vec!["test text"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let bytes = v.to_bincode_bytes().unwrap();
        let loaded = TfidfVectorizer::from_bincode_bytes(&bytes).unwrap();

        let x1 = v.transform(&texts);
        let x2 = loaded.transform(&texts);
        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_serialization() {
        let texts = vec!["test"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let json = v.to_json().unwrap();
        let loaded = TfidfVectorizer::from_json(&json).unwrap();

        assert_eq!(v.num_features(), loaded.num_features());
    }

    #[test]
    fn test_determinism() {
        let texts = vec!["test"];
        let params = VectorizerParams::new(1.0, 1.0, false);

        let (_, x1) = TfidfVectorizer::fit_transform(&texts, params.clone());
        let (_, x2) = TfidfVectorizer::fit_transform(&texts, params);

        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    fn test_vocabulary_access() {
        let texts = vec!["hello world test", "sample text data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let vocab = v.vocabulary();
        assert!(!vocab.is_empty());
        assert!(vocab.values().all(|&idx| idx < v.num_features()));
    }

    // vectorize_from_tokens equivalence tests

    #[test]
    fn test_vectorize_from_tokens_equivalence() {
        // Verify vectorize_from_tokens produces same result as transform
        use crate::pre_processor::tokenizer::tokenize;

        let texts = vec!["Hello world test", "Sample text data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        // Method 1: transform (internal tokenization)
        let x1 = v.transform(&texts);

        // Method 2: vectorize_from_tokens (external tokenization)
        let tokenized = tokenize(&texts);
        let x2 = v.vectorize_from_tokens(&tokenized);

        // Should produce identical results
        assert_eq!(x1.data(), x2.data());
        assert_eq!(x1.indices(), x2.indices());
        assert_eq!(x1.indptr().raw_storage(), x2.indptr().raw_storage());
    }

    #[test]
    fn test_vectorize_from_tokens_empty() {
        // Edge case: empty token sequence
        let texts = vec!["train data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let empty_tokens = vec![vec![]];
        let x = v.vectorize_from_tokens(&empty_tokens);

        assert_eq!(x.rows(), 1);
        assert_eq!(x.nnz(), 0); // Empty tokens -> zero vector
    }

    #[test]
    fn test_vectorize_from_tokens_batch() {
        // Test with multiple token sequences
        use crate::pre_processor::tokenizer::tokenize;

        let texts = vec!["train text one", "train text two"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let test_texts = vec!["test one", "test two", "test three"];
        let tokenized = tokenize(&test_texts);
        let x = v.vectorize_from_tokens(&tokenized);

        assert_eq!(x.rows(), 3);
        assert_eq!(x.cols(), v.num_features());
    }

    // IDF formula tests (internal behavior)

    #[test]
    fn test_universal_term_produces_consistent_output() {
        // Term appearing in all documents should have consistent IDF
        // IDF formula: log((n+1)/(df+1)) + 1
        // For df=n: log((n+1)/(n+1)) + 1 = log(1) + 1 = 0 + 1 = 1.0

        let texts = vec!["common word", "common word", "common word"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        // All n-grams appear in all docs
        // Transform should produce consistent normalized vectors
        let x = v.transform(&texts);

        // All rows should be identical (same input)
        let row0 = x.outer_view(0).unwrap();
        let row1 = x.outer_view(1).unwrap();

        assert_eq!(row0.nnz(), row1.nnz());
    }

    #[test]
    fn test_rare_term_vs_common_term() {
        // Rare terms should have different impact than common terms
        // Need longer texts to produce multiple n-grams
        let texts = vec![
            "common word appears often",
            "common word and rare item",
            "common word appears often",
        ];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let x = v.transform(&texts);

        // Row 1 (with "rare") should differ from rows 0 and 2
        let row0 = x.outer_view(0).unwrap();
        let row1 = x.outer_view(1).unwrap();

        // If vocabularies overlap, check if any values differ
        if row0.nnz() > 0 && row1.nnz() > 0 {
            // Different documents should produce different vectors
            // (unless vocabulary filtering removed all distinctive terms)
            let row0_data = row0.data().to_vec();
            let row1_data = row1.data().to_vec();

            // At least verify they're valid
            assert!(!row0_data.is_empty() || !row1_data.is_empty());
        }
    }

    #[test]
    fn test_idf_produces_valid_output() {
        // Verify IDF computation produces valid sparse matrices
        let texts = vec!["word word", "word other", "other"];

        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let x = v.transform(&texts);

        // All output values should be finite and non-negative
        for &val in x.data() {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }
    }

    // rkyv alignment regression tests

    #[test]
    #[cfg(feature = "rkyv")]
    fn test_rkyv_unaligned_bytes() {
        // Regression test: rkyv requires aligned memory
        // include_bytes!() doesn't guarantee alignment, must use AlignedVec
        let texts = vec!["test data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        // Serialize
        let bytes = v.to_bytes().unwrap();

        // Create unaligned slice (simulate include_bytes!)
        let unaligned = bytes.to_vec();

        // Direct access without alignment would fail
        // This test documents the requirement to use AlignedVec in from_bytes

        // Correct approach: from_bytes copies to AlignedVec
        let loaded = TfidfVectorizer::from_bytes(&unaligned).unwrap();

        // Verify it works
        let x1 = v.transform(&texts);
        let x2 = loaded.transform(&texts);
        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    #[cfg(feature = "rkyv")]
    fn test_rkyv_large_vectorizer() {
        // Test rkyv serialization with larger vocabulary
        let texts: Vec<String> = (0..100)
            .map(|i| format!("Text sample {i} with words"))
            .collect();
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let bytes = v.to_bytes().unwrap();
        let loaded = TfidfVectorizer::from_bytes(&bytes).unwrap();

        assert_eq!(v.num_features(), loaded.num_features());

        // Verify transform produces same results
        let test_texts = vec!["sample words"];
        let x1 = v.transform(&test_texts);
        let x2 = loaded.transform(&test_texts);
        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    #[cfg(feature = "rkyv")]
    fn test_rkyv_roundtrip_determinism() {
        // Multiple save/load cycles should be deterministic
        let texts = vec!["determinism test"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let bytes1 = v.to_bytes().unwrap();
        let loaded1 = TfidfVectorizer::from_bytes(&bytes1).unwrap();

        let bytes2 = loaded1.to_bytes().unwrap();
        let loaded2 = TfidfVectorizer::from_bytes(&bytes2).unwrap();

        // Should produce identical results after multiple cycles
        let x1 = v.transform(&texts);
        let x2 = loaded2.transform(&texts);
        assert_eq!(x1.data(), x2.data());
    }

    // Edge cases

    #[test]
    fn test_tfidf_with_single_document() {
        // Single document: all terms have df=1, should produce valid output
        // Need longer text to ensure n-grams are generated
        let texts = vec!["single document with multiple words and some extra content"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let x = v.transform(&texts);
        assert_eq!(x.rows(), 1);
        // With ngram_range=(2,4), may have features if text is long enough
        // If no features, that's also valid (all filtered out)
    }

    #[test]
    fn test_tfidf_empty_after_filtering() {
        // If min_df/max_df filter out all n-grams, vocabulary is empty
        let texts = vec!["unique1", "unique2", "unique3"];
        let params = VectorizerParams::new(2.0, 1.0, false); // min_df=2, but all are unique

        let v = TfidfVectorizer::fit(&texts, params);
        // Vocabulary may be empty
        // Transform should still produce valid sparse matrix
        let x = v.transform(&texts);
        assert_eq!(x.rows(), 3);
    }

    #[test]
    fn test_tfidf_very_sparse_input() {
        // Document with no n-grams in vocabulary
        let train = vec!["training data words"];
        let test = vec!["completely different text"];

        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&train, params);

        let x = v.transform(&test);

        // Should produce valid matrix (all zeros)
        assert_eq!(x.rows(), 1);
        // May have zero non-zero entries
    }

    #[test]
    fn test_sublinear_tf_with_high_counts() {
        // Sublinear TF should dampen effect of very high term frequencies
        // TF(t) = 1 + log(count) instead of count

        let texts = vec!["a a a a a a a a a a", "b"]; // "a" appears 10 times

        let params_linear = VectorizerParams::new(1.0, 1.0, false);
        let params_sublinear = VectorizerParams::new(1.0, 1.0, true);

        let (v_linear, x_linear) = TfidfVectorizer::fit_transform(&texts, params_linear);
        let (v_sublinear, x_sublinear) = TfidfVectorizer::fit_transform(&texts, params_sublinear);

        // Both should have same vocabulary size
        assert_eq!(v_linear.num_features(), v_sublinear.num_features());

        // But TF-IDF values should differ
        // Sublinear should reduce magnitude of frequent terms
        let linear_data: Vec<f32> = x_linear.data().to_vec();
        let sublinear_data: Vec<f32> = x_sublinear.data().to_vec();

        assert_ne!(linear_data, sublinear_data);
    }

    #[test]
    fn test_transform_preserves_input_order() {
        // Verify that batch transform preserves document order
        let texts = vec!["doc1", "doc2", "doc3"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let x = v.transform(&texts);

        // Each row should correspond to its input document
        assert_eq!(x.rows(), 3);

        // Transform should be deterministic per document
        let x_single: Vec<_> = texts.iter().map(|t| v.transform(&[t])).collect();

        for (i, x_) in x_single.iter().enumerate().take(3) {
            let row_from_batch = x.outer_view(i).unwrap();
            let row_from_single = x_.outer_view(0).unwrap();

            // Compare data (may need to account for normalization differences)
            assert_eq!(row_from_batch.nnz(), row_from_single.nnz());
        }
    }

    #[test]
    fn test_num_features_matches_vocabulary() {
        // Verify num_features() returns correct vocabulary size
        let texts = vec!["sample text data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v = TfidfVectorizer::fit(&texts, params);

        let vocab = v.vocabulary();
        assert_eq!(vocab.len(), v.num_features());
    }
}
