use ahash::HashMap;
use sprs::CsMat;
use tracing::debug;

use super::{count_vectorizer::CountVectorizer, params::VectorizerParams};

/// TF-IDF vectorizer for text preprocessing.
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
/// - Serialize with `to_bytes()` (bincode) or `to_json()` (serde feature)
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

#[cfg(feature = "bincode")]
impl TfidfVectorizer {
    /// Serialize vectorizer to bytes using bincode format.
    ///
    /// Used for fast binary serialization. Preferred format for Rust-to-Rust communication.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize vectorizer from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
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
