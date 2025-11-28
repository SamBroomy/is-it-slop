use ahash::HashMap;
use sprs::CsMat;
use tracing::debug;

use super::{count_vectorizer::CountVectorizer, params::VectorizerParams};

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Vec<f64>,
}

impl TfidfVectorizer {
    pub fn fit<T: AsRef<str> + Sync>(
        texts: &[T],
        count_vectorizer_params: VectorizerParams,
    ) -> Self {
        debug!(num_texts = texts.len(), "Fitting TfidfVectorizer");
        let (count_vectorizer, tf_matrix) =
            CountVectorizer::fit_transform(texts, count_vectorizer_params);
        debug!("Calculating IDF values");

        // Calculate IDF: log((n_docs + 1) / (df + 1)) + 1
        let n_docs = texts.len() as f64;
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
            .map(|&doc_freq| ((n_docs + 1.0) / (doc_freq as f64 + 1.0)).ln() + 1.0)
            .collect();
        debug!("IDF calculation complete");

        Self {
            count_vectorizer,
            idf,
        }
    }

    pub fn transform<T: AsRef<str> + Sync>(&self, texts: &[T]) -> CsMat<f64> {
        debug!(
            num_texts = texts.len(),
            "Transforming texts using TfidfVectorizer"
        );
        let mut tf_matrix = self.count_vectorizer.transform(texts);

        // Apply TF-IDF transformation in f64
        for mut row_vec in tf_matrix.outer_iterator_mut() {
            // Apply sublinear TF scaling if enabled: tf -> 1 + log(tf)
            if self.count_vectorizer.params().sublinear_tf() {
                for (_, val) in row_vec.iter_mut() {
                    if *val > 0.0 {
                        *val = 1.0 + val.ln();
                    }
                }
            }

            // Apply IDF (already in f64)
            for (col_idx, val) in row_vec.iter_mut() {
                *val *= self.idf[col_idx];
            }

            // Normalize row vector (L2 norm) - calculate in f64 to avoid precision loss
            let norm = row_vec.iter().map(|(_, &v)| v * v).sum::<f64>().sqrt();
            // Normalize
            if norm > 0.0 {
                for (_, val) in row_vec.iter_mut() {
                    *val /= norm;
                }
            }
        }

        tf_matrix
    }

    fn fit_transform<T: AsRef<str> + Sync>(
        texts: &[T],
        count_vectorizer_params: VectorizerParams,
    ) -> (Self, CsMat<f64>) {
        let vectorizer = Self::fit(texts, count_vectorizer_params);
        let transformed = vectorizer.transform(texts);
        (vectorizer, transformed)
    }

    #[must_use]
    pub fn num_features(&self) -> usize {
        self.count_vectorizer.num_features()
    }

    #[must_use]
    pub fn vocabulary(&self) -> HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }

    #[must_use]
    pub fn params(&self) -> &VectorizerParams {
        self.count_vectorizer.params()
    }
}

#[cfg(feature = "bincode")]
impl TfidfVectorizer {
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (vectorizer, _): (Self, usize) =
            bincode::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(vectorizer)
    }
}

#[cfg(feature = "serde")]
impl TfidfVectorizer {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}
