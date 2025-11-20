use ahash::HashMap;

use sprs::CsMat;
use tracing::debug;

use super::{count_vectorizer::CountVectorizer, params::VectorizerParams};

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, )]
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
        // Compute IDF values
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
        let tf_matrix = self.count_vectorizer.transform(texts);

        let mut tf_matrix = tf_matrix.to_owned();

        // Apply TF-IDF transformation
        for mut row_vec in tf_matrix.outer_iterator_mut() {
            // Apply IDF
            for (col_idx, val) in row_vec.iter_mut() {
                *val *= self.idf[col_idx];
            }
            // Normalize row vector (L2 norm)
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

    pub fn num_features(&self) -> usize {
        self.count_vectorizer.num_features()
    }

    pub fn vocabulary(&self) -> HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }
}
