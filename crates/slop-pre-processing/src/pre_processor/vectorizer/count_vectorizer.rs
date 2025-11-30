use ahash::HashMap;
use sprs::CsMat;
use tracing::{debug, warn};

use super::{
    ngrams::{self, NgramKey},
    params::VectorizerParams,
    tokenizer,
};
use crate::pre_processor::vectorizer::tokenizer::reverse_tokenize;

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct CountVectorizer {
    params: VectorizerParams,
    /// Vocabulary mapping n-gram (as `SmallVec`) to feature index
    /// Using `SmallVec` eliminates string conversion overhead
    #[cfg_attr(feature = "serde", serde(with = "serde_vocab"))]
    vocab: HashMap<NgramKey, usize>,
    // /// Cached decoded vocabulary for fast access
    // /// Only computed when vocabulary() is called
    // #[cfg_attr(feature = "serde", serde(skip))]
    // decoded_vocab: Option<HashMap<String, usize>>,
}

impl CountVectorizer {
    pub fn fit<T: AsRef<str> + Sync>(texts: &[T], params: VectorizerParams) -> Self {
        debug!(num_texts = texts.len(), "Fitting CountVectorizer");
        let tokenized_texts = tokenizer::tokenize(texts);
        Self::fit_from_tokenized::<ahash::RandomState>(&tokenized_texts, params, None)
    }

    /// Internal method to fit from pre-tokenized texts.
    /// Used by `fit_transform` to avoid double tokenization.
    ///
    /// # Arguments
    /// * `tokenized_texts` - Pre-tokenized documents
    /// * `params` - Vectorizer parameters
    /// * `precomputed_ngrams` - Optional pre-computed n-grams to avoid recomputation
    fn fit_from_tokenized<H: std::hash::BuildHasher>(
        tokenized_texts: &[Vec<u32>],
        params: VectorizerParams,
        precomputed_ngrams: Option<&[std::collections::HashMap<NgramKey, usize, H>]>,
    ) -> Self {
        debug!("Building vocabulary from tokenized texts");
        if params.ngram_range().1 > crate::NGRAM_CONST_KEY {
            warn!(
                max_ngram_size = params.ngram_range().1,
                ngram_const_key = crate::NGRAM_CONST_KEY,
                "Requested n-gram size exceeds NGRAM_CONST_KEY; this may lead to suboptimal performance as the n-gram keys will not fit in the optimized SmallVec size"
            );
        }

        // Use pre-computed n-grams if available, otherwise compute them
        let vocab_df = precomputed_ngrams.map_or_else(
            || ngrams::build_vocabulary(tokenized_texts, params.ngram_counts()),
            |ngram_maps| {
                // Fast path: reuse pre-computed n-grams
                debug!("Using pre-computed n-grams for vocabulary building");
                let vocab_df = dashmap::DashMap::with_hasher(ahash::RandomState::default());

                for ngram_map in ngram_maps {
                    for ngram_key in ngram_map.keys() {
                        vocab_df
                            .entry(ngram_key.clone())
                            .and_modify(|df| *df += 1)
                            .or_insert(1);
                    }
                }
                vocab_df
            },
        );

        let vocab_size = vocab_df.len();
        let num_docs = tokenized_texts.len();

        // Calculate min_df threshold: terms appearing in fewer than this many docs are filtered
        // - If min_df < 1.0: treat as proportion of documents
        // - If min_df >= 1.0: treat as absolute document count
        let min_df_threshold = if params.min_df() < 1.0 {
            (params.min_df() * num_docs as f64).ceil() as usize
        } else {
            params.min_df() as usize
        };

        // Calculate max_df threshold: terms appearing in more than this many docs are filtered
        // - If max_df <= 1.0: treat as proportion of documents
        // - If max_df > 1.0: treat as absolute document count
        let max_df_threshold = if params.max_df() <= 1.0 {
            (params.max_df() * num_docs as f64).ceil() as usize
        } else {
            params.max_df() as usize
        };

        debug!(
            min_df = params.min_df(),
            min_df_threshold = min_df_threshold,
            min_df_interpretation = if params.min_df() < 1.0 {
                "proportion"
            } else {
                "absolute"
            },
            max_df = params.max_df(),
            max_df_threshold = max_df_threshold,
            max_df_interpretation = if params.max_df() <= 1.0 {
                "proportion"
            } else {
                "absolute"
            },
            num_docs = num_docs,
            "Applying min_df and max_df filtering"
        );

        let filtered_vocab = vocab_df
            .into_iter()
            .filter(|(_, df)| *df >= min_df_threshold && *df <= max_df_threshold)
            .map(|(token, _)| token)
            .collect::<Vec<_>>();

        debug!(
            original_size = vocab_size,
            filtered_size = filtered_vocab.len(),
            "Vocabulary filtered by min_df and max_df"
        );

        let mut sorted_tokens = filtered_vocab;
        sorted_tokens.sort();
        let vocab = sorted_tokens
            .into_iter()
            .enumerate()
            .map(|(idx, token)| (token, idx))
            .collect::<HashMap<NgramKey, usize>>();

        debug!(vocab_size = vocab.len(), "CountVectorizer fitting complete");

        Self {
            params,
            vocab,
            // decoded_vocab: None, // Lazy initialization
        }
    }

    pub fn transform<T: AsRef<str> + Sync>(&self, texts: &[T]) -> CsMat<f64> {
        debug!(
            num_texts = texts.len(),
            "Transforming texts using CountVectorizer"
        );
        let tokenized_texts = tokenizer::tokenize(texts);
        self.transform_from_tokenized::<ahash::RandomState>(&tokenized_texts, texts.len(), None)
    }

    /// Internal method to transform from pre-tokenized texts.
    /// Used by `fit_transform` to avoid double tokenization and n-gram computation.
    ///
    /// # Arguments
    /// * `tokenized_texts` - Pre-tokenized documents
    /// * `num_texts` - Number of texts (for CSR matrix sizing)
    /// * `precomputed_ngrams` - Optional pre-computed n-grams to avoid recomputation
    fn transform_from_tokenized<H: std::hash::BuildHasher>(
        &self,
        tokenized_texts: &[Vec<u32>],
        num_texts: usize,
        precomputed_ngrams: Option<&[std::collections::HashMap<NgramKey, usize, H>]>,
    ) -> CsMat<f64> {
        // Build CSR format directly
        let mut indptr = Vec::with_capacity(num_texts + 1);

        // Estimate capacity: assume ~5% of features per document on average
        // This is a conservative estimate for text data with typical sparsity
        let estimated_nnz = (num_texts * self.num_features() / 20).max(num_texts * 10);
        let mut indices = Vec::with_capacity(estimated_nnz);
        let mut data = Vec::with_capacity(estimated_nnz);

        indptr.push(0);

        // Get n-grams for all documents
        // Use pre-computed if available, otherwise compute now
        if let Some(ngram_maps) = precomputed_ngrams {
            // Fast path: use pre-computed n-grams
            for ngrams in ngram_maps {
                // Use SmallVec to avoid heap allocation for typical document sizes
                // Most documents have <256 unique n-grams in vocab
                let mut row_entries = smallvec::SmallVec::<[(usize, f64); 256]>::new();

                for (ngram_key, &count) in ngrams {
                    if let Some(&col_idx) = self.vocab.get(ngram_key) {
                        row_entries.push((col_idx, count as f64));
                    }
                }

                row_entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);
                for (col_idx, count) in row_entries {
                    indices.push(col_idx);
                    data.push(count);
                }
                indptr.push(indices.len());
            }
        } else {
            // Slow path: compute n-grams now
            // Note: This allocates but is only used when not in fit_transform
            for tokens in tokenized_texts {
                let ngrams = ngrams::count_ngrams(tokens, self.params.ngram_counts());

                // Use SmallVec to avoid heap allocation for typical document sizes
                let mut row_entries = smallvec::SmallVec::<[(usize, f64); 256]>::new();

                for (ngram_key, &count) in &ngrams {
                    if let Some(&col_idx) = self.vocab.get(ngram_key) {
                        row_entries.push((col_idx, count as f64));
                    }
                }

                row_entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);
                for (col_idx, count) in row_entries {
                    indices.push(col_idx);
                    data.push(count);
                }
                indptr.push(indices.len());
            }
        }

        debug!(
            non_zero_entries = data.len(),
            "Text transformation complete"
        );
        CsMat::new((num_texts, self.num_features()), indptr, indices, data)
    }

    /// Optimized `fit_transform` that computes n-grams only once.
    ///
    /// This method tokenizes once, computes n-grams once, then reuses them
    /// for both vocabulary building and transformation, achieving ~2x speedup
    /// over calling `fit()` followed by `transform()`.
    pub fn fit_transform<T: AsRef<str> + Sync>(
        texts: &[T],
        params: VectorizerParams,
    ) -> (Self, CsMat<f64>) {
        debug!(
            num_texts = texts.len(),
            "Optimized fit_transform: tokenizing and computing n-grams once"
        );

        // Step 1: Tokenize once
        let tokenized_texts = tokenizer::tokenize(texts);

        // Step 2: Compute n-grams once and cache them
        debug!("Computing n-grams for all documents");
        let ngram_maps: Vec<_> = tokenized_texts
            .iter()
            .map(|tokens| ngrams::count_ngrams(tokens, params.ngram_counts()))
            .collect();

        // Step 3: Fit from pre-computed n-grams
        debug!("Fitting vectorizer from cached n-grams");
        let vectorizer = Self::fit_from_tokenized::<ahash::RandomState>(
            &tokenized_texts,
            params,
            Some(&ngram_maps[..]),
        );

        // Step 4: Transform using the same pre-computed n-grams
        debug!("Transforming using cached n-grams");
        let transformed = vectorizer.transform_from_tokenized::<ahash::RandomState>(
            &tokenized_texts,
            texts.len(),
            Some(&ngram_maps[..]),
        );

        debug!("fit_transform complete with single n-gram computation");
        (vectorizer, transformed)
    }

    pub fn num_features(&self) -> usize {
        self.vocab.len()
    }

    /// Get the vocabulary as a mapping of human-readable text to feature index.
    ///
    /// Would be nice to cache this but makes serialization more complex.
    ///
    /// This isn't called frequently enough to justify the added complexity.
    pub fn vocabulary(&self) -> HashMap<String, usize> {
        // // Lazy initialization of decoded vocabulary
        // if self.decoded_vocab.is_none() {
        debug!(
            vocab_size = self.vocab.len(),
            "Decoding vocabulary for the first time (will be cached)"
        );

        self.vocab
            .iter()
            .map(|(ngram_key, &idx)| {
                // Convert SmallVec back to text via reverse tokenization
                let text = reverse_tokenize(ngram_key.as_slice());
                (text, idx)
            })
            .collect()

        // self.decoded_vocab = Some(decoded);
        // debug!("Vocabulary decoded and cached");
        // }

        // // Return cached vocabulary (unwrap is safe because we just initialized it)
        // self.decoded_vocab.as_ref().unwrap()
    }

    pub fn params(&self) -> &VectorizerParams {
        &self.params
    }
}

#[cfg(feature = "serde")]
mod serde_vocab {
    use ahash::HashMapExt;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{HashMap, NgramKey};

    pub fn serialize<S>(vocab: &HashMap<NgramKey, usize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as array of [key, value] pairs for JSON compatibility
        // Sort by value (feature index) to ensure deterministic serialization
        let mut pairs: Vec<(Vec<u32>, usize)> =
            vocab.iter().map(|(k, v)| (k.to_vec(), *v)).collect();
        pairs.sort_by_key(|(_, idx)| *idx);
        pairs.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<NgramKey, usize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use smallvec::SmallVec;

        let pairs: Vec<(Vec<u32>, usize)> = Vec::deserialize(deserializer)?;
        let mut map = HashMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            map.insert(SmallVec::from_vec(k), v);
        }
        Ok(map)
    }
}
