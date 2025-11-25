use ahash::HashMap;
use sprs::CsMat;
use tracing::debug;

use super::{
    ngrams::{self, NgramKey},
    params::VectorizerParams,
    tokenizer,
};
use crate::pre_processor::vectorizer::{ tokenizer::reverse_tokenize};

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct CountVectorizer {
    params: VectorizerParams,
    /// Vocabulary mapping n-gram (as `SmallVec`) to feature index
    /// Using `SmallVec` eliminates string conversion overhead
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

        // Use pre-computed n-grams if available, otherwise compute them
        let vocab_df = precomputed_ngrams.map_or_else(
            || ngrams::build_vocabulary(tokenized_texts, params.ngram_range()),
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

        debug!(min_df = params.min_df(), "Applying min_df filtering");
        let filtered_vocab = vocab_df
            .into_iter()
            .filter(|(_, df)| *df >= params.min_df())
            .map(|(token, _)| token)
            .collect::<Vec<_>>();
        debug!(
            original_size = vocab_size,
            filtered_size = filtered_vocab.len(),
            "Vocabulary filtered by min_df"
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
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        // Get n-grams for all documents
        // Use pre-computed if available, otherwise compute now
        if let Some(ngram_maps) = precomputed_ngrams {
            // Fast path: use pre-computed n-grams
            for ngrams in ngram_maps {
                let mut row_entries = ngrams
                    .iter()
                    .filter_map(|(ngram_key, &count)| {
                        self.vocab
                            .get(ngram_key)
                            .map(|&col_idx| (col_idx, count as f64))
                    })
                    .collect::<Vec<_>>();

                row_entries.sort_by_key(|(col_idx, _)| *col_idx);
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
                let ngrams = ngrams::count_ngrams(tokens, self.params.ngram_range());
                let mut row_entries = ngrams
                    .iter()
                    .filter_map(|(ngram_key, &count)| {
                        self.vocab
                            .get(ngram_key)
                            .map(|&col_idx| (col_idx, count as f64))
                    })
                    .collect::<Vec<_>>();

                row_entries.sort_by_key(|(col_idx, _)| *col_idx);
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
            .map(|tokens| ngrams::count_ngrams(tokens, params.ngram_range()))
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
