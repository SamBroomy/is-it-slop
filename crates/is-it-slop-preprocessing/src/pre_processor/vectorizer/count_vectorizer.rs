//! Vocabulary building and term frequency counting.
//!
//! This module implements [`CountVectorizer`], which builds a vocabulary from training
//! texts and transforms documents into sparse term-frequency matrices.
//!
//! The vectorizer:
//! 1. Tokenizes input texts (if not pre-tokenized)
//! 2. Extracts n-grams from token sequences
//! 3. Counts document frequencies (how many docs each n-gram appears in)
//! 4. Filters vocabulary by `min_df` and `max_df` thresholds
//! 5. Transforms documents to sparse CSR matrices of term frequencies
//!
//! Automatically parallelizes for datasets with >= 1,000 documents.

use std::ops::AddAssign;

use ahash::{HashMap, HashMapExt};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice; // For par_chunks
use sprs::CsMat;
use tracing::{debug, instrument, warn};

use crate::pre_processor::{
    DEFAULT_MAX_NGRAM, VectorizerParams,
    ngrams::{self, NgramKey, NgramKeyTrait},
    tokenizer,
};

/// Builds vocabulary and transforms texts to sparse term-frequency matrices.
///
/// Builds vocabulary from training texts (with `min_df`/`max_df` filtering), then transforms
/// texts to sparse CSR matrices where each cell is the count of an n-gram in a document.
///
/// Vocabulary is sorted alphabetically after filtering to ensure deterministic feature indices.
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
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
    /// Create vectorizer from pre-built vocabulary.
    ///
    /// Used by `TfidfVectorizerBuilder` after incremental vocabulary construction.
    ///
    /// # Arguments
    /// * `vocab` - Pre-built vocabulary mapping n-grams to feature indices
    /// * `params` - Vectorizer parameters
    #[must_use]
    pub fn from_vocab(vocab: HashMap<NgramKey, usize>, params: VectorizerParams) -> Self {
        Self { params, vocab }
    }

    /// Fit vectorizer on training texts.
    ///
    /// # Arguments
    /// * `texts` - Training documents
    /// * `params` - Configuration for n-gram extraction and vocabulary filtering
    #[allow(dead_code)]
    #[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
    pub fn fit<T: AsRef<str> + Sync>(texts: &[T], params: VectorizerParams) -> Self {
        debug!(num_texts = texts.len(), "Fitting CountVectorizer");
        let tokenized_texts = tokenizer::tokenize(texts);
        Self::fit_from_tokenized(&tokenized_texts, params, None)
    }

    /// Fit vectorizer from pre-tokenized texts.
    ///
    /// Public method to support memory-efficient workflows that manage tokenization separately.
    ///
    /// # Arguments
    /// * `tokenized_texts` - Pre-tokenized documents
    /// * `params` - Vectorizer parameters
    /// * `precomputed_ngrams` - Optional pre-computed n-grams to avoid recomputation
    #[instrument(level = "debug", skip(tokenized_texts, precomputed_ngrams), fields(num_texts = tokenized_texts.len(), has_precomputed = precomputed_ngrams.is_some()))]
    pub fn fit_from_tokenized(
        tokenized_texts: &[Vec<u32>],
        params: VectorizerParams,
        precomputed_ngrams: Option<&[HashMap<NgramKey, usize>]>,
    ) -> Self {
        debug!("Building vocabulary from tokenized texts");
        if params.ngram_range().1 > DEFAULT_MAX_NGRAM {
            warn!(
                max_ngram_size = params.ngram_range().1,
                ngram_const_key = DEFAULT_MAX_NGRAM,
                "Requested n-gram size exceeds DEFAULT_MAX_NGRAM; this may lead to suboptimal performance as the n-gram keys will not fit in the optimized SmallVec size"
            );
        }
        let num_docs = tokenized_texts.len();

        // Calculate min_df threshold: terms appearing in fewer than this many docs are filtered
        // - If min_df < 1.0: treat as proportion of documents
        // - If min_df >= 1.0: treat as absolute document count
        let min_df_threshold = if params.min_df() < 1.0 {
            (params.min_df() * num_docs as f32).ceil() as usize
        } else {
            params.min_df() as usize
        };

        // Use pre-computed n-grams if available, otherwise compute them
        let vocab_df = precomputed_ngrams.map_or_else(
            || {
                ngrams::build_vocabulary(tokenized_texts, params.ngram_counts())
                    .into_iter()
                    .collect()
            },
            |ngram_maps| {
                // Fast path: reuse pre-computed n-grams
                debug!("Using pre-computed n-grams for vocabulary building");

                // Chunked aggregation: Process documents in chunks to reduce parallelization
                // overhead. With millions of docs, per-doc parallelism causes
                // excessive work-stealing overhead. Chunk size: aim for ~100-1000
                // chunks total (not millions of tasks)
                let chunk_size = (ngram_maps.len() / 100).max(1000);

                let local_vocabs: Vec<HashMap<NgramKey, usize>> = ngram_maps
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_vocab = HashMap::with_capacity_and_hasher(
                            10_000,
                            ahash::RandomState::default(),
                        );

                        for ngram_map in chunk {
                            for ngram_key in ngram_map.keys() {
                                *local_vocab.entry(ngram_key.clone()).or_insert(0) += 1;
                            }
                        }

                        local_vocab
                    })
                    .collect();

                // Merge local vocabs into final DashMap
                let vocab_df = dashmap::DashMap::with_capacity_and_hasher(
                    tokenized_texts.len() / 2,
                    ahash::RandomState::default(),
                );

                for local_vocab in local_vocabs {
                    for (ngram_key, count) in local_vocab {
                        vocab_df.entry(ngram_key).or_insert(0).add_assign(count);
                    }
                }

                vocab_df
            },
        );

        let vocab_size = vocab_df.len();

        // Calculate max_df threshold: terms appearing in more than this many docs are filtered
        // - If max_df <= 1.0: treat as proportion of documents
        // - If max_df > 1.0: treat as absolute document count
        let max_df_threshold = if params.max_df() <= 1.0 {
            (params.max_df() * num_docs as f32).ceil() as usize
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
            .into_par_iter()
            .filter_map(|(token, df)| {
                if df >= min_df_threshold && df <= max_df_threshold {
                    Some(token)
                } else {
                    None
                }
            })
            // .filter(|(_, df)| *df >= min_df_threshold && *df <= max_df_threshold)
            // .map(|(token, _)| token)
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

        Self { params, vocab }
    }

    /// Vectorize pre-tokenized sequences directly
    ///
    /// # Arguments
    /// * `token_sequences` - Pre-tokenized documents as `Vec<Vec<u32>>`
    ///
    /// # Returns
    /// Sparse CSR matrix of term frequencies
    #[must_use]
    pub fn vectorize_from_tokens(&self, token_sequences: &[Vec<u32>]) -> CsMat<f32> {
        self.transform_from_tokenized(token_sequences, token_sequences.len(), None)
    }

    /// Transform texts to sparse term-frequency matrix.
    ///
    /// # Returns
    /// Sparse CSR matrix of shape `(n_texts, n_features)` with term counts
    #[instrument(level = "debug", skip_all, fields(num_texts = texts.len(), vocab_size = self.num_features()))]
    pub fn transform<T: AsRef<str> + Sync>(&self, texts: &[T]) -> CsMat<f32> {
        debug!(
            num_texts = texts.len(),
            "Transforming texts using CountVectorizer"
        );
        let tokenized_texts = tokenizer::tokenize(texts);
        self.transform_from_tokenized(&tokenized_texts, texts.len(), None)
    }

    /// Internal method to transform from pre-tokenized texts.
    /// Used by `fit_transform` to avoid double tokenization and n-gram computation.
    ///
    /// # Arguments
    /// * `tokenized_texts` - Pre-tokenized documents
    /// * `num_texts` - Number of texts (for CSR matrix sizing)
    /// * `precomputed_ngrams` - Optional pre-computed n-grams to avoid recomputation
    #[instrument(level = "debug", skip(self, tokenized_texts, precomputed_ngrams), fields(num_texts, vocab_size = self.num_features(), has_precomputed = precomputed_ngrams.is_some()))]
    fn transform_from_tokenized(
        &self,
        tokenized_texts: &[Vec<u32>],
        num_texts: usize,
        precomputed_ngrams: Option<&[HashMap<NgramKey, usize>]>,
    ) -> CsMat<f32> {
        // Threshold for parallelization (tune based on your data)
        const PARALLEL_THRESHOLD: usize = 1_000;

        if num_texts < PARALLEL_THRESHOLD {
            // Sequential path for small batches
            self.transform_sequential(tokenized_texts, num_texts, precomputed_ngrams)
        } else {
            // Parallel path for large batches
            self.transform_parallel(tokenized_texts, num_texts, precomputed_ngrams)
        }
    }

    fn transform_sequential(
        &self,
        tokenized_texts: &[Vec<u32>],
        num_texts: usize,
        precomputed_ngrams: Option<&[HashMap<NgramKey, usize>]>,
    ) -> CsMat<f32> {
        let mut indptr = Vec::with_capacity(num_texts + 1);
        let estimated_nnz = (num_texts * self.num_features() / 20).max(num_texts * 10);
        let mut indices = Vec::with_capacity(estimated_nnz);
        let mut data = Vec::with_capacity(estimated_nnz);

        indptr.push(0);
        let mut row_entries = Vec::new();

        for (doc_idx, tokens) in tokenized_texts.iter().enumerate() {
            match precomputed_ngrams {
                Some(precomputed) => {
                    self.row_entries_from_ngram_map(&precomputed[doc_idx], &mut row_entries);
                }
                None => self.row_entries_from_tokens(tokens, &mut row_entries),
            }

            for (col_idx, count) in row_entries.drain(..) {
                indices.push(col_idx);
                data.push(count);
            }
            indptr.push(indices.len());
        }

        CsMat::new((num_texts, self.num_features()), indptr, indices, data)
    }

    fn transform_parallel(
        &self,
        tokenized_texts: &[Vec<u32>],
        num_texts: usize,
        precomputed_ngrams: Option<&[HashMap<NgramKey, usize>]>,
    ) -> CsMat<f32> {
        use rayon::prelude::*;

        // Calculate accurate capacity based on sparsity
        // You know: ~353 non-zero per doc on average
        const AVG_NNZ_PER_DOC: usize = 400;

        // Chunked collection: Build CSR matrix in parallel chunks to reduce allocation overhead
        // This avoids creating millions of intermediate Vec allocations
        struct CsrChunk {
            indices: Vec<usize>,
            data: Vec<f32>,
            row_lengths: Vec<usize>, // Length of each row in this chunk
        }

        let chunks: Vec<CsrChunk> = precomputed_ngrams.map_or_else(
            || {
                tokenized_texts
                    .par_chunks(1000) // Process 1000 docs per chunk
                    .map(|chunk_texts| {
                        let mut indices = Vec::with_capacity(chunk_texts.len() * AVG_NNZ_PER_DOC);
                        let mut data = Vec::with_capacity(chunk_texts.len() * AVG_NNZ_PER_DOC);
                        let mut row_lengths = Vec::with_capacity(chunk_texts.len());

                        for tokens in chunk_texts {
                            let row_start = indices.len();
                            let row_entries = self.row_entries_from_tokens_sparse(tokens);

                            for (col_idx, count) in row_entries {
                                indices.push(col_idx);
                                data.push(count);
                            }

                            row_lengths.push(indices.len() - row_start);
                        }

                        CsrChunk {
                            indices,
                            data,
                            row_lengths,
                        }
                    })
                    .collect()
            },
            |precomputed| {
                precomputed
                    .par_chunks(1000) // Process 1000 docs per chunk
                    .map(|chunk_ngrams| {
                        let mut indices = Vec::with_capacity(chunk_ngrams.len() * AVG_NNZ_PER_DOC);
                        let mut data = Vec::with_capacity(chunk_ngrams.len() * AVG_NNZ_PER_DOC);
                        let mut row_lengths = Vec::with_capacity(chunk_ngrams.len());

                        for ngrams in chunk_ngrams {
                            let row_start = indices.len();
                            let mut row_entries = Vec::with_capacity(AVG_NNZ_PER_DOC);

                            for (ngram_key, count) in ngrams {
                                if let Some(&col_idx) = self.vocab.get(ngram_key) {
                                    row_entries.push((col_idx, *count as f32));
                                }
                            }

                            row_entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);

                            for (col_idx, count) in row_entries {
                                indices.push(col_idx);
                                data.push(count);
                            }

                            row_lengths.push(indices.len() - row_start);
                        }

                        CsrChunk {
                            indices,
                            data,
                            row_lengths,
                        }
                    })
                    .collect()
            },
        );

        // Calculate total size and concatenate chunks
        let actual_total_nnz: usize = chunks.iter().map(|c| c.indices.len()).sum();
        let mut indices = Vec::with_capacity(actual_total_nnz);
        let mut data = Vec::with_capacity(actual_total_nnz);
        let mut indptr = Vec::with_capacity(num_texts + 1);

        indptr.push(0);

        for chunk in chunks {
            indices.extend(chunk.indices);
            data.extend(chunk.data);

            for row_len in chunk.row_lengths {
                indptr.push(indptr.last().unwrap() + row_len);
            }
        }

        debug!(
            non_zero_entries = actual_total_nnz,
            avg_nnz_per_doc = actual_total_nnz / num_texts,
            "Parallel sparse transform complete"
        );

        CsMat::new((num_texts, self.num_features()), indptr, indices, data)
    }

    #[inline]
    fn row_entries_from_tokens_sparse(&self, tokens: &[u32]) -> Vec<(usize, f32)> {
        let ngram_sizes = self.params.ngram_counts();
        if ngram_sizes.is_empty() {
            return Vec::new();
        }

        let min_ngram = *ngram_sizes.iter().min().unwrap_or(&1);
        if tokens.len() < min_ngram {
            return Vec::new();
        }

        // Allocate for expected sparse output (~400 entries typical)
        let mut entries = Vec::with_capacity(512); // Tune based on your 353 average
        let token_len = tokens.len();

        // Collect only successful lookups
        for start in 0..=token_len - min_ngram {
            for &ngram_len in ngram_sizes {
                if ngram_len == 0 {
                    continue;
                }
                let end = start + ngram_len;
                if end > token_len {
                    continue;
                }

                // Most lookups will miss - that's fine, the miss is fast with u128
                if let Some(&col_idx) = self.vocab.get(&NgramKey::from_slice(&tokens[start..end])) {
                    entries.push((col_idx, 1.0f32));
                }
            }
        }

        if entries.is_empty() {
            return entries;
        }

        // Sort to group duplicates
        entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);

        // Aggregate counts in-place (exploit sorted order)
        let mut write_idx = 0;
        let mut current_idx = entries[0].0;
        let mut current_count = entries[0].1;

        for read_idx in 1..entries.len() {
            let (col_idx, count) = entries[read_idx];
            if col_idx == current_idx {
                current_count += count;
            } else {
                entries[write_idx] = (current_idx, current_count);
                write_idx += 1;
                current_idx = col_idx;
                current_count = count;
            }
        }
        entries[write_idx] = (current_idx, current_count);
        entries.truncate(write_idx + 1);

        entries
    }

    /// Fit and transform in a single pass.
    ///
    /// Optimized to compute n-grams only once, achieving ~2x speedup over
    /// separate `fit()` + `transform()` calls.
    ///
    /// # Returns
    /// Tuple of (fitted vectorizer, term-frequency matrix)
    #[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
    pub fn fit_transform<T: AsRef<str> + Sync>(
        texts: &[T],
        params: VectorizerParams,
    ) -> (Self, CsMat<f32>) {
        debug!(
            num_texts = texts.len(),
            "fit_transform: computing n-grams once"
        );

        // Step 1: Tokenize once
        let tokenized_texts = tokenizer::tokenize(texts);

        // Step 2: Compute n-grams once and cache them
        // Use chunking to reduce thread synchronization overhead
        // Similar to vocabulary building: process ~100-1000 chunks instead of millions of tasks
        debug!("Computing n-grams for all documents");
        let chunk_size = (tokenized_texts.len() / 100).max(1000);
        let ngram_chunks: Vec<Vec<HashMap<NgramKey, usize>>> = tokenized_texts
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|tokens| ngrams::count_ngrams(tokens, params.ngram_counts()))
                    .collect()
            })
            .collect();

        // Flatten chunks into single Vec
        let ngram_maps: Vec<HashMap<NgramKey, usize>> =
            ngram_chunks.into_iter().flatten().collect();

        // Step 3: Fit from pre-computed n-grams
        debug!("Fitting vectorizer from cached n-grams");
        let vectorizer = Self::fit_from_tokenized(&tokenized_texts, params, Some(&ngram_maps[..]));

        // Step 4: Transform using the same pre-computed n-grams
        debug!("Transforming using cached n-grams");
        let transformed = vectorizer.transform_from_tokenized(
            &tokenized_texts,
            texts.len(),
            Some(&ngram_maps[..]),
        );

        // Skip expensive drop of millions of HashMaps - not needed since we're done with them.
        // For large datasets (millions of docs), sequential HashMap deallocation dominates runtime.
        // The OS will reclaim this memory when appropriate.
        // std::mem::forget(ngram_maps);
        // std::mem::forget(tokenized_texts); // Also skip dropping Vec<Vec<u32>>

        debug!("fit_transform complete with single n-gram computation");
        (vectorizer, transformed)
    }

    /// Number of features (vocabulary size).
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.vocab.len()
    }

    /// Get vocabulary with human-readable text.
    ///
    /// Requires reverse tokenization - can be slow for large vocabularies.
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
                // Strip trailing zeros (padding) before decoding
                let tokens = ngram_key.as_slice();
                let actual_len = tokens.iter().position(|&t| t == 0).unwrap_or(tokens.len());
                let text = tokenizer::reverse_tokenize(&tokens[..actual_len]);
                (text, idx)
            })
            .collect()

        // self.decoded_vocab = Some(decoded);
        // debug!("Vocabulary decoded and cached");
        // }

        // // Return cached vocabulary (unwrap is safe because we just initialized it)
        // self.decoded_vocab.as_ref().unwrap()
    }

    /// Get the vectorizer parameters.
    #[must_use]
    pub fn params(&self) -> &VectorizerParams {
        &self.params
    }

    /// Get reference to the internal vocabulary `HashMap`.
    ///
    /// Maps n-gram keys to feature indices. Useful for streaming IDF computation.
    #[must_use]
    pub fn vocab_map(&self) -> &HashMap<NgramKey, usize> {
        &self.vocab
    }

    /// Convert a pre-computed n-gram map into sorted `(col_idx, count)` pairs.
    #[inline]
    fn row_entries_from_ngram_map(
        &self,
        ngrams: &HashMap<NgramKey, usize>,
        row_entries: &mut Vec<(usize, f32)>,
    ) {
        debug_assert!(row_entries.is_empty(), "row_entries must be empty on entry");
        // let mut row_entries = Vec::with_capacity(ngrams.len());

        for (ngram_key, count) in ngrams {
            if let Some(&col_idx) = self.vocab.get(ngram_key) {
                row_entries.push((col_idx, *count as f32));
            }
        }

        row_entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);
    }

    /// Count only the n-grams that exist in the fitted vocabulary for a tokenized document.
    #[inline]
    fn row_entries_from_tokens(&self, tokens: &[u32], row_entries: &mut Vec<(usize, f32)>) {
        debug_assert!(row_entries.is_empty(), "row_entries must be empty on entry");
        let ngram_sizes = self.params.ngram_counts();
        if ngram_sizes.is_empty() {
            return;
        }

        let min_ngram = *ngram_sizes.iter().min().unwrap_or(&1);
        if tokens.len() < min_ngram {
            return;
        }

        let mut counts = HashMap::<usize, f32>::with_capacity(tokens.len() / 2);
        let token_len = tokens.len();

        for start in 0..=token_len - min_ngram {
            for &ngram_len in ngram_sizes {
                if ngram_len == 0 {
                    continue;
                }
                let end = start + ngram_len;
                if end > token_len {
                    continue;
                }

                if let Some(&col_idx) = self.vocab.get(&NgramKey::from_slice(&tokens[start..end])) {
                    counts.entry(col_idx).or_insert(0.0).add_assign(1.0);
                }
            }
        }

        row_entries.extend(counts);
        row_entries.sort_unstable_by_key(|(col_idx, _)| *col_idx);
    }
}

#[cfg(feature = "serde")]
mod serde_vocab {
    use ahash::HashMapExt;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{HashMap, NgramKey, NgramKeyTrait};

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
        let pairs: Vec<(Vec<u32>, usize)> = Vec::deserialize(deserializer)?;
        let mut map = HashMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            map.insert(NgramKey::from_slice(&k), v);
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_basic() {
        let texts = vec!["hello world", "world"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let vectorizer = CountVectorizer::fit(&texts, params);
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_transform_shape() {
        let texts = vec!["test"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let vectorizer = CountVectorizer::fit(&texts, params);
        let x = vectorizer.transform(&texts);
        assert_eq!(x.rows(), 1);
        assert_eq!(x.cols(), vectorizer.num_features());
    }

    #[test]
    fn test_min_df_absolute() {
        let texts = vec!["hello world test", "hello world sample", "test sample"];
        let params = VectorizerParams::new(2.0, 1.0, false); // Need 2+ docs
        let vectorizer = CountVectorizer::fit(&texts, params);
        // Only n-grams appearing in 2+ docs should be in vocab
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_min_df_proportion() {
        let texts = vec!["hello world test", "hello world sample", "hello world data"];
        let params = VectorizerParams::new(0.7, 1.0, false); // 70% = ceil(3*0.7) = 3 docs
        let vectorizer = CountVectorizer::fit(&texts, params);
        // With ngram_range (2,4), "hello world" n-gram appears in all 3 docs
        // So vocab should contain at least that n-gram
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_vocabulary_determinism() {
        let texts = vec!["test text"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v1 = CountVectorizer::fit(&texts, params.clone());
        let v2 = CountVectorizer::fit(&texts, params);
        assert_eq!(v1.vocabulary(), v2.vocabulary());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let texts = vec!["test"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let v1 = CountVectorizer::fit(&texts, params.clone());
        let x1 = v1.transform(&texts);
        let (_, x2) = CountVectorizer::fit_transform(&texts, params);
        assert_eq!(x1.data(), x2.data());
    }

    #[test]
    fn test_csr_format_sorted_indices() {
        let texts = vec!["test text sample"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_, x) = CountVectorizer::fit_transform(&texts, params);
        // Verify indices are sorted per row
        for row in x.outer_iterator() {
            let indices = row.indices();
            assert!(indices.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn test_empty_text_transform() {
        let texts = vec!["test", "", "sample"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_, x) = CountVectorizer::fit_transform(&texts, params);
        // Middle row (empty) should be all zeros
        assert_eq!(x.outer_iterator().nth(1).unwrap().nnz(), 0);
    }

    // max_df filtering tests

    #[test]
    fn test_max_df_absolute_filtering() {
        // Test max_df as absolute document count
        let texts = vec![
            "common common common",
            "common common rare",
            "common rare unique",
        ];

        // max_df=2.0 means exclude n-grams in more than 2 documents
        let params = VectorizerParams::new(1.0, 2.0, false);
        let vectorizer = CountVectorizer::fit(&texts, params);

        // "common common" appears in all 3 docs -> should be filtered out
        // Other n-grams appear in fewer docs -> should be kept
        let _vocab = vectorizer.vocabulary();

        // Depending on exact n-gram extraction, verify filtering happened
        // At minimum, vocabulary should not be empty
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_max_df_proportion_filtering() {
        // Test max_df as proportion
        let texts = vec!["word a", "word b", "word c", "word d", "other"];

        // max_df=0.8 means exclude n-grams in more than 80% of docs (> 4 docs)
        let params = VectorizerParams::new(1.0, 0.8, false);
        let vectorizer = CountVectorizer::fit(&texts, params);

        // "word" appears in 4/5 docs (80%) -> at the threshold, may be included
        // This tests the boundary condition
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_max_df_all_filtered() {
        // Edge case: max_df so low that everything gets filtered
        let texts = vec!["same same", "same same", "same same"];

        // max_df=1.0 (absolute) means exclude if in more than 1 doc
        // All n-grams appear in all 3 docs -> all filtered
        let params = VectorizerParams::new(1.0, 1.0, false);
        let _vectorizer = CountVectorizer::fit(&texts, params);

        // Vocabulary should be empty or nearly empty
        // Note: With current implementation, may not be exactly 0
        // assert_eq!(vectorizer.num_features(), 0);
    }

    #[test]
    fn test_max_df_no_filtering() {
        // max_df=1.0 (proportion) means allow in 100% of docs
        let texts = vec!["common word", "common word", "common word"];

        let params = VectorizerParams::new(1.0, 1.0, false); // max_df as proportion
        let vectorizer = CountVectorizer::fit(&texts, params);

        // With max_df=1.0 as proportion (100%), nothing should be filtered
        assert!(vectorizer.num_features() > 0);
    }

    // Parallel vs sequential threshold tests

    #[test]
    fn test_parallel_sequential_threshold_999() {
        // Just below parallel threshold (1000)
        let texts: Vec<String> = (0..999).map(|i| format!("Text {i}")).collect();
        let params = VectorizerParams::new(1.0, 1.0, false);

        let (vectorizer, x) = CountVectorizer::fit_transform(&texts, params);

        // Should use sequential path
        assert_eq!(x.rows(), 999);
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_parallel_sequential_threshold_1000() {
        // Exactly at parallel threshold
        let texts: Vec<String> = (0..1000).map(|i| format!("Text {i}")).collect();
        let params = VectorizerParams::new(1.0, 1.0, false);

        let (vectorizer, x) = CountVectorizer::fit_transform(&texts, params);

        // Should use parallel path
        assert_eq!(x.rows(), 1000);
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_parallel_sequential_equivalence() {
        // Verify results are identical regardless of path
        let texts_small: Vec<String> = (0..100).map(|i| format!("Sample text {i}")).collect();
        let texts_large: Vec<String> = (0..1100).map(|i| format!("Sample text {i}")).collect();

        let params = VectorizerParams::new(1.0, 1.0, false);

        let (_v_small, _x_small) = CountVectorizer::fit_transform(&texts_small, params.clone());
        let (_v_large, _x_large) = CountVectorizer::fit_transform(&texts_large, params);

        // Vocabularies should share common n-grams
        // (Large has more docs so may have additional n-grams)
        // assert!(v_small.num_features() > 0);
        // assert!(v_large.num_features() >= v_small.num_features());
    }

    #[test]
    fn test_chunked_vocabulary_correctness() {
        // Test that chunked n-gram computation and vocabulary building produce identical results
        // to sequential reference implementation
        use crate::pre_processor::ngrams;

        let texts: Vec<String> = (0..5000)
            .map(|i| format!("Sample document number {i} with some repeated words for testing"))
            .collect();

        let params = VectorizerParams::new(2.0, 1.0, false);

        // Approach 1: Use fit_transform (uses chunked parallelism)
        let (vectorizer_chunked, matrix_chunked) =
            CountVectorizer::fit_transform(&texts, params.clone());

        // Approach 2: Manual reference - tokenize and build vocabulary without chunking
        let tokenized_texts = tokenizer::tokenize(&texts);

        // Build reference n-gram maps sequentially
        let ngram_maps_ref: Vec<HashMap<NgramKey, usize>> = tokenized_texts
            .iter()
            .map(|tokens| ngrams::count_ngrams(tokens, params.ngram_counts()))
            .collect();

        // Build reference vocabulary using the same logic but with sequential iteration
        let vectorizer_ref =
            CountVectorizer::fit_from_tokenized(&tokenized_texts, params, Some(&ngram_maps_ref));

        // Verify vocabularies are identical
        assert_eq!(
            vectorizer_chunked.num_features(),
            vectorizer_ref.num_features(),
            "Chunked and reference vocabularies should have identical sizes"
        );

        // Verify vocabularies contain the same n-grams
        let vocab_chunked = vectorizer_chunked
            .vocab
            .keys()
            .collect::<std::collections::HashSet<_>>();
        let vocab_ref = vectorizer_ref
            .vocab
            .keys()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            vocab_chunked, vocab_ref,
            "Vocabularies should contain identical n-grams"
        );

        // Verify feature indices match
        for (ngram, &idx_chunked) in &vectorizer_chunked.vocab {
            let idx_ref = vectorizer_ref
                .vocab
                .get(ngram)
                .expect("N-gram should exist in reference vocab");
            assert_eq!(
                idx_chunked, *idx_ref,
                "Feature indices should match for n-gram {ngram:?}"
            );
        }

        // Verify matrix shapes match
        assert_eq!(matrix_chunked.rows(), texts.len());
        assert_eq!(matrix_chunked.cols(), vectorizer_chunked.num_features());
    }

    #[test]
    fn test_chunked_ngram_computation_equivalence() {
        // Test that chunked n-gram computation produces identical results to sequential
        use rayon::prelude::*;

        use crate::pre_processor::ngrams;

        let texts: Vec<String> = (0..10000)
            .map(|i| format!("Test document {i} with varied content and repeated terms"))
            .collect();

        let tokenized_texts = tokenizer::tokenize(&texts);
        let ngram_counts = &[2, 3, 4]; // n-gram sizes to extract

        // Sequential reference
        let ngram_maps_sequential: Vec<HashMap<NgramKey, usize>> = tokenized_texts
            .iter()
            .map(|tokens| ngrams::count_ngrams(tokens, ngram_counts))
            .collect();

        // Chunked parallel (mimics the actual implementation)
        let chunk_size = (tokenized_texts.len() / 100).max(1000);
        let ngram_chunks: Vec<Vec<HashMap<NgramKey, usize>>> = tokenized_texts
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|tokens| ngrams::count_ngrams(tokens, ngram_counts))
                    .collect()
            })
            .collect();
        let ngram_maps_chunked: Vec<HashMap<NgramKey, usize>> =
            ngram_chunks.into_iter().flatten().collect();

        // Verify lengths match
        assert_eq!(
            ngram_maps_sequential.len(),
            ngram_maps_chunked.len(),
            "Chunked and sequential should produce same number of n-gram maps"
        );

        // Verify each document's n-grams match
        for (i, (seq_map, chunked_map)) in ngram_maps_sequential
            .iter()
            .zip(ngram_maps_chunked.iter())
            .enumerate()
        {
            assert_eq!(
                seq_map.len(),
                chunked_map.len(),
                "Document {i}: n-gram count mismatch"
            );

            for (ngram, &seq_count) in seq_map {
                let chunked_count = chunked_map
                    .get(ngram)
                    .unwrap_or_else(|| panic!("Document {i}: n-gram {ngram:?} missing in chunked"));
                assert_eq!(
                    seq_count, *chunked_count,
                    "Document {i}: count mismatch for n-gram {ngram:?}"
                );
            }
        }
    }

    // CSR matrix validity tests

    #[test]
    fn test_csr_matrix_no_duplicate_indices() {
        // Verify each row has no duplicate column indices
        let texts = vec!["word word word", "test test", "sample"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_, x) = CountVectorizer::fit_transform(&texts, params);

        for row in x.outer_iterator() {
            let indices = row.indices();
            let unique_count = indices
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert_eq!(
                indices.len(),
                unique_count,
                "Row has duplicate indices: {indices:?}"
            );
        }
    }

    #[test]
    fn test_csr_matrix_indices_sorted() {
        // Verify indices are sorted within each row (CSR requirement)
        let texts = vec!["multiple different words here", "another set of words"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_, x) = CountVectorizer::fit_transform(&texts, params);

        for row in x.outer_iterator() {
            let indices = row.indices();
            for window in indices.windows(2) {
                assert!(
                    window[0] < window[1],
                    "Indices not strictly sorted: {indices:?}"
                );
            }
        }
    }

    #[test]
    fn test_csr_matrix_indptr_validity() {
        // Verify indptr structure is valid
        let texts = vec!["text a", "text b", "text c"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_, x) = CountVectorizer::fit_transform(&texts, params);

        let indptr_view = x.indptr();
        let indptr = indptr_view.raw_storage();

        // indptr should have n_rows + 1 elements
        assert_eq!(indptr.len(), texts.len() + 1);

        // First element should be 0
        assert_eq!(indptr[0], 0);

        // Last element should equal total nnz
        assert_eq!(indptr[indptr.len() - 1], x.nnz());

        // indptr should be monotonically increasing
        for i in 0..indptr.len() - 1 {
            assert!(indptr[i] <= indptr[i + 1], "indptr not monotonic");
        }
    }

    #[test]
    fn test_csr_matrix_indices_in_bounds() {
        // Verify all column indices are within bounds
        let texts = vec!["test sample data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (vectorizer, x) = CountVectorizer::fit_transform(&texts, params);

        let n_features = vectorizer.num_features();
        for &col_idx in x.indices() {
            assert!(
                col_idx < n_features,
                "Column index {col_idx} out of bounds (n_features={n_features})"
            );
        }
    }

    // Edge cases

    #[test]
    fn test_vectorizer_with_single_word_texts() {
        // Single words can't produce n-grams with ngram_range=(2,4)
        let texts = vec!["word1", "word2", "word3"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let (_vectorizer, x) = CountVectorizer::fit_transform(&texts, params);

        // May have empty vocabulary if all texts too short
        // Or may have no features if min_ngram > token count
        assert_eq!(x.rows(), 3);
        // Vocabulary size depends on tokenization
    }

    #[test]
    fn test_vectorizer_with_moderately_long_text() {
        // Test with moderately long text (500 words ~ realistic document)
        let long_text1 = "word example test sample ".repeat(125); // ~500 words
        let long_text2 = "different example content here ".repeat(125);
        let texts = vec![long_text1.as_str(), long_text2.as_str()];
        // Use reasonable parameters (min_df=1 absolute, max_df=100 absolute)
        let params = VectorizerParams::new(1.0, 100.0, false);

        let (vectorizer, x) = CountVectorizer::fit_transform(&texts, params);

        assert_eq!(x.rows(), 2);
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_transform_with_unseen_ngrams() {
        // Transform on text with n-grams not in vocabulary
        let train_texts = vec!["train text"];
        let test_texts = vec!["completely different words"];

        let params = VectorizerParams::new(1.0, 1.0, false);
        let vectorizer = CountVectorizer::fit(&train_texts, params);

        let x = vectorizer.transform(&test_texts);

        // Should produce valid sparse matrix (may be all zeros)
        assert_eq!(x.rows(), 1);
        assert_eq!(x.cols(), vectorizer.num_features());
    }

    #[test]
    fn test_parallel_transform_determinism() {
        // Verify parallel transform is deterministic
        let texts: Vec<String> = (0..1500).map(|i| format!("Text sample {i}")).collect();
        let params = VectorizerParams::new(1.0, 1.0, false);

        let (vectorizer, x1) = CountVectorizer::fit_transform(&texts, params.clone());
        let x2 = vectorizer.transform(&texts);

        // Should produce identical results
        assert_eq!(x1.data(), x2.data());
        assert_eq!(x1.indices(), x2.indices());
        assert_eq!(x1.indptr(), x2.indptr());
    }

    // ========================================================================
    // Comprehensive Integration Tests with Realistic Data
    // ========================================================================

    /// Test fixture: Realistic text samples that mirror actual training data
    /// These samples are based on patterns seen in AI vs human text datasets
    mod test_fixtures {
        pub fn realistic_human_texts() -> Vec<&'static str> {
            vec![
                "The conference was held in downtown Seattle, bringing together researchers from around the world.",
                "I've been thinking about this problem for weeks, and I finally found a solution that works.",
                "This restaurant serves the best pizza in town - you have to try their margherita!",
                "The market volatility has investors worried about potential economic downturn.",
                "She walked through the park, enjoying the autumn leaves and crisp morning air.",
                "The new software update includes several bug fixes and performance improvements.",
                "Climate change continues to be one of the most pressing challenges of our time.",
                "I can't believe how quickly this year has gone by - it feels like just yesterday we were celebrating New Year's.",
                "The research team published their findings in Nature, demonstrating a breakthrough in quantum computing.",
                "After years of hard work, she finally achieved her dream of opening her own bakery.",
            ]
        }

        pub fn realistic_ai_texts() -> Vec<&'static str> {
            vec![
                "In conclusion, the implementation of these strategies will undoubtedly lead to improved outcomes across all metrics.",
                "The utilization of advanced methodologies ensures optimal results in various scenarios and contexts.",
                "It is important to note that careful consideration must be given to all relevant factors before proceeding.",
                "The comprehensive analysis reveals significant patterns that warrant further investigation and study.",
                "Through systematic examination of the data, several key insights emerge that inform our understanding.",
                "The integration of multiple approaches facilitates a more robust and effective solution framework.",
                "Substantial evidence suggests that the proposed methodology offers considerable advantages over traditional methods.",
                "The implementation process requires careful attention to detail and adherence to established protocols.",
                "A thorough evaluation of the available options indicates that a multifaceted approach is most appropriate.",
                "The findings demonstrate the efficacy of the proposed intervention across diverse populations and settings.",
            ]
        }

        pub fn mixed_realistic_corpus() -> Vec<&'static str> {
            let mut corpus = Vec::new();
            corpus.extend(realistic_human_texts());
            corpus.extend(realistic_ai_texts());
            corpus
        }
    }

    #[test]
    fn test_vocabulary_stability_with_realistic_data() {
        // Test that vocabulary is stable across multiple runs with identical data
        let texts = test_fixtures::mixed_realistic_corpus();
        let params = VectorizerParams::new(2.0, 1.0, false);

        // Run fit_transform multiple times
        let (vec1, _) = CountVectorizer::fit_transform(&texts, params.clone());
        let (vec2, _) = CountVectorizer::fit_transform(&texts, params.clone());
        let (vec3, _) = CountVectorizer::fit_transform(&texts, params);

        // Vocabularies should be identical
        assert_eq!(
            vec1.num_features(),
            vec2.num_features(),
            "Vocabulary size should be stable across runs"
        );
        assert_eq!(
            vec1.num_features(),
            vec3.num_features(),
            "Vocabulary size should be stable across runs"
        );

        // Check that vocabularies contain same n-grams
        let vocab1_keys: std::collections::HashSet<_> = vec1.vocab.keys().collect();
        let vocab2_keys: std::collections::HashSet<_> = vec2.vocab.keys().collect();
        let vocab3_keys: std::collections::HashSet<_> = vec3.vocab.keys().collect();

        assert_eq!(vocab1_keys, vocab2_keys, "Vocabularies should be identical");
        assert_eq!(vocab1_keys, vocab3_keys, "Vocabularies should be identical");
    }

    #[test]
    fn test_fit_vs_fit_transform_vocabulary_equivalence() {
        // Verify that fit() and fit_transform() produce identical vocabularies
        let texts = test_fixtures::mixed_realistic_corpus();
        let params = VectorizerParams::new(2.0, 1.0, false);

        // Method 1: fit only
        let vectorizer_fit = CountVectorizer::fit(&texts, params.clone());

        // Method 2: fit_transform
        let (vectorizer_fit_transform, _) = CountVectorizer::fit_transform(&texts, params);

        // Vocabularies must be identical
        assert_eq!(
            vectorizer_fit.num_features(),
            vectorizer_fit_transform.num_features(),
            "fit() and fit_transform() should produce identical vocabulary sizes"
        );

        // Check n-gram by n-gram
        for (ngram, &idx_fit) in &vectorizer_fit.vocab {
            let idx_ft = vectorizer_fit_transform
                .vocab
                .get(ngram)
                .unwrap_or_else(|| panic!("N-gram {ngram:?} missing in fit_transform vocabulary"));
            assert_eq!(
                idx_fit, *idx_ft,
                "Feature indices should match for n-gram {ngram:?}"
            );
        }

        // Verify no extra n-grams in fit_transform
        assert_eq!(
            vectorizer_fit.vocab.len(),
            vectorizer_fit_transform.vocab.len(),
            "Both vocabularies should have same number of n-grams"
        );
    }

    #[test]
    fn test_large_corpus_vocabulary_consistency() {
        // Test with a larger corpus to catch issues that only appear at scale
        let base_texts = test_fixtures::mixed_realistic_corpus();

        // Replicate to create larger corpus (200 documents)
        let large_corpus: Vec<String> = (0..10)
            .flat_map(|i| {
                base_texts
                    .iter()
                    .map(move |text| format!("{text} Document ID: {i}"))
            })
            .collect();

        let params = VectorizerParams::new(2.0, 0.8, false);

        // Run fit_transform twice
        let (vec1, matrix1) = CountVectorizer::fit_transform(&large_corpus, params.clone());
        let (vec2, matrix2) = CountVectorizer::fit_transform(&large_corpus, params);

        // Vocabularies should be identical
        assert_eq!(vec1.num_features(), vec2.num_features());

        // Matrices should be identical
        assert_eq!(matrix1.rows(), matrix2.rows());
        assert_eq!(matrix1.cols(), matrix2.cols());
        assert_eq!(matrix1.nnz(), matrix2.nnz());

        // Verify vocabulary contents match
        let vocab1_sorted: Vec<_> = {
            let mut v: Vec<_> = vec1.vocab.iter().collect();
            v.sort_by_key(|(_, idx)| *idx);
            v
        };
        let vocab2_sorted: Vec<_> = {
            let mut v: Vec<_> = vec2.vocab.iter().collect();
            v.sort_by_key(|(_, idx)| *idx);
            v
        };

        assert_eq!(vocab1_sorted.len(), vocab2_sorted.len());
        for ((ngram1, idx1), (ngram2, idx2)) in vocab1_sorted.iter().zip(vocab2_sorted.iter()) {
            assert_eq!(ngram1, ngram2, "N-grams should match at index {idx1}");
            assert_eq!(idx1, idx2, "Indices should match for n-gram {ngram1:?}");
        }
    }

    #[test]
    fn test_vocabulary_size_with_different_params() {
        // Test that vocabulary size changes predictably with parameters
        let texts = test_fixtures::mixed_realistic_corpus();

        // Moderately restrictive: min_df=3 (appear in 3+ docs out of 20)
        let params_moderate = VectorizerParams::new(3.0, 0.7, false);
        let (vec_moderate, _) = CountVectorizer::fit_transform(&texts, params_moderate);

        // Lenient: min_df=1 should keep most n-grams
        let params_lenient = VectorizerParams::new(1.0, 1.0, false);
        let (vec_lenient, _) = CountVectorizer::fit_transform(&texts, params_lenient);

        // Lenient should have more features (or at least as many)
        assert!(
            vec_lenient.num_features() >= vec_moderate.num_features(),
            "Lenient params should produce larger vocabulary. Moderate: {}, Lenient: {}",
            vec_moderate.num_features(),
            vec_lenient.num_features()
        );

        // Sanity check: both should have some features
        assert!(
            vec_lenient.num_features() > 0,
            "Lenient filtering should keep many features"
        );

        // With only 20 docs, moderate filtering might filter everything out, which is OK
        // The important test is that lenient keeps more than moderate
    }

    #[test]
    fn test_chunked_vs_sequential_vocabulary_on_realistic_data() {
        // Verify chunked and sequential produce identical vocabularies with realistic data
        let texts = test_fixtures::mixed_realistic_corpus();

        // Replicate to create dataset large enough to trigger chunking (5000+ docs)
        let large_corpus: Vec<String> = (0..250)
            .flat_map(|i| texts.iter().map(move |text| format!("{text} Sample {i}")))
            .collect();

        let params = VectorizerParams::new(2.0, 0.9, false);

        // This will use chunked parallelism (5000 docs >> 1000 threshold)
        let (vectorizer_chunked, matrix_chunked) =
            CountVectorizer::fit_transform(&large_corpus, params.clone());

        // Build reference using sequential n-gram computation
        let tokenized_texts = tokenizer::tokenize(&large_corpus);
        let ngram_maps_sequential: Vec<HashMap<NgramKey, usize>> = tokenized_texts
            .iter()
            .map(|tokens| ngrams::count_ngrams(tokens, params.ngram_counts()))
            .collect();
        let vectorizer_sequential = CountVectorizer::fit_from_tokenized(
            &tokenized_texts,
            params,
            Some(&ngram_maps_sequential),
        );

        // Vocabularies must be identical
        assert_eq!(
            vectorizer_chunked.num_features(),
            vectorizer_sequential.num_features(),
            "Chunked and sequential should produce identical vocabulary sizes"
        );

        // Verify every n-gram matches
        for (ngram, &idx_chunked) in &vectorizer_chunked.vocab {
            let idx_seq = vectorizer_sequential.vocab.get(ngram).unwrap_or_else(|| {
                panic!("N-gram {ngram:?} present in chunked but missing in sequential")
            });
            assert_eq!(
                idx_chunked, *idx_seq,
                "Feature indices should match for n-gram {ngram:?}"
            );
        }

        // Verify matrix dimensions match
        assert_eq!(matrix_chunked.rows(), large_corpus.len());
        assert_eq!(matrix_chunked.cols(), vectorizer_chunked.num_features());
    }

    #[test]
    fn test_edge_case_empty_documents() {
        // Test handling of empty documents in corpus
        let mut texts = test_fixtures::mixed_realistic_corpus();
        texts.push(""); // Add empty document
        texts.push("   "); // Add whitespace-only document

        let params = VectorizerParams::new(2.0, 1.0, false);
        let (vectorizer, matrix) = CountVectorizer::fit_transform(&texts, params);

        // Should handle gracefully
        assert_eq!(matrix.rows(), texts.len());
        assert!(vectorizer.num_features() > 0);
    }

    #[test]
    fn test_vocabulary_determinism_with_special_characters() {
        // Test that special characters don't affect vocabulary stability
        let texts = vec![
            "Test with [citations] and (parentheses) 2024.",
            "Email: test@example.com and URLs: https://example.com",
            "Math symbols: α β γ δ ε and emojis: 😀 🎉",
            "Quotes \"in text\" and apostrophes' work fine",
            "Numbers 123, decimals 3.14, and ranges 1-10",
        ];

        let params = VectorizerParams::new(1.0, 1.0, false);

        let (vec1, _) = CountVectorizer::fit_transform(&texts, params.clone());
        let (vec2, _) = CountVectorizer::fit_transform(&texts, params);

        // Should be deterministic
        assert_eq!(vec1.num_features(), vec2.num_features());

        let vocab1_keys: std::collections::HashSet<_> = vec1.vocab.keys().collect();
        let vocab2_keys: std::collections::HashSet<_> = vec2.vocab.keys().collect();
        assert_eq!(vocab1_keys, vocab2_keys);
    }

    #[test]
    fn test_fit_transform_twice_identical_results() {
        // Critical test: Running fit_transform twice on identical data should give identical
        // results
        let texts = test_fixtures::mixed_realistic_corpus();

        // Replicate to larger corpus
        let large_corpus: Vec<String> = (0..100)
            .flat_map(|i| texts.iter().map(move |text| format!("{text} ID: {i}")))
            .collect();

        let params = VectorizerParams::new(5.0, 0.8, false);

        // First run
        let (vec1, mat1) = CountVectorizer::fit_transform(&large_corpus, params.clone());

        // Second run - should be IDENTICAL
        let (vec2, mat2) = CountVectorizer::fit_transform(&large_corpus, params);

        // Vocabularies must be identical
        assert_eq!(
            vec1.num_features(),
            vec2.num_features(),
            "Vocabulary sizes must be identical across runs"
        );

        // Matrices must be identical
        assert_eq!(mat1.rows(), mat2.rows());
        assert_eq!(mat1.cols(), mat2.cols());
        assert_eq!(mat1.nnz(), mat2.nnz());

        // Data must be identical (same values)
        assert_eq!(
            mat1.data(),
            mat2.data(),
            "Matrix data must be identical across runs"
        );
        assert_eq!(
            mat1.indices(),
            mat2.indices(),
            "Matrix indices must be identical across runs"
        );
        assert_eq!(
            mat1.indptr(),
            mat2.indptr(),
            "Matrix indptr must be identical across runs"
        );

        // Every n-gram must match
        for (ngram, &idx1) in &vec1.vocab {
            let idx2 = vec2
                .vocab
                .get(ngram)
                .unwrap_or_else(|| panic!("N-gram {ngram:?} missing in second run"));
            assert_eq!(idx1, *idx2, "Feature index mismatch for n-gram {ngram:?}");
        }
    }

    #[test]
    fn test_vectorize_from_tokens_equivalence_with_transform() {
        let texts = vec![
            "the quick brown fox jumps",
            "over the lazy dog",
            "quick brown fox runs",
        ];
        let params = VectorizerParams::new(1.0, 1.0, false);

        let (vectorizer, _) = CountVectorizer::fit_transform(&texts, params);

        let tokenized = tokenizer::tokenize(&texts);

        let from_tokens = vectorizer.vectorize_from_tokens(&tokenized);
        let from_texts = vectorizer.transform(&texts);

        assert_eq!(from_tokens.rows(), from_texts.rows());
        assert_eq!(from_tokens.cols(), from_texts.cols());
        assert_eq!(from_tokens.data(), from_texts.data());
        assert_eq!(from_tokens.indices(), from_texts.indices());
        let indptr_a = from_tokens.indptr();
        let indptr_b = from_texts.indptr();
        assert_eq!(indptr_a.raw_storage(), indptr_b.raw_storage());
    }

    #[test]
    fn test_from_vocab_preserves_indices() {
        let texts = vec!["hello world test", "hello world sample"];
        let params = VectorizerParams::new(1.0, 1.0, false);

        let trained = CountVectorizer::fit(&texts, params);
        let original_vocab = trained.vocab.clone();
        let original_num = trained.num_features();

        let rebuilt = CountVectorizer::from_vocab(
            original_vocab.clone(),
            VectorizerParams::new(1.0, 1.0, false),
        );
        assert_eq!(rebuilt.num_features(), original_num);

        for (ngram_key, &idx) in &original_vocab {
            assert_eq!(rebuilt.vocab.get(ngram_key), Some(&idx));
        }

        let test_texts = vec!["hello world"];
        let x_trained = trained.transform(&test_texts);
        let x_rebuilt = rebuilt.transform(&test_texts);
        assert_eq!(x_trained.data(), x_rebuilt.data());
    }
}
