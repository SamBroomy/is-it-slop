use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use dashmap::DashMap;
#[cfg(feature = "progress-bars")]
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::NGRAM_CONST_KEY;

/// Type alias for n-gram keys using `SmallVec` for stack allocation of small n-grams
pub type NgramKey = SmallVec<[u32; NGRAM_CONST_KEY]>;

/// Count n-grams in a sequence of tokens.
///
/// This optimized version uses `SmallVec` to avoid heap allocations for typical n-gram sizes (≤8 tokens).
/// The function pre-allocates `HashMap` capacity and uses the efficient `and_modify` pattern.
///
/// # Arguments
/// * `tokens` - Sequence of token IDs
/// * `ngram_range` - Range of n-gram sizes to extract
///
/// # Returns
/// `HashMap` mapping n-gram (as `SmallVec`) to count
pub fn count_ngrams(tokens: &[u32], ngram_range: &[usize]) -> HashMap<NgramKey, usize> {
    // Pre-compute capacity to reduce HashMap resizing
    let max_possible_ngrams: usize = ngram_range
        .iter()
        .map(|&n| tokens.len().saturating_sub(n.saturating_sub(1)))
        .sum();

    // Assume ~33% unique n-grams (typical for text data)
    // This reduces rehashing overhead significantly
    let estimated_capacity = (max_possible_ngrams / 3).max(16);
    let mut ngram_counter = HashMap::with_capacity(estimated_capacity);

    for &n in ngram_range {
        // Early exit if not enough tokens for this n-gram size
        if n == 0 || n > tokens.len() {
            continue;
        }

        for window in tokens.windows(n) {
            // SmallVec::from_slice uses stack allocation for n ≤ 8
            let key = SmallVec::from_slice(window);

            // and_modify pattern is more efficient than or_insert + increment
            ngram_counter
                .entry(key)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }
    ngram_counter
}

/// Extract unique n-grams from a sequence of tokens (no counting).
///
/// This is more efficient than `count_ngrams` when you only need to know
/// which n-grams appear, not how many times (e.g., for document frequency).
///
/// # Arguments
/// * `tokens` - Sequence of token IDs
/// * `ngram_range` - Range of n-gram sizes to extract
///
/// # Returns
/// `HashSet` of unique n-grams (as `SmallVec`)
pub fn unique_ngrams(tokens: &[u32], ngram_range: &[usize]) -> HashSet<NgramKey> {
    // Pre-compute capacity to reduce HashSet resizing
    let max_possible_ngrams: usize = ngram_range
        .iter()
        .map(|&n| tokens.len().saturating_sub(n.saturating_sub(1)))
        .sum();

    // Assume ~50% unique n-grams for HashSet (higher than HashMap since no count aggregation)
    let estimated_capacity = (max_possible_ngrams / 2).max(16);
    let mut unique = HashSet::with_capacity(estimated_capacity);

    for &n in ngram_range {
        // Early exit if not enough tokens for this n-gram size
        if n == 0 || n > tokens.len() {
            continue;
        }

        for window in tokens.windows(n) {
            // SmallVec::from_slice uses stack allocation for n ≤ 8
            unique.insert(SmallVec::from_slice(window));
        }
    }
    unique
}

/// Build vocabulary from tokenized texts using `SmallVec` keys.
///
/// Optimized to use `unique_ngrams` instead of `count_ngrams` since we only
/// need to track which n-grams appear in each document, not how many times.
///
/// # Arguments
/// * `tokenized_texts` - Slice of tokenized documents
/// * `ngram_range` - Range of n-gram sizes to extract
///
/// # Returns
/// `DashMap` mapping n-gram (as `SmallVec`) to document frequency
#[cfg(feature = "progress-bars")]
pub fn build_vocabulary(
    tokenized_texts: &[Vec<u32>],
    ngram_range: &[usize],
) -> DashMap<NgramKey, usize, ahash::RandomState> {
    let vocab_df = DashMap::with_hasher(ahash::RandomState::default());

    // Parallel iteration over documents with progress bar
    tokenized_texts.par_iter().progress().for_each(|tokens| {
        // Use unique_ngrams instead of count_ngrams - we only need presence, not counts
        let ngrams = unique_ngrams(tokens, ngram_range);

        // For each unique n-gram in this document, increment its document frequency
        for ngram_key in ngrams {
            vocab_df
                .entry(ngram_key)
                .and_modify(|df| *df += 1)
                .or_insert(1);
        }
    });

    vocab_df
}

#[cfg(not(feature = "progress-bars"))]
pub fn build_vocabulary(
    tokenized_texts: &[Vec<u32>],
    ngram_range: &[usize],
) -> DashMap<NgramKey, usize, ahash::RandomState> {
    let vocab_df = DashMap::with_hasher(ahash::RandomState::default());

    // Parallel iteration over documents (no progress bar)
    tokenized_texts.par_iter().for_each(|tokens| {
        // Use unique_ngrams instead of count_ngrams - we only need presence, not counts
        let ngrams = unique_ngrams(tokens, ngram_range);

        // For each unique n-gram in this document, increment its document frequency
        for ngram_key in ngrams {
            vocab_df
                .entry(ngram_key)
                .and_modify(|df| *df += 1)
                .or_insert(1);
        }
    });

    vocab_df
}

// TODO: profile this function against the current implementation of `build_vocabulary`
// Removes dashmap dependency if we use this over the other one
pub fn build_vocabulary_merge(
    tokenized_texts: &[Vec<u32>],
    ngram_range: &[usize],
) -> HashMap<NgramKey, usize> {
    // Phase 1: Parallel - each thread extracts unique n-grams (no counting needed)
    let partial_results: Vec<HashSet<_>> = tokenized_texts
        .par_iter()
        .map(|tokens| unique_ngrams(tokens, ngram_range))
        .collect();

    let estimated_size = partial_results
        .iter()
        .map(std::collections::HashSet::len)
        .sum::<usize>()
        .max(16);

    // Phase 2: Sequential merge - count document frequency
    let mut vocab_df =
        HashMap::with_capacity_and_hasher(estimated_size, ahash::RandomState::default());
    for partial in partial_results {
        for key in partial {
            *vocab_df.entry(key).or_insert(0) += 1;
        }
    }
    vocab_df
}
