//! Token n-gram extraction and vocabulary building.
//!
//! This module provides efficient n-gram extraction from token sequences (not characters
//! or words). N-grams are sequences of consecutive BPE token IDs.
//!
//! # Key Concepts
//!
//! - **Token n-grams**: Sequences of 2-4 consecutive token IDs (e.g., `[tok1, tok2, tok3]`)
//! - **u128 storage**: N-grams stored as compact 128-bit keys (4 tokens × 32 bits)
//! - **Parallel vocabulary building**: Uses `DashMap` for thread-safe concurrent updates
//!
//! # Storage Format
//!
//! N-grams up to 4 tokens are packed into a `u128`:
//! ```text
//! token[0] | token[1] | token[2] | token[3]
//! 32 bits  | 32 bits  | 32 bits  | 32 bits
//! ```
//!
//! Shorter n-grams (2-3 tokens) have trailing zeros that must be stripped before decoding.
//!
//! # Usage
//!
//! This module is internal to the preprocessing pipeline. N-grams are extracted
//! automatically during vectorization via [`TfidfVectorizer`] or [`CountVectorizer`].
//!
//! [`TfidfVectorizer`]: crate::pre_processor::TfidfVectorizer
//! [`CountVectorizer`]: crate::pre_processor::CountVectorizer

use std::{fmt::Debug, hash::Hash, ops::AddAssign};

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use dashmap::DashMap;
#[cfg(feature = "progress-bars")]
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use tracing::instrument;

/// Trait for n-gram key types supporting different storage strategies.
pub trait NgramKeyTrait:
    Hash + Eq + PartialEq + PartialOrd + Ord + Clone + Debug + Send + Sync
{
    /// Creates an n-gram key from a slice of token IDs.
    fn from_slice(tokens: &[u32]) -> Self;
    /// Returns the token IDs as a slice.
    fn as_slice(&self) -> &[u32];
    /// Converts the token IDs to a vector.
    #[allow(dead_code)]
    fn to_vec(&self) -> Vec<u32>;
}

/// Compact n-gram storage using u128 for up to 4 tokens.
///
/// Stores token sequences by packing four u32 token IDs into a single u128:
/// - Bits 0-31: token\[0\]
/// - Bits 32-63: token\[1\]
/// - Bits 64-95: token\[2\]
/// - Bits 96-127: token\[3\]
///
/// For n-grams with fewer than 4 tokens, the unused positions contain zeros.
/// These trailing zeros must be stripped before decoding to prevent artifacts.
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    rkyv(derive(Debug, PartialEq, Eq, Hash))
)]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NgramKey(u128);

impl NgramKeyTrait for NgramKey {
    fn from_slice(tokens: &[u32]) -> Self {
        debug_assert!(tokens.len() <= 4, "NgramKeyShort supports up to 4 tokens");
        let mut key = 0u128;
        for (i, &token) in tokens.iter().enumerate().take(4) {
            key |= u128::from(token) << (i * 32);
        }
        Self(key)
    }

    fn as_slice(&self) -> &[u32] {
        // SAFETY: Reinterpret the u128 storage as a 4-element u32 array.
        // Cast from `*const u128` to `*const u32` preserves proper alignment for u32.
        let ptr = &raw const self.0;
        let ptr = ptr.cast::<u32>();
        let len = size_of::<u128>() / size_of::<u32>(); // 4
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    fn to_vec(&self) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(4);
        for i in 0..4 {
            let token = ((self.0 >> (i * 32)) & 0xFFFF_FFFF) as u32;
            tokens.push(token);
        }
        tokens
    }
}
/// Count n-grams in a sequence of tokens.
///
/// This optimized version uses `SmallVec` to avoid heap allocations for typical n-gram sizes (≤8
/// tokens). The function pre-allocates `HashMap` capacity and uses the efficient `and_modify`
/// pattern.
///
/// # Arguments
/// * `tokens` - Sequence of token IDs
/// * `ngram_range` - Range of n-gram sizes to extract
///
/// # Returns
/// `HashMap` mapping n-gram (as `SmallVec`) to count
#[instrument(level = "trace", skip(tokens, ngram_range), fields(num_tokens = tokens.len(), ngram_sizes = ngram_range.len()))]
pub fn count_ngrams(tokens: &[u32], ngram_range: &[usize]) -> HashMap<NgramKey, usize> {
    if tokens.is_empty() || ngram_range.is_empty() {
        return HashMap::new();
    }

    let min_n = *ngram_range.iter().min().unwrap();

    // Early exit if not enough tokens
    if tokens.len() < min_n {
        return HashMap::new();
    }

    // Pre-compute capacity to reduce HashMap resizing
    let max_possible_ngrams: usize = ngram_range
        .iter()
        .map(|&n| tokens.len().saturating_sub(n - 1))
        .sum();

    let unique_ratio = if tokens.len() < 100 {
        0.6
    } else if tokens.len() < 1000 {
        0.4
    } else {
        0.25
    };

    let capacity = ((max_possible_ngrams as f32 * unique_ratio) as usize).max(16);
    let mut ngram_counter = HashMap::with_capacity(capacity);

    for &n in ngram_range {
        // Early exit if not enough tokens for this n-gram size
        if n == 0 || n > tokens.len() {
            continue;
        }

        for window in tokens.windows(n) {
            let key = NgramKey::from_slice(window);

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
#[instrument(level = "trace", skip(tokens, ngram_range), fields(num_tokens = tokens.len(), ngram_sizes = ngram_range.len()))]
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
            unique.insert(NgramKey::from_slice(window));
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
/// `HashMap` mapping n-gram (as `SmallVec`) to document frequency
#[instrument(level = "debug", skip(tokenized_texts, ngram_range), fields(num_texts = tokenized_texts.len(), ngram_sizes = ngram_range.len()))]
pub fn build_vocabulary(
    tokenized_texts: &[Vec<u32>],
    ngram_range: &[usize],
) -> DashMap<NgramKey, usize, ahash::RandomState> {
    let vocab_df = DashMap::with_hasher(ahash::RandomState::default());

    // Parallel iteration over documents with progress bar
    let iter = tokenized_texts.par_iter();
    #[cfg(feature = "progress-bars")]
    let iter = iter.progress();

    iter.for_each(|tokens| {
        // Use unique_ngrams instead of count_ngrams - we only need presence, not counts
        let ngrams = unique_ngrams(tokens, ngram_range);

        // For each unique n-gram in this document, increment its document frequency
        for ngram_key in ngrams {
            vocab_df.entry(ngram_key).or_insert(0).add_assign(1);
        }
    });
    vocab_df
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_ngrams_basic() {
        let tokens = vec![1, 2, 3, 4];
        let ngrams = count_ngrams(&tokens, &[2]);
        assert_eq!(ngrams.len(), 3); // [1,2], [2,3], [3,4]
    }

    #[test]
    fn test_count_ngrams_repeated() {
        let tokens = vec![1, 2, 1, 2];
        let ngrams = count_ngrams(&tokens, &[2]);
        let key = NgramKey::from_slice(&[1, 2]);
        assert_eq!(ngrams.get(&key), Some(&2));
    }

    #[test]
    fn test_count_ngrams_empty() {
        let ngrams = count_ngrams(&[], &[2]);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_unique_ngrams() {
        let tokens = vec![1, 2, 1, 2];
        let ngrams = unique_ngrams(&tokens, &[2]);
        assert_eq!(ngrams.len(), 2); // [1,2] and [2,1]
    }

    #[test]
    fn test_build_vocabulary() {
        let docs = vec![vec![1, 2], vec![2, 3], vec![1, 2]];
        let vocab = build_vocabulary(&docs, &[2]);
        let key = NgramKey::from_slice(&[1, 2]);
        assert_eq!(*vocab.get(&key).unwrap(), 2); // df=2 (appears in 2 docs)
    }

    #[test]
    fn test_ngram_key_round_trip() {
        let tokens = vec![1, 2, 3];
        let key = NgramKey::from_slice(&tokens);
        let result = key.to_vec();
        assert_eq!(&result[..3], &tokens[..]);
    }

    // u128 storage edge cases

    #[test]
    fn test_ngram_key_trailing_zeros_2gram() {
        // Regression test for v5.0 vocabulary decode bug
        // Issue: NgramKey(u128) has trailing zeros for n-grams < 4 tokens
        // Fix: Strip trailing zeros before reverse_tokenize()
        let tokens = vec![100, 200];
        let key = NgramKey::from_slice(&tokens);
        let as_slice = key.as_slice();

        // as_slice returns all 4 u32s, but only first 2 are meaningful
        assert_eq!(as_slice.len(), 4);
        assert_eq!(as_slice[0], 100);
        assert_eq!(as_slice[1], 200);
        assert_eq!(as_slice[2], 0); // Trailing zero
        assert_eq!(as_slice[3], 0); // Trailing zero

        // to_vec also returns 4 elements with trailing zeros
        let as_vec = key.to_vec();
        assert_eq!(as_vec.len(), 4);
        assert_eq!(&as_vec[..2], &tokens[..]);
    }

    #[test]
    fn test_ngram_key_trailing_zeros_3gram() {
        let tokens = vec![100, 200, 300];
        let key = NgramKey::from_slice(&tokens);
        let as_slice = key.as_slice();

        assert_eq!(as_slice.len(), 4);
        assert_eq!(&as_slice[..3], &tokens[..]);
        assert_eq!(as_slice[3], 0); // Trailing zero
    }

    #[test]
    fn test_ngram_key_exactly_4_tokens() {
        // Maximum capacity: 4 tokens fit exactly in u128
        let tokens = vec![100, 200, 300, 400];
        let key = NgramKey::from_slice(&tokens);
        let as_slice = key.as_slice();

        assert_eq!(as_slice.len(), 4);
        assert_eq!(as_slice, &tokens[..]);

        // No trailing zeros when all 4 slots used
        let as_vec = key.to_vec();
        assert_eq!(as_vec, tokens);
    }

    #[test]
    fn test_ngram_key_as_slice_correct_length() {
        // Verify as_slice always returns 4 elements (internal representation)
        let key_1 = NgramKey::from_slice(&[1]);
        assert_eq!(key_1.as_slice().len(), 4);

        let key_2 = NgramKey::from_slice(&[1, 2]);
        assert_eq!(key_2.as_slice().len(), 4);

        let key_3 = NgramKey::from_slice(&[1, 2, 3]);
        assert_eq!(key_3.as_slice().len(), 4);

        let key_4 = NgramKey::from_slice(&[1, 2, 3, 4]);
        assert_eq!(key_4.as_slice().len(), 4);
    }

    #[test]
    fn test_ngram_key_zero_token_value() {
        // Token ID 0 is valid, should not be confused with padding
        let tokens = vec![0, 1, 2];
        let key = NgramKey::from_slice(&tokens);
        let as_slice = key.as_slice();

        assert_eq!(as_slice[0], 0); // Actual token, not padding
        assert_eq!(as_slice[1], 1);
        assert_eq!(as_slice[2], 2);
        assert_eq!(as_slice[3], 0); // Padding
    }

    // Capacity and performance

    #[test]
    fn test_count_ngrams_capacity_estimate() {
        // Test capacity estimation logic
        let tokens: Vec<u32> = (0..50).collect();
        let ngrams = count_ngrams(&tokens, &[2, 3]);

        // For 50 tokens:
        // 2-grams: 49 possible
        // 3-grams: 48 possible
        // Total: 97 possible, but many will be unique
        // With unique_ratio = 0.6 for len < 100: capacity = 97 * 0.6 ≈ 58
        assert!(ngrams.capacity() >= 16, "Should have minimum capacity");
        assert!(
            ngrams.len() <= ngrams.capacity(),
            "Len should not exceed capacity"
        );
    }

    #[test]
    fn test_count_ngrams_no_excessive_allocations() {
        // Test that capacity estimation prevents excessive HashMap resizing
        let tokens: Vec<u32> = (0..1000).collect();
        let ngrams = count_ngrams(&tokens, &[2, 3, 4]);

        // For 1000 tokens:
        // 2-grams: 999 possible
        // 3-grams: 998 possible
        // 4-grams: 997 possible
        // Total: 2994 possible
        // With unique_ratio = 0.25 for len >= 1000: capacity = 2994 * 0.25 ≈ 748
        assert!(
            ngrams.capacity() > 500,
            "Should pre-allocate reasonable capacity"
        );
        assert!(ngrams.len() <= ngrams.capacity());
    }

    // Parallel determinism

    #[test]
    fn test_build_vocabulary_parallel_deterministic() {
        // Run multiple times to ensure parallel vocabulary building is deterministic
        let docs = vec![
            vec![1, 2, 3, 4, 5],
            vec![2, 3, 4, 5, 6],
            vec![1, 3, 5, 7, 9],
            vec![2, 4, 6, 8, 10],
        ];

        let vocab1 = build_vocabulary(&docs, &[2, 3]);
        let vocab2 = build_vocabulary(&docs, &[2, 3]);
        let vocab3 = build_vocabulary(&docs, &[2, 3]);

        // All runs should produce identical vocabularies
        assert_eq!(vocab1.len(), vocab2.len());
        assert_eq!(vocab1.len(), vocab3.len());

        for entry in &vocab1 {
            let key = entry.key();
            let count = entry.value();
            assert_eq!(vocab2.get(key).map(|v| *v), Some(*count));
            assert_eq!(vocab3.get(key).map(|v| *v), Some(*count));
        }
    }

    #[test]
    fn test_build_vocabulary_dashmap_no_race_conditions() {
        // Test with many documents to stress-test parallel DashMap updates
        let docs: Vec<Vec<u32>> = (0..100)
            .map(|i| vec![i % 10, (i + 1) % 10, (i + 2) % 10])
            .collect();

        // Run multiple times
        let vocab1 = build_vocabulary(&docs, &[2]);
        let vocab2 = build_vocabulary(&docs, &[2]);

        // Results should be identical (no race conditions)
        assert_eq!(vocab1.len(), vocab2.len());
        for entry in &vocab1 {
            assert_eq!(vocab2.get(entry.key()).map(|v| *v), Some(*entry.value()));
        }
    }

    // Edge cases

    #[test]
    fn test_ngram_empty_tokens() {
        let ngrams = count_ngrams(&[], &[2, 3]);
        assert!(ngrams.is_empty());

        let unique = unique_ngrams(&[], &[2, 3]);
        assert!(unique.is_empty());
    }

    #[test]
    fn test_ngram_single_token() {
        // Single token can't produce bigrams or higher
        let tokens = vec![42];
        let ngrams = count_ngrams(&tokens, &[2, 3]);
        assert!(
            ngrams.is_empty(),
            "Single token should produce no n-grams for n>=2"
        );

        let unique = unique_ngrams(&tokens, &[2, 3]);
        assert!(unique.is_empty());
    }

    #[test]
    fn test_ngram_all_same_token() {
        // Repeated tokens should produce n-grams
        let tokens = vec![5, 5, 5, 5];
        let ngrams = count_ngrams(&tokens, &[2]);

        // Should produce [5,5] three times (windows of size 2)
        assert_eq!(ngrams.len(), 1); // Only one unique n-gram
        let key = NgramKey::from_slice(&[5, 5]);
        assert_eq!(ngrams.get(&key), Some(&3)); // Count = 3
    }

    #[test]
    fn test_ngram_insufficient_tokens_for_size() {
        // 3 tokens, requesting 4-grams
        let tokens = vec![1, 2, 3];
        let ngrams = count_ngrams(&tokens, &[4]);
        assert!(
            ngrams.is_empty(),
            "Should produce no n-grams when tokens < n"
        );

        // Should still work for valid sizes
        let ngrams_valid = count_ngrams(&tokens, &[2, 3]);
        assert!(!ngrams_valid.is_empty());
    }

    #[test]
    fn test_unique_ngrams_no_duplicates() {
        // unique_ngrams should deduplicate
        let tokens = vec![1, 2, 1, 2];
        let unique = unique_ngrams(&tokens, &[2]);

        // [1,2] appears twice, [2,1] appears once
        assert_eq!(unique.len(), 2); // Only unique n-grams

        // count_ngrams should track counts
        let counts = count_ngrams(&tokens, &[2]);
        let key_12 = NgramKey::from_slice(&[1, 2]);
        assert_eq!(counts.get(&key_12), Some(&2));
    }

    #[test]
    fn test_ngram_range_ordering() {
        // N-gram range order shouldn't matter
        let tokens = vec![1, 2, 3, 4];
        let ngrams1 = count_ngrams(&tokens, &[2, 3, 4]);
        let ngrams2 = count_ngrams(&tokens, &[4, 2, 3]);
        let ngrams3 = count_ngrams(&tokens, &[3, 4, 2]);

        assert_eq!(ngrams1.len(), ngrams2.len());
        assert_eq!(ngrams1.len(), ngrams3.len());

        for (key, count) in &ngrams1 {
            assert_eq!(ngrams2.get(key), Some(count));
            assert_eq!(ngrams3.get(key), Some(count));
        }
    }

    #[test]
    fn test_ngram_key_as_slice_sound() {
        let key = NgramKey::from_slice(&[1, 2, 3]);
        assert_eq!(key.0, 0x00000000_00000003_00000002_00000001);
        let s = key.as_slice();
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], 1);
        assert_eq!(s[1], 2);
        assert_eq!(s[2], 3);
        assert_eq!(s[3], 0);

        let key_full = NgramKey::from_slice(&[u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(key_full.0, u128::MAX);
        let s = key_full.as_slice();
        assert_eq!(s, &[u32::MAX; 4]);
    }
}
