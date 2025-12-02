//! Text tokenization using tiktoken BPE encoding.
//!
//! Uses the `o200k_base` encoding from tiktoken to convert text into token IDs.
//! Automatically switches between sequential and parallel processing based on workload:
//!
//! - Parallel: >= 100 texts OR >= 10,000 total characters
//! - Sequential: smaller workloads (avoids thread overhead)

#[cfg(feature = "progress-bars")]
use std::borrow::Cow;

#[cfg(feature = "progress-bars")]
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::prelude::*;
use tiktoken_rs::o200k_base_singleton;
use tracing::debug;

/// Minimum number of texts to consider parallelization
const MIN_TEXTS_FOR_PARALLEL: usize = 100;

/// Minimum total character count to consider parallelization
const MIN_CHARS_FOR_PARALLEL: usize = 10_000;

#[cfg(feature = "progress-bars")]
fn progress_bar_setup(len: usize, message: impl Into<Cow<'static, str>>) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(message);
    pb
}

#[cfg(feature = "progress-bars")]
fn tokenize_texts_par<T: AsRef<str> + Sync>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using parallel tokenization");
    let bpe = o200k_base_singleton();
    let pb = progress_bar_setup(texts.len(), "Tokenizing texts in parallel");
    let result = texts
        .par_iter()
        .progress_with(pb.clone())
        .map(|text| bpe.encode_ordinary(text.as_ref()))
        .collect();
    pb.finish_with_message("Parallel tokenization complete");
    result
}

#[cfg(not(feature = "progress-bars"))]
fn tokenize_texts_par<T: AsRef<str> + Sync>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using parallel tokenization");
    let bpe = o200k_base_singleton();
    texts
        .par_iter()
        .map(|text| bpe.encode_ordinary(text.as_ref()))
        .collect()
}

#[cfg(feature = "progress-bars")]
fn tokenize_texts<T: AsRef<str>>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using sequential tokenization");
    let bpe = o200k_base_singleton();
    let pb = progress_bar_setup(texts.len(), "Tokenizing texts");

    let result = texts
        .iter()
        .progress_with(pb.clone())
        .map(|text| bpe.encode_ordinary(text.as_ref()))
        .collect();
    pb.finish_with_message("Tokenization complete");
    result
}

#[cfg(not(feature = "progress-bars"))]
fn tokenize_texts<T: AsRef<str>>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using sequential tokenization");
    let bpe = o200k_base_singleton();
    texts
        .iter()
        .map(|text| bpe.encode_ordinary(text.as_ref()))
        .collect()
}

/// Determine if parallel processing should be used based on workload characteristics.
///
/// Parallelization is beneficial when:
/// - There are many texts (>= 100), OR
/// - The total character count is large (>= 10,000 chars)
///
/// This heuristic balances thread spawning overhead against tokenization work.
#[inline]
fn should_use_parallel<T: AsRef<str>>(texts: &[T]) -> bool {
    let num_texts = texts.len();

    // If we have many texts, always parallelize
    if num_texts >= MIN_TEXTS_FOR_PARALLEL {
        return true;
    }

    // For fewer texts, check total workload
    // Sample first few to estimate average length if we have many
    let total_chars: usize = if num_texts > 20 {
        // Estimate based on first 20 texts to avoid iterating all
        let sample_chars: usize = texts.iter().take(20).map(|s| s.as_ref().len()).sum();
        (sample_chars * num_texts) / 20 // estimated total
    } else {
        texts.iter().map(|s| s.as_ref().len()).sum()
    };

    total_chars >= MIN_CHARS_FOR_PARALLEL
}

/// Tokenize texts using tiktoken `o200k_base` encoding.
///
/// Automatically parallelizes for large workloads (>= 100 texts or >= 10k chars).
///
/// # Arguments
/// * `texts` - Input documents
///
/// # Returns
/// Vector of token ID sequences, one per input text
pub fn tokenize<T: AsRef<str> + Sync>(texts: &[T]) -> Vec<Vec<u32>> {
    if should_use_parallel(texts) {
        tokenize_texts_par(texts)
    } else {
        tokenize_texts(texts)
    }
}

/// Decode token IDs back to text.
///
/// Used for vocabulary inspection. Not called during training/inference.
pub fn reverse_tokenize(tokens: &[u32]) -> String {
    let bpe = o200k_base_singleton();

    bpe.decode(tokens.to_vec()).unwrap_or_default()
}
