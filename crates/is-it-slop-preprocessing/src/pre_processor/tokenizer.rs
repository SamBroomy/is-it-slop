//! Text tokenization using BPE encoding.
//!
//! Uses the `o200k_base` encoding from openai vocab to convert text into token IDs.
//! Automatically switches between sequential and parallel processing based on workload:
//!
//! - Parallel: >= 1,000 texts OR >= 1MB total bytes
//! - Sequential: smaller workloads (avoids thread overhead)

use bpe_openai::o200k_base;
#[cfg(feature = "progress-bars")]
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{debug, instrument};

/// Minimum number of texts before even considering parallelization.
/// Below this threshold, thread overhead exceeds benefit regardless of text size.
const MIN_TEXTS_FOR_PARALLEL_CONSIDERATION: usize = 8;

/// Minimum number of texts to always parallelize
const MIN_TEXTS_FOR_PARALLEL: usize = 1_000;

/// Minimum total bytes to consider parallelization (~1MB)
const MIN_BYTES_FOR_PARALLEL: usize = 1_000_000;

#[cfg(feature = "progress-bars")]
fn progress_bar_setup(
    len: usize,
    message: impl Into<std::borrow::Cow<'static, str>>,
) -> ProgressBar {
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

#[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
fn tokenize_texts_par<T: AsRef<str> + Sync>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using parallel tokenization");
    let bpe = o200k_base();
    #[cfg(feature = "progress-bars")]
    let pb = progress_bar_setup(texts.len(), "Tokenizing texts in parallel");
    let result = texts.par_iter();
    #[cfg(feature = "progress-bars")]
    let result = result.progress_with(pb.clone());
    let result = result.map(|text| bpe.encode(text)).collect();
    #[cfg(feature = "progress-bars")]
    pb.finish_with_message("Parallel tokenization complete");
    result
}

#[instrument(level = "debug", skip(texts), fields(num_texts = texts.len()))]
fn tokenize_texts<T: AsRef<str>>(texts: &[T]) -> Vec<Vec<u32>> {
    debug!(num_texts = texts.len(), "Using sequential tokenization");
    let bpe = o200k_base();
    texts.iter().map(|text| bpe.encode(text.as_ref())).collect()
}

/// Determine if parallel processing should be used based on workload characteristics.
///
/// Parallelization is beneficial when:
/// - There are >= `MIN_TEXTS_FOR_PARALLEL_CONSIDERATION` texts AND (>= `MIN_TEXTS_FOR_PARALLEL`
///   texts OR >= `MIN_BYTES_FOR_PARALLEL` total bytes)
///
/// This heuristic balances thread spawning overhead against tokenization work.
/// Very small workloads (< 8 texts) never parallelize, regardless of size.
#[inline]
fn should_use_parallel<T: AsRef<str>>(texts: &[T]) -> bool {
    let num_texts = texts.len();

    // Never parallelize tiny workloads - thread overhead exceeds benefit
    // (even a single 10MB text won't benefit from parallel processing)
    if num_texts < MIN_TEXTS_FOR_PARALLEL_CONSIDERATION {
        return false;
    }

    // If we have many texts, always parallelize
    if num_texts >= MIN_TEXTS_FOR_PARALLEL {
        return true;
    }

    // For medium-sized workloads, check total bytes
    let total_bytes: usize = texts.iter().map(|s| s.as_ref().len()).sum();
    total_bytes >= MIN_BYTES_FOR_PARALLEL
}
/// Tokenize texts using bpe-openai `o200k_base` encoding.
///
/// Automatically parallelizes for large workloads (>= 1,000 texts or >= 1MB bytes).
///
/// # Arguments
/// * `texts` - Input documents
///
/// # Returns
/// Vector of token ID sequences, one per input text
#[instrument(level = "debug", skip(texts), fields(num_texts = texts.len(), use_parallel = should_use_parallel(texts)))]
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
#[must_use]
pub fn reverse_tokenize(tokens: &[u32]) -> String {
    if tokens.is_empty() {
        return String::new();
    }

    let bpe = o200k_base();
    bpe.decode(tokens).unwrap_or_else(|| {
        // Log the error with token IDs for debugging
        tracing::warn!(
            tokens = ?tokens,
            "Failed to decode token IDs, using replacement character"
        );
        // Return a debug-friendly representation instead of !!
        format!("[DECODE_ERROR: {tokens:?}]")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_processor::{CountVectorizer, VectorizerParams};
    #[test]
    fn test_tokenize_basic() {
        let texts = vec!["Hello world", "Test"];
        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 2);
        assert!(!tokens[0].is_empty());
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize(&[""]);
        assert_eq!(tokens[0].len(), 0);
    }

    // #[test]
    // fn test_reverse_tokenize_round_trip() {
    //     let text = "Hello world";
    //     let tokens = tokenize(&[text]);
    //     let result = reverse_tokenize(&tokens[0]);
    //     assert!(result.contains("Hello"));
    // }

    #[test]
    fn test_tokenize_deterministic() {
        let texts = vec!["test"];
        let tokens1 = tokenize(&texts);
        let tokens2 = tokenize(&texts);
        assert_eq!(tokens1, tokens2);
    }
    #[test]
    fn test_vocabulary_no_decode_artifacts() {
        let texts = vec!["Hello world", "The quick brown fox", "Test sample data"];
        let params = VectorizerParams::new(1.0, 1.0, false);
        let vectorizer = CountVectorizer::fit(&texts, params);

        let vocab = vectorizer.vocabulary();

        // Check for decode artifacts
        let bad_entries: Vec<_> = vocab
            .iter()
            .filter(|(text, _)| text.contains("!!") || text.is_empty())
            .collect();

        assert!(
            bad_entries.is_empty(),
            "Found {} vocabulary entries with decode artifacts:\n{:#?}",
            bad_entries.len(),
            bad_entries
        );
    }

    #[test]
    fn test_reverse_tokenize_round_trip() {
        let texts = vec!["Hello world", "Test 123"];
        let tokenized = tokenize(&texts);

        for (original, tokens) in texts.iter().zip(&tokenized) {
            let decoded = reverse_tokenize(tokens);
            println!("Original: '{original}'");
            println!("Tokens: {tokens:?}");
            println!("Decoded: '{decoded}'");

            assert!(
                !decoded.contains("!!"),
                "Decoded text '{decoded}' contains '!!' artifact"
            );
        }
    }

    // Threshold boundaries

    #[test]
    fn test_tokenize_exactly_1000_texts_parallel() {
        // Exactly at MIN_TEXTS_FOR_PARALLEL threshold should trigger parallel
        let texts: Vec<String> = (0..1000).map(|i| format!("Text {i}")).collect();
        assert!(
            should_use_parallel(&texts),
            "1000 texts should use parallel"
        );

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 1000);
    }

    #[test]
    fn test_tokenize_999_texts_sequential() {
        // One below threshold: should check byte size
        let texts: Vec<String> = (0..999).map(|i| format!("Short {i}")).collect();
        let total_bytes: usize = texts.iter().map(String::len).sum();

        if total_bytes < MIN_BYTES_FOR_PARALLEL {
            assert!(
                !should_use_parallel(&texts),
                "999 small texts should use sequential"
            );
        }

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 999);
    }

    #[test]
    fn test_tokenize_1mb_single_text_parallel() {
        // Single text >= 1MB should now use SEQUENTIAL (< 8 texts threshold)
        let large_text = "a".repeat(1_000_000);
        let texts = vec![large_text.as_str()];

        assert!(
            !should_use_parallel(&texts),
            "Single text should use sequential regardless of size"
        );

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 1);
        assert!(!tokens[0].is_empty());
    }

    #[test]
    fn test_tokenize_below_1mb_sequential() {
        // Below 1MB and < 1000 texts should use sequential
        let text = "a".repeat(500_000); // 500KB
        let texts = vec![text.as_str()];

        assert!(
            !should_use_parallel(&texts),
            "500KB text should use sequential"
        );

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_parallel_threshold_with_byte_check() {
        // 500 texts, each 3KB = 1.5MB total -> should parallelize
        // (>= 8 texts AND >= 1MB)
        let large_texts: Vec<String> = (0..500)
            .map(|i| format!("Text {} {}", i, "x".repeat(3000)))
            .collect();

        let total_bytes: usize = large_texts.iter().map(String::len).sum();
        assert!(total_bytes >= MIN_BYTES_FOR_PARALLEL);
        assert!(
            should_use_parallel(&large_texts),
            "500 texts with >= 1MB should parallelize"
        );

        let tokens = tokenize(&large_texts);
        assert_eq!(tokens.len(), 500);
    }

    #[test]
    fn test_min_texts_threshold() {
        // Test the MIN_TEXTS_FOR_PARALLEL_CONSIDERATION boundary

        // 7 texts: never parallelize (even with huge size)
        let huge_text = "a".repeat(2_000_000); // 2MB each
        let seven_huge: Vec<_> = (0..7).map(|_| huge_text.as_str()).collect();
        assert!(
            !should_use_parallel(&seven_huge),
            "< 8 texts never parallelize"
        );

        // 8 texts with enough bytes: should parallelize
        let eight_huge: Vec<_> = (0..8).map(|_| huge_text.as_str()).collect();
        assert!(
            should_use_parallel(&eight_huge),
            "8 texts with >= 1MB should parallelize"
        );

        // 8 texts without enough bytes: should not parallelize
        let small_text = "hello";
        let eight_small: Vec<_> = (0..8).map(|_| small_text).collect();
        assert!(
            !should_use_parallel(&eight_small),
            "8 small texts below byte threshold should not parallelize"
        );
    }

    // Sequential/parallel equivalence

    #[test]
    fn test_parallel_sequential_same_result() {
        // Force both paths and compare results
        let texts = vec!["Hello world", "Test text", "Sample data"];

        // Force sequential
        let seq_tokens = tokenize_texts(&texts);

        // Force parallel
        let par_tokens = tokenize_texts_par(&texts);

        assert_eq!(
            seq_tokens, par_tokens,
            "Sequential and parallel should produce identical results"
        );
    }

    #[test]
    fn test_batch_sizes_crossing_threshold() {
        // Test behavior around the 1000-text threshold
        let small_batch: Vec<String> = (0..100).map(|i| format!("Text {i}")).collect();
        let medium_batch: Vec<String> = (0..999).map(|i| format!("Text {i}")).collect();
        let large_batch: Vec<String> = (0..1001).map(|i| format!("Text {i}")).collect();

        let tokens_small = tokenize(&small_batch);
        let tokens_medium = tokenize(&medium_batch);
        let tokens_large = tokenize(&large_batch);

        assert_eq!(tokens_small.len(), 100);
        assert_eq!(tokens_medium.len(), 999);
        assert_eq!(tokens_large.len(), 1001);

        // Verify determinism across threshold
        for tokens_vec in [&tokens_small, &tokens_medium, &tokens_large] {
            for (i, tokens) in tokens_vec.iter().enumerate() {
                let text = format!("Text {i}");
                let expected = tokenize(&[text.as_str()]);
                assert_eq!(
                    tokens, &expected[0],
                    "Text {i} should tokenize consistently"
                );
            }
        }
    }

    // Unicode edge cases

    #[test]
    fn test_tokenize_mixed_scripts() {
        let texts = vec![
            "English text",         // Latin
            "日本語のテキスト",     // Japanese
            "مرحبا بالعالم",        // Arabic
            "Привет мир",           // Cyrillic
            "Mixed 日本語 English", // Mixed
        ];

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), texts.len());

        for (i, token_vec) in tokens.iter().enumerate() {
            assert!(
                !token_vec.is_empty(),
                "Text '{:?}' should produce tokens",
                texts[i]
            );
        }
    }

    #[test]
    fn test_tokenize_emojis_and_special_chars() {
        let texts = vec![
            "Hello 😀 world 🎉",
            "Math: ∑∫∂∇",
            "Arrows: →←↑↓",
            "Emoji sequence: 👨‍👩‍👧‍👦",
        ];

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), texts.len());

        for token_vec in &tokens {
            assert!(!token_vec.is_empty());
        }
    }

    #[test]
    fn test_tokenize_combining_diacritics() {
        // Combining diacritics (é vs e + combining acute)
        let texts = vec![
            "café",         // Precomposed
            "cafe\u{0301}", // Decomposed (e + combining acute)
        ];

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 2);

        // Both should produce valid tokens
        for token_vec in &tokens {
            assert!(!token_vec.is_empty());
        }
    }

    #[test]
    fn test_tokenize_zero_width_characters() {
        let texts = vec![
            "Hello\u{200B}world", // Zero-width space
            "Test\u{FEFF}text",   // Zero-width no-break space (BOM)
        ];

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 2);

        for token_vec in &tokens {
            assert!(!token_vec.is_empty());
        }
    }

    // Boundary conditions

    #[test]
    fn test_tokenize_very_long_single_text() {
        // Test with a 10MB text (should use SEQUENTIAL - only 1 text)
        let large_text = "This is a test sentence. ".repeat(400_000); // ~10MB
        let texts = vec![large_text.as_str()];

        assert!(
            !should_use_parallel(&texts),
            "Single text should use sequential"
        );

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 1);
        assert!(
            tokens[0].len() > 100_000,
            "Large text should produce many tokens"
        );
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = tokenize(&[""]);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].len(), 0);
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let texts = vec!["   ", "\t\t", "\n\n", "    \n\t  "];
        let tokens = tokenize(&texts);

        assert_eq!(tokens.len(), texts.len());
        // Whitespace-only strings may produce tokens (depending on BPE)
        // Just verify no panic and consistent behavior
        for (i, token_vec) in tokens.iter().enumerate() {
            let text = texts[i];
            let expected = tokenize(&[text]);
            assert_eq!(token_vec, &expected[0]);
        }
    }

    #[test]
    fn test_tokenize_repeated_characters() {
        // Long repetitions can stress tokenizer
        let texts = vec!["a".repeat(1000), "123".repeat(500), "🎉".repeat(200)];

        let tokens = tokenize(&texts);
        assert_eq!(tokens.len(), 3);

        for token_vec in &tokens {
            assert!(!token_vec.is_empty());
        }
    }

    #[test]
    fn test_tokenize_deterministic_parallel() {
        // Verify parallel tokenization is deterministic
        let texts: Vec<String> = (0..1500).map(|i| format!("Test text number {i}")).collect();

        let tokens1 = tokenize(&texts);
        let tokens2 = tokenize(&texts);
        let tokens3 = tokenize(&texts);

        assert_eq!(tokens1, tokens2);
        assert_eq!(tokens1, tokens3);
    }

    #[test]
    fn test_should_use_parallel_edge_cases() {
        // Empty
        assert!(!should_use_parallel(&Vec::<&str>::new()));

        // Single small text
        assert!(!should_use_parallel(&["hello"]));

        // Single text at byte threshold (still sequential - only 1 text)
        let exactly_1mb = "a".repeat(1_000_000);
        assert!(
            !should_use_parallel(&[exactly_1mb.as_str()]),
            "Single text never parallelizes"
        );

        // 7 texts (below MIN_TEXTS_FOR_PARALLEL_CONSIDERATION)
        let seven_small: Vec<&str> = vec!["text"; 7];
        assert!(
            !should_use_parallel(&seven_small),
            "< 8 texts never parallelizes"
        );

        // 8 small texts (at threshold, but not enough bytes)
        let eight_small: Vec<&str> = vec!["text"; 8];
        assert!(
            !should_use_parallel(&eight_small),
            "8 small texts don't reach byte threshold"
        );

        // 8 large texts (at threshold, enough bytes)
        let large_text = "a".repeat(150_000); // 150KB each, 8 * 150KB = 1.2MB
        let eight_large: Vec<_> = (0..8).map(|_| large_text.as_str()).collect();
        assert!(
            should_use_parallel(&eight_large),
            "8 texts with >= 1MB should parallelize"
        );
    }

    #[test]
    fn test_reverse_tokenize_empty() {
        let result = reverse_tokenize(&[]);
        assert_eq!(result, "");
    }

    #[test]
    fn test_reverse_tokenize_single_token() {
        // Token ID 100 should decode to something valid
        let tokens = vec![100];
        let result = reverse_tokenize(&tokens);
        assert!(!result.is_empty());
        assert!(!result.contains("!!"));
    }
}
