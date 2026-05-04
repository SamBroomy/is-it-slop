//! Utility functions for statistical feature extraction.
//!
//! Provides shared utilities like Shannon entropy calculation and text segmentation.

/// Compute Shannon entropy of a frequency distribution.
///
/// Formula: ``H = -Σ(p_i * log2(p_i))``
/// where `p_i` is the probability of element `i`
///
/// Returns 0.0 for empty distributions.
///
/// # Arguments
/// * `counts` - Frequency counts for each element
///
/// # Example
/// ```
/// # use std::collections::HashMap;
/// # use is_it_slop_preprocessing::pre_processor::features::shannon_entropy;
/// let mut counts = HashMap::new();
/// counts.insert('a', 2);
/// counts.insert('b', 2);
/// let entropy = shannon_entropy(&counts);
/// assert!((entropy - 1.0).abs() < 1e-6); // Perfect balance = 1.0 bit
/// ```
pub fn shannon_entropy<T: std::hash::Hash + Eq, S: std::hash::BuildHasher>(
    counts: &std::collections::HashMap<T, usize, S>,
) -> f32 {
    if counts.is_empty() {
        return 0.0;
    }

    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f32;

    counts
        .values()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f32 / total_f;
            -p * p.log2()
        })
        .sum()
}

/// Split text into words (whitespace-separated tokens).
///
/// Returns lowercase words for case-insensitive analysis.
pub fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_lowercase).collect()
}

/// Split text into sentences using common sentence terminators.
///
/// Splits on `.`, `!`, `?` and filters empty strings.
pub fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract punctuation characters from text.
///
/// Returns vector of punctuation characters for distribution analysis.
#[must_use]
pub fn extract_punctuation(text: &str) -> Vec<char> {
    const PUNCTUATION: &str = ".,!?;:'\"-()[]{}";
    text.chars().filter(|c| PUNCTUATION.contains(*c)).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    // Basic entropy tests
    #[test]
    fn test_shannon_entropy_empty() {
        let counts: HashMap<char, usize> = HashMap::new();
        let entropy = shannon_entropy(&counts);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_shannon_entropy_single_symbol() {
        // Single symbol = no uncertainty = 0 bits
        let mut counts = HashMap::new();
        counts.insert('a', 1);
        let entropy = shannon_entropy(&counts);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_shannon_entropy_uniform_symbol_repeated() {
        // Multiple occurrences of same symbol = still 0 bits
        let mut counts = HashMap::new();
        counts.insert('a', 5);
        let entropy = shannon_entropy(&counts);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_shannon_entropy_two_equal() {
        // Two equally likely symbols = 1 bit
        let mut counts = HashMap::new();
        counts.insert('a', 1);
        counts.insert('b', 1);
        let entropy = shannon_entropy(&counts);
        assert!((entropy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shannon_entropy_aab() {
        // Based on entropy crate test: "aab" → 0.9182958
        // p_a = 2/3, p_b = 1/3
        // H = -(2/3 * log2(2/3) + 1/3 * log2(1/3)) = 0.9182958
        let mut counts = HashMap::new();
        counts.insert('a', 2);
        counts.insert('b', 1);
        let entropy = shannon_entropy(&counts);
        assert!((entropy - 0.9182958).abs() < 1e-6);
    }

    #[test]
    fn test_shannon_entropy_four_equal() {
        // Four equally likely symbols = 2 bits
        let mut counts = HashMap::new();
        counts.insert('a', 1);
        counts.insert('b', 1);
        counts.insert('c', 1);
        counts.insert('d', 1);
        let entropy = shannon_entropy(&counts);
        assert!((entropy - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_shannon_entropy_skewed() {
        let mut counts = HashMap::new();
        counts.insert('a', 9);
        counts.insert('b', 1);
        let entropy = shannon_entropy(&counts);
        // Should be less than 1 bit (not equally distributed)
        assert!(entropy < 1.0);
        assert!(entropy > 0.0);
        // Exact value: -(9/10 * log2(9/10) + 1/10 * log2(1/10)) ≈ 0.469
        assert!((entropy - 0.469).abs() < 0.01);
    }

    #[test]
    fn test_shannon_entropy_zero_counts_filtered() {
        // Zero counts should be filtered out
        let mut counts = HashMap::new();
        counts.insert('a', 2);
        counts.insert('b', 0); // This should be filtered
        let entropy = shannon_entropy(&counts);
        assert_eq!(entropy, 0.0); // Only one effective symbol
    }

    #[test]
    fn test_split_words() {
        let text = "Hello World Test";
        let words = split_words(text);
        assert_eq!(words, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_split_sentences() {
        let text = "First sentence. Second sentence! Third? Fourth.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
    }

    #[test]
    fn test_extract_punctuation() {
        let text = "Hello, world! How are you?";
        let punct = extract_punctuation(text);
        assert_eq!(punct, vec![',', '!', '?']);
    }
}
