//! Statistical feature computation for AI text detection.
//!
//! Implements 9 validated features capturing writing style patterns:
//! - Document-level features (6): Computed once per document, replicated across chunks
//! - Chunk-level features (3): Computed per chunk

use ahash::HashMap;

use super::utils::{extract_punctuation, shannon_entropy, split_sentences, split_words};

// =============================================================================
// Document-Level Features (6)
// =============================================================================

/// Feature 1: Bigram repetition rate (strongest signal: d=-0.419, coef=+1.98)
///
/// Proportion of word bigrams that appear more than once.
/// AI text has LESS repetition (more varied phrasing).
///
/// Formula: (count of bigrams appearing >1 time) / (total unique bigrams)
#[must_use]
pub fn compute_bigram_repetition_rate(text: &str) -> f32 {
    let words = split_words(text);

    if words.len() < 2 {
        return 0.0;
    }

    // Create bigrams
    let bigrams: Vec<String> = words
        .windows(2)
        .map(|window| format!("{} {}", window[0], window[1]))
        .collect();

    if bigrams.is_empty() {
        return 0.0;
    }

    // Count bigram frequencies
    let mut bigram_counts: HashMap<String, usize> = HashMap::default();
    for bigram in bigrams {
        *bigram_counts.entry(bigram).or_insert(0) += 1;
    }

    // Count how many bigrams repeat (count > 1)
    let repeated_bigrams = bigram_counts.values().filter(|&&count| count > 1).count();

    repeated_bigrams as f32 / bigram_counts.len() as f32
}

/// Feature 2: Punctuation entropy (second strongest: d=-0.365, coef=+0.23)
///
/// Shannon entropy of punctuation mark distribution.
/// AI uses MORE diverse punctuation (higher entropy).
///
/// Formula: `H = -Σ(p_i * log2(p_i))` for punctuation characters
#[must_use]
pub fn compute_punctuation_entropy(text: &str) -> f32 {
    let punctuation = extract_punctuation(text);

    if punctuation.is_empty() {
        return 0.0;
    }

    // Count punctuation frequencies
    let mut punct_counts: HashMap<char, usize> = HashMap::default();
    for c in punctuation {
        *punct_counts.entry(c).or_insert(0) += 1;
    }

    shannon_entropy(&punct_counts)
}

/// Feature 3: Lexical diversity (d=+0.165, coef=-0.17)
///
/// Ratio of unique tokens to total tokens.
/// Higher diversity = more human-like.
///
/// Formula: `unique_words / total_words`
#[must_use]
pub fn compute_lexical_diversity(text: &str) -> f32 {
    let words = split_words(text);

    if words.is_empty() {
        return 0.0;
    }

    let unique_words: std::collections::HashSet<_> = words.iter().collect();
    unique_words.len() as f32 / words.len() as f32
}

/// Feature 4: Vocabulary richness (d=+0.154, coef=-0.09)
///
/// Length-normalized lexical diversity using square root.
/// Less sensitive to document length than raw lexical diversity.
///
/// Formula: `sqrt(unique_words) / total_words`
#[must_use]
pub fn compute_vocab_richness(text: &str) -> f32 {
    let words = split_words(text);

    if words.is_empty() {
        return 0.0;
    }

    let unique_words: std::collections::HashSet<_> = words.iter().collect();
    (unique_words.len() as f32).sqrt() / words.len() as f32
}

/// Feature 5: Word repetition rate (d=-0.084, coef=+0.07)
///
/// Proportion of unique words that appear multiple times.
/// AI has LESS word repetition (more varied vocabulary).
///
/// Formula: (count of words appearing >1 time) / (total unique words)
#[must_use]
pub fn compute_word_repetition_rate(text: &str) -> f32 {
    let words = split_words(text);

    if words.is_empty() {
        return 0.0;
    }

    // Count word frequencies
    let mut word_counts: HashMap<String, usize> = HashMap::default();
    for word in words {
        *word_counts.entry(word).or_insert(0) += 1;
    }

    // Count how many words repeat (count > 1)
    let repeated_words = word_counts.values().filter(|&&count| count > 1).count();

    repeated_words as f32 / word_counts.len() as f32
}

/// Feature 6: Sentence length coefficient of variation (d=-0.115, coef=+0.07)
///
/// Normalized burstiness measure: `std(sentence_lengths) / mean(sentence_lengths)`.
/// AI has LESS variation (more uniform sentence structure).
///
/// Formula: CV = std / mean (0.0 if < 2 sentences)
#[must_use]
pub fn compute_sentence_length_cv(text: &str) -> f32 {
    let sentences = split_sentences(text);

    if sentences.len() < 2 {
        return 0.0;
    }

    // Compute sentence lengths (in words)
    let lengths: Vec<f32> = sentences
        .iter()
        .map(|s| s.split_whitespace().count() as f32)
        .collect();

    let mean = lengths.iter().sum::<f32>() / lengths.len() as f32;

    if mean == 0.0 {
        return 0.0;
    }

    let variance =
        lengths.iter().map(|&len| (len - mean).powi(2)).sum::<f32>() / lengths.len() as f32;
    let std = variance.sqrt();

    std / mean
}

// =============================================================================
// Chunk-Level Features (3)
// =============================================================================

/// Feature 7: Average word length (dominant coefficient: +8.23)
///
/// Mean character length per word.
/// Simple but highly discriminative at chunk level.
///
/// Formula: mean(len(word) for word in chunk)
#[must_use]
pub fn compute_avg_word_length(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return 0.0;
    }

    let total_chars: usize = words.iter().map(|w| w.len()).sum();
    total_chars as f32 / words.len() as f32
}

/// Feature 8: Chunk punctuation entropy (d=-0.080, coef=+0.07)
///
/// Same as document-level punctuation entropy but computed per chunk.
/// Captures local punctuation patterns.
///
/// Formula: ``H = -Σ(p_i * log2(p_i))`` for punctuation in chunk
#[must_use]
pub fn compute_chunk_punctuation_entropy(text: &str) -> f32 {
    // Same implementation as document-level
    compute_punctuation_entropy(text)
}

/// Feature 9: Word frequency entropy (d=-0.098, coef=+0.08)
///
/// Shannon entropy of word frequency distribution.
/// AI has LESS diverse word usage (lower entropy).
///
/// Formula: ``H = -Σ(p_i * log2(p_i))`` for word frequencies
#[must_use]
pub fn compute_word_frequency_entropy(text: &str) -> f32 {
    let words = split_words(text);

    if words.is_empty() {
        return 0.0;
    }

    // Count word frequencies
    let mut word_counts: HashMap<String, usize> = HashMap::default();
    for word in words {
        *word_counts.entry(word).or_insert(0) += 1;
    }

    shannon_entropy(&word_counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigram_repetition_rate() {
        // No repetition
        let text = "the cat sat. the dog ran.";
        let rate = compute_bigram_repetition_rate(text);
        assert!((0.0..=1.0).contains(&rate));

        // With repetition
        let text2 = "the cat the cat the dog";
        let rate2 = compute_bigram_repetition_rate(text2);
        assert!(rate2 > 0.0);
    }

    #[test]
    fn test_punctuation_entropy() {
        let text = "Hello! How are you? I'm fine.";
        let entropy = compute_punctuation_entropy(text);
        assert!(entropy > 0.0);

        // No punctuation
        let text2 = "Hello world";
        let entropy2 = compute_punctuation_entropy(text2);
        assert_eq!(entropy2, 0.0);
    }

    #[test]
    fn test_lexical_diversity() {
        let text = "the cat sat on the mat";
        let diversity = compute_lexical_diversity(text);
        // 5 unique words / 6 total = 0.833...
        assert!((diversity - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_vocab_richness() {
        let text = "the cat sat on the mat";
        let richness = compute_vocab_richness(text);
        // sqrt(5) / 6 = 0.373
        assert!((richness - 0.373).abs() < 0.01);
    }

    #[test]
    fn test_word_repetition_rate() {
        let text = "the cat sat on the mat";
        let rate = compute_word_repetition_rate(text);
        // "the" repeats: 1 repeated word / 5 unique = 0.2
        assert!((rate - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_sentence_length_cv() {
        let text = "Short. This is a longer sentence. Very long sentence here.";
        let cv = compute_sentence_length_cv(text);
        assert!(cv > 0.0);

        // Single sentence
        let text2 = "Only one sentence";
        let cv2 = compute_sentence_length_cv(text2);
        assert_eq!(cv2, 0.0);
    }

    #[test]
    fn test_avg_word_length() {
        let text = "cat dog";
        let avg = compute_avg_word_length(text);
        assert_eq!(avg, 3.0); // Both words are 3 chars
    }

    #[test]
    fn test_word_frequency_entropy() {
        let text = "the cat sat on the mat";
        let entropy = compute_word_frequency_entropy(text);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_empty_inputs() {
        let empty = "";
        assert_eq!(compute_bigram_repetition_rate(empty), 0.0);
        assert_eq!(compute_punctuation_entropy(empty), 0.0);
        assert_eq!(compute_lexical_diversity(empty), 0.0);
        assert_eq!(compute_vocab_richness(empty), 0.0);
        assert_eq!(compute_word_repetition_rate(empty), 0.0);
        assert_eq!(compute_sentence_length_cv(empty), 0.0);
        assert_eq!(compute_avg_word_length(empty), 0.0);
        assert_eq!(compute_word_frequency_entropy(empty), 0.0);
    }
}
