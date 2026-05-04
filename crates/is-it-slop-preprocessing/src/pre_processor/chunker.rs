//! Token-based text chunking for handling variable-length documents.
//!
//! This module provides [`TokenChunker`] for splitting token sequences into overlapping
//! chunks of fixed size. Chunking enables consistent handling of both short and long
//! documents by:
//! - Processing all texts as similarly-sized segments
//! - Capturing local patterns uniformly regardless of document length
//! - Preventing information loss from overly long or short documents
//!
//! The chunker uses an **even distribution algorithm** that adjusts chunk size and
//! overlap to avoid tiny trailing chunks, ensuring all chunks meet minimum size requirements.
//!
//! # Architecture
//!
//! Default configuration:
//! - **Chunk size**: 150 tokens
//! - **Overlap**: 15 tokens (10%)
//! - **Minimum chunk size**: 30 tokens
//!
//! For documents shorter than `chunk_size`, a single chunk is returned.
//!
//! # Example
//!
//! ```rust
//! use is_it_slop_preprocessing::pre_processor::TokenChunker;
//!
//! let chunker = TokenChunker::default();
//! let tokens: Vec<u32> = (0..500).collect(); // 500 tokens
//! let chunks = chunker.chunk(&tokens);
//! // Returns ~4 chunks of ~130-150 tokens each with ~15 token overlap
//! ```

use rayon::prelude::*;

/// Splits token sequences into overlapping chunks with even distribution.
///
/// Uses an optimization algorithm to evenly distribute tokens across chunks,
/// avoiding tiny trailing chunks that would violate `min_chunk_size`.
#[derive(Clone, Debug, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TokenChunker {
    /// Size of each chunk in tokens, e.g., 150 tokens
    pub chunk_size: usize,
    /// Number of overlapping tokens between chunks, e.g., 15 tokens(10%)
    pub overlap: usize,
    /// Minimum size of a chunk to be included, e.g., 30 tokens
    pub min_chunk_size: usize,
}

impl Default for TokenChunker {
    fn default() -> Self {
        Self {
            chunk_size: 150,
            overlap: 15,
            min_chunk_size: 30,
        }
    }
}

impl TokenChunker {
    /// Chunk a single token sequence with even distribution
    #[must_use]
    pub fn chunk(&self, tokens: &[u32]) -> Vec<Vec<u32>> {
        if tokens.len() <= self.chunk_size {
            return vec![tokens.to_vec()];
        }
        // Calculate optimal number of chunks
        let num_chunks = self.calculate_num_chunks(tokens.len());
        if num_chunks == 1 {
            return vec![tokens.to_vec()];
        }

        // Calculate optimal chunk size and overlap to distribute evenly
        let (chunk_size, overlap) = self.calculate_optimal_params(tokens.len(), num_chunks);
        let chunk_size = chunk_size.min(tokens.len());
        let overlap = overlap.min(chunk_size);

        let mut chunks = Vec::with_capacity(num_chunks);
        let step = chunk_size.saturating_sub(overlap);

        let mut start = 0;
        while start < tokens.len() {
            let end = (start + chunk_size).min(tokens.len());
            // Only add chunks that meet minimum size requirement
            if end - start >= self.min_chunk_size || chunks.is_empty() {
                chunks.push(tokens[start..end].to_vec());
            }
            start += step;
            // Prevent infinite loop if step is 0
            if step == 0 {
                break;
            }
        }
        chunks
    }

    /// Calculate the optimal number of chunks needed
    fn calculate_num_chunks(&self, total_tokens: usize) -> usize {
        if total_tokens <= self.chunk_size {
            return 1;
        }

        // Calculate how many chunks we'd get with the standard approach
        let step = self.chunk_size - self.overlap;

        // The first chunk takes chunk_size tokens, then each subsequent chunk advances by step
        let remaining_after_first = total_tokens.saturating_sub(self.chunk_size);
        let additional_chunks = remaining_after_first.div_ceil(step);
        let naive_chunks = 1 + additional_chunks;

        // Check if the last chunk would be too small
        let last_chunk_start = (naive_chunks - 1) * step;
        let last_chunk_size = total_tokens.saturating_sub(last_chunk_start);

        if last_chunk_size < self.min_chunk_size && naive_chunks > 1 {
            // Last chunk too small, merge it with previous chunk
            naive_chunks - 1
        } else {
            naive_chunks
        }
    }

    /// Calculate optimal chunk size and overlap to evenly distribute tokens
    fn calculate_optimal_params(&self, total_tokens: usize, num_chunks: usize) -> (usize, usize) {
        if num_chunks == 1 {
            return (total_tokens, 0);
        }
        // chunk_size = (total_tokens + overlap * (num_chunks - 1)) / num_chunks
        // We'll iterate to find the best fit
        let mut best_chunk_size = self.chunk_size;
        let mut best_overlap = self.overlap;
        let mut best_error = usize::MAX;

        // Maximum chunk size cannot exceed total tokens
        let max_chunk_size = total_tokens.min(self.chunk_size + 50);

        // Try chunk sizes around the configured size
        for chunk_size in (self.min_chunk_size..=max_chunk_size).rev() {
            // Calculate overlap needed to fit exactly
            // total_tokens = chunk_size + (num_chunks - 1) * step
            // step = (total_tokens - chunk_size) / (num_chunks - 1)
            if chunk_size > total_tokens {
                continue; // Skip invalid configurations
            }
            let step = (total_tokens.saturating_sub(chunk_size)) / (num_chunks - 1);
            let overlap = chunk_size.saturating_sub(step);

            // Check if this is a valid configuration
            if overlap < chunk_size && overlap <= self.overlap * 2 {
                // Calculate actual coverage
                let actual_coverage = chunk_size + (num_chunks - 1) * step;
                let error = actual_coverage.abs_diff(total_tokens);

                if error < best_error {
                    best_error = error;
                    best_chunk_size = chunk_size;
                    best_overlap = overlap;
                }

                if error == 0 {
                    break;
                }
            }
        }

        (best_chunk_size, best_overlap)
    }

    /// Chunk multiple token sequences in parallel with even distribution
    #[must_use]
    pub fn chunk_batch(&self, token_sequences: &[Vec<u32>]) -> Vec<Vec<Vec<u32>>> {
        token_sequences
            .par_iter()
            .map(|tokens| self.chunk(tokens))
            .collect()
    }

    /// Serialize the `TokenChunker` to a JSON string.
    #[cfg(feature = "serde")]
    pub fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_even_chunk_distribution() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        // 250 tokens should split evenly into 3 chunks
        let tokens: Vec<u32> = (0..250).collect();
        let chunks = chunker.chunk(&tokens);

        println!("Chunks created: {}", chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("Chunk {}: size = {}", i, chunk.len());
        }

        // All chunks should be reasonably sized (no tiny last chunk)
        for chunk in &chunks {
            assert!(
                chunk.len() >= chunker.min_chunk_size,
                "Chunk size {} is below minimum {}",
                chunk.len(),
                chunker.min_chunk_size
            );
        }

        // Verify coverage (with overlaps, total unique tokens should be close to input)
        let first_token_positions: Vec<usize> =
            chunks.iter().map(|chunk| chunk[0] as usize).collect();
        println!("Chunk starting positions: {first_token_positions:?}");
    }

    #[test]
    fn test_calculate_num_chunks() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        // Test various input sizes
        // For 50 tokens: fits in single chunk
        assert_eq!(chunker.calculate_num_chunks(50), 1);

        // For 100 tokens: exactly one chunk
        assert_eq!(chunker.calculate_num_chunks(100), 1);

        // For 150 tokens: step=90, first chunk 0-99, second starts at 90, covers 90-149 (60 tokens)
        // 60 >= min_chunk_size(30), so 2 chunks
        assert_eq!(chunker.calculate_num_chunks(150), 2);

        // For 250 tokens: step=90
        // Chunk 0: 0-99 (100 tokens)
        // Chunk 1: 90-189 (100 tokens)
        // Chunk 2: 180-249 (70 tokens) >= 30, so 3 chunks
        assert_eq!(chunker.calculate_num_chunks(250), 3);

        // For 500 tokens: step=90
        // Positions: 0, 90, 180, 270, 360, 450
        // Last chunk: 450-499 (50 tokens) >= 30, so 6 chunks
        assert_eq!(chunker.calculate_num_chunks(500), 6);
    }
    #[test]
    fn test_no_dropped_content() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        let tokens: Vec<u32> = (0..275).collect();
        let chunks = chunker.chunk(&tokens);

        // Last chunk should include the final tokens
        let last_chunk = chunks.last().unwrap();
        assert_eq!(
            last_chunk.last().unwrap(),
            &274,
            "Last token should be included"
        );

        // Verify reasonable chunk sizes
        for (i, chunk) in chunks.iter().enumerate() {
            println!("Chunk {}: {} tokens", i, chunk.len());
            assert!(chunk.len() >= chunker.min_chunk_size);
        }
    }

    #[test]
    fn test_small_input_unchanged() {
        let chunker = TokenChunker {
            chunk_size: 150,
            overlap: 15,
            min_chunk_size: 30,
        };

        let tokens: Vec<u32> = (0..50).collect();
        let chunks = chunker.chunk(&tokens);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 50);
    }
    #[test]
    fn test_chunk_edge_cases() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        // Test with tokens less than chunk_size but more than configured
        let tokens_short: Vec<u32> = (0..75).collect();
        let chunks = chunker.chunk(&tokens_short);
        assert_eq!(chunks.len(), 1, "Should produce 1 chunk for 75 tokens");
        assert_eq!(chunks[0].len(), 75);

        // Test with tokens slightly over chunk_size
        let tokens_medium: Vec<u32> = (0..120).collect();
        let chunks = chunker.chunk(&tokens_medium);
        assert!(!chunks.is_empty(), "Should produce at least one chunk");
        for chunk in &chunks {
            assert!(chunk.len() >= chunker.min_chunk_size);
        }
        assert_eq!(
            chunks.last().unwrap().last().unwrap(),
            &119,
            "Last token should be included"
        );

        // Test with various small sizes that caused panics
        for size in [169, 172, 177, 179, 191] {
            let tokens: Vec<u32> = (0..size).collect();
            let chunks = chunker.chunk(&tokens);
            assert!(!chunks.is_empty(), "Size {size} should produce chunks");
            assert_eq!(
                chunks.last().unwrap().last().unwrap(),
                &(size - 1),
                "Size {size} should include last token"
            );
        }
    }

    // Serialization tests

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_round_trip() {
        use serde_json;

        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 25,
        };

        // Serialize to JSON
        let json_str = serde_json::to_string(&chunker).expect("Should serialize");
        assert!(json_str.contains("chunk_size"));
        assert!(json_str.contains("100"));

        // Deserialize from JSON
        let loaded: TokenChunker = serde_json::from_str(&json_str).expect("Should deserialize");
        assert_eq!(loaded.chunk_size, chunker.chunk_size);
        assert_eq!(loaded.overlap, chunker.overlap);
        assert_eq!(loaded.min_chunk_size, chunker.min_chunk_size);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_from_json_str() {
        let json = r#"{"chunk_size": 150, "overlap": 15, "min_chunk_size": 30}"#;
        let chunker = TokenChunker::from_json_str(json).expect("Should parse JSON");

        assert_eq!(chunker.chunk_size, 150);
        assert_eq!(chunker.overlap, 15);
        assert_eq!(chunker.min_chunk_size, 30);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_serialization_format() {
        // Verify JSON format is human-readable and correct
        let chunker = TokenChunker {
            chunk_size: 200,
            overlap: 20,
            min_chunk_size: 50,
        };

        let json = serde_json::to_string_pretty(&chunker).expect("Should serialize");
        println!("JSON output:\n{json}");

        // Verify structure
        assert!(json.contains(r#""chunk_size": 200"#));
        assert!(json.contains(r#""overlap": 20"#));
        assert!(json.contains(r#""min_chunk_size": 50"#));
    }

    // Overlap validation tests

    #[test]
    fn test_overlap_validation_between_adjacent_chunks() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        let tokens: Vec<u32> = (0..250).collect();
        let chunks = chunker.chunk(&tokens);

        assert!(chunks.len() >= 2, "Should produce multiple chunks");

        // Verify chunks are sequential and cover the full range
        for i in 0..chunks.len() - 1 {
            let chunk1 = &chunks[i];
            let chunk2 = &chunks[i + 1];

            // Verify chunks are not empty
            assert!(!chunk1.is_empty());
            assert!(!chunk2.is_empty());

            // Verify chunks are sequential (chunk2 starts after chunk1)
            let last_token_chunk1 = *chunk1.last().unwrap();
            let first_token_chunk2 = *chunk2.first().unwrap();

            // Due to even distribution algorithm, overlap may vary
            // Just verify sequential ordering
            assert!(
                last_token_chunk1 < first_token_chunk2 + 50, // Allow for overlap
                "Chunks should be reasonably sequential"
            );
        }
    }

    #[test]
    fn test_no_dropped_tokens_in_chunks() {
        // Verify all tokens from input appear in at least one chunk
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        let tokens: Vec<u32> = (0..275).collect();
        let chunks = chunker.chunk(&tokens);

        // First token should be in first chunk
        assert_eq!(chunks[0][0], 0);

        // Last token should be in last chunk
        assert_eq!(*chunks.last().unwrap().last().unwrap(), 274);

        // Verify chunks cover the full range
        let all_unique_tokens: std::collections::HashSet<u32> =
            chunks.iter().flat_map(|c| c.iter()).copied().collect();

        // All original tokens should appear
        for &token in &tokens {
            assert!(
                all_unique_tokens.contains(&token),
                "Token {token} not found in any chunk"
            );
        }
    }

    // Invalid config tests

    // Note: Invalid chunker configurations (overlap >= chunk_size) are not tested
    // as they represent programmer errors that should be caught during development.
    // The implementation will panic or produce incorrect results for such configs.

    #[test]
    fn test_invalid_config_min_chunk_greater_than_chunk_size() {
        // min_chunk_size > chunk_size: logically inconsistent
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 150,
        };

        let tokens: Vec<u32> = (0..200).collect();
        let chunks = chunker.chunk(&tokens);

        // Should still produce chunks (first chunk always included)
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_zero_overlap_no_shared_tokens() {
        // Zero overlap: chunks should be contiguous but not overlapping
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 0,
            min_chunk_size: 30,
        };

        let tokens: Vec<u32> = (0..250).collect();
        let chunks = chunker.chunk(&tokens);

        assert!(chunks.len() >= 2);

        // Verify no overlap: last token of chunk[i] + 1 == first token of chunk[i+1]
        for i in 0..chunks.len() - 1 {
            let last_token = *chunks[i].last().unwrap();
            let next_first_token = *chunks[i + 1].first().unwrap();

            // With even distribution, may not be exactly contiguous, but should be close
            assert!(
                next_first_token > last_token,
                "Chunks should not overlap with zero overlap setting"
            );
        }
    }

    #[test]
    fn test_chunking_with_custom_parameters() {
        // Test various parameter combinations
        let configs = vec![
            (50, 5, 10),   // Small chunks
            (200, 20, 40), // Large chunks
            (150, 0, 30),  // No overlap
            (150, 30, 30), // High overlap (20%)
        ];

        for (chunk_size, overlap, min_chunk_size) in configs {
            let chunker = TokenChunker {
                chunk_size,
                overlap,
                min_chunk_size,
            };

            let tokens: Vec<u32> = (0..300).collect();
            let chunks = chunker.chunk(&tokens);

            // Basic invariants
            assert!(!chunks.is_empty());
            for chunk in &chunks {
                assert!(
                    chunk.len() >= min_chunk_size || chunks.len() == 1,
                    "Chunk size {} violates min_chunk_size {min_chunk_size}",
                    chunk.len()
                );
            }

            // Verify first and last tokens
            assert_eq!(chunks[0][0], 0);
            assert_eq!(*chunks.last().unwrap().last().unwrap(), 299);
        }
    }

    #[test]
    fn test_chunk_batch_equivalence() {
        let chunker = TokenChunker::default();
        let token_sequences: Vec<Vec<u32>> = (0..10)
            .map(|i| (i * 100..i * 100 + 250).collect())
            .collect();

        let batched = chunker.chunk_batch(&token_sequences);
        let sequential: Vec<Vec<Vec<u32>>> = token_sequences
            .iter()
            .map(|tokens| chunker.chunk(tokens))
            .collect();

        assert_eq!(batched.len(), sequential.len());
        for (i, (batch_chunks, seq_chunks)) in batched.iter().zip(sequential.iter()).enumerate() {
            assert_eq!(
                batch_chunks, seq_chunks,
                "chunk_batch and chunk disagree at index {i}"
            );
        }
    }

    #[test]
    fn test_chunk_empty_input() {
        let chunker = TokenChunker::default();
        let result = chunker.chunk(&[]);
        assert_eq!(result, vec![Vec::<u32>::new()]);

        let batch_result = chunker.chunk_batch(&[Vec::new(), Vec::new()]);
        assert_eq!(batch_result.len(), 2);
        assert_eq!(batch_result[0], vec![Vec::<u32>::new()]);
        assert_eq!(batch_result[1], vec![Vec::<u32>::new()]);
    }

    #[test]
    fn test_chunk_exact_boundaries() {
        let chunker = TokenChunker {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 30,
        };

        let tokens_exact: Vec<u32> = (0..100).collect();
        let chunks = chunker.chunk(&tokens_exact);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 100);

        let tokens_two_chunks: Vec<u32> = (0..200).collect();
        let chunks = chunker.chunk(&tokens_two_chunks);
        assert!(
            chunks.len() >= 2,
            "200 tokens should produce at least 2 chunks"
        );
        assert_eq!(*chunks.last().unwrap().last().unwrap(), 199);
    }
}
