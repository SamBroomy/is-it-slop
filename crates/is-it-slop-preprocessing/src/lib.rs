//! Fast TF-IDF text vectorization using token-level n-grams.
//!
//! This crate provides high-performance text preprocessing for machine learning,
//! using tiktoken BPE tokenization and sparse matrix operations. Designed for both
//! training (via Python bindings) and inference (native Rust).
//!
//! # Architecture
//!
//! The pipeline consists of:
//! 1. **Tokenization** - Text → BPE token IDs using tiktoken `o200k_base`
//! 2. **N-gram extraction** - Token IDs → token n-grams (sequences of consecutive tokens)
//! 3. **Counting** (`CountVectorizer`) - N-grams → sparse term-frequency matrix
//! 4. **TF-IDF weighting** ([`TfidfVectorizer`]) - TF matrix → normalized TF-IDF features
//!
//! # Key Features
//!
//! - **Token n-grams**: Uses sequences of BPE tokens (not characters or words)
//! - **Parallel processing**: Automatic parallelization via rayon for large datasets
//! - **Sparse matrices**: Memory-efficient `sprs::CsMat` representation
//! - **sklearn-compatible**: Python API matches sklearn's `TfidfVectorizer`
//!
//! # Example (Rust)
//!
//! ```rust
//! use is_it_slop_preprocessing::pre_processor::{TfidfVectorizer, VectorizerParams};
//!
//! let texts = vec!["example text 1", "example text 2"];
//! let params = VectorizerParams::new(2..=4, 10.0, 0.8, true);
//! let (vectorizer, tfidf_matrix) = TfidfVectorizer::fit_transform(&texts, params);
//! ```
//!
//! [`TfidfVectorizer`]: pre_processor::TfidfVectorizer

#[cfg(feature = "python")]
mod python;

pub mod pre_processor;

/// Internal constant for optimizing n-gram key storage.
///
/// Uses the default max n-gram size to set the inline capacity of `SmallVec` for n-gram keys.
/// This avoids heap allocations for the common case where n-grams fit within the default range.
pub(crate) const NGRAM_CONST_KEY: usize = pre_processor::DEFAULT_MAX_NGRAM;
