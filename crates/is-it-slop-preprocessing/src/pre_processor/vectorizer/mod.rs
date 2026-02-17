//! TF-IDF vectorization with token n-grams.
//!
//! This module implements sparse TF-IDF (Term Frequency-Inverse Document Frequency)
//! vectorization using token-level n-grams. The implementation follows sklearn's
//! `TfidfVectorizer` API and parameters for compatibility with Python training workflows.
//!
//! # Pipeline
//!
//! 1. **Counting** ([`CountVectorizer`]): Build vocabulary and count term frequencies
//! 2. **Weighting** ([`TfidfVectorizer`]): Apply IDF weights and L2 normalization
//!
//! # Features
//!
//! - **Sparse matrices**: Memory-efficient CSR format via `sprs` crate
//! - **Parallel processing**: Automatic parallelization for large datasets
//! - **sklearn-compatible**: IDF formula matches sklearn: `log((n+1)/(df+1)) + 1`
//! - **Sublinear TF**: Optional `log(tf + 1)` scaling
//! - **Vocabulary filtering**: `min_df` and `max_df` thresholds
//!
//! # Example
//!
//! ```rust
//! use is_it_slop_preprocessing::pre_processor::{TfidfVectorizer, VectorizerParams};
//!
//! let texts = vec!["first document", "second document"];
//! let params = VectorizerParams::new(2..=4, 5.0, 0.9, true);
//! let (vectorizer, matrix) = TfidfVectorizer::fit_transform(&texts, params);
//! ```

mod count_vectorizer;

mod params;
mod tfidf_vectorizer;

pub use count_vectorizer::CountVectorizer;
pub use params::{DEFAULT_MAX_NGRAM, DEFAULT_MIN_NGRAM, VectorizerParams};
pub use tfidf_vectorizer::TfidfVectorizer;
