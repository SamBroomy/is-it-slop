//! ! Pre-processing module for is-it-slop
//!
//! This module provides text vectorization using TF-IDF with token-level n-grams.

mod vectorizer;

pub use vectorizer::{DEFAULT_MAX_NGRAM, DEFAULT_MIN_NGRAM, TfidfVectorizer, VectorizerParams};
