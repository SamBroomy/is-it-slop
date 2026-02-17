//! Text preprocessing pipeline for machine learning.
//!
//! This module provides a complete text processing pipeline combining:
//! - Text cleaning (HTML entities, encoding artifacts, dataset-specific patterns)
//! - Tokenization (tiktoken BPE encoding)
//! - Token-based chunking (splitting long documents into overlapping segments)
//! - TF-IDF vectorization with token n-grams
//!
//! The pipeline is designed for both training (Python bindings) and inference (native Rust),
//! with automatic parallelization and sparse matrix representations for efficiency.
mod chunker;
mod cleaner;
mod ngrams;
mod tokenizer;
mod vectorizer;
pub use chunker::TokenChunker;
pub use cleaner::{TextCleaner, text_cleaner_for_inference, text_cleaner_for_training};
pub use tokenizer::{reverse_tokenize, tokenize};
pub use vectorizer::{
    CountVectorizer, DEFAULT_MAX_NGRAM, DEFAULT_MIN_NGRAM, TfidfVectorizer, VectorizerParams,
};
