//! Text preprocessing pipeline for the slop classifier.
//!
//! - Text cleaning (HTML entities, encoding artifacts, dataset-specific patterns)
//! - Tokenization (tiktoken `o200k_base` BPE encoding)
//! - Token-based chunking (splitting long documents into overlapping segments)
mod chunker;
mod cleaner;
mod tokenizer;
pub use chunker::TokenChunker;
pub use cleaner::{TextCleaner, text_cleaner_for_inference, text_cleaner_for_training};
pub use tokenizer::tokenize;
