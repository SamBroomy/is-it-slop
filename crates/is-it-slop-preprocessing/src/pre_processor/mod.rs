//! Text preprocessing pipeline for the slop classifier.
//!
//! - Text cleaning (HTML entities, encoding artifacts, dataset-specific patterns)
//! - Tokenization (tiktoken `o200k_base` BPE encoding)
//!
//! Chunking lives in [`crate::encoder`]: documents become fixed-size token grids there, and the
//! model's dynamic chunk axis takes arbitrarily many chunks per document.
mod cleaner;
mod tokenizer;
pub use cleaner::{
    TextCleaner, text_cleaner_for_inference, text_cleaner_for_training,
    text_cleaner_for_training_keeping_markdown,
};
pub use tokenizer::{token_len, tokenize};
