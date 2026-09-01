//! Text preprocessing for the is-it-slop classifier.
//!
//! Turns raw text into the token-level representation the model consumes:
//!
//! 1. **Cleaning** — HTML entities, encoding artifacts, dataset-specific patterns
//! 2. **Tokenization** — text → BPE token IDs using tiktoken `o200k_base`
//! 3. **Chunking** — token IDs → fixed-size segments for long documents
//!
//! Used from both training (via Python bindings) and inference (native Rust).
//! Large batches parallelize automatically via rayon.

#[cfg(feature = "python")]
mod python;

pub mod pre_processor;
