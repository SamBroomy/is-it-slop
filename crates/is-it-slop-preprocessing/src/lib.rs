//! Text preprocessing for the is-it-slop classifier.
//!
//! Turns raw text into the token-level representation the model consumes:
//!
//! 1. **Cleaning** — HTML entities, encoding artifacts, dataset-specific patterns
//! 2. **Tokenization** — text → BPE token IDs using tiktoken `o200k_base`
//! 3. **Chunking** — token IDs → fixed-size segments for long documents
//!
//! Used from both training and inference so text becomes the same token grid
//! everywhere. Large batches parallelize automatically via rayon.

pub mod encoder;
pub mod pre_processor;

#[cfg(feature = "python")]
mod python;

/// Borrow any string slice as `&str`s.
///
/// Public batch functions accept `impl AsRef<str>` for ergonomics but do their work on `&[&str]`,
/// so only this shell is monomorphised per caller type and the bodies compile once.
pub fn as_strs<T: AsRef<str>>(texts: &[T]) -> Vec<&str> {
    texts.iter().map(AsRef::as_ref).collect()
}
