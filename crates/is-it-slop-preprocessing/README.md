# is-it-slop-preprocessing

Text preprocessing for AI text detection: cleaning, BPE tokenization, and document encoding.

> **Note:** This is an internal crate used by [`is-it-slop`](https://crates.io/crates/is-it-slop). For AI text detection, use the main [`is-it-slop`](https://crates.io/crates/is-it-slop) crate instead.

## Purpose

This crate provides the text side of the is-it-slop pipeline — the one implementation both training and inference encode through:

- **Text cleaning** (two-stage: universal + dataset artifacts)
- **Tokenization** (tiktoken BPE o200k_base)
- **Document encoding** (frequency-ranked vocabulary + fixed-size token-grid chunking)

## Quick Start

```rust
use is_it_slop_preprocessing::encoder::{DocumentEncoder, Vocabulary};

// Training: rank a vocabulary on the training corpus and encode documents.
let texts = ["some document", "another document"];
let vocabulary = Vocabulary::build(&texts, 16_384);
let encoder = DocumentEncoder::new(vocabulary);
let encoded = encoder.encode_batch(&texts, &[0.0, 1.0]);

// Inference: reconstruct the exact shipped encoding from bundle artifacts.
// let encoder = DocumentEncoder::from_bundle(vocab_bytes, config_json)?;
```

## Architecture

**Two-stage Cleaning:**

- **Universal cleaner** (always): HTML entities, encoding artifacts, whitespace normalization
- **Dataset artifact cleaner** (training only): Citations, news datelines, academic headers

> Two cleaners ensure that the model learns actual patterns rather than overfitting to specific dataset quirks (which there are a lot that I could find and still likely missed many).

**Encoding:**
Tokens map through a frequency-ranked vocabulary onto compact embedding rows (rare tokens share one OOV row), then fill fixed-size chunk grids — padded, never back-filled — that the model consumes directly.

## For AI Text Detection

For end-to-end AI text detection (inference), use the main crate:

```bash
cargo add is-it-slop
```

Or the CLI:

```bash
cargo install is-it-slop --features cli
is-it-slop "Your text here"
```

See the [is-it-slop documentation](https://docs.rs/is-it-slop) for the complete detection pipeline.

## Documentation

- [API Documentation](https://docs.rs/is-it-slop-preprocessing)
- [Main Project](https://github.com/SamBroomy/is-it-slop)

## License

MIT
