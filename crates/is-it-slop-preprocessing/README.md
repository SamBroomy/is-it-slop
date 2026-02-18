# is-it-slop-preprocessing

Fast TF-IDF text preprocessing for AI text detection models.

> **Note:** This is primarily an internal crate used by [`is-it-slop`](https://crates.io/crates/is-it-slop). For AI text detection, use the main [`is-it-slop`](https://crates.io/crates/is-it-slop) crate instead.

## Purpose

This crate provides the preprocessing pipeline for the is-it-slop AI text detection system:

- **Text cleaning** (two-stage: universal + dataset artifacts)
- **Tokenization** (tiktoken BPE o200k_base)
- **Token-based chunking** (overlapping 150-token windows)
- **Token n-gram extraction** (2-4 token sequences)
- **TF-IDF vectorization** (sklearn-compatible)

## Use Cases

1. **Training custom AI detection models**: Use this crate to preprocess training data with the same pipeline used in inference
2. **Python bindings**: Exposes Rust preprocessing to Python via PyO3 for training workflows
3. **Internal use**: Powers the [`is-it-slop`](https://crates.io/crates/is-it-slop) inference pipeline

## Features

- `python` (default): PyO3 bindings for Python training workflows
- `rkyv` (default): Zero-copy serialization for vectorizers
- `serde`: Alternative JSON/bincode serialization
- `bincode`: Legacy bincode serialization support
- `progress-bars`: Progress indicators for long-running operations
- `mimalloc`: Use mimalloc as the global allocator

## Quick Start

```rust
use is_it_slop_preprocessing::{
    TfidfVectorizer, VectorizerParams, text_cleaner_for_inference,
};

// Clean and vectorize text
let cleaner = text_cleaner_for_inference();
let clean_text = cleaner.clean("Raw input text");

// Configure vectorizer (n-gram range is fixed at 2-4 tokens)
let params = VectorizerParams::new(10.0, 0.9, true); // min_df, max_df, sublinear_tf

// Fit vectorizer on training data
let vectorizer = TfidfVectorizer::fit(&train_texts, &params)?;

// Transform text to TF-IDF features
let features = vectorizer.transform(&[clean_text])?;

// Save for inference
vectorizer.save_rkyv("vectorizer.rkyv")?;
```

## Python Bindings

For model training in Python:

```bash
uv add is-it-slop-preprocessing
```

or using pip:

```bash
pip install is-it-slop-preprocessing
```

```python
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams

# Configure and fit vectorizer
params = VectorizerParams(min_df=10, max_df=0.9, sublinear_tf=True)
vectorizer, X_train = TfidfVectorizer.fit_transform(train_texts, params)

# Save for Rust inference
vectorizer.save("vectorizer.rkyv")
```

See the [Python package documentation](https://pypi.org/project/is-it-slop-preprocessing/) for complete API reference.

## Architecture

**Token N-grams:**
Unlike character or word n-grams, this uses sequences of tiktoken BPE tokens (2-4 consecutive tokens). This captures AI writing patterns at the subword level.

**Two-stage Cleaning:**

- **Universal cleaner** (always): HTML entities, encoding artifacts, whitespace normalization
- **Dataset artifact cleaner** (training only): Citations, news datelines, academic headers

> Two cleaners ensure that the model learns actual patterns rather than overfitting to specific dataset quirks (which there are a lot that I could find and still likely missed many).

**Chunking:**
Splits long texts into overlapping 150-token chunks with 15-token overlap, enabling consistent feature extraction regardless of document length.

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
- [Python Package](https://pypi.org/project/is-it-slop-preprocessing/)

## License

MIT
