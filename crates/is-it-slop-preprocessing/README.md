# is-it-slop-preprocessing

Text preprocessing for AI text detection: cleaning, BPE tokenization, and chunking.

> **Note:** This is primarily an internal crate used by [`is-it-slop`](https://crates.io/crates/is-it-slop). For AI text detection, use the main [`is-it-slop`](https://crates.io/crates/is-it-slop) crate instead.

## Purpose

This crate provides the text side of the is-it-slop pipeline:

- **Text cleaning** (two-stage: universal + dataset artifacts)
- **Tokenization** (tiktoken BPE o200k_base)
- **Token-based chunking** (fixed-size token windows for long documents)

## Use Cases

1. **Dataset curation**: The Python bindings expose cleaning and tokenization for building training datasets
2. **Internal use**: Powers the [`is-it-slop`](https://crates.io/crates/is-it-slop) inference pipeline and the Rust training crate

## Features

- `python`: PyO3 bindings for Python dataset workflows
- `serde`: JSON serialization for configuration types
- `progress-bars`: Progress indicators for long-running operations

## Quick Start

```rust
use is_it_slop_preprocessing::pre_processor::{text_cleaner_for_inference, tokenize};

let cleaner = text_cleaner_for_inference();
let clean_text = cleaner.clean("Raw input text");
let tokens = tokenize(&[clean_text]);
```

## Python Bindings

For dataset curation in Python:

```bash
uv add is-it-slop-preprocessing
```

```python
from is_it_slop_preprocessing import CleaningMode, TextCleaner, tokenize

cleaner = TextCleaner(CleaningMode.TRAINING)
clean_texts = cleaner.clean_batch(train_texts)
tokens = tokenize(clean_texts)
```

See the [Python package documentation](https://pypi.org/project/is-it-slop-preprocessing/) for the complete API reference.

## Architecture

**Two-stage Cleaning:**

- **Universal cleaner** (always): HTML entities, encoding artifacts, whitespace normalization
- **Dataset artifact cleaner** (training only): Citations, news datelines, academic headers

> Two cleaners ensure that the model learns actual patterns rather than overfitting to specific dataset quirks (which there are a lot that I could find and still likely missed many).

**Chunking:**
Splits long token sequences into fixed-size windows, enabling consistent handling regardless of document length.

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
