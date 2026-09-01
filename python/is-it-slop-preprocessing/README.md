# is-it-slop-preprocessing

Text cleaning, BPE tokenization, and chunking for AI text detection datasets.

Implementation in Rust with Python bindings.

> **Note for inference users:** If you only want to use the AI text detection model for predictions, install [`is-it-slop`](https://pypi.org/project/is-it-slop/) instead. This preprocessing library exposes the pipeline's text side for dataset curation.

The Python bindings allow us to use the same Rust-based text preprocessing when building datasets and at inference time, ensuring consistency between the data the model sees in training and deployment.

## Features

- **Text cleaning**: Two-stage cleaning (universal artifacts always; dataset artifacts for training data)
- **Tokenization**: tiktoken BPE `o200k_base` token IDs
- **Chunking**: Token-level chunking for long documents
- **Parallel processing**: Automatic multi-threading via Rust/rayon

## Installation

```bash
pip install is-it-slop-preprocessing
```

## Quick Start

```python
from is_it_slop_preprocessing import CleaningMode, TextCleaner, TokenChunker, tokenize

# Clean training data (removes dataset-specific artifacts)
cleaner = TextCleaner(CleaningMode.TRAINING)
clean_texts = cleaner.clean_batch(train_texts)

# Tokenize
tokens = tokenize(clean_texts)

# Chunk long documents
chunker = TokenChunker(chunk_size=150, overlap=15)
chunked = chunker.chunk_batch(tokens)
```

## Platform Support

Pre-built wheels available for:

- **Linux**: x86_64, aarch64 (manylinux_2_28)
- **macOS**: Apple Silicon (ARM64)
- **Windows**: x86_64

## License

MIT
