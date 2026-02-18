# is-it-slop-preprocessing

Fast TF-IDF text vectorization for training AI text detection models.

Implementation in Rust with Python bindings.

> **Note for inference users:** If you only want to use the AI text detection model for predictions, install [`is-it-slop`](https://pypi.org/project/is-it-slop/) instead. This preprocessing library is primarily for the training step or accessing the preprocessing pipeline directly.

The Python bindings allow us to use the same Rust-based text preprocessing at training and inference time, ensuring consistency between model training and deployment.

## Features

- **Token n-grams**: Uses tiktoken BPE token sequences (not characters/words)
- **sklearn-compatible API**: Drop-in replacement for training pipelines
- **Parallel processing**: Automatic multi-threading via Rust/rayon
- **Multiple serialization formats**: rkyv (default), bincode, and JSON support

## Installation

```bash
pip install is-it-slop-preprocessing
```

## Quick Start

```python
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams

# Configure vectorizer (n-gram range is fixed at 2-4 tokens)
params = VectorizerParams(
    min_df=10,           # Ignore terms in < 10 docs
    max_df=0.8,          # Ignore terms in > 80% of docs
    sublinear_tf=True    # Apply log scaling to term frequencies
)

# Fit and transform training data
vectorizer, X_train = TfidfVectorizer.fit_transform(train_texts, params)

# Transform test data
X_test = vectorizer.transform(test_texts)

# Save vectorizer for inference
vectorizer.save("tfidf_vectorizer.rkyv")
```

## Platform Support

Pre-built wheels available for:

- **Linux**: x86_64, aarch64 (manylinux_2_28)
- **macOS**: Apple Silicon (ARM64)
- **Windows**: x86_64

## License

MIT
