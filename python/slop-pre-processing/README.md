# slop-pre-processing

Fast TF-IDF text vectorization for training AI text detection models.

Implementation in Rust with Python bindings.

The python bindings allow us to use the same Rust-based text preprocessing at training and inference time.

## Features

- **Token n-grams**: Uses tiktoken BPE token sequences (not characters/words)
- **sklearn-compatible API**: Drop-in replacement for training pipelines
- **Parallel processing**: Automatic multi-threading via Rust/rayon

## Installation

```bash
pip install slop-pre-processing
```

## Quick Start

```python
from slop_pre_processing import TfidfVectorizer, VectorizerParams

# Configure vectorizer
params = VectorizerParams(
    ngram_range=(3, 5),  # 3-5 token n-grams
    min_df=10,           # Ignore terms in < 10 docs
    max_df=0.8,          # Ignore terms in > 80% of docs
    sublinear_tf=True    # Apply log scaling to term frequencies
)

# Fit and transform training data
vectorizer, X_train = TfidfVectorizer.fit_transform(train_texts, params)

# Transform test data
X_test = vectorizer.transform(test_texts)

# Save vectorizer for inference
vectorizer.save("tfidf_vectorizer.bin")
```

## API Overview

### VectorizerParams

Configuration for text processing:

- `ngram_range`: Tuple of (min_n, max_n) for token n-gram range
- `min_df`: Minimum document frequency (proportion or count)
- `max_df`: Maximum document frequency (proportion or count)
- `sublinear_tf`: Apply `1 + log(tf)` scaling

### TfidfVectorizer

Main vectorizer class:

- `fit_transform(texts, params)`: Fit and transform in one pass (faster)
- `fit(texts, params)`: Fit vocabulary only
- `transform(texts)`: Transform to TF-IDF matrix
- `save(path)`: Save to bincode format
- `load(path)`: Load from bincode format

## Why Token N-grams?

Unlike character n-grams or word n-grams, this uses **sequences of BPE tokens**:

- With `ngram_range=(3,5)`, extracts 3-5 consecutive tiktoken tokens
- Better captures AI patterns spanning multiple sub-word units
- More compact vocabulary than character n-grams

## License

MIT
