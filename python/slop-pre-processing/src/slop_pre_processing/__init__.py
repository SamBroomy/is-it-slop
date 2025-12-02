"""Fast TF-IDF text vectorization using Rust-backed implementation.

This package provides high-performance text preprocessing for machine learning,
using tiktoken BPE tokenization and sparse matrix operations.

Key Features
------------
- Token n-grams: Uses tiktoken BPE token sequences (not characters/words)
- Parallel processing: Automatic multi-threading via Rust/rayon
- sklearn-compatible: Drop-in replacement for training pipelines

Quick Start
-----------
>>> from slop_pre_processing import TfidfVectorizer, VectorizerParams
>>> params = VectorizerParams(ngram_range=(3, 5), min_df=10, max_df=0.8)
>>> vectorizer, X_train = TfidfVectorizer.fit_transform(train_texts, params)
>>> X_test = vectorizer.transform(test_texts)

"""

# Import only user-facing wrapper classes
from ._internal import TfidfVectorizer, VectorizerParams, __version__

__all__ = ["TfidfVectorizer", "VectorizerParams", "__version__"]
