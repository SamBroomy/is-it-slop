"""Fast TF-IDF text vectorization using Rust-backed implementation.

This package provides high-performance text preprocessing for machine learning,
using tiktoken BPE tokenization and sparse matrix operations.

Key Features
------------
- Token n-grams: Uses tiktoken BPE token sequences (not characters/words)
- Parallel processing: Automatic multi-threading via Rust/rayon
- sklearn-compatible: Drop-in replacement for training pipelines
- Text cleaning: Remove dataset artifacts and encoding issues

Quick Start
-----------
>>> from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams
>>> from is_it_slop_preprocessing import TextCleaner, CleaningMode
>>>
>>> # Clean training data
>>> cleaner = TextCleaner(CleaningMode.TRAINING)
>>> clean_texts = cleaner.clean_batch(train_texts)
>>>
>>> # Vectorize
>>> params = VectorizerParams(ngram_range=(3, 5), min_df=10, max_df=0.8)
>>> vectorizer, X_train = TfidfVectorizer.fit_transform(clean_texts, params)
>>>
>>> # At inference time
>>> cleaner = TextCleaner(CleaningMode.INFERENCE)
>>> clean_input = cleaner.clean(user_input)
>>> X_test = vectorizer.transform([clean_input])

"""

# Import only user-facing wrapper classes
from ._internal import (
    CleaningMode,
    TextCleaner,
    TfidfVectorizer,
    TokenChunker,
    VectorizerParams,
    __version__,
    reverse_tokenize,
    tokenize,
)

__all__ = [
    "CleaningMode",
    "TextCleaner",
    "TfidfVectorizer",
    "TokenChunker",
    "VectorizerParams",
    "__version__",
    "reverse_tokenize",
    "tokenize",
]
