"""Internal interface and types for the `is_it_slop_preprocessing` package.

This module provides the main interface to the Rust bindings for text vectorization
using TF-IDF.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
from scipy.sparse import csr_matrix, vstack

if TYPE_CHECKING:
    from numpy.typing import NDArray


from ._is_it_slop_preprocessing_rust_bindings import (
    RustCleaningMode,
    RustTextCleaner,
    RustTfidfVectorizer,
    RustTfidfVectorizerBuilder,
    RustTokenChunker,
    RustVectorizerParams,
    __version__,
)
from ._is_it_slop_preprocessing_rust_bindings import reverse_tokenize as reverse_tokenize_internal
from ._is_it_slop_preprocessing_rust_bindings import tokenize as tokenize_internal

__all__ = [
    "CleaningMode",
    "TextCleaner",
    "TfidfVectorizer",
    "TfidfVectorizerBuilder",
    "TokenChunker",
    "VectorizerParams",
    "__version__",
    "extract_combined_batch",
    "reverse_tokenize",
    "tokenize",
]


class SupportsToList(Protocol):
    """Protocol for Polars-like & Numpy-like objects.

    Requires: len(), slicing, to_list() and tolist() conversion.
    Matches: Polars Series, DataFrame columns, NumPy arrays, custom array types, etc.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, key: int | slice) -> Any: ...  # noqa: ANN401
    def to_list(self) -> list[str]: ...
    def tolist(self) -> list[str]: ...


if TYPE_CHECKING:
    ValidTexts: TypeAlias = list[str] | NDArray[np.str_] | NDArray[np.object_] | SupportsToList

logger = logging.getLogger(__name__)


def _validate_texts(texts: ValidTexts) -> list[str]:
    """Validate the input texts for fitting or transforming.

    Validated here so we dont pass invalid data to the Rust side.

    Args:
        texts: Input texts to validate. Accepts:
            - list[str]
            - NumPy array of strings (dtype 'U' or 'S')
            - Polars Series (via Protocol - no dependency required)
            - Any object with .to_list() or .tolist() method


    Returns:
        The texts as a list of strings.

    Raises:
        TypeError: If the input is not a valid text container.
        ValueError: If the input NumPy array is not 1-dimensional or has an invalid dtype.


    """
    # Handle Protocol objects (Polars Series, etc.) via duck typing
    if not isinstance(texts, (list, np.ndarray)):
        # Try to_list() first (Polars convention)
        if hasattr(texts, "to_list") and callable(texts.to_list):  # type: ignore[union-attr]
            try:
                texts = texts.to_list()  # type: ignore[union-attr]
            except Exception as e:
                msg = f"Failed to convert input via .to_list(): {e}"
                raise TypeError(msg) from e
        # Fallback to tolist() (NumPy-like convention)
        elif hasattr(texts, "tolist") and callable(texts.tolist):  # type: ignore[union-attr]
            try:
                texts = texts.tolist()  # type: ignore[union-attr]
            except Exception as e:
                msg = f"Failed to convert input via .tolist(): {e}"
                raise TypeError(msg) from e
        else:
            msg = (
                "Input must be a list of strings, NumPy array, or an object "
                "with .to_list() or .tolist() method (e.g., Polars Series)."
            )
            raise TypeError(msg)

    elif isinstance(texts, np.ndarray):
        if texts.dtype.kind == "O":
            texts = texts.astype(str)
        if texts.dtype.kind not in {"U", "S"}:
            msg = "NumPy array must have dtype 'str' or 'unicode'."
            raise TypeError(msg)
        if texts.ndim != 1:
            msg = "Input NumPy array must be 1-dimensional."
            raise ValueError(msg)
        return texts.tolist()

    if not isinstance(texts, list):
        msg = "Input must be a list of strings or a 1D NumPy array of strings."
        raise TypeError(msg)

    if not all(isinstance(t, str) for t in texts):
        msg = "All elements in the input list must be strings."
        raise TypeError(msg)

    return texts


class VectorizerParams:
    """Parameters for configuring the text vectorizer.

    Both min_df and max_df can be specified as either:
    - A float in (0.0, 1.0) representing a proportion of documents
    - A float >= 1.0 representing an absolute document count
    - Ngram range defaults to (2, 4)

    Args:
        min_df: Minimum document frequency (proportion or count)
        max_df: Maximum document frequency (proportion or count)
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf))

    Examples:
        min_df=0.05  # Filter terms appearing in < 5% of documents
        min_df=10.0  # Filter terms appearing in < 10 documents
        max_df=0.9   # Filter terms appearing in > 90% of documents
        max_df=500.0 # Filter terms appearing in > 500 documents
        sublinear_tf=True  # Use log scaling for term frequency

    """

    __slots__ = ("_inner",)

    def __init__(self, *, min_df: float, max_df: float, sublinear_tf: bool = True) -> None:
        self._inner = RustVectorizerParams(min_df, max_df, sublinear_tf)

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self._inner.ngram_range

    @property
    def min_df(self) -> float:
        return self._inner.min_df

    @property
    def max_df(self) -> float:
        return self._inner.max_df

    @property
    def sublinear_tf(self) -> bool:
        return self._inner.sublinear_tf

    def __repr__(self) -> str:
        return (
            f"VectorizerParams(ngram_range={self.ngram_range}, min_df={self.min_df}, "
            f"max_df={self.max_df}, sublinear_tf={self.sublinear_tf})"
        )

    def as_rust(self) -> RustVectorizerParams:
        """Return the underlying Rust object.

        Returns:
            RustVectorizerParams: The underlying Rust parameters object

        """
        return self._inner


class TfidfVectorizer:
    """TF-IDF text vectorizer with Rust-backed implementation.

    This vectorizer is always fitted - you cannot create an unfitted instance.
    Use the static `fit()` method to create a fitted vectorizer from training texts.

    Examples
    --------
    >>> from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams
    >>> params = VectorizerParams(min_df=10, max_df=0.8)
    >>> vectorizer = TfidfVectorizer.fit(train_texts, params)
    >>> X_test = vectorizer.transform(test_texts)

    """

    __slots__ = ("_parameters", "_vectorizer")

    def __init__(self, params: VectorizerParams, rust_vectorizer: RustTfidfVectorizer) -> None:
        """Private constructor. Use TfidfVectorizer.fit() to create instances.

        Args:
            params: VectorizerParams instance containing vectorizer configuration.
            rust_vectorizer: Fitted RustTfidfVectorizer instance.

        """
        self._parameters = params
        self._vectorizer = rust_vectorizer

    @staticmethod
    def fit(
        texts: ValidTexts, params: VectorizerParams, batch_size: int = 50_000, auto_batch_threshold: int = 100_000
    ) -> TfidfVectorizer:
        """Fit a new TF-IDF vectorizer to the provided texts with automatic batching for large datasets.

        Args:
            texts: Training texts to fit the vectorizer.
            params: Vectorizer parameters.
            batch_size: Texts per batch when batching (default: 50,000).
            auto_batch_threshold: Dataset size threshold for batching (default: 100,000).

        Returns:
            A fitted TfidfVectorizer instance.

        Notes:
            For datasets >= auto_batch_threshold, automatically uses batched processing
            to avoid OOM. Validates texts in batches to avoid memory spikes.

        """
        num_texts = len(texts)

        # Small datasets: validate all at once, use regular fit
        if num_texts < auto_batch_threshold:
            validated_texts = _validate_texts(texts)
            rust_vectorizer = RustTfidfVectorizer(validated_texts, params.as_rust())
            return TfidfVectorizer(params, rust_vectorizer)

        # Large datasets: validate in batches, use batched fit
        logger.info(
            "Dataset size (%s) >= threshold (%s), using batched training with batch_size=%s",
            f"{num_texts:,}",
            f"{auto_batch_threshold:,}",
            f"{batch_size:,}",
        )

        builder = TfidfVectorizerBuilder(params)
        num_batches = (num_texts + batch_size - 1) // batch_size

        for i in range(0, num_texts, batch_size):
            batch_idx = i // batch_size
            batch_slice = texts[i : i + batch_size]
            batch = _validate_texts(batch_slice)

            logger.info(
                "Processing batch %d/%d (%s texts, total_docs=%s)",
                batch_idx + 1,
                num_batches,
                f"{len(batch):,}",
                f"{builder.total_docs:,}",
            )

            builder.partial_fit(batch)

        logger.info(
            "Finalizing vectorizer (total_docs=%s, raw_vocab=%s)",
            f"{builder.total_docs:,}",
            f"{builder.raw_vocab_size:,}",
        )

        return builder.finalize()

    @staticmethod
    def fit_transform(
        texts: ValidTexts, params: VectorizerParams, batch_size: int = 50_000, auto_batch_threshold: int = 100_000
    ) -> tuple[TfidfVectorizer, csr_matrix]:
        """Fit a new TF-IDF vectorizer and transform the texts in one optimized step.

        This is more efficient than calling fit() followed by transform() because
        it only computes n-grams once instead of twice.

        For large datasets (>= auto_batch_threshold), falls back to batched fit() + transform()
        to avoid OOM, at the cost of computing n-grams twice.

        Args:
            texts: Training texts to fit the vectorizer and transform.
            params: Vectorizer parameters.
            batch_size: Number of texts per batch when using batched mode (default: 50,000).
            auto_batch_threshold: Switch to batching if len(texts) >= this (default: 100,000).

        Returns:
            A tuple of (fitted_vectorizer, transformed_matrix).

        """
        num_texts = len(texts)

        # Small datasets: use optimized fit_transform (single n-gram computation)
        if num_texts < auto_batch_threshold:
            validated_texts = _validate_texts(texts)
            rust_vectorizer, transform_result = RustTfidfVectorizer.fit_transform(validated_texts, params.as_rust())

            shape: tuple[int, int]
            data: NDArray[np.float32]
            indices: NDArray[np.uintp]
            indptr: NDArray[np.uintp]

            shape, data, indices, indptr = transform_result  # type: ignore[assignment]
            transformed_matrix = csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

            vectorizer = TfidfVectorizer(params, rust_vectorizer)
            return vectorizer, transformed_matrix

        logger.warning(
            "Dataset size (%s)) >= threshold (%s). fit_transform() will use batched fit() + transform() (computes n-grams twice). "
            "This avoids OOM but is slower than optimized fit_transform().",
            f"{num_texts:,}",
            f"{auto_batch_threshold:,}",
        )

        vectorizer = TfidfVectorizer.fit(texts, params, batch_size, auto_batch_threshold)
        transformed_matrix = vectorizer.transform(texts, batch_size, auto_batch_threshold)

        return vectorizer, transformed_matrix

    def transform(self, texts: ValidTexts, batch_size: int = 50_000, auto_batch_threshold: int = 100_000) -> csr_matrix:
        """Transform new texts into TF-IDF feature vectors.

        Automatically batches validation and transformation for large datasets to avoid OOM.

        Args:
            texts: Texts to transform.
            batch_size: Number of texts per batch when using batched mode (default: 50,000).
            auto_batch_threshold: Switch to batching if len(texts) >= this (default: 100,000).

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        """
        num_texts = len(texts)

        # Small datasets: validate all, transform at once
        if num_texts < auto_batch_threshold:
            validated_texts = _validate_texts(texts)

            shape: tuple[int, int]
            data: NDArray[np.float32]
            indices: NDArray[np.uintp]
            indptr: NDArray[np.uintp]

            shape, data, indices, indptr = self._vectorizer.transform(validated_texts)  # type: ignore[assignment]
            return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

        # Large datasets: validate and transform in batches

        logger.info("Transforming %s texts in batches of %s", f"{num_texts:,}", f"{batch_size:,}")

        batch_matrices = []
        num_batches = (num_texts + batch_size - 1) // batch_size

        for i in range(0, num_texts, batch_size):
            batch_idx = i // batch_size
            batch_slice = texts[i : i + batch_size]
            batch = _validate_texts(batch_slice)

            logger.info(
                "Transforming batch %s/%s (%s texts)", f"{batch_idx + 1}", f"{num_batches:,}", f"{len(batch):,}"
            )

            # Transform batch
            shape, data, indices, indptr = self._vectorizer.transform(batch)  # type: ignore[assignment]
            batch_matrix = csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)
            batch_matrices.append(batch_matrix)

        logger.info("Stacking %s batch matrices", f"{len(batch_matrices):,}")
        # Stack all batch matrices vertically
        stacked = vstack(batch_matrices, format="csr", dtype=np.float32)
        return csr_matrix(stacked)  # Ensure csr_matrix type

    def vectorize_from_tokens(self, token_sequences: list[list[int]]) -> csr_matrix:
        """Transform pre-tokenized sequences to TF-IDF matrix.

        Used after token-level chunking to vectorize chunks directly
        without re-tokenization.

        Args:
            token_sequences: Pre-tokenized (and optionally chunked) sequences

        Returns:
            Sparse TF-IDF matrix

        Examples:
            >>> # Chunk and vectorize workflow
            >>> tokens = vectorizer.tokenize_batch(texts)
            >>> chunker = TokenChunker()
            >>> chunked = chunker.chunk_batch(tokens)
            >>> # Flatten chunks
            >>> flat_chunks = [chunk for chunks in chunked for chunk in chunks]
            >>> X = vectorizer.vectorize_from_tokens(flat_chunks)

        """
        shape: tuple[int, int]
        data: NDArray[np.float32]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = self._vectorizer.vectorize_from_tokens(token_sequences)  # type: ignore[assignment]
        return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

    @property
    def params(self) -> VectorizerParams:
        """Return the vectorizer parameters."""
        return self._parameters

    @property
    def num_features(self) -> int:
        """Return the number of features (vocabulary size) of the fitted vectorizer."""
        return self._vectorizer.num_features

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary of the fitted vectorizer as a mapping of terms to indices."""
        return self._vectorizer.vocabulary

    def __getstate__(self) -> dict[str, bytes]:
        """Get state for pickling.

        Returns:
            Dictionary containing the serialized vectorizer.

        """
        return {"vectorizer_bytes": bytes(self._vectorizer.to_bytes())}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        """Set state from unpickling.

        Args:
            state: Dictionary containing the serialized vectorizer.

        """
        rust_vectorizer_bytes: bytes = state["vectorizer_bytes"]
        self._vectorizer = RustTfidfVectorizer.from_bytes(rust_vectorizer_bytes)
        self._parameters = VectorizerParams(
            min_df=self._vectorizer.params.min_df,
            max_df=self._vectorizer.params.max_df,
            sublinear_tf=self._vectorizer.params.sublinear_tf,
        )

    def save(self, path: str | Path) -> None:
        """Save raw rkyv bytes for direct Rust consumption.

        This method saves the vectorizer as raw rkyv bytes without JSON wrapping,
        which can be loaded directly in Rust using TfidfVectorizer::from_bytes().

        Args:
            path: File path to save the raw rkyv bytes.

        Raises:
            ValueError: If the file extension is not .json or .rkyv.

        """
        path = Path(path)

        try:
            self._vectorizer.save(str(path))
        except Exception as e:
            msg = f"Failed to save vectorizer: {e}"
            raise ValueError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> TfidfVectorizer:
        """Load a fitted vectorizer from raw rkyv bytes.

        Args:
            path: File path to load the raw rkyv bytes from.

        Returns:
            A loaded TfidfVectorizer instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .json or .rkyv.

        """
        path = Path(path)

        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        try:
            rust_vectorizer = RustTfidfVectorizer.load(str(path))
        except Exception as e:
            msg = f"Failed to load vectorizer: {e}"
            raise ValueError(msg) from e
        params = VectorizerParams(
            min_df=rust_vectorizer.params.min_df,
            max_df=rust_vectorizer.params.max_df,
            sublinear_tf=rust_vectorizer.params.sublinear_tf,
        )
        return cls(params, rust_vectorizer)

    def __repr__(self) -> str:
        return (
            f"TfidfVectorizer(ngram_range={self._parameters.ngram_range}, "
            f"min_df={self._parameters.min_df}, max_df={self._parameters.max_df}, "
            f"num_features={self.num_features})"
        )

    def __str__(self) -> str:
        return self._vectorizer.__str__()


class TfidfVectorizerBuilder:
    """Builder for incremental TF-IDF vectorizer training.

    Supports sklearn-style partial_fit() for large datasets that don't fit in memory.
    Accumulates document frequencies across multiple batches, then builds final vectorizer.

    Examples
    --------
    >>> from is_it_slop_preprocessing import TfidfVectorizerBuilder, VectorizerParams
    >>> params = VectorizerParams(min_df=10, max_df=0.9)
    >>> builder = TfidfVectorizerBuilder(params)
    >>>
    >>> # Process data in batches
    >>> for batch in batches:
    ...     builder.partial_fit(batch)
    >>>
    >>> # Finalize: apply min_df/max_df filtering and calculate IDF
    >>> vectorizer = builder.finalize()

    """

    __slots__ = ("_builder", "_parameters")

    def __init__(self, params: VectorizerParams) -> None:
        """Create a new builder with the given parameters.

        Args:
            params: Vectorizer configuration (ngram_range, min_df, max_df, sublinear_tf)

        """
        self._parameters = params
        self._builder = RustTfidfVectorizerBuilder(params.as_rust())

    def partial_fit(self, texts: ValidTexts) -> None:
        """Process a batch of texts, updating document frequencies.

        Can be called multiple times with different batches. Document frequencies
        accumulate across all calls.

        Args:
            texts: Batch of documents to process

        """
        validated_texts = _validate_texts(texts)
        self._builder.partial_fit(validated_texts)

    def finalize(self) -> TfidfVectorizer:
        """Finalize the vectorizer: apply min_df/max_df filtering and calculate IDF weights.

        Returns a fitted TfidfVectorizer ready for transform() calls.
        After calling this, the builder cannot be used again.

        Returns:
            Fitted TfidfVectorizer

        Raises:
            RuntimeError: If no documents have been processed (call partial_fit at least once)

        """  # noqa: DOC502
        rust_vectorizer = self._builder.finalize()
        return TfidfVectorizer(self._parameters, rust_vectorizer)

    @property
    def total_docs(self) -> int:
        """Get current number of documents processed."""
        return self._builder.total_docs

    @property
    def raw_vocab_size(self) -> int:
        """Get current vocabulary size (before filtering)."""
        return self._builder.raw_vocab_size

    def __repr__(self) -> str:
        return self._builder.__repr__()


def tokenize(text: ValidTexts, batch_size: int = 50_000, auto_batch_threshold: int = 100_000) -> list[list[int]]:
    """Tokenize text into token IDs with automatic batching for large datasets.

    Automatically switches between regular tokenization and batched processing based
    on dataset size. For datasets smaller than the threshold, uses the faster
    single-pass tokenization. For larger datasets, processes data in batches to
    avoid OOM at the PyO3 FFI boundary.

    Validates texts in batches to avoid OOM on large datasets.

    Args:
        text: List of input texts to tokenize.
        batch_size: Number of texts per batch when using batched mode (default: 50,000).
        auto_batch_threshold: Switch to batching if len(texts) >= this (default: 100,000).

    Returns:
        List of lists of token IDs.

    Notes:
        For datasets >= auto_batch_threshold, automatically uses batched processing
        to avoid OOM at PyO3 FFI boundary.

    """
    num_texts = len(text)

    # Small datasets: validate all, tokenize at once
    if num_texts < auto_batch_threshold:
        validated_texts = _validate_texts(text)
        return tokenize_internal(validated_texts)

    # Large datasets: validate and tokenize in batches
    logger.info(
        "Dataset size (%s) >= threshold (%s), using batched tokenization with batch_size=%s",
        f"{num_texts:,}",
        f"{auto_batch_threshold:,}",
        f"{batch_size:,}",
    )

    all_tokens = []
    num_batches = (num_texts + batch_size - 1) // batch_size

    for i in range(0, num_texts, batch_size):
        batch_idx = i // batch_size
        batch_slice = text[i : i + batch_size]
        batch = _validate_texts(batch_slice)
        logger.info(
            "Tokenizing batch %s/%s (%s texts, %s completed)",
            f"{batch_idx + 1}",
            f"{num_batches:,}",
            f"{len(batch):,}",
            f"{len(all_tokens):,}",
        )

        batch_tokens = tokenize_internal(batch)
        all_tokens.extend(batch_tokens)

    logger.info("Tokenization complete: %s documents", f"{len(all_tokens):,}")
    return all_tokens


def reverse_tokenize(tokens: list[int]) -> str:
    """Reverse tokenize token IDs back into text.

    Used for vocabulary inspection. Not called during training/inference.

    Args:
        tokens: List of lists of token IDs to reverse tokenize.

    Returns:
        List of reconstructed texts.

    """
    return reverse_tokenize_internal(tokens)


class TokenChunker:
    """Token-level text chunker with Rust-backed implementation.

    Splits tokenized sequences into overlapping chunks while preserving
    token boundaries. Used for processing long texts in the training pipeline.

    Examples:
        >>> from is_it_slop_preprocessing import TokenChunker
        >>> chunker = TokenChunker(chunk_size=150, overlap=15)
        >>>
        >>> # Chunk a single token sequence
        >>> tokens = [1, 2, 3, ..., 200]  # Long sequence
        >>> chunks = chunker.chunk(tokens)  # Returns [[1..150], [135..200]]
        >>>
        >>> # Chunk multiple sequences in parallel
        >>> token_sequences = [[...], [...], ...]
        >>> chunked = chunker.chunk_batch(token_sequences)

    """

    __slots__ = ("_inner",)

    def __init__(self, chunk_size: int = 150, overlap: int = 15, min_chunk_size: int = 30) -> None:
        self._inner = RustTokenChunker(chunk_size, overlap, min_chunk_size)

    def to_dict(self) -> dict[str, int]:
        return self._inner.to_dict()

    def chunk(self, tokens: list[int]) -> list[list[int]]:
        """Chunk a single token sequence.

        Args:
            tokens: List of token IDs

        Returns:
            List of chunked token sequences

        """
        return self._inner.chunk(tokens)

    def chunk_batch(self, token_sequences: list[list[int]]) -> list[list[list[int]]]:
        """Chunk multiple token sequences in parallel.

        Args:
            token_sequences: List of token ID sequences

        Returns:
            List of lists of chunked token sequences

        """
        return self._inner.chunk_batch(token_sequences)


class CleaningMode(Enum):
    """Enum class for text cleaning modes.

    Attributes:
        TRAINING: Apply all cleaning rules (universal + dataset artifacts).
            Use during model training to remove dataset-specific artifacts.
        INFERENCE: Apply only universal cleaning rules.
            Use at inference time to avoid removing legitimate text patterns.

    """

    TRAINING = RustCleaningMode.Training
    INFERENCE = RustCleaningMode.Inference


class TextCleaner:
    """Fast text cleaning with Rust-backed implementation.

    Provides two cleaning modes:
    - Training: Aggressive cleaning to remove dataset artifacts (citations,
      datelines, academic headers, etc.)
    - Inference: Conservative cleaning of only universal artifacts (encoding
      issues, HTML entities, whitespace)

    Examples
    --------
    >>> from is_it_slop_preprocessing import TextCleaner, CleaningMode
    >>>
    >>> # For training data
    >>> cleaner = TextCleaner(CleaningMode.TRAINING)
    >>> clean_texts = cleaner.clean_batch(train_texts)
    >>>
    >>> # For inference/production
    >>> cleaner = TextCleaner(CleaningMode.INFERENCE)
    >>> clean_text = cleaner.clean("User input text...")

    """

    __slots__ = ("_inner",)

    def __init__(self, mode: RustCleaningMode = CleaningMode.INFERENCE) -> None:
        """Initialize a text cleaner with the specified mode.

        Args:
            mode: Cleaning mode (TRAINING or INFERENCE). Defaults to INFERENCE.

        """
        self._inner = RustTextCleaner(mode.value)

    def clean(self, text: str) -> str:
        """Clean a single text string.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text string.

        Examples:
        --------
        >>> cleaner = TextCleaner(CleaningMode.INFERENCE)
        >>> cleaner.clean("Text with &#39;HTML&#39; entities")
        "Text with 'HTML' entities"

        """
        return self._inner.clean(text)

    def clean_batch(self, texts: ValidTexts) -> list[str]:
        """Clean multiple texts in parallel.

        Automatically parallelizes cleaning across CPU cores using Rust's rayon.

        Args:
            texts: List of input texts or NumPy array of strings to clean.

        Returns:
            List of cleaned text strings.

        Examples:
        --------
        >>> cleaner = TextCleaner(CleaningMode.TRAINING)
        >>> clean_texts = cleaner.clean_batch(train_texts)

        """
        validated_texts = _validate_texts(texts)
        return self._inner.clean_batch(validated_texts)

    def __repr__(self) -> str:
        # Determine mode by checking if it's training or inference
        # (no direct way to get mode from Rust side, but doesn't matter for repr)
        return "TextCleaner(mode=TRAINING)" if self._inner.is_training_mode() else "TextCleaner(mode=INFERENCE)"


def extract_combined_batch(full_texts: ValidTexts, chunk_tokens_batch: list[list[list[int]]]) -> NDArray[np.float32]:
    """Extract combined statistical features for a batch of documents.

    Requires the 'statistical-features' Rust feature to be enabled during build.

    Extracts 9 writing style features that capture patterns orthogonal
    to content-based TF-IDF features:

    Document-Level Features (6):
        1. bigram_repetition_rate - Proportion of repeating word bigrams
        2. punctuation_entropy - Shannon entropy of punctuation distribution
        3. lexical_diversity - Unique words / total words
        4. vocab_richness - sqrt(unique words) / total words
        5. word_repetition_rate - Proportion of repeating words
        6. sentence_length_cv - Coefficient of variation for sentence lengths

    Chunk-Level Features (3):
        7. chunk_avg_word_length - Mean character length per word
        8. chunk_punctuation_entropy - Local punctuation entropy
        9. chunk_word_frequency_entropy - Shannon entropy of word frequencies

    Args:
        full_texts: List of full document texts (NOT tokens)
        chunk_tokens_batch: List of chunked token sequences for each document

    Returns:
        Numpy array of shape (total_chunks, 9) with combined features

    Raises:
        ImportError: If statistical features were not compiled in.
                    Rebuild with: maturin develop --features statistical-features

    Examples:
        >>> from is_it_slop_preprocessing import extract_combined_batch, tokenize
        >>> texts = ["First document with multiple sentences.", "Second document also with text."]
        >>> tokens_batch = tokenize(texts)
        >>> # Assume we have chunks for each document
        >>> chunks_batch = [[tokens_batch[0]], [tokens_batch[1]]]
        >>> features = extract_combined_batch(texts, chunks_batch)
        >>> features.shape  # (2, 9) - one row per chunk
        (2, 9)

    """
    # Try to import statistical features - may not be available if not compiled
    try:
        from ._is_it_slop_preprocessing_rust_bindings import (  # noqa: PLC0415
            rust_extract_combined_batch as rust_extract_combined_batch_internal,
        )

        validated_texts = _validate_texts(full_texts)
        return rust_extract_combined_batch_internal(validated_texts, chunk_tokens_batch)
    except (ImportError, AttributeError) as e:
        msg = (
            "Statistical features are not available. "
            "The 'statistical-features' Rust feature was not enabled during build. "
            "Rebuild with: maturin develop --features statistical-features"
        )
        raise ImportError(msg) from e
