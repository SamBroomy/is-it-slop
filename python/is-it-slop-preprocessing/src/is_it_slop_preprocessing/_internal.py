"""Internal interface and types for the `is_it_slop_preprocessing` package.

This module provides the main interface to the Rust bindings for text vectorization
using TF-IDF.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


from ._is_it_slop_preprocessing_rust_bindings import (
    RustCleaningMode,
    RustTextCleaner,
    RustTfidfVectorizer,
    RustTokenChunker,
    RustVectorizerParams,
    __version__,
)
from ._is_it_slop_preprocessing_rust_bindings import reverse_tokenize as reverse_tokenize_internal
from ._is_it_slop_preprocessing_rust_bindings import tokenize as tokenize_internal

__all__ = ["CleaningMode", "TextCleaner", "TfidfVectorizer", "TokenChunker", "VectorizerParams", "__version__"]


class ToList(Protocol):
    """Protocol for objects with to_list() or tolist() methods.

    Supports both Polars Series (to_list) and NumPy arrays (tolist).
    """

    def tolist(self) -> list[str]: ...
    def to_list(self) -> list[str]: ...


if TYPE_CHECKING:
    ValidTexts: TypeAlias = list[str] | NDArray[np.str_] | NDArray[np.object_] | ToList


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
        if hasattr(texts, "to_list") and callable(texts.to_list):
            try:
                texts = texts.to_list()
            except Exception as e:
                msg = f"Failed to convert input via .to_list(): {e}"
                raise TypeError(msg) from e
        # Fallback to tolist() (NumPy-like convention)
        elif hasattr(texts, "tolist") and callable(texts.tolist):
            try:
                texts = texts.tolist()
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
    >>> params = VectorizerParams(ngram_range=(3, 5), min_df=10)
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
    def fit(texts: ValidTexts, params: VectorizerParams) -> TfidfVectorizer:
        """Fit a new TF-IDF vectorizer to the provided texts.

        Args:
            texts: Training texts to fit the vectorizer.
            params: Vectorizer parameters.

        Returns:
            A fitted TfidfVectorizer instance.

        """
        validated_texts = _validate_texts(texts)
        rust_vectorizer = RustTfidfVectorizer(validated_texts, params.as_rust())
        return TfidfVectorizer(params, rust_vectorizer)

    @staticmethod
    def fit_transform(texts: ValidTexts, params: VectorizerParams) -> tuple[TfidfVectorizer, csr_matrix]:
        """Fit a new TF-IDF vectorizer and transform the texts in one optimized step.

        This is more efficient than calling fit() followed by transform() because
        it only computes n-grams once instead of twice.

        Args:
            texts: Training texts to fit the vectorizer and transform.
            params: Vectorizer parameters.

        Returns:
            A tuple of (fitted_vectorizer, transformed_matrix).

        """
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

    def transform(self, texts: ValidTexts) -> csr_matrix:
        """Transform new texts into TF-IDF feature vectors.

        Args:
            texts: Texts to transform.

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        """
        validated_texts = _validate_texts(texts)

        shape: tuple[int, int]
        data: NDArray[np.float32]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = self._vectorizer.transform(validated_texts)  # type: ignore[assignment]
        return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

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


def tokenize(text: ValidTexts) -> list[list[int]]:
    """Tokenize text into token IDs.

    Used for vocabulary inspection. Not called during training/inference.

    Args:
        text: List of input texts to tokenize.

    Returns:
        List of lists of token IDs.

    """
    validated_texts = _validate_texts(text)
    return tokenize_internal(validated_texts)


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
