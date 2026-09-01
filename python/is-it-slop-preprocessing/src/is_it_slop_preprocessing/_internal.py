"""Internal interface and types for the `is_it_slop_preprocessing` package.

Wraps the Rust bindings for text cleaning, tokenization, and chunking.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


from ._is_it_slop_preprocessing_rust_bindings import RustCleaningMode, RustTextCleaner, RustTokenChunker, __version__
from ._is_it_slop_preprocessing_rust_bindings import tokenize as tokenize_internal

__all__ = ["CleaningMode", "TextCleaner", "TokenChunker", "__version__", "tokenize"]


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
