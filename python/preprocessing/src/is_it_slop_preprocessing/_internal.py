"""Internal interface for the `is_it_slop_preprocessing` package.

Thin Python wrappers around the Rust bindings, so IDEs get real type hints and documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._is_it_slop_preprocessing_rust_bindings import __version__
from ._is_it_slop_preprocessing_rust_bindings import clean_for_training as _clean_for_training
from ._is_it_slop_preprocessing_rust_bindings import count_tokens as _count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["__version__", "clean_for_training", "count_tokens"]


def clean_for_training(texts: Iterable[str]) -> list[str]:
    """Clean `texts` with the training-time cleaner, in parallel.

    Args:
        texts: Raw texts (any iterable of strings, e.g. a list or a polars Series).

    Returns:
        list[str]: One cleaned text per input, in order. A text that cleans to nothing is "".

    """
    return _clean_for_training(texts)


def count_tokens(texts: Iterable[str]) -> list[int]:
    """Count `o200k_base` tokens in each text, in parallel.

    Args:
        texts: Texts to count (any iterable of strings).

    Returns:
        list[int]: Token count per input, in order.

    """
    return _count_tokens(texts)
