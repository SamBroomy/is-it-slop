"""The is-it-slop training-time text cleaner and tokenizer (internal, unstable)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _native
from ._native import __version__

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["__version__", "clean_for_training", "count_tokens"]


def clean_for_training(texts: Iterable[str]) -> list[str]:
    """Clean `texts` with the training-time cleaner, in parallel.

    Args:
        texts: Raw texts.

    Returns:
        list[str]: One cleaned text per input, in order. A text that cleans to nothing is "".

    """
    return _native.clean_for_training(list(texts))


def count_tokens(texts: Iterable[str]) -> list[int]:
    """Count `o200k_base` tokens in each text, in parallel.

    Args:
        texts: Texts to count.

    Returns:
        list[int]: Token count per input, in order.

    """
    return _native.count_tokens(list(texts))
