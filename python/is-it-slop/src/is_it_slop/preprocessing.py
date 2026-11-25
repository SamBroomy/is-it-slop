"""Text preprocessing helpers for the dataset pipeline.

Internal and unstable: these exist so the data pipeline cleans text with exactly the Rust cleaner the
model is trained and evaluated with. They are not part of the supported public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._is_it_slop_rust_bindings import clean_for_training as _clean_for_training
from ._is_it_slop_rust_bindings import count_tokens as _count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["clean_for_training", "count_tokens"]


def clean_for_training(texts: Iterable[str]) -> list[str]:
    """Clean `texts` with the training-time cleaner, in parallel.

    Args:
        texts: Raw texts.

    Returns:
        list[str]: One cleaned text per input, in order. A text that cleans to nothing is "".

    """
    return _clean_for_training(list(texts))


def count_tokens(texts: Iterable[str]) -> list[int]:
    """Count `o200k_base` tokens in each text, in parallel.

    Args:
        texts: Texts to count.

    Returns:
        list[int]: Token count per input, in order.

    """
    return _count_tokens(list(texts))
