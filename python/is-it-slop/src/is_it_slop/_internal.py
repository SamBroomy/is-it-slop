"""Internal interface for the `is_it_slop` package.

Thin Python wrappers around the Rust bindings, so IDEs get real type hints and
documentation. All inference runs in Rust (a Burn model embedded in the extension
module, no runtime dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ._is_it_slop_rust_bindings import CLASSIFICATION_THRESHOLD as _CLASSIFICATION_THRESHOLD
from ._is_it_slop_rust_bindings import MODEL_VERSION as _MODEL_VERSION
from ._is_it_slop_rust_bindings import PredictionResult as _PredictionResult
from ._is_it_slop_rust_bindings import __version__
from ._is_it_slop_rust_bindings import is_this_slop as _is_this_slop
from ._is_it_slop_rust_bindings import is_this_slop_batch as _is_this_slop_batch
from ._is_it_slop_rust_bindings import is_this_slop_each as _is_this_slop_each

if TYPE_CHECKING:
    from collections.abc import Iterable

CLASSIFICATION_THRESHOLD: Final[float] = _CLASSIFICATION_THRESHOLD
"""Default classification threshold between 0.0 and 1.0.

If P(AI) >= threshold, the text is classified as AI-generated.
Lower thresholds are more sensitive (classify more as AI), higher thresholds are more
conservative (classify more as Human). The default is fitted on calibrated validation
probabilities and is used whenever no threshold is passed.
"""

MODEL_VERSION: Final[str] = _MODEL_VERSION
"""Version of the underlying ML model used for predictions."""

__all__ = [
    "CLASSIFICATION_THRESHOLD",
    "MODEL_VERSION",
    "Classification",
    "Prediction",
    "__version__",
    "is_this_slop",
    "is_this_slop_batch",
    "is_this_slop_each",
]


class Classification(StrEnum):
    """Whether a text was judged human-written or AI-generated.

    A `str` enum: members compare equal to their labels, so
    `result.classification == "AI"` and `== Classification.AI` both work.
    """

    HUMAN = "Human"
    AI = "AI"

    def __str__(self) -> str:
        """Return the label.

        Returns:
            str: "Human" or "AI".

        """
        return self.value


@dataclass(frozen=True, slots=True)
class Prediction:
    """Immutable result of a prediction.

    Attributes:
        human_probability: Probability that the text is human-written (0.0 to 1.0).
        ai_probability: Probability that the text is AI-generated (0.0 to 1.0).
        classification: Human or AI, from `ai_probability >= threshold`.
        confidence: Calibrated margin from the 50% coin-flip point (0.0 to 1.0):
            0.0 means the model is guessing, 1.0 means it is certain either way.
        num_chunks: Number of 192-token chunks the document produced (1 for short texts).
        chunk_dispersion: Standard deviation of per-chunk AI probabilities. High values
            mean chunks disagree, a possible sign of mixed human/AI content. None for
            single-chunk texts.

    Examples:
        >>> result = is_this_slop("Some text")
        >>> result.classification
        <Classification.HUMAN: 'Human'>
        >>> if result.classification is Classification.AI:
        ...     print(f"Detected slop with {result.ai_probability:.1%} probability")

    """

    human_probability: float
    ai_probability: float
    classification: Classification
    confidence: float
    num_chunks: int
    chunk_dispersion: float | None

    def __str__(self) -> str:
        """Return a short summary.

        Returns:
            str: For example `AI (AI: 91.3%)`.

        """
        return f"{self.classification} (AI: {self.ai_probability * 100:.1f}%)"


def _to_prediction(inner: _PredictionResult) -> Prediction:
    return Prediction(
        human_probability=inner.human_probability,
        ai_probability=inner.ai_probability,
        classification=Classification(inner.classification),
        confidence=inner.confidence,
        num_chunks=inner.num_chunks,
        chunk_dispersion=inner.chunk_dispersion,
    )


def is_this_slop(text: str, threshold: float | None = None) -> Prediction:
    """Predict whether text is AI-generated (slop) or human-written.

    Args:
        text: Input text to classify. Must be non-empty.
        threshold: Classification threshold in [0.0, 1.0]. Leave unset: the default,
            `CLASSIFICATION_THRESHOLD`, was chosen by training to maximise overall F1. A custom
            value is an escape hatch for a deliberate precision/recall trade-off.
            If P(AI) >= threshold the text is classified as AI.

    Returns:
        Prediction: Probabilities, classification and confidence details.

    Raises:
        ValueError: If text is empty or whitespace-only, or threshold is not in [0, 1].

    Examples:
        >>> result = is_this_slop("This is some text")
        >>> print(f"{result.classification}: {result.ai_probability:.2%}")
        Human: 15.23%
        >>> is_this_slop("Generated content", threshold=0.7).classification
        <Classification.HUMAN: 'Human'>

    """
    if not text or not text.strip():
        msg = "Input text must be non-empty"
        raise ValueError(msg)
    return _to_prediction(_is_this_slop(text, threshold))


def is_this_slop_batch(texts: Iterable[str], threshold: float | None = None) -> list[Prediction]:
    """Predict many texts at once; all or nothing.

    Faster than calling `is_this_slop` in a loop. If any text is blank the whole
    call raises; use `is_this_slop_each` to get a result per text instead.

    Args:
        texts: Input texts to classify. Must contain at least one text.
        threshold: Classification threshold in [0.0, 1.0], applied to every text.

    Returns:
        list[Prediction]: One prediction per input text, in input order.

    Raises:
        ValueError: If texts is empty, any text is blank, or threshold is not in [0, 1].

    Examples:
        >>> texts = ["First text", "Second text"]
        >>> for text, result in zip(texts, is_this_slop_batch(texts)):
        ...     print(f"{text}: {result.classification}")

    """
    items = list(texts)
    if not items:
        msg = "Input texts must be a non-empty list"
        raise ValueError(msg)
    return [_to_prediction(r) for r in _is_this_slop_batch(items, threshold)]


def is_this_slop_each(texts: Iterable[str], threshold: float | None = None) -> list[Prediction | None]:
    """Predict many texts at once, with a result per text.

    Like `is_this_slop_batch`, but a blank text yields `None` at its position
    instead of raising for the whole batch. An out-of-range threshold still raises ValueError.

    Args:
        texts: Input texts to classify (may be empty, giving an empty list).
        threshold: Classification threshold in [0.0, 1.0], applied to every text.

    Returns:
        list[Prediction | None]: One entry per input text, in input order.

    Examples:
        >>> results = is_this_slop_each(["Some text", ""])
        >>> results[1] is None
        True

    """
    items = list(texts)
    if not items:
        return []
    return [None if r is None else _to_prediction(r) for r in _is_this_slop_each(items, threshold)]
