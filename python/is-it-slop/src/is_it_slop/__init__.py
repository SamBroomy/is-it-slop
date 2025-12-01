"""is-it-slop: AI-generated text detection.

Fast Rust-backed inference for detecting AI-generated text (slop detection).
"""

from ._internal import CLASSIFICATION_THRESHOLD, Prediction, __version__, is_this_slop, is_this_slop_batch

__all__ = ["CLASSIFICATION_THRESHOLD", "Prediction", "__version__", "is_this_slop", "is_this_slop_batch"]
