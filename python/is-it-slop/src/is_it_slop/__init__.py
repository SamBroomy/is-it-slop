"""is-it-slop: AI-generated text detection.

Fast Rust-backed inference for detecting AI-generated text (slop detection).

This package provides Python bindings to a Rust-based ML inference engine
that detects AI-generated text with high accuracy and speed.

Key Features
------------
- Fast inference: Rust-backed ONNX runtime
- Pre-trained model: Embedded at compile time
- Simple API: Single function call for predictions
- Batch processing: Efficient multi-text inference

Quick Start
-----------
>>> from is_it_slop import is_this_slop
>>> result = is_this_slop("Your text here")
>>> print(result.classification)
'Human'
>>> print(f"AI probability: {result.ai_probability:.2%}")
AI probability: 15.23%

"""

from ._internal import (
    CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    Prediction,
    __version__,
    is_this_slop,
    is_this_slop_batch,
)

__all__ = [
    "CLASSIFICATION_THRESHOLD",
    "MODEL_VERSION",
    "Prediction",
    "__version__",
    "is_this_slop",
    "is_this_slop_batch",
]
