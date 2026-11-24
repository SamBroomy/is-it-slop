"""is-it-slop: AI-generated text detection.

Fast Rust-backed inference for detecting AI-generated text (slop detection).

This package provides Python bindings to a Rust-based ML inference engine
that detects AI-generated text with high accuracy and speed. Includes both
a Python API and a command-line interface.

Key Features
------------
- Fast inference: a small Burn model run natively in Rust
- Pre-trained model: embedded in the package, no downloads
- Simple API: Single function call for predictions
- Batch processing: Efficient multi-text inference
- Command-line interface: Use via `is-it-slop` command or `uvx is-it-slop`

Quick Start (Python API)
-------------------------
>>> from is_it_slop import is_this_slop
>>> result = is_this_slop("Your text here")
>>> print(result.classification)
Human
>>> print(f"AI probability: {result.ai_probability:.2%}")
AI probability: 15.23%

Quick Start (CLI)
-----------------
After installation, use the `is-it-slop` command:

    $ is-it-slop "Your text here"

      AI  [██████████████░░░░░░]  Human
           67.8%          32.2%

      Confidence    35.6%

    $ is-it-slop "Your text" -o label
    AI

Or run directly without installing:

    $ uvx is-it-slop "Your text here"

Run `is-it-slop --help` for all options.

"""

from ._internal import (
    CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    Classification,
    Prediction,
    __version__,
    is_this_slop,
    is_this_slop_batch,
    is_this_slop_each,
)

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
