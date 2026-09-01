"""Rust-backed text preprocessing for the is-it-slop classifier.

Provides the text-side of the pipeline: cleaning, tiktoken BPE tokenization,
and token-level chunking. All heavy lifting happens in Rust with automatic
parallelization via rayon.

Quick Start
-----------
>>> from is_it_slop_preprocessing import TextCleaner, CleaningMode, tokenize
>>>
>>> cleaner = TextCleaner(CleaningMode.INFERENCE)
>>> clean_text = cleaner.clean(user_input)
>>> tokens = tokenize([clean_text])

"""

# Import only user-facing wrapper classes and functions
from ._internal import CleaningMode, TextCleaner, TokenChunker, __version__, tokenize

__all__ = ["CleaningMode", "TextCleaner", "TokenChunker", "__version__", "tokenize"]
