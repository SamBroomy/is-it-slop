"""is-it-slop-preprocessing: the text cleaner and tokenizer the model is trained with.

Internal and unstable: the dataset pipeline uses it so training data is cleaned by exactly the
code the Rust training crate and the model expect. Not part of the supported public API.

>>> from is_it_slop_preprocessing import clean_for_training, count_tokens
>>> clean_for_training(["  Some   raw text "])
['Some raw text']
>>> count_tokens(["Some raw text"])
[3]
"""

from ._internal import __version__, clean_for_training, count_tokens

__all__ = ["__version__", "clean_for_training", "count_tokens"]
