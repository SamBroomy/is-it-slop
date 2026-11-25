# is-it-slop-preprocessing

The text cleaner and tokenizer the is-it-slop model is trained with, as a small Python package. The
dataset pipeline (`python/curation`) uses it so training data is cleaned by exactly the code the
Rust training crate and the model expect. It is internal: not published, and not a supported API.

```python
from is_it_slop_preprocessing import clean_for_training, count_tokens

clean_for_training(["  Some   raw text "])  # -> ["Some raw text"]
count_tokens(["Some raw text"])  # -> [3]   (o200k_base)
```

The bindings are the `python` feature of the `is-it-slop-preprocessing` crate
(`crates/is-it-slop-preprocessing/src/python.rs`), built with maturin. Both functions take any iterable
of strings. The inference package `is-it-slop` does not include them.
