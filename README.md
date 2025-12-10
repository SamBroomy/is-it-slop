
<div align=center>
<img src="https://cdn.pixabay.com/photo/2014/04/02/17/04/pink-307853_1280.png" alt-text="pigs love slop", width="350px"/>

[![Crates.io](https://img.shields.io/crates/v/is-it-slop?style=for-the-badge)](https://crates.io/crates/is-it-slop)
[![Crates.io Downloads](https://img.shields.io/crates/d/is-it-slop?style=for-the-badge&label=crates.io%20downloads)](https://crates.io/crates/is-it-slop)
[![Docs.rs](https://img.shields.io/docsrs/is-it-slop?style=for-the-badge)](https://docs.rs/crate/is-it-slop/latest)

[![PyPI](https://img.shields.io/pypi/v/is-it-slop?style=for-the-badge)](https://pypi.org/project/is-it-slop/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/is-it-slop?style=for-the-badge&label=pypi%20downloads)](https://pypi.org/project/is-it-slop/)
[![License](https://img.shields.io/crates/l/is-it-slop?style=for-the-badge)](./LICENSE)

</div>

---

`is-it-slop`
A fast and accurate ***AI text detector*** built with Rust and Python.

A CLI tool and library, classifying whether a given text was written by AI or a human.

# is-it-slop

Fast AI text detection using TF-IDF and ensemble classifiers.

## Features

- **Fast**: Rust-based preprocessing and ONNX inference
- **Minimal**: Just a ~13 MB ML model + 3 MB vectorizer for pre-processing — no heavy transformer models or GPU required
- **Self-Contained**: Single ~35 MB binary with ONNX runtime bundled. No Python, external dependencies, or network access needed at runtime
- **Robust**: [Trained on](./notebooks/train.ipynb) [15+ curated datasets](./notebooks/dataset_curation.ipynb)
- **Accurate**: 96%+ accuracy (F1 0.96, MCC 0.93)
- **Portable**: ONNX model embedded in CLI binary
- **Dual APIs**: Rust library + Python bindings

## Installation

### CLI (Rust)

```bash
cargo install is-it-slop --features cli
```

Model artifacts (14.1 MB zip archive) are downloaded automatically during build from GitHub releases.

### Python Package

```bash
uv add is-it-slop
# or
pip install is-it-slop
```

### Rust Library

```bash
cargo add is-it-slop
```

## Quick Start

### CLI

```bash
is-it-slop "Your text here"
# Output: 0.234 (AI probability)

is-it-slop "Text" --format class
# Output: 0 (Human) or 1 (AI)
```

### Python

```python
from is_it_slop import is_this_slop
result = is_this_slop("Your text here")
print(result.classification)
>>> 'Human'
print(f"AI probability: {result.ai_probability:.2%}")
>>> AI probability: 15.23%
```

### Rust

```rust
use is_it_slop::Predictor;

let predictor = Predictor::new();
let prediction = predictor.predict("Your text here")?;
println!("AI probability: {}", prediction.ai_probability());
```

## Architecture

```
Training (Python):
  Texts -> RustTfidfVectorizer -> TF-IDF -> sklearn models ->  ONNX

Inference (Rust CLI):
  Texts -> TfidfVectorizer (Rust) -> TF-IDF -> ONNX Runtime -> Prediction
```

**Why separate artifacts?**

- Vectorizer: Fast Rust preprocessing.

> Python bindings make it easy to train a model in Python and use it in Rust.

- Model: Portable ONNX format (no Python runtime needed)

## Training

See [`notebooks/dataset_curation.ipynb`](notebooks/dataset_curation.ipynb) for which datasets were used.
See [`notebooks/train.ipynb`](notebooks/train.ipynb) for training pipeline.

Great care was taken to use multiple diverse datasets to avoid overfitting to any single source of human or AI-generated text. Great care was also taken to avoid the underlying model just learning artifacts of specific datasets.

For more information about look in the `notebooks/` directory.

## License

[MIT](./LICENSE)

# Description

# Getting Started

## Dependencies

## Installation

##
