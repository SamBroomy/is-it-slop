# is-it-slop

Fast AI text detection using TF-IDF and ensemble classifiers.

## Features

- **Fast**: Rust-based preprocessing
- **Accurate**: 96%+ accuracy (F1 0.96, MCC 0.93)
- **Portable**: ONNX model embedded in CLI binary
- **Dual APIs**: Rust library + Python bindings

## Installation

### CLI (Rust)

```bash
cargo install is-it-slop --features cli
```

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

See [`notebooks/train.ipynb`](notebooks/train.ipynb) for training pipeline.

## License

[MIT](./LICENSE)
