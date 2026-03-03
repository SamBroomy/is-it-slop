# is-it-slop

Fast and accurate AI text detection using machine learning.

Python bindings for the is-it-slop AI text detection library, powered by Rust and ONNX Runtime.

> **This is the main inference library.** For most users, this is all you need. If you want to access the preprocessing pipeline directly, see [`is-it-slop-preprocessing`](https://pypi.org/project/is-it-slop-preprocessing/).

## Features

- **Fast inference**: Rust-backed ONNX runtime with optimized preprocessing
- **Pre-trained model**: Embedded at compile time, no downloads required
- **Simple API**: Single function call for predictions
- **Batch processing**: Efficient multi-text inference
- **Text chunking**: Handles variable-length documents (50-5000+ tokens)
- **Cross-platform**: Linux, macOS (Apple Silicon), Windows

## Installation

```bash
uv add is-it-slop
```

or using pip:

```bash
pip install is-it-slop
```

## Quick Start

```python
from is_it_slop import is_this_slop

# Predict on single text
result = is_this_slop("Your text here")
print(result.classification)  # "Human" or "AI"
print(f"AI probability: {result.ai_probability:.2%}")

# Use custom threshold
result = is_this_slop("Your text here", threshold=0.7)
print(result.classification)

# Batch processing
from is_it_slop import is_this_slop_batch

texts = ["First text", "Second text", "Third text"]
results = is_this_slop_batch(texts)
for text, result in zip(texts, results):
    print(f"{text}: {result.classification} ({result.ai_probability:.1%})")
```

## API Reference

### Functions

**`is_this_slop(text, threshold=None)`**

Predict whether a single text is AI-generated or human-written.

**Parameters:**

- `text` (str): Input text string
- `threshold` (float, optional): Classification threshold (0.0-1.0). If not provided, uses the default optimized threshold.

**Returns:** `PredictionResult` object

**Example:**

```python
result = is_this_slop("This text was written by a human.")
print(result.classification)  # "Human" or "AI"
print(result.ai_probability)  # 0.0 to 1.0
print(result.human_probability)  # 0.0 to 1.0
```

---

**`is_this_slop_batch(texts, threshold=None)`**

Predict whether multiple texts are AI-generated or human-written.

**Parameters:**

- `texts` (list[str]): List of text strings
- `threshold` (float, optional): Classification threshold (0.0-1.0)

**Returns:** List of `PredictionResult` objects

**Example:**

```python
results = is_this_slop_batch([
    "First document to check",
    "Second document to check",
    "Third document to check"
])

for i, result in enumerate(results):
    print(f"Text {i+1}: {result.classification}")
```

### PredictionResult Object

Result object with classification and probabilities:

**Attributes:**

- `classification` (str): Either `"Human"` or `"AI"`
- `human_probability` (float): Probability of human-written text (0.0-1.0)
- `ai_probability` (float): Probability of AI-generated text (0.0-1.0)

**Note:** `human_probability + ai_probability == 1.0`

**String representation:**

```python
>>> result = is_this_slop("Some text")
>>> print(result)
Human (AI: 12.3%)

>>> repr(result)
'PredictionResult(human=0.877, ai=0.123, class=Human)'
```

### Constants

**`CLASSIFICATION_THRESHOLD`**

The default threshold value used for classification. This threshold is optimized for overall F1 score based on validation data.

```python
from is_it_slop import CLASSIFICATION_THRESHOLD

print(f"Default threshold: {CLASSIFICATION_THRESHOLD}")
```

**`MODEL_VERSION`**

The version of the embedded model.

```python
from is_it_slop import MODEL_VERSION

print(f"Model version: {MODEL_VERSION}")
```

## How It Works

1. **Text Cleaning**: Normalizes HTML entities, encoding artifacts, and whitespace
2. **Tokenization**: Uses tiktoken (o200k_base) BPE encoding
3. **Chunking**: Splits long texts into 150-token overlapping chunks
4. **Vectorization**: TF-IDF with 2-4 token n-grams
5. **Inference**: ONNX Runtime with LogisticRegression model
6. **Aggregation**: Combines chunk predictions using weighted mean

This pipeline ensures consistent preprocessing between training and inference.

## Command-Line Interface

For CLI usage, install the Rust binary:

```bash
cargo install is-it-slop --features cli
```

```bash
# Basic usage
is-it-slop "Your text here"

# JSON output
is-it-slop "Your text here" --format json

# Classification only (0 or 1)
is-it-slop "Your text here" --format class
```

## Platform Support

Pre-built wheels available for:

- **Linux**: x86_64, aarch64 (manylinux_2_28)
- **macOS**: Apple Silicon (ARM64)
- **Windows**: x86_64

## License

MIT

## Links

- [PyPI Package](https://pypi.org/project/is-it-slop/)
- [GitHub Repository](https://github.com/SamBroomy/is-it-slop)
- [Preprocessing Library](https://pypi.org/project/is-it-slop-preprocessing/)
