# is-it-slop

Python bindings for the is-it-slop AI text detection library.

## Features

- **Fast inference**: Rust-backed ONNX runtime
- **Pre-trained model**: Embedded at compile time
- **Simple API**: Single function call for predictions
- **Batch processing**: Efficient multi-text inference

## Installation

```bash
uv add is-it-slop
# or
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
    print(f"{text}: {result.classification}")
```

## API Overview

### Functions

**`is_this_slop(text, threshold=None)`**

Predict whether a single text is AI-generated or human-written.

- `text`: Input text string
- `threshold`: Optional classification threshold (0.0-1.0)
- Returns: `Prediction` object

**`is_this_slop_batch(texts, threshold=None)`**

Predict whether multiple texts are AI-generated or human-written.

- `texts`: List of text strings
- `threshold`: Optional classification threshold (0.0-1.0)
- Returns: List of `Prediction` objects

### Prediction Object

Result object with classification and probabilities:

- `classification`: String, either "Human" or "AI"
- `human_probability`: Float (0.0-1.0), probability of human-written
- `ai_probability`: Float (0.0-1.0), probability of AI-generated

### Constants

**`CLASSIFICATION_THRESHOLD`**

Default threshold value used for classification. This threshold is set to optimize for overall F1 score based on validation data.

## CLI Tool

For command-line usage, install the Rust binary:

```bash
cargo install is-it-slop --features cli
is-it-slop "Your text here"
```

## License

MIT
