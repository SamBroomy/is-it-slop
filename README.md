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

# is-it-slop

***Unsure if the article you just read was AI generated slop?***

`is-it-slop` is a small, fast, and accurate text classifier that detects AI-generated text. Using classic ML - TF-IDF and logistic regression on token n-grams.

**No transformers, no GPU, no Python runtime required. Just a single ~60 MB Rust binary with embedded model artifacts.**

> Inspired by [Magika](https://github.com/google/magika) for serving a small, fast model via ONNX Runtime in Rust.

## Features

- **Fast**: Rust-based multi-threaded preprocessing and batched ONNX Runtime inference
- **Small**: ~11.4 MB of model artifacts, no GPU or transformers needed
- **Portable**: Single ~60 MB binary with embedded model and no Python runtime required
- **Accurate**: 95.6% accuracy on holdout test set (F1 0.958, MCC 0.912)
- **Chunk-aware**: Handles long documents via overlapping 150-token chunks with weighted aggregation
- **Cross-platform**: macOS (ARM64), Linux (x86_64, ARM64), Windows (x86_64, ARM64)
- **Multiple interfaces**: Command-line tool, Python library, Rust library, Android AAR

## Installation

### Command Line Tool

**Quick Install** (shell script — downloads pre-built binary):

```shell
curl -fsSL https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.sh | sh
```

**Windows Quick Install** (PowerShell — downloads pre-built binary):

```powershell
Invoke-RestMethod https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.ps1 | Invoke-Expression
```

> Both install to `~/.local/bin`.

### **Python** (CLI + library)

```shell
pip install is-it-slop
# or with uv:
uv tool install is-it-slop
# or run directly:
uvx is-it-slop "Your text here"
# or add to project:
uv add is-it-slop
```

### **Homebrew** (macOS/Linux)

```shell
brew tap SamBroomy/is-it-slop
brew install is-it-slop
```

### **Rust/Cargo**

Pre-built binary via `cargo-binstall` (recommended for Rust users):

```shell
cargo binstall is-it-slop
```

Build from source (requires [Rust toolchain](https://www.rust-lang.org/tools/install)):

```shell
cargo install is-it-slop --locked --features cli
```

> Model artifacts (~11.4 MB) download automatically during build and are embedded in the binary. No runtime downloads, no Python required.

### Library Install

#### Python Library

```shell
pip install is-it-slop
# or with uv:
uv add is-it-slop
```

#### Rust Library

```shell
cargo add is-it-slop
```

## Usage

```bash
$ is-it-slop "Your text here"

  Human  [███████████████████░]  AI
          95.1%           4.9%

Confidence Metrics
  Model       95.1%
  Sample      87.3%
  Overall     85.1%
```

Other output modes:

```bash
# Classification label only
$ is-it-slop "Your text" --label
Human

# Label with AI probability score
$ is-it-slop "Your text" --label --score
Human (0.0880)

# Bare float for shell scripting
$ is-it-slop "Your text" --score
0.0880

# Exit code mode (0=AI, 1=Human, 2=error)
$ is-it-slop "Your text" --classify

# Full JSON (includes chunk predictions and confidence metrics)
$ is-it-slop "Your text" --json

# JSON lines format (for streaming)
$ is-it-slop "Your text" --jsonl

# Batch from file (auto-detects .json vs line-delimited)
$ is-it-slop -b texts.txt

# Custom classification threshold
$ is-it-slop "Your text" --threshold 0.7

# Verbose: show model internals (threshold, entropy)
$ is-it-slop "Your text" --verbose

# Quiet: bar only, no metrics
$ is-it-slop "Your text" --quiet
```

Run `is-it-slop --help` for all options.

### Python

```python
from is_it_slop import is_this_slop

result = is_this_slop("Your text here")
print(result.classification)     # 'Human' or 'AI'
print(f"AI: {result.ai_probability:.1%}")  # AI: 8.8%
```

### Rust

```rust
use is_it_slop::Predictor;

let predictor = Predictor::new();
let result = predictor.predict("Your text here")?;
println!("AI probability: {:.2}%", result.prediction.ai_probability() * 100.0);
```

### Android

Download `is-it-slop-aarch64-linux-android.aar` from the [latest release](https://github.com/SamBroomy/is-it-slop/releases/latest)
and drop it into `app/libs/`, then add to `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation(files("libs/is-it-slop-aarch64-linux-android.aar"))
}
```

```kotlin
import ai.isitlop.SlopDetector

val result = SlopDetector.predict("Some text")
// {"aiProbability":0.92,"humanProbability":0.08,"classification":"AI","numChunks":1,"chunkAgreement":1.0}

val label = SlopDetector.classify("Some text")
// "Human" or "AI"

val results = SlopDetector.predictBatch("""["text 1","text 2"]""")
```

## Upgrading

If you installed via the shell script or cargo-binstall, upgrade to the latest version with:

```shell
is-it-slop self update
```

Or re-run the install script:

```shell
# Linux / macOS / WSL
curl -fsSL https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.sh | sh
```
```powershell
# Windows PowerShell
Invoke-RestMethod https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.ps1 | Invoke-Expression
```

For package manager installations (pip, uv, Homebrew, Cargo), use your package manager's upgrade command:

```shell
pip install --upgrade is-it-slop       # Python
uv tool install is-it-slop@latest      # uv
brew upgrade is-it-slop                # Homebrew
cargo install is-it-slop --locked --force --features cli  # Cargo
```

## How It Works

**Training (Python):**

```text
Texts → Clean → Tokenize (BPE) → Chunk → TF-IDF → Stacked Ensemble → ONNX
```

**Inference (Rust):**

```text
Text → Clean → Tokenize → Chunk (150 tokens, 15 overlap) → TF-IDF per chunk → ONNX → Aggregate → Result
```

### Why BPE Tokenization?

We use tiktoken's BPE tokenization `o200k_base` to convert text into sequences of 2-4 consecutive tokens. This captures sub-word patterns that character or word n-grams miss, particularly useful for the predictable token sequences that AI models produce.

> The idea here is that LLMs operate on tokens, and token-level n-grams can capture patterns that character or word n-grams might miss, especially for AI-generated text. Humans often have more varied token usage, while AI-generated text may have more predictable token sequences.

### Why Chunking?

Variable-length documents (50-5000 tokens) lose information in fixed-size feature vectors. Splitting into overlapping 150-token chunks ensures consistent feature extraction regardless of document length. Chunk predictions are aggregated via weighted mean.

### Why Separate Artifacts?

- **TF-IDF preprocessing in Rust**: Avoids complex sklearn-to-ONNX conversion and keeps preprocessing during inference fast without Python dependencies.
- **sklearn → ONNX model**: Portable format, no Python at inference
- **Two-stage text cleaning**: Universal (always) + dataset artifacts (training only to remove dataset-specific noise)

This also avoids complex sklearn-to-ONNX preprocessing conversion while keeping inference fast.

> We use try and clean specific artifacts from the training datasets (e.g. "HuggingFace", "arXiv", "Film Reviews") to prevent the model from learning dataset-specific patterns that wouldn't generalize. While I have tried my best to ensure that the model is learning generalizable features of AI-generated text, there may still be some residual dataset-specific artifacts that could be cleaned in future iterations. The two-stage cleaning process allows us to remove universal noise while also targeting specific artifacts from the training data.

## Architecture

```text
crates/
├── is-it-slop-preprocessing/  # Text → TF-IDF pipeline (PyO3 bindings for training)
│   ├── cleaner.rs            # Two-stage text cleaning
│   ├── tokenizer.rs          # tiktoken BPE (o200k_base)
│   ├── chunker.rs            # Token-based chunking
│   ├── ngrams.rs             # Token n-gram extraction
│   └── vectorizer/           # TF-IDF vectorizer with rkyv serialization
└── is-it-slop/               # ONNX inference + CLI + bindings
    ├── bin/                  # CLI binary entrypoint
    ├── cli/                  # Command-line argument parsing
    ├── model/                # Embedded artifacts (build.rs downloads)
    ├── pipeline/             # Prediction, aggregation, error types
    ├── python/               # PyO3 bindings (PyPI package)
    ├── kotlin/               # JNI bindings (Android .so / .aar)
    └── lib.rs                # Predictor, Threshold, public re-exports

python/                       # Two PyO3 packages (inference + preprocessing)
android/                      # Kotlin wrapper for Android JNI
notebooks/                    # Dataset curation + training
```

## Training

### Dataset

Trained on **25+ diverse datasets** (~687K samples across 118K test, 95K validation):

- **Human sources**: News (newswire, ag_news, imdb), essays (ivy panda, ASAP, PERSUADE), quotes, reviews
- **AI sources**: GPT-3.5/4, Claude, Llama 3.1/3.2, Gemini 2, SmolLM2, Qwen 2.5
- **Class balance**: ~48% human, ~52% AI

**Data quality caveat:** Model performance depends on dataset label accuracy. We assume training data labels are correct (human text is genuinely human-written, AI text is genuinely AI-generated), but mislabelled examples may exist.

See [`notebooks/dataset_curation.ipynb`](notebooks/dataset_curation.ipynb) for details.

![Embedding visualization](./plots/embedding_visualization.png)

### Training Pipeline

See [`notebooks/train.ipynb`](notebooks/train.ipynb) for the complete training pipeline.

### Model Architecture

The classifier is a **stacked ensemble** of calibrated linear models trained on token n-gram TF-IDF features:

1. **Base models** (4 classifiers):
   - SGD Classifier (stochastic gradient descent)
   - Logistic Regression
   - Calibrated Linear SVC (with probability calibration)
   - Multinomial Naive Bayes

2. **Meta-learner**: Logistic Regression combines base model predictions via 5-fold stacking

3. **Feature extraction**: Token n-grams (2-4 tokens) → TF-IDF vectors
   - Uses tiktoken's `o200k_base` BPE encoding
   - Captures subword patterns across ~105k features (2-4 grams, min_df=0.07%, 99.9% sparse)

**Why this works:** AI-generated text exhibits predictable token sequence patterns. By combining multiple linear models with different learning characteristics, the ensemble captures these patterns robustly across diverse writing styles.

### Model Artifacts

Exported artifacts (embedded at build time):

- `tfidf_vectorizer.rkyv` - Vectorizer with vocabulary
- `slop-classifier.onnx` - Stacked ensemble model
- `classification_threshold.txt` - Document-level threshold
- `chunk_classification_threshold.txt` - Per-chunk threshold
- `token_chunker_config.json` - Chunking parameters

Not embedded but also available in `model_artifacts/`:

- `model_metadata.json` - Metadata (training datasets, performance metrics)

### `slop-classifier.onnx`

![Training pipeline visualization](./plots/slop-classifier.onnx.svg)

The diagram shows the full ONNX graph: input → 5 parallel classifiers → probability calibration → meta-learner → final prediction.

### **Additional visualizations:**

>See [`plots/`](./plots/) for embedding visualizations, feature distributions, and model analysis.
>

## Development

```bash
# Build
cargo build --release -p is-it-slop --features cli

# Test
just test

# Full CI check (fmt + clippy + tests)
just check

# Training pipeline
just model-pipeline
```

## License

[MIT](./LICENSE)
