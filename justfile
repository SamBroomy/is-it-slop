# Slop Detection CLI Examples

model-pipeline: build-pre-processing-bindings dataset-curation training-pipeline build-bindings build-cli-release

build-pre-processing-bindings:
    uv run --directory python/slop-pre-processing maturin develop --release

dataset-curation:
    uv run jupyter nbconvert --to script notebooks/dataset_curation.ipynb
    uv run python notebooks/dataset_curation.py

training-pipeline:
    uv run jupyter nbconvert --to script notebooks/train.ipynb
    uv run python notebooks/train.py --force-retrain-vectorizer

build-bindings:
    uv run --directory python/is-it-slop maturin develop --release

build-cli-release:
    cargo build --release --bin slop-cli

# Run CLI with different output formats and options
run-cli:
    @echo "=== Running slop-cli examples ==="
    @echo ""
    @echo "1. Default output (AI probability as float 0-1):"
    cargo run -q --release --bin slop-cli -- "This is a test text to check if it's AI generated."
    @echo ""
    @echo "2. Class format (just 0 or 1):"
    cargo run -q --release --bin slop-cli -- "This is a test text." --format class
    @echo ""
    @echo "3. JSON format (detailed output):"
    cargo run -q --release --bin slop-cli -- "This is a test text." --format json
    @echo ""
    @echo "4. Human-readable format:"
    cargo run -q --release --bin slop-cli -- "This is a test text." --format human
    @echo ""
    @echo "5. Verbose mode (shows timing):"
    cargo run -q --release --bin slop-cli -- "This is a test." --verbose
    @echo ""
    @echo "6. Custom labels with JSON:"
    cargo run -q --release --bin slop-cli -- "This is a test." --labels real fake --format json
    @echo ""
    @echo "7. Quiet mode with class output:"
    cargo run -q --release --bin slop-cli -- "This is a test." --quiet --format class
    @echo ""
    @echo "=== All examples complete ==="

# Build the CLI in release mode
build-cli:
    cargo build --release --bin slop-cli

# Run a quick test with custom text
test-cli TEXT:
    cargo run -q --release --bin slop-cli -- "{{TEXT}}"

# Show CLI help
cli-help:
    cargo run -q --release --bin slop-cli -- --help
