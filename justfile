# Slop Detection CLI Examples

# Default recipe shows available commands
default:
    @just --list

# =============================================================================
# Development Pipeline
# =============================================================================

model-pipeline: && build-pre-processing-bindings dataset-curation training-pipeline build-bindings build-cli-release
    uv sync -U --dev --all-extras --all-groups

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
    cargo build --release --features cli --bin is-it-slop

# Run CLI with different output formats and options
run-cli:
    @echo "=== Running slop-cli examples ==="
    @echo ""
    @echo "1. Default output (AI probability as float 0-1):"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test text to check if it's AI generated."
    @echo ""
    @echo "2. Class format (just 0 or 1):"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test text." --format class
    @echo ""
    @echo "3. JSON format (detailed output):"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test text." --format json
    @echo ""
    @echo "4. Human-readable format:"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test text." --format human
    @echo ""
    @echo "5. Verbose mode (shows timing):"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test." --verbose
    @echo ""
    @echo "6. Custom labels with JSON:"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test." --labels real fake --format json
    @echo ""
    @echo "7. Quiet mode with class output:"
    cargo run -q --release --features cli --bin is-it-slop -- "This is a test." --quiet --format class
    @echo ""
    @echo "=== All examples complete ==="

# Build the CLI in release mode
build-cli:
    cargo build --release --features cli --bin is-it-slop

# Run a quick test with custom text
test-cli TEXT:
    cargo run -q --release --features cli --bin is-it-slop -- "{{TEXT}}"

# Show CLI help
cli-help:
    cargo run -q --release --features cli --bin is-it-slop -- --help

# =============================================================================
# Publishing & Release
# =============================================================================

# Extract version from workspace Cargo.toml
CARGO_PKG_VERSION := `grep "^version" Cargo.toml | head -1 | cut -d'"' -f2`

# Verify model artifacts exist and are valid
check-model-artifacts:
    @echo "=== Checking model artifacts for version {{CARGO_PKG_VERSION}} ==="
    @echo "\n📦 Checking for required files..."
    @test -f crates/is-it-slop/model_artifacts/{{CARGO_PKG_VERSION}}/slop-classifier.onnx || (echo "❌ Missing model ONNX file" && exit 1)
    @test -f crates/is-it-slop/model_artifacts/{{CARGO_PKG_VERSION}}/tfidf_vectorizer.bin || (echo "❌ Missing vectorizer bin file" && exit 1)
    @test -f crates/is-it-slop/model_artifacts/{{CARGO_PKG_VERSION}}/classification_threshold.txt || (echo "❌ Missing threshold file" && exit 1)
    @echo "✅ All artifact files present"
    @echo "\n📊 Artifact sizes:"
    @ls -lh crates/is-it-slop/model_artifacts/{{CARGO_PKG_VERSION}}/ | tail -n +2
    @echo "\n🔍 Validating ONNX model..."
    @uv run python -c "import onnx; onnx.checker.check_model('crates/is-it-slop/model_artifacts/{{CARGO_PKG_VERSION}}/slop-classifier.onnx'); print('✅ ONNX model is valid')"
    @echo "\n✅ All model artifacts are valid"

# Run all pre-publish checks (tests, lints, builds)
pre-publish-check: check-model-artifacts
    @echo "=== Running pre-publish checks ==="
    @echo "\n📋 Running Rust tests..."
    cargo test --all-features
    @echo "\n🔍 Running Rust clippy..."
    cargo clippy --all-features -- -D warnings
    @echo "\n🎨 Checking Rust formatting..."
    cargo fmt --check
    @echo "\n📦 Building Rust packages..."
    cargo build --release --all-features
    @echo "\n🐍 Building Python packages..."
    just build-python-wheels
    @echo "\n✅ All pre-publish checks passed!"

# Build all Python wheels
build-python-wheels:
    @echo "Building slop-pre-processing wheel..."
    uv run --directory python/slop-pre-processing maturin build --release
    @echo "Building is-it-slop wheel..."
    uv run --directory python/is-it-slop maturin build --release

# Publish Rust crates to crates.io (dry-run)
publish-rust-dry-run:
    @echo "=== Dry-run publishing Rust crates ==="
    @echo "\n📦 Publishing is-it-slop-preprocessing..."
    cargo publish -p is-it-slop-preprocessing --dry-run
    @echo "\n📦 Publishing is-it-slop (library + binary)..."
    cargo publish -p is-it-slop --dry-run

# Publish Rust crates to crates.io (REAL)
publish-rust:
    @echo "=== Publishing Rust crates to crates.io ==="
    @echo "\n⚠️  This will publish to crates.io (no undo!)"
    @echo "Press Enter to continue or Ctrl+C to abort..."
    @read confirmation
    @echo "\n📦 Publishing is-it-slop-preprocessing..."
    cargo publish -p is-it-slop-preprocessing
    @echo "\nWaiting 30s for crates.io to update..."
    @sleep 30
    @echo "\n📦 Publishing is-it-slop..."
    cargo publish -p is-it-slop

# Publish Python packages to PyPI (using uv)
publish-python:
    @echo "=== Publishing Python packages to PyPI ==="
    @echo "\n⚠️  This will publish to PyPI (no undo!)"
    @echo "Press Enter to continue or Ctrl+C to abort..."
    @read confirmation
    @echo "\n🐍 Publishing slop-pre-processing..."
    uv run --directory python/slop-pre-processing maturin publish
    @echo "\n🐍 Publishing is-it-slop..."
    uv run --directory python/is-it-slop maturin publish

# Publish Python packages to TestPyPI (for testing)
publish-python-test:
    @echo "=== Publishing Python packages to TestPyPI ==="
    @echo "\n🐍 Publishing slop-pre-processing to TestPyPI..."
    uv run --directory python/slop-pre-processing maturin publish --repository testpypi
    @echo "\n🐍 Publishing is-it-slop to TestPyPI..."
    uv run --directory python/is-it-slop maturin publish --repository testpypi

# Full release workflow (all platforms)
release: pre-publish-check
    @echo "\n=== Ready to publish! ==="
    @echo "\nRun these commands to publish:"
    @echo "  just publish-rust        # Publish to crates.io"
    @echo "  just publish-python      # Publish to PyPI"
    @echo "\nOr test first with:"
    @echo "  just publish-rust-dry-run"
    @echo "  just publish-python-test  # TestPyPI"

# Quick install from source (Rust binary only)
install-cli:
    cargo install --path crates/is-it-slop --features cli --force

# Test installation from crates.io (simulates user experience)
test-install-rust:
    @echo "=== Testing Rust installation ==="
    @echo "\nLibrary usage:"
    cargo add is-it-slop --dry-run
    @echo "\nBinary installation:"
    cargo install is-it-slop --dry-run

# Show current version across all packages
show-versions:
    @echo "=== Package Versions ==="
    @echo "\nRust workspace:"
    @grep "^version" Cargo.toml | head -1
    @echo "\nPython packages:"
    @echo "  slop-pre-processing:" && grep "^version" python/slop-pre-processing/pyproject.toml
    @echo "  is-it-slop:" && grep "^version" python/is-it-slop/pyproject.toml
