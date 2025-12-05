# Slop Detection CLI Examples

# Default recipe shows available commands
default:
    @just --list

bootstrap: && install-pre-commit
    uv sync --dev --all-extras --all-groups

# Install Git hooks using prefligit
[group('git-hooks')]
install-pre-commit:
    #!/usr/bin/env sh
    if ! command -v prefligit &> /dev/null; then
        echo "Installing prefligit..."
        cargo install --locked --git https://github.com/j178/prefligit
    else
        echo "prefligit is already installed"
    fi
    prefligit install
    prefligit run --all-files

# Run the pre-commit hooks
[group('git-hooks')]
run-pre-commit:
    prefligit run --all-files

# Run the pre-push hooks
[group('git-hooks')]
run-pre-push:
    prefligit run --hook-stage pre-push

# =============================================================================
# Development Pipeline
# =============================================================================

model-pipeline: && build-pre-processing-bindings dataset-curation training-pipeline build-bindings build-cli-release
    uv sync -U --dev --all-extras --all-groups

build-pre-processing-bindings:
    uv run --directory python/is-it-slop-preprocessing maturin develop --release --uv

dataset-curation:
    uv run jupyter nbconvert --to script notebooks/dataset_curation.ipynb
    uv run python notebooks/dataset_curation.py --force-retrain-vectorizer

training-pipeline:
    uv run jupyter nbconvert --to script notebooks/train.ipynb
    uv run python notebooks/train.py --force-retrain-vectorizer

build-bindings:
    uv run --directory python/is-it-slop maturin develop --release --uv

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
    cargo run --release --features cli --bin is-it-slop -- "{{ TEXT }}"

# Show CLI help
cli-help:
    cargo run --features cli --bin is-it-slop -- --help

# =============================================================================
# Model Artifacts Management
# =============================================================================

# Package model artifacts into a tarball
[group('release')]
package-artifacts:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== Packaging model artifacts ==="

    # Read MODEL_VERSION from build.rs (not Cargo.toml version!)
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    if [ -z "${MODEL_VERSION}" ]; then
        echo "Failed to determine MODEL_VERSION from build.rs"
        exit 1
    fi

    ART_DIR="crates/is-it-slop/model_artifacts/${MODEL_VERSION}"
    if [ ! -d "${ART_DIR}" ]; then
        echo "No artifacts directory found at ${ART_DIR}"
        exit 1
    fi

    # Clean up macOS artifacts before packaging
    find "${ART_DIR}" -name '._*' -delete 2>/dev/null || true
    find "${ART_DIR}" -name '.DS_Store' -delete 2>/dev/null || true

    # Calculate uncompressed size
    echo ""
    echo "📊 Artifact contents:"
    echo "----------------------------------------"
    UNCOMPRESSED_SIZE=0
    for f in "${ART_DIR}"/*; do
        if [ -f "$f" ]; then
            SIZE=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
            SIZE_HUMAN=$(du -h "$f" | cut -f1)
            FILENAME=$(basename "$f")
            printf "  %-35s %10s\n" "$FILENAME" "$SIZE_HUMAN"
            UNCOMPRESSED_SIZE=$((UNCOMPRESSED_SIZE + SIZE))
        fi
    done
    echo "----------------------------------------"
    UNCOMPRESSED_HUMAN=$(echo "$UNCOMPRESSED_SIZE" | awk '{
        if ($1 >= 1073741824) printf "%.2f GB", $1/1073741824
        else if ($1 >= 1048576) printf "%.2f MB", $1/1048576
        else if ($1 >= 1024) printf "%.2f KB", $1/1024
        else printf "%d B", $1
    }')
    printf "  %-35s %10s\n" "TOTAL (uncompressed)" "$UNCOMPRESSED_HUMAN"
    echo ""

    TAR_PATH="crates/is-it-slop/model_artifacts/model-v${MODEL_VERSION}.tar.gz"
    echo "Creating tarball at ${TAR_PATH}..."
    rm -f "${TAR_PATH}"
    # Use COPYFILE_DISABLE to prevent macOS from adding resource forks
    COPYFILE_DISABLE=1 tar -czf "${TAR_PATH}" -C crates/is-it-slop/model_artifacts "${MODEL_VERSION}"

    # Calculate compressed size and ratio
    COMPRESSED_SIZE=$(stat -f%z "${TAR_PATH}" 2>/dev/null || stat -c%s "${TAR_PATH}" 2>/dev/null)
    COMPRESSED_HUMAN=$(du -h "${TAR_PATH}" | cut -f1)

    if [ "$UNCOMPRESSED_SIZE" -gt 0 ]; then
        RATIO=$(echo "scale=1; (1 - $COMPRESSED_SIZE / $UNCOMPRESSED_SIZE) * 100" | bc)
        COMPRESSION_FACTOR=$(echo "scale=2; $UNCOMPRESSED_SIZE / $COMPRESSED_SIZE" | bc)
    else
        RATIO="0"
        COMPRESSION_FACTOR="1"
    fi

    echo ""
    echo "✅ Created ${TAR_PATH}"
    echo ""
    echo "📦 Compression stats:"
    echo "----------------------------------------"
    printf "  %-25s %10s\n" "Uncompressed size:" "$UNCOMPRESSED_HUMAN"
    printf "  %-25s %10s\n" "Compressed size:" "$COMPRESSED_HUMAN"
    printf "  %-25s %9s%%\n" "Space saved:" "$RATIO"
    printf "  %-25s %9sx\n" "Compression ratio:" "$COMPRESSION_FACTOR"
    echo "----------------------------------------"

# Check if MODEL_VERSION bump is needed (warns if model files changed but version didn't)
[group('release')]
check-model-version:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    ART_DIR="crates/is-it-slop/model_artifacts/${MODEL_VERSION}"

    echo "=== Model Version Safety Check ==="
    echo "Current MODEL_VERSION: ${MODEL_VERSION}"
    echo ""

    # Check if a release already exists for this model version
    if gh release view "model-v${MODEL_VERSION}" &>/dev/null; then
        echo "⚠️  Release model-v${MODEL_VERSION} already exists on GitHub!"
        echo ""

        # Download the existing release to compare
        TEMP_DIR=$(mktemp -d)
        trap "rm -rf ${TEMP_DIR}" EXIT
        echo "Downloading existing release for comparison..."
        curl -sL "https://github.com/SamBroomy/is-it-slop/releases/download/model-v${MODEL_VERSION}/model-v${MODEL_VERSION}.tar.gz" \
            | tar -xz -C "${TEMP_DIR}" 2>/dev/null || {
                echo "❌ Failed to download existing release"
                exit 1
            }

        # Compare checksums
        echo ""
        echo "Comparing local vs released artifacts:"
        echo "----------------------------------------"
        MISMATCH=0
        for f in "${ART_DIR}"/*; do
            if [ -f "$f" ]; then
                FILENAME=$(basename "$f")
                LOCAL_SHA=$(shasum -a 256 "$f" | cut -d' ' -f1)
                REMOTE_FILE="${TEMP_DIR}/${MODEL_VERSION}/${FILENAME}"

                if [ -f "${REMOTE_FILE}" ]; then
                    REMOTE_SHA=$(shasum -a 256 "${REMOTE_FILE}" | cut -d' ' -f1)
                    if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
                        printf "  %-35s ✅ match\n" "$FILENAME"
                    else
                        printf "  %-35s ❌ MISMATCH!\n" "$FILENAME"
                        MISMATCH=1
                    fi
                else
                    printf "  %-35s ⚠️  new file\n" "$FILENAME"
                    MISMATCH=1
                fi
            fi
        done
        echo "----------------------------------------"

        if [ "$MISMATCH" -eq 1 ]; then
            echo ""
            echo "❌ LOCAL ARTIFACTS DIFFER FROM RELEASED VERSION!"
            echo ""
            echo "This is a BREAKING CHANGE. You must:"
            echo "  1. Bump MODEL_VERSION in crates/is-it-slop/build.rs"
            echo "  2. Create a new model_artifacts/{NEW_VERSION}/ directory"
            echo "  3. Run 'just release-model' to create a new release"
            echo ""
            echo "Current MODEL_VERSION: ${MODEL_VERSION}"
            echo "Suggested new version: $(echo ${MODEL_VERSION} | awk -F. '{print $1"."$2+1".0"}')"
            exit 1
        else
            echo ""
            echo "✅ Local artifacts match released version"
        fi
    else
        echo "ℹ️  No existing release for model-v${MODEL_VERSION}"
        echo "   This appears to be a new model version - safe to proceed"
    fi

# Create GitHub release with model artifacts (with safety check)
[group('release')]
create-model-release: check-model-version
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    TAR_PATH="crates/is-it-slop/model_artifacts/model-v${MODEL_VERSION}.tar.gz"

    if [ ! -f "${TAR_PATH}" ]; then
        echo "Tarball not found at ${TAR_PATH}"
        echo "Run 'just package-artifacts' first"
        exit 1
    fi

    echo "=== Creating GitHub model release model-v${MODEL_VERSION} ==="

    # Check if release already exists
    if gh release view "model-v${MODEL_VERSION}" &>/dev/null; then
        echo ""
        echo "⚠️  Release model-v${MODEL_VERSION} already exists!"
        echo "    Artifacts were verified to match in check-model-version"
        echo "    Skipping upload (nothing to update)"
        echo ""
    else
        echo "Creating new release model-v${MODEL_VERSION}..."
        gh release create "model-v${MODEL_VERSION}" \
            --title "Model v${MODEL_VERSION}" \
            --notes "Model artifacts for is-it-slop. Download automatically during build or manually from this release." \
            "${TAR_PATH}"
        echo ""
        echo "✅ Model release model-v${MODEL_VERSION} created"
    fi

# Test downloading model artifacts from GitHub
[group('release')]
test-artifact-download:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)

    echo "=== Testing artifact download ==="

    LOCAL_ART_DIR="crates/is-it-slop/model_artifacts/${MODEL_VERSION}"
    BACKUP_DIR=""

    # Cleanup function - always restore artifacts on exit
    cleanup() {
        if [ -n "${BACKUP_DIR}" ] && [ -d "${BACKUP_DIR}" ]; then
            echo "Restoring local artifacts..."
            rm -rf "${LOCAL_ART_DIR}" 2>/dev/null || true
            mv "${BACKUP_DIR}" "${LOCAL_ART_DIR}"
        fi
    }
    trap cleanup EXIT

    # Move local artifacts aside (if they exist)
    if [ -d "${LOCAL_ART_DIR}" ]; then
        BACKUP_DIR="${LOCAL_ART_DIR}.backup.$$"
        echo "Temporarily moving local artifacts aside..."
        mv "${LOCAL_ART_DIR}" "${BACKUP_DIR}"
    fi

    # Clean and rebuild
    echo "Cleaning build cache..."
    cargo clean -p is-it-slop

    echo "Rebuilding (should trigger download from GitHub)..."
    cargo build -p is-it-slop

    echo ""
    echo "✅ Download test passed! Build succeeded."

# Full model release workflow (run when model artifacts change)
[group('release')]
release-model: package-artifacts create-model-release test-artifact-download
    @echo ""
    @echo "✅ Model release complete!"
    @echo ""
    @echo "Model artifacts are now available on GitHub."
    @echo "Rust/Python releases are handled automatically by CI:"
    @echo "  - Rust: release-plz creates releases on merge to main"
    @echo "  - Python: wheels built when release-plz pushes a tag"

# =============================================================================
# Local Development & Testing
# =============================================================================

# Build all Python wheels locally (for testing)
[group('dev')]
build-python-wheels:
    @echo "Building is-it-slop-preprocessing wheel..."
    uv run --directory python/is-it-slop-preprocessing maturin build --release
    @echo "Building is-it-slop wheel..."
    uv run --directory python/is-it-slop maturin build --release

# Quick install CLI from source
[group('dev')]
install-cli:
    cargo install --path crates/is-it-slop --features cli --force

# Show current version across all packages
[group('dev')]
show-versions:
    @echo "=== Package Versions ==="
    @echo "\nRust workspace:"
    @grep "^version" Cargo.toml | head -1
    @echo "\nModel version:"
    @grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1
    @echo "\nPython packages:"
    @echo "  is-it-slop-preprocessing:" && grep "^version" python/is-it-slop-preprocessing/pyproject.toml
    @echo "  is-it-slop:" && grep "^version" python/is-it-slop/pyproject.toml

# Run all checks locally (mirrors CI)
[group('dev')]
check: rust-lint-fix
    cargo fmt --all --check
    cargo clippy --all-targets --workspace --features all-testable -- -D warnings
    cargo test --all-targets --workspace --features all-testable

# =============================================================================
# CI & Linting
# =============================================================================

# Remove unused dependencies
[group('ci')]
[group('lint')]
[group('precommit')]
cargo-machete:
    cargo machete --with-metadata --fix

# Check the docs
[group('ci')]
[group('lint')]
cargo-docs:
    cargo doc --all-targets --workspace --features all-testable --no-deps

# Cargo audit
[group('ci')]
[group('lint')]
cargo-audit:
    cargo audit --deny unsound --deny yanked

# Check maturin can build Python bindings
[group('lint')]
[group('precommit')]
maturin-check: build-pre-processing-bindings build-bindings
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Checking maturin build..."
    uv sync --dev --quiet
    echo "✅ Maturin build check passed"

# Fix Rust code with clippy and fmt (for precommit)
[group('lint')]
[group('precommit')]
rust-lint-fix:
    cargo clippy --workspace --features all-testable --all-targets --fix --allow-staged --allow-dirty --quiet -- -D warnings
    cargo clippy --workspace --all-targets --no-default-features --fix --allow-staged --allow-dirty --quiet -- -D warnings
