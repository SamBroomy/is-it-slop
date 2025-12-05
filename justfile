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
    uv run --directory python/slop-pre-processing maturin develop --release

dataset-curation:
    uv run jupyter nbconvert --to script notebooks/dataset_curation.ipynb
    uv run python notebooks/dataset_curation.py --force-retrain-vectorizer

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
    cargo run --release --features cli --bin is-it-slop -- "{{ TEXT }}"

# Show CLI help
cli-help:
    cargo run --features cli --bin is-it-slop -- --help

# =============================================================================
# Publishing & Release
# =============================================================================

# Full release: model artifacts → crates.io → git tag → triggers PyPI via CI
publish: pre-publish-check release-model _publish-rust-crates _create-and-push-tag
    @echo ""
    @echo "✅ Release complete!"
    @echo ""
    @echo "🐍 Python wheels are being built and published by CI."
    @echo "   Watch progress at: https://github.com/SamBroomy/is-it-slop/actions"

# Internal: publish Rust crates to crates.io
_publish-rust-crates:
    #!/usr/bin/env bash
    set -e
    echo ""
    echo "📦 Publishing is-it-slop-preprocessing to crates.io..."
    cargo publish -p is-it-slop-preprocessing

    echo ""
    echo "⏳ Waiting 15s for crates.io to index..."
    sleep 15

    echo ""
    echo "📦 Publishing is-it-slop to crates.io..."
    cargo publish -p is-it-slop

# Internal: create and push git tag
_create-and-push-tag:
    #!/usr/bin/env bash
    set -e
    CRATE_VERSION=$(grep "^version" Cargo.toml | head -1 | cut -d'"' -f2)
    TAG="v${CRATE_VERSION}"

    echo ""
    echo "🏷️  Creating git tag ${TAG}..."

    # Delete existing tag if it exists (allows re-running after CI fix)
    if git rev-parse "${TAG}" >/dev/null 2>&1; then
        echo "⚠️  Tag ${TAG} already exists locally, deleting..."
        git tag -d "${TAG}"
    fi
    if git ls-remote --tags origin | grep -q "refs/tags/${TAG}$"; then
        echo "⚠️  Tag ${TAG} already exists on remote, deleting..."
        git push origin --delete "${TAG}" || true
    fi

    git tag -a "${TAG}" -m "Release ${TAG}"

    echo "📤 Pushing to origin..."
    git push origin HEAD
    git push origin "${TAG}"

    echo ""
    echo "✅ Tag ${TAG} pushed - CI will build and publish Python wheels"

# Run all pre-publish checks
pre-publish-check:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    cd "${REPO_ROOT}"

    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    CRATE_VERSION=$(grep "^version" Cargo.toml | head -1 | cut -d'"' -f2)
    TAG="v${CRATE_VERSION}"

    echo "=== Pre-publish checks ==="
    echo ""
    echo "  Crate version:  ${CRATE_VERSION}"
    echo "  Model version:  ${MODEL_VERSION}"
    echo "  Git tag:        ${TAG}"
    echo ""

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "❌ You have uncommitted changes!"
        echo "   Please commit or stash them before publishing."
        exit 1
    fi

    echo "📋 Running Rust tests..."

    cargo test --all-targets --workspace --features all-testable

    echo ""
    echo "🔍 Running Rust clippy..."
    cargo clippy --all-targets --workspace --features all-testable -- -D warnings

    echo ""
    echo "🎨 Checking Rust formatting..."
    cargo fmt --check

    echo ""
    echo "📦 Building release..."
    cargo build --release --all-targets --workspace --features all-testable

    just build-python-wheels

    echo ""
    echo "✅ All pre-publish checks passed!"
    echo ""
    echo "⚠️  This will:"
    echo "    1. Release model artifacts to GitHub (if new)"
    echo "    2. Publish Rust crates to crates.io"
    echo "    3. Create/update git tag ${TAG}"
    echo "    4. Trigger CI to build and publish Python wheels to PyPI"
    echo ""
    echo "Press Enter to continue or Ctrl+C to abort..."
    read confirmation

# Dry run: test everything without publishing
publish-dry-run: pre-publish-check
    @echo "=== Dry-run publishing ==="
    @echo ""
    @echo "📦 Testing is-it-slop-preprocessing..."
    cargo publish -p is-it-slop-preprocessing --dry-run
    @echo ""
    @echo "📦 Testing is-it-slop..."
    cargo publish -p is-it-slop --dry-run
    @echo ""
    @echo "✅ Dry run complete - everything looks good!"
    @echo ""
    @echo "Run 'just publish' to publish for real."

# Trigger CI to rebuild Python wheels (useful after CI fixes)
retrigger-release:
    #!/usr/bin/env bash
    set -e
    CRATE_VERSION=$(grep "^version" Cargo.toml | head -1 | cut -d'"' -f2)
    TAG="v${CRATE_VERSION}"

    echo "=== Re-triggering release for ${TAG} ==="
    echo ""
    echo "This will delete and recreate the tag to trigger CI."
    echo "Use this after fixing CI issues."
    echo ""
    echo "Press Enter to continue or Ctrl+C to abort..."
    read confirmation

    # Delete and recreate tag
    if git ls-remote --tags origin | grep -q "refs/tags/${TAG}$"; then
        echo "Deleting remote tag ${TAG}..."
        git push origin --delete "${TAG}"
    fi
    if git rev-parse "${TAG}" >/dev/null 2>&1; then
        echo "Deleting local tag ${TAG}..."
        git tag -d "${TAG}"
    fi

    echo "Creating new tag ${TAG}..."
    git tag -a "${TAG}" -m "Release ${TAG}"
    git push origin "${TAG}"

    echo ""
    echo "✅ Tag ${TAG} recreated - CI will rebuild Python wheels"
    echo "   Watch: https://github.com/SamBroomy/is-it-slop/actions"

# Build all Python wheels
build-python-wheels:
    @echo "Building slop-pre-processing wheel..."
    uv run --directory python/slop-pre-processing maturin build --release
    @echo "Building is-it-slop wheel..."
    uv run --directory python/is-it-slop maturin build --release

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
    echo ""
    echo "Next steps:"
    echo "  1. Create GitHub release:  gh release create model-v${MODEL_VERSION} --title 'Model v${MODEL_VERSION}' --notes 'Model artifacts for is-it-slop'"
    echo "  2. Upload tarball:         gh release upload model-v${MODEL_VERSION} ${TAR_PATH}"

# Check if MODEL_VERSION bump is needed (warns if model files changed but version didn't)
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
        sleep 10

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

# Full model release workflow
release-model: package-artifacts create-model-release test-artifact-download
    @echo ""
    @echo "✅ Model release complete!"
    @echo ""
    @echo "You can now publish the crate:"
    @echo "  just publish-rust-dry-run"
    @echo "  just publish-rust"

# Publish Rust crates to crates.io (dry-run)
publish-rust-dry-run:
    @echo "=== Dry-run publishing Rust crates ==="
    @echo "\n📦 Publishing is-it-slop-preprocessing..."
    cargo publish -p is-it-slop-preprocessing --dry-run
    @echo "\n📦 Publishing is-it-slop (library + binary)..."
    cargo publish -p is-it-slop --dry-run

# Quick install from source (Rust binary only)
install-cli:
    cargo install --path crates/is-it-slop --features cli --force

# Show current version across all packages
show-versions:
    @echo "=== Package Versions ==="
    @echo "\nRust workspace:"
    @grep "^version" Cargo.toml | head -1
    @echo "\nPython packages:"
    @echo "  slop-pre-processing:" && grep "^version" python/slop-pre-processing/pyproject.toml
    @echo "  is-it-slop:" && grep "^version" python/is-it-slop/pyproject.toml

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
