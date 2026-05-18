# Slop Detection CLI Examples

# Default recipe shows available commands
default:
    @just --list

# Setup: install dependencies and pre-commit hooks
[group('setup')]
bootstrap: && install-pre-commit
    uv sync --dev --all-extras --all-groups

# Install pre-commit hooks
[group('git-hooks')]
[group('setup')]
install-pre-commit:
    uv run prek install
    uv run prek run --all-files

# =============================================================================
# Development Pipeline
# =============================================================================

model-pipeline: && build-pre-processing-bindings dataset-curation training-pipeline build-bindings build-cli-release
    uv sync --all-extras --all-groups

build-pre-processing-bindings:
    uv run --directory python/is-it-slop-preprocessing maturin develop --release --uv

dataset-curation: sync-notebooks
    uv run python notebooks/dataset_curation.py --force-retrain-vectorizer --bump-major

training-pipeline: sync-notebooks
    uv run python notebooks/train.py --force-retrain-vectorizer --bump-major

# Generate additional visualizations from trained model
generate-extra-plots: sync-notebooks
    uv run python notebooks/generate_extra_plots.py

build-bindings:
    uv run --directory python/is-it-slop maturin develop --release --uv

build-cli-release:
    cargo build --profile dist --features cli --bin is-it-slop

# Run CLI with different output formats and options
run-cli:
    @echo "=== Running is-it-slop examples ==="
    @echo ""
    @echo "1. Default output (human-readable):"
    cargo run --release --features cli --bin is-it-slop -- "This is a test text to check if it's AI generated."
    @echo ""
    @echo "2. Classification label only:"
    cargo run --release --features cli --bin is-it-slop -- --label "This is a test text."
    @echo ""
    @echo "3. Label with score:"
    cargo run --release --features cli --bin is-it-slop -- --label --score "This is a test text."
    @echo ""
    @echo "4. JSON format:"
    cargo run --release --features cli --bin is-it-slop -- --json "This is a test text."
    @echo ""
    @echo "5. JSONL format (streaming):"
    cargo run --release --features cli --bin is-it-slop -- --jsonl "This is a test text."
    @echo ""
    @echo "6. Bare score for scripting:"
    cargo run --release --features cli --bin is-it-slop -- --score "This is a test text."
    @echo ""
    @echo "=== Examples complete ==="

# Build the CLI in dist mode (optimized for distribution)
build-cli:
    cargo build --profile dist --features cli --bin is-it-slop

# Run a quick test with custom text
test-cli TEXT:
    cargo run --release --features cli --bin is-it-slop -- "{{ TEXT }}" --format json --verbose

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

    ART_DIR="model_artifacts/${MODEL_VERSION}"
    if [ ! -d "${ART_DIR}" ]; then
        echo "No artifacts directory found at ${ART_DIR}"
        exit 1
    fi

    # Only these files are required for the release
    REQUIRED_FILES=(
        "classification_threshold.txt"
        "slop-classifier.onnx"
        "tfidf_vectorizer.rkyv"
        "chunk_classification_threshold.txt"
        "token_chunker_config.json"
        "model_metadata.json"
    )

    # Verify required files exist
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${ART_DIR}/${f}" ]; then
            echo "❌ Missing required file: ${ART_DIR}/${f}"
            exit 1
        fi
    done

    # Clean up macOS artifacts before packaging
    find "${ART_DIR}" -name '._*' -delete 2>/dev/null || true
    find "${ART_DIR}" -name '.DS_Store' -delete 2>/dev/null || true

    # Calculate uncompressed size (only required files)
    echo ""
    echo "📊 Artifact contents:"
    echo "----------------------------------------"
    UNCOMPRESSED_SIZE=0
    for f in "${REQUIRED_FILES[@]}"; do
        FILEPATH="${ART_DIR}/${f}"
        SIZE=$(stat -f%z "$FILEPATH" 2>/dev/null || stat -c%s "$FILEPATH" 2>/dev/null)
        SIZE_HUMAN=$(du -h "$FILEPATH" | cut -f1)
        printf "  %-35s %10s\n" "$f" "$SIZE_HUMAN"
        UNCOMPRESSED_SIZE=$((UNCOMPRESSED_SIZE + SIZE))
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

    # Create a temporary directory with only the required files
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf ${TEMP_DIR}" EXIT

    mkdir -p "${TEMP_DIR}/${MODEL_VERSION}"
    for f in "${REQUIRED_FILES[@]}"; do
        cp "${ART_DIR}/${f}" "${TEMP_DIR}/${MODEL_VERSION}/"
    done

    TAR_PATH="model_artifacts/model-v${MODEL_VERSION}.tar.gz"
    echo "Creating tarball at ${TAR_PATH}..."
    rm -f "${TAR_PATH}"
    # Use COPYFILE_DISABLE to prevent macOS from adding resource forks
    COPYFILE_DISABLE=1 tar -czf "${TAR_PATH}" -C "${TEMP_DIR}" "${MODEL_VERSION}"

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

# Package training data into a tarball (uncompressed - parquet is already compressed)
[group('release')]
package-training-data:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== Packaging training data ==="

    # Read MODEL_VERSION from build.rs (same version as model artifacts)
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    if [ -z "${MODEL_VERSION}" ]; then
        echo "Failed to determine MODEL_VERSION from build.rs"
        exit 1
    fi

    DATA_DIR="data/${MODEL_VERSION}"
    if [ ! -d "${DATA_DIR}" ]; then
        echo "No data directory found at ${DATA_DIR}"
        exit 1
    fi

    # Required files for dataset release
    REQUIRED_FILES=(
        "train.parquet"
        "test.parquet"
        "validation.parquet"
        "dataset_metadata.json"
    )

    # Verify required files exist
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${DATA_DIR}/${f}" ]; then
            echo "❌ Missing required file: ${DATA_DIR}/${f}"
            exit 1
        fi
    done

    # Calculate total size
    echo ""
    echo "📊 Training data contents:"
    echo "----------------------------------------"
    TOTAL_SIZE=0
    for f in "${REQUIRED_FILES[@]}"; do
        FILEPATH="${DATA_DIR}/${f}"
        SIZE=$(stat -f%z "$FILEPATH" 2>/dev/null || stat -c%s "$FILEPATH" 2>/dev/null)
        SIZE_HUMAN=$(du -h "$FILEPATH" | cut -f1)
        printf "  %-35s %10s\n" "$f" "$SIZE_HUMAN"
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    done
    echo "----------------------------------------"
    TOTAL_HUMAN=$(echo "$TOTAL_SIZE" | awk '{
        if ($1 >= 1073741824) printf "%.2f GB", $1/1073741824
        else if ($1 >= 1048576) printf "%.2f MB", $1/1048576
        else if ($1 >= 1024) printf "%.2f KB", $1/1024
        else printf "%d B", $1
    }')
    printf "  %-35s %10s\n" "TOTAL" "$TOTAL_HUMAN"
    echo ""

    # Use uncompressed tar (parquet files are already compressed internally)
    TAR_PATH="data/data-v${MODEL_VERSION}.tar"
    echo "Creating tarball at ${TAR_PATH}..."
    rm -f "${TAR_PATH}" "${TAR_PATH}.gz"

    # Create a temporary directory with just the files we want
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf ${TEMP_DIR}" EXIT

    mkdir -p "${TEMP_DIR}/${MODEL_VERSION}"
    for f in "${REQUIRED_FILES[@]}"; do
        cp "${DATA_DIR}/${f}" "${TEMP_DIR}/${MODEL_VERSION}/"
    done

    # Use COPYFILE_DISABLE to prevent macOS from adding resource forks
    COPYFILE_DISABLE=1 tar -cf "${TAR_PATH}" -C "${TEMP_DIR}" "${MODEL_VERSION}"

    TAR_SIZE=$(du -h "${TAR_PATH}" | cut -f1)
    echo ""
    echo "✅ Created ${TAR_PATH} (${TAR_SIZE})"
    echo ""
    echo "ℹ️  Using uncompressed tar since parquet files are already compressed internally"

# Check if MODEL_VERSION bump is needed (warns if model files changed but version didn't)
[group('release')]
check-model-version:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)
    ART_DIR="model_artifacts/${MODEL_VERSION}"

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
    MODEL_TAR_PATH="model_artifacts/model-v${MODEL_VERSION}.tar.gz"
    DATA_TAR_PATH="data/data-v${MODEL_VERSION}.tar"

    if [ ! -f "${MODEL_TAR_PATH}" ]; then
        echo "Model tarball not found at ${MODEL_TAR_PATH}"
        echo "Run 'just package-artifacts' first"
        exit 1
    fi

    # Check for optional data tarball
    DATA_TARBALL_ARGS=()
    if [ -f "${DATA_TAR_PATH}" ]; then
        DATA_TARBALL_ARGS=("${DATA_TAR_PATH}")
        echo "Found training data tarball at ${DATA_TAR_PATH}"
    else
        echo "No training data tarball found (optional)"
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
            --notes "Model artifacts for is-it-slop. Download automatically during build or manually from this release. Training data included for reproducibility." \
            "${MODEL_TAR_PATH}" "${DATA_TARBALL_ARGS[@]}"
        echo ""
        echo "✅ Model release model-v${MODEL_VERSION} created"
        if [ ${#DATA_TARBALL_ARGS[@]} -gt 0 ]; then
            echo "   - Model artifacts: model-v${MODEL_VERSION}.tar.gz"
            echo "   - Training data: data-v${MODEL_VERSION}.tar"
        fi
    fi

# Test downloading model artifacts from GitHub
[group('release')]
test-artifact-download:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_VERSION=$(grep 'const MODEL_VERSION' crates/is-it-slop/build.rs | head -1 | cut -d'"' -f2)

    echo "=== Testing artifact download ==="

    LOCAL_ART_DIR="model_artifacts/${MODEL_VERSION}"
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
release-model: package-artifacts package-training-data create-model-release test-artifact-download
    @echo ""
    @echo "✅ Model release complete!"
    @echo ""
    @echo "Model artifacts and training data are now available on GitHub."
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

# Test binary packaging locally (without uploading)
[group('dev')]
test-binary-package TARGET:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building for target: {{ TARGET }}"
    cargo build --profile dist --features cli --target {{ TARGET }}

    # Determine binary name (add .exe for Windows)
    # Note: dist profile outputs to target/{TARGET}/dist/ not release/
    if [[ "{{ TARGET }}" == *"windows"* ]]; then
        BINARY="is-it-slop.exe"
        ARCHIVE="is-it-slop-{{ TARGET }}.zip"
        echo "Creating zip archive..."
        cd target/{{ TARGET }}/dist
        zip "../../../${ARCHIVE}" "${BINARY}"
        cd ../../..
    else
        BINARY="is-it-slop"
        ARCHIVE="is-it-slop-{{ TARGET }}.tar.gz"
        echo "Creating tar.gz archive..."
        tar -czf "target/${ARCHIVE}" -C "target/{{ TARGET }}/dist" "${BINARY}"
    fi

    SIZE=$(du -h "target/${ARCHIVE}" | cut -f1)
    echo "✅ Test package created: target/${ARCHIVE} (${SIZE})"

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

# Quick test (just run all tests without detailed output)
[group('dev')]
test: test-rust test-python

# Test only Rust crates (without Python bindings for speed)
[group('dev')]
test-rust:
    @echo "Testing preprocessing crate (unit tests)..."
    cargo test --lib -p is-it-slop-preprocessing --no-default-features --features rkyv,serde,bincode
    @echo ""
    @echo "Testing main crate (unit tests)..."
    cargo test --lib -p is-it-slop --all-features
    @echo ""
    @echo "Testing main crate (integration tests)..."
    cargo test -p is-it-slop --test integration_test

# Test only Python packages
[group('dev')]
test-python: build-pre-processing-bindings build-bindings
    @echo "Testing Python preprocessing package..."
    uv run --directory python/is-it-slop-preprocessing pytest tests/ -v
    @echo ""
    @echo "Testing Python inference package..."
    uv run --directory python/is-it-slop pytest tests/ -v

# Run all checks locally (mirrors CI)
[group('dev')]
check: rust-lint-fix
    cargo fmt --all --check
    cargo clippy --all-targets --workspace --features all-testable -- -D warnings
    cargo test --all-targets --workspace --features all-testable

# Pre-release validation - run all checks before pushing
[group('dev')]
[group('release')]
pre-release:
    @echo "Running pre-release checks..."
    cargo fmt --all --check
    cargo clippy -p is-it-slop-preprocessing --no-default-features --features rkyv,serde,bincode --all-targets -- -D warnings
    cargo clippy -p is-it-slop --no-default-features --features cli --all-targets -- -D warnings
    cargo test -p is-it-slop-preprocessing --lib --no-default-features --features rkyv,serde,bincode
    cargo test -p is-it-slop --lib --all-features
    cargo test -p is-it-slop --doc --no-default-features --features cli
    cargo test -p is-it-slop-preprocessing --doc --no-default-features --features rkyv,serde
    cargo doc --workspace --no-deps --all-features
    cargo package -p is-it-slop-preprocessing --no-verify --allow-dirty
    @echo "✅ All checks passed!"

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
    cargo doc --all-features --workspace --no-deps

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

# =============================================================================
# Notebook Management
# =============================================================================

# Convert notebooks to Python scripts (for version control)
[group('notebooks')]
notebooks-to-scripts:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Converting notebooks to Python scripts..."
    for nb in notebooks/*.ipynb; do
        if [ -f "$nb" ]; then
            echo "  Converting $(basename "$nb")..."
            uv run jupyter nbconvert --to script "$nb" --output-dir notebooks/
        fi
    done
    echo "✅ All notebooks converted"

# Convert Python scripts back to notebooks (restore dev environment)
[group('notebooks')]
scripts-to-notebooks:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Converting Python scripts to notebooks..."
    for py in notebooks/*.py; do
        if [ -f "$py" ] && [ "$(basename "$py")" != "__init__.py" ]; then
            nb="${py%.py}.ipynb"
            if [ ! -f "$nb" ]; then
                echo "  Converting $(basename "$py") -> $(basename "$nb")..."
                uv run jupytext --to notebook "$py" --output "$nb"
            else
                echo "  Skipping $(basename "$py") (notebook exists)"
            fi
        fi
    done
    echo "✅ Scripts converted to notebooks"

# Sync notebooks: convert to scripts and strip outputs for cleaner diffs
[group('notebooks')]
[group('precommit')]
sync-notebooks: notebooks-to-scripts
    #!/usr/bin/env bash
    set -euo pipefail
    # Optionally strip outputs from notebooks to reduce size
    if command -v nbstripout &> /dev/null; then
        echo "Stripping notebook outputs..."
        nbstripout notebooks/*.ipynb 2>/dev/null || true
    fi
    echo "✅ Notebooks synced"

# We can see if an essay is AI-generated or not.
is-the-essay-slop:
    cargo run --release --features cli --bin is-it-slop -- "This article was very thought provoking and caused me to thoroughly evaluate the idea of gender and the role it plays in our society. The article discussed peers using teasing as a way to enforce gender norms. I do not necessarily see this as a problem. God made male and female and made us differently from each other on purpose and for a purpose. God is very intentional with what He makes, and I believe trying to change that would only do more harm. Gender roles and tendencies should not be considered “stereotypes”. Women naturally want to do womanly things because God created us with those womanly desires in our hearts. The same goes for men. God created men in the image of His courage and strength, and He created women in the image of His beauty. He intentionally created women differently than men and we should live our lives with that in mind. It is frustrating to me when I read articles like this and discussion posts from my classmates of so many people trying to conform to the same mundane opinion, so they do not step on people’s toes. I think that is a cowardly and insincere way to live. It is important to use the freedom of speech we have been given in this country, and I personally believe that eliminating gender in our society would be detrimental, as it pulls us farther from God’s original plan for humans. It is perfectly normal for kids to follow gender “stereotypes” because that is how God made us. The reason so many girls want to feel womanly and care for others in a motherly way is not because they feel pressured to fit into social norms. It is because God created and chose them to reflect His beauty and His compassion in that way. In Genesis, God says that it is not good for man to be alone, so He created a helper for man (which is a woman). Many people assume the word “helper” in this context to be condescending and offensive to women. However, the original word in Hebrew is “ezer kenegdo” and that directly translates to “helper equal to”. Additionally, God describes Himself in the Bible using “ezer kenegdo”, or “helper”, and He describes His Holy Spirit as our Helper as well. This shows the importance God places on the role of the helper (women’s roles). God does not view women as less significant than men. He created us with such intentionally and care and He made women in his image of being a helper, and in the image of His beauty. If leaning into that role means I am “following gender stereotypes” then I am happy to be following a stereotype that aligns with the gifts and abilities God gave me as a woman. I do not think men and women are pressured to be more masculine or feminine. I strongly disagree with the idea from the article that encouraging acceptance of diverse gender expressions could improve students’ confidence. Society pushing the lie that there are multiple genders and everyone should be whatever they want to be is demonic and severely harms American youth. I do not want kids to be teased or bullied in school. However, pushing the lie that everyone has their own truth and everyone can do whatever they want and be whoever they want is not biblical whatsoever. The Bible says that our lives are not our own but that our lives and bodies belong to the Lord for His glory. I live my life based on this truth and firmly believe that there would be less gender issues and insecurities in children if they were raised knowing that they do not belong to themselves, but they belong to the Lord. Overall, reading articles such as this one encourage me to one day raise my children knowing that they have a Heavenly Father who loves them and cherishes them deeply and that having their identity firmly rooted in who He is will give them the satisfaction and acceptance that the world can never provide for them. My prayer for the world and specifically for American society and youth is that they would not believe the lies being spread from Satan that make them believe they are better off as another gender than what God made them. I pray that they feel God’s love and acceptance as who He originally created them to be." --format json --verbose
