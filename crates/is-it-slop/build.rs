//! Build script for is-it-slop crate.
//!
//! This build script handles downloading and managing model artifacts required for runtime
//! inference.

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use tempfile::tempdir;

/// Model version to use
/// Update this when releasing new model versions
/// Crate version doesn't need to change for patch updates
const MODEL_VERSION: &str = "4.0.0";

/// The shipped weights are the half-precision export: half the binary for a measured
/// validation-AUC cost in the 1e-6 range. `model.safetensors` (f32) stays in the release
/// tarball as the canonical reference but is not embedded.
const MODEL_WEIGHTS_FILENAME: &str = "model.f16.safetensors";
const VOCABULARY_FILENAME: &str = "token_vocabulary.safetensors";
const ENCODER_CONFIG_FILENAME: &str = "encoder_config.json";
const MODEL_CONFIG_FILENAME: &str = "model_config.json";
const THRESHOLD_FILENAME: &str = "classification_threshold.txt";
const CALIBRATION_TEMPERATURE_FILENAME: &str = "calibration_temperature.txt";

/// Required artifact filenames (relative to version directory)
const REQUIRED_ARTIFACTS: &[&str] = &[
    MODEL_WEIGHTS_FILENAME,
    VOCABULARY_FILENAME,
    ENCODER_CONFIG_FILENAME,
    MODEL_CONFIG_FILENAME,
    THRESHOLD_FILENAME,
    CALIBRATION_TEMPERATURE_FILENAME,
];

/// Base URL for downloading model artifacts from GitHub releases
fn default_artifact_url(version: &str) -> String {
    format!(
        "{}/releases/download/model-v{}/model-v{}.tar.gz",
        env!("CARGO_PKG_REPOSITORY"),
        version,
        version
    )
}

/// Get the user cache directory for model artifacts
/// Returns ~/.cache/is-it-slop/models/ on Linux, ~/Library/Caches/is-it-slop/models/ on macOS, etc.
fn get_cache_dir() -> Option<PathBuf> {
    // Try directories crate location
    directories::ProjectDirs::from("", "", "is-it-slop").map(|dirs| dirs.cache_dir().join("models"))
}

/// Check if all required artifacts exist in a directory
fn artifacts_exist(dir: &Path) -> bool {
    REQUIRED_ARTIFACTS.iter().all(|f| dir.join(f).exists())
}

/// Copy artifacts from source to destination directory
fn copy_artifacts(src_dir: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dest_dir)?;
    for filename in REQUIRED_ARTIFACTS {
        let src = src_dir.join(filename);
        let dest = dest_dir.join(filename);
        if src.exists() {
            fs::copy(&src, &dest)?;
        }
    }
    Ok(())
}

/// Create dummy/stub artifacts for docs-only builds
fn create_stub_artifacts(target_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(target_dir)?;

    // Create empty/minimal stub files
    // These will compile but panic at runtime if used
    fs::write(target_dir.join(MODEL_WEIGHTS_FILENAME), b"")?;
    fs::write(target_dir.join(VOCABULARY_FILENAME), b"")?;
    fs::write(
        target_dir.join(ENCODER_CONFIG_FILENAME),
        r#"{"preprocessing_version": "0.0.0", "tokenizer": "o200k_base", "chunk_size": 192, "max_chunks_seen_in_training": 8, "min_fill": 0.1, "overlap": 0, "vocab_size": 0, "oov_id": 0, "pad_id": 1}"#,
    )?;
    fs::write(target_dir.join(MODEL_CONFIG_FILENAME), b"{}")?;
    fs::write(target_dir.join(THRESHOLD_FILENAME), "0.5")?;
    fs::write(target_dir.join(CALIBRATION_TEMPERATURE_FILENAME), "1.0")?;

    Ok(())
}

/// Download artifacts to target directory
fn download_artifacts(
    target_dir: &Path,
    model_version: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let url =
        env::var("MODEL_ARTIFACT_URL").unwrap_or_else(|_| default_artifact_url(model_version));

    println!("cargo:warning=Downloading model artifacts from {url}");

    let temp_dir = tempdir()?;

    // Download the tarball
    let mut tar_gz_data = Vec::new();
    ureq::get(&url)
        .call()?
        .into_body()
        .into_reader()
        .read_to_end(&mut tar_gz_data)?;

    // Decompress and extract
    let tar = flate2::read::GzDecoder::new(&tar_gz_data[..]);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(temp_dir.path())?;

    fs::create_dir_all(target_dir)?;

    // Try with version subdir first, then flat
    let extracted_version_dir = temp_dir.path().join(model_version);
    let src = if extracted_version_dir.exists() {
        extracted_version_dir
    } else {
        temp_dir.path().to_path_buf()
    };

    for entry in fs::read_dir(&src)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if REQUIRED_ARTIFACTS.contains(&file_name_str.as_ref()) {
            fs::copy(entry.path(), target_dir.join(&file_name))?;
        }
    }

    println!("cargo:warning=Model artifacts downloaded successfully");
    Ok(())
}

/// Ensure artifacts exist in `OUT_DIR`.
/// Priority:
/// 1. Already in `OUT_DIR` → use as-is
/// 2. Copy from local source/override dir → copy to `OUT_DIR`
/// 3. Copy from user cache dir → copy to `OUT_DIR`
/// 4. Download from GitHub → save to cache AND `OUT_DIR`
fn ensure_artifacts_in_out_dir(
    out_artifacts_dir: &Path,
    source_artifacts_dir: &Path,
    model_version: &str,
) {
    // 1. Already exist in OUT_DIR? Done.
    if artifacts_exist(out_artifacts_dir) {
        return;
    }

    // 2. Exist in source/local dir (dev environment)? Copy to OUT_DIR
    let source_version_dir = source_artifacts_dir.join(model_version);
    if artifacts_exist(&source_version_dir) {
        copy_artifacts(&source_version_dir, out_artifacts_dir)
            .expect("Failed to copy artifacts from source to OUT_DIR");
        return;
    }

    // 3. Exist in user cache? Copy to OUT_DIR
    if let Some(cache_dir) = get_cache_dir() {
        let cache_version_dir = cache_dir.join(model_version);
        if artifacts_exist(&cache_version_dir) {
            println!(
                "cargo:warning=Using cached model artifacts from {}",
                cache_version_dir.display()
            );
            copy_artifacts(&cache_version_dir, out_artifacts_dir)
                .expect("Failed to copy artifacts from cache to OUT_DIR");
            return;
        }
    }

    // 4. Download from GitHub
    println!("cargo:warning=Model artifacts not found locally, downloading...");

    // First, try to download to cache (so future builds don't re-download)
    if let Some(cache_dir) = get_cache_dir() {
        let cache_version_dir = cache_dir.join(model_version);
        if let Err(e) = download_artifacts(&cache_version_dir, model_version) {
            println!("cargo:warning=Failed to download to cache: {e}");
            println!("cargo:warning=Falling back to direct download to OUT_DIR...");
            // Fall through to direct download below
        } else {
            // Successfully downloaded to cache, now copy to OUT_DIR
            copy_artifacts(&cache_version_dir, out_artifacts_dir)
                .expect("Failed to copy artifacts from cache to OUT_DIR");
            println!(
                "cargo:warning=Model artifacts cached at {}",
                cache_version_dir.display()
            );
            return;
        }
    }

    // Fallback: download directly to OUT_DIR (no caching)
    if let Err(e) = download_artifacts(out_artifacts_dir, model_version) {
        let default_url = default_artifact_url(model_version);
        eprintln!("cargo:warning=Failed to download model artifacts: {e}");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=To build this crate, you need model artifacts:");
        eprintln!("cargo:warning=  curl -LO {default_url}");
        panic!("Model artifacts required but not found");
    }
}

/// Read a single float artifact (threshold, temperature) with a fallback default.
fn read_float_artifact(path: &Path, name: &str, default: f32) -> f32 {
    fs::read_to_string(path).map_or_else(
        |_| {
            println!("cargo:warning={name} file not found, using default {default}");
            default
        },
        |contents| {
            contents.trim().parse::<f32>().unwrap_or_else(|_| {
                println!("cargo:warning=Could not parse {name}, using default {default}");
                default
            })
        },
    )
}

fn main() {
    // Allow override for testing new model versions
    let model_version = env::var("MODEL_VERSION").unwrap_or_else(|_| MODEL_VERSION.to_string());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_artifacts_dir = out_dir.join("model_artifacts").join(&model_version);

    // Source artifacts directory (local dev or env override)
    let source_artifacts_dir = env::var("MODEL_ARTIFACTS_DIR").map_or_else(
        |_| {
            // Model artifacts are at workspace root, not in crate directory
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent() // crates/
                .and_then(|p| p.parent()) // workspace root
                .expect("Unable to find workspace root")
                .join("model_artifacts")
        },
        PathBuf::from,
    );

    // Check if the model doesn't need to be downloaded (e.g docs-only build)
    let is_docs_rs = env::var("DOCS_RS").is_ok();
    let is_stub_model =
        env::var("IS_IT_SLOP_STUB_MODEL").is_ok() || env::var("CARGO_FEATURE_STUB_MODEL").is_ok();
    if is_docs_rs || is_stub_model {
        println!("cargo:warning=Building with stub model artifacts (docs-only build)");
        println!(
            "cargo:warning=Runtime inference will panic! This build is for compilation/docs only."
        );
        create_stub_artifacts(&out_artifacts_dir).expect("Failed to create stub artifacts");
    } else {
        // Ensure artifacts exist in OUT_DIR
        ensure_artifacts_in_out_dir(&out_artifacts_dir, &source_artifacts_dir, &model_version);
    }
    // Expose env vars for include_bytes! macros
    println!("cargo:rustc-env=MODEL_VERSION={model_version}");
    println!(
        "cargo:rustc-env=MODEL_ARTIFACTS_DIR={}",
        out_artifacts_dir.display()
    );
    println!("cargo:rustc-env=MODEL_WEIGHTS_FILENAME={MODEL_WEIGHTS_FILENAME}");
    println!("cargo:rustc-env=VOCABULARY_FILENAME={VOCABULARY_FILENAME}");
    println!("cargo:rustc-env=ENCODER_CONFIG_FILENAME={ENCODER_CONFIG_FILENAME}");
    println!("cargo:rustc-env=MODEL_CONFIG_FILENAME={MODEL_CONFIG_FILENAME}");

    // Bake the decision contract into threshold.rs
    let threshold_val = read_float_artifact(
        &out_artifacts_dir.join(THRESHOLD_FILENAME),
        "Threshold",
        0.5,
    );
    let temperature_val = read_float_artifact(
        &out_artifacts_dir.join(CALIBRATION_TEMPERATURE_FILENAME),
        "Calibration temperature",
        1.0,
    );

    let threshold_rs = format!(
        "// This file is auto-generated by build.rs

/// Default classification threshold between 0.0 and 1.0.
///
/// If P(AI) >= threshold, the text is classified as AI-generated.
pub const CLASSIFICATION_THRESHOLD: f32 = {threshold_val};

/// Temperature the document logit is divided by before the sigmoid.
///
/// Fitted on the validation split; the classification threshold above was fitted on the
/// calibrated probabilities, so both must be applied together.
pub const CALIBRATION_TEMPERATURE: f32 = {temperature_val};\n"
    );
    fs::write(out_dir.join("threshold.rs"), threshold_rs).expect("Failed to write threshold.rs");

    // Only rerun if source artifacts change or env vars change
    let source_version_dir = source_artifacts_dir.join(&model_version);
    for filename in REQUIRED_ARTIFACTS {
        let source_file = source_version_dir.join(filename);
        println!("cargo:rerun-if-changed={}", source_file.display());
    }
    println!("cargo:rerun-if-env-changed=MODEL_ARTIFACTS_DIR");
    println!("cargo:rerun-if-env-changed=MODEL_ARTIFACT_URL");
    println!("cargo:rerun-if-env-changed=MODEL_VERSION");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=IS_IT_SLOP_STUB_MODEL");
}
