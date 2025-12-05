use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use tempfile::tempdir;

/// Model version to use
/// Update this when releasing new model versions
/// Crate version doesn't need to change for patch updates
const MODEL_VERSION: &str = "1.0.0";

const CLASSIFIER_MODEL_FILENAME: &str = "slop-classifier.onnx";
const TOKENIZER_FILENAME: &str = "tfidf_vectorizer.bin";
const THRESHOLD_FILENAME: &str = "classification_threshold.txt";

/// Required artifact filenames (relative to version directory)
const REQUIRED_ARTIFACTS: &[&str] = &[
    CLASSIFIER_MODEL_FILENAME,
    TOKENIZER_FILENAME,
    THRESHOLD_FILENAME,
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

fn main() {
    const DEFAULT_THRESHOLD: f32 = 0.5;

    // Allow override for testing new model versions
    let model_version = env::var("MODEL_VERSION").unwrap_or_else(|_| MODEL_VERSION.to_string());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_artifacts_dir = out_dir.join("model_artifacts").join(&model_version);

    // Source artifacts directory (local dev or env override)
    let source_artifacts_dir = env::var("MODEL_ARTIFACTS_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model_artifacts"),
        PathBuf::from,
    );

    // Ensure artifacts exist in OUT_DIR
    ensure_artifacts_in_out_dir(&out_artifacts_dir, &source_artifacts_dir, &model_version);

    // Expose env vars for include_bytes! macros
    println!("cargo:rustc-env=MODEL_VERSION={model_version}");
    println!(
        "cargo:rustc-env=MODEL_ARTIFACTS_DIR={}",
        out_artifacts_dir.display()
    );
    println!("cargo:rustc-env=CLASSIFIER_MODEL_FILENAME={CLASSIFIER_MODEL_FILENAME}");
    println!("cargo:rustc-env=TOKENIZER_FILENAME={TOKENIZER_FILENAME}");

    // Read and write threshold to OUT_DIR
    let threshold_path = out_artifacts_dir.join(THRESHOLD_FILENAME);
    let val = fs::read_to_string(&threshold_path).map_or_else(
        |_| {
            println!("cargo:warning=Threshold file not found, using default 0.5");
            DEFAULT_THRESHOLD
        },
        |contents| {
            contents.trim().parse::<f32>().unwrap_or_else(|_| {
                println!("cargo:warning=Could not parse threshold, using default 0.5");
                DEFAULT_THRESHOLD
            })
        },
    );

    let threshold_rs = format!(
        "// This file is auto-generated by build.rs

/// Default classification threshold between 0.0 and 1.0.
///
/// If P(AI) >= threshold, the text is classified as AI-generated.
pub const CLASSIFICATION_THRESHOLD: f32 = {val};\n"
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
}
