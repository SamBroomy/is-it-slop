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

/// Download artifacts directly to target directory
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
    let extracted_version_dir = temp_dir.path().join(MODEL_VERSION);
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

/// Create dummy/placeholder artifact files for publish verification.
/// These files allow `include_bytes!` to succeed during `cargo publish --dry-run`
/// but are NOT included in the published crate (excluded via Cargo.toml).
/// Real users will download actual artifacts when building from crates.io.
fn create_dummy_artifacts(target_dir: &Path) {
    fs::create_dir_all(target_dir).expect("Failed to create artifacts directory");

    println!("cargo:warning=Creating dummy artifacts for publish verification...");

    for filename in REQUIRED_ARTIFACTS {
        let file_path = target_dir.join(filename);
        let dummy_content: &[u8] = match *filename {
            "classification_threshold.txt" => b"0.5",
            _ => b"DUMMY_FOR_PUBLISH_VERIFY",
        };
        fs::write(&file_path, dummy_content)
            .unwrap_or_else(|e| panic!("Failed to create dummy {filename}: {e}"));
    }

    println!("cargo:warning=Dummy artifacts created - DO NOT use this build for actual inference!");
}

/// Ensure artifacts exist in `OUT_DIR`.
/// Priority:
/// 1. Already in `OUT_DIR` → skip
/// 2. Copy from local source/override dir → copy to `OUT_DIR`
/// 3. Download from GitHub → directly to `OUT_DIR`
/// 4. Create dummies → in `OUT_DIR` (publish verification only)
fn ensure_artifacts_in_out_dir(
    out_artifacts_dir: &Path,
    source_artifacts_dir: &Path,
    model_version: &str,
    skip_download: bool,
) {
    // 1. Already exist in OUT_DIR?
    if artifacts_exist(out_artifacts_dir) {
        return;
    }

    // 2. Exist in source/local dir? Copy to OUT_DIR
    let source_version_dir = source_artifacts_dir.join(MODEL_VERSION);
    if artifacts_exist(&source_version_dir) {
        copy_artifacts(&source_version_dir, out_artifacts_dir)
            .expect("Failed to copy artifacts to OUT_DIR");
        return;
    }

    // 3. Skip download mode? Create dummies
    if skip_download {
        create_dummy_artifacts(out_artifacts_dir);
        return;
    }

    // 4. Download directly to OUT_DIR (warn - this is notable for users)
    println!("cargo:warning=Model artifacts not found locally, downloading...");
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
    let out_artifacts_dir = out_dir.join("model_artifacts").join(MODEL_VERSION);

    // Source artifacts directory (local dev or env override)
    let source_artifacts_dir = env::var("MODEL_ARTIFACTS_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model_artifacts"),
        PathBuf::from,
    );

    let skip_download = env::var("SKIP_MODEL_DOWNLOAD").is_ok();

    // Ensure artifacts exist in OUT_DIR
    ensure_artifacts_in_out_dir(
        &out_artifacts_dir,
        &source_artifacts_dir,
        &model_version,
        skip_download,
    );

    // Expose env vars
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
    let source_version_dir = source_artifacts_dir.join(MODEL_VERSION);
    for filename in REQUIRED_ARTIFACTS {
        let source_file = source_version_dir.join(filename);
        println!("cargo:rerun-if-changed={}", source_file.display());
    }
    println!("cargo:rerun-if-env-changed=MODEL_ARTIFACTS_DIR");
    println!("cargo:rerun-if-env-changed=MODEL_ARTIFACT_URL");
    println!("cargo:rerun-if-env-changed=SKIP_MODEL_DOWNLOAD");
}
