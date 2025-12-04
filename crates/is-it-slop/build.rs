use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use tempfile::tempdir;

const MODEL_VERSION: &str = "0.1.0";

const CLASSIFIER_MODEL_FILENAME: &str = "slop-classifier.onnx";
const TOKENIZER_FILENAME: &str = "tfidf_vectorizer.bin";
const THRESHOLD_FILENAME: &str = "classification_threshold.txt";

/// Required artifact filenames (relative to version directory)
const REQUIRED_ARTIFACTS: &[&str] = &[
    CLASSIFIER_MODEL_FILENAME,
    TOKENIZER_FILENAME,
    THRESHOLD_FILENAME,
];
// https://github.com/SamBroomy/is-it-slop/releases/download/v0.1.0/v0.1.0.tar.gz
/// Base URL for downloading model artifacts from GitHub releases
/// Set `MODEL_ARTIFACT_URL` env var to override (e.g., for mirrors or different versions)
fn default_artifact_url() -> String {
    format!(
        "{}/releases/download/model-v{}/model-v{}.tar.gz",
        env!("CARGO_PKG_REPOSITORY"),
        MODEL_VERSION,
        MODEL_VERSION
    )
}

fn download_artifacts(artifacts_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = env::var("MODEL_ARTIFACT_URL").unwrap_or_else(|_| default_artifact_url());

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

    let target_version_dir = artifacts_dir.join(MODEL_VERSION);
    fs::create_dir_all(&target_version_dir)?;

    let extracted_version_dir = temp_dir.path().join(MODEL_VERSION);

    if extracted_version_dir.exists() {
        copy_artifacts(&extracted_version_dir, &target_version_dir)?;
    } else {
        // Try without version subdir (flat archive)
        copy_artifacts(temp_dir.path(), &target_version_dir)?;
    }

    println!("cargo:warning=Model artifacts downloaded successfully");
    Ok(())
}
fn copy_artifacts(src_dir: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(src_dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if REQUIRED_ARTIFACTS.contains(&file_name_str.as_ref()) {
            let src_path = entry.path();
            let dest_path = dest_dir.join(&file_name);
            fs::copy(&src_path, &dest_path)?;
            println!(
                "cargo:warning=Extracted {} to {}",
                file_name_str,
                dest_path.display()
            );
        }
    }
    Ok(())
}

/// Create dummy/placeholder artifact files for publish verification.
/// These files allow `include_bytes!` to succeed during `cargo publish --dry-run`
/// but are NOT included in the published crate (excluded via Cargo.toml).
/// Real users will download actual artifacts when building from crates.io.
fn create_dummy_artifacts(artifacts_dir: &Path) {
    let version_dir = artifacts_dir.join(MODEL_VERSION);
    fs::create_dir_all(&version_dir).expect("Failed to create artifacts directory");

    println!("cargo:warning=Creating dummy artifacts for publish verification...");

    // Create minimal dummy files
    for filename in REQUIRED_ARTIFACTS {
        let file_path = version_dir.join(filename);
        if !file_path.exists() {
            let dummy_content: &[u8] = match *filename {
                // Threshold file needs valid content
                THRESHOLD_FILENAME => b"0.5",
                // Other files just need to exist (content doesn't matter for publish verify)
                _ => b"DUMMY_FOR_PUBLISH_VERIFY",
            };
            fs::write(&file_path, dummy_content)
                .unwrap_or_else(|e| panic!("Failed to create dummy {filename}: {e}"));
            println!("cargo:warning=Created dummy: {filename}");
        }
    }
    println!("cargo:warning=Dummy artifacts created - DO NOT use this build for actual inference!");
}

fn ensure_artifacts_exist(artifacts_dir: &Path, skip_download: bool) {
    let version_dir = artifacts_dir.join(MODEL_VERSION);

    let all_exist = REQUIRED_ARTIFACTS.iter().all(|filename| {
        let file_path = version_dir.join(filename);
        file_path.exists()
    });
    if all_exist {
        // println!(
        //     "cargo:warning=Using local model artifacts at {}",
        //     version_dir.display()
        // );
        return;
    }

    if skip_download {
        // Create dummy files for publish verification
        create_dummy_artifacts(artifacts_dir);
        return;
    }

    // Try to download if missing
    println!("cargo:warning=Model artifacts not found locally, attempting download...");
    if let Err(e) = download_artifacts(artifacts_dir) {
        let default_url = default_artifact_url();
        eprintln!("cargo:warning=Failed to download model artifacts: {e}");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=To build this crate, you need model artifacts:");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=Option 1: Download manually");
        eprintln!("cargo:warning=  wget {default_url} -O model_artifacts.tar.gz");
        eprintln!(
            "cargo:warning=  tar -xzf model_artifacts.tar.gz -C {}",
            artifacts_dir.display()
        );
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=Option 2: Build from repository");
        eprintln!("cargo:warning=  git clone {}", env!("CARGO_PKG_REPOSITORY"));
        eprintln!("cargo:warning=  cd is-it-slop && cargo build");
        eprintln!("cargo:warning=");
        panic!("Model artifacts required but not found");
    }
}

fn main() {
    const DEFAULT_THRESHOLD: f32 = 0.5;
    // Expose MODEL_VERSION to the main crate
    println!("cargo:rustc-env=MODEL_VERSION={MODEL_VERSION}");
    // Allow override (useful in CI or when building from other crates)
    let artifacts_dir = env::var("MODEL_ARTIFACTS_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model_artifacts"),
        PathBuf::from,
    );
    // Expose artifacts directory
    println!(
        "cargo:rustc-env=MODEL_ARTIFACTS_DIR={}",
        artifacts_dir.display()
    );
    // Expose file names
    println!("cargo:rustc-env=CLASSIFIER_MODEL_FILENAME={CLASSIFIER_MODEL_FILENAME}");
    println!("cargo:rustc-env=TOKENIZER_FILENAME={TOKENIZER_FILENAME}");
    // Dont need to expose threshold filename, as this is put into threshold.rs
    // println!("cargo:rustc-env=THRESHOLD_FILENAME={THRESHOLD_FILENAME}");

    // Ensure artifacts exist (download if necessary)
    // Skip download if SKIP_MODEL_DOWNLOAD is set (for crates.io publishing dry-run)
    let skip_download = env::var("SKIP_MODEL_DOWNLOAD").is_ok();

    // Ensure artifacts exist (download or create dummies)
    ensure_artifacts_exist(&artifacts_dir, skip_download);

    // Read classification threshold into a constant
    let threshold_path = artifacts_dir
        .join(MODEL_VERSION)
        .join("classification_threshold.txt");
    let out_path = "src/model/threshold.rs";

    let val = fs::read_to_string(&threshold_path).map_or_else(
        |_| {
            println!(
                "cargo:warning=Threshold file not found at {}, using default 0.5",
                threshold_path.display()
            );
            DEFAULT_THRESHOLD
        },
        |contents| {
            let trimmed = contents.trim();
            trimmed.parse::<f32>().unwrap_or_else(|_| {
                println!(
                    "cargo:warning=Could not parse threshold value in {}, using default 0.5",
                    threshold_path.display()
                );
                DEFAULT_THRESHOLD
            })
        },
    );

    let threshold_const = format!("// This file is auto-generated and updated on build by build.rs

/// Default classification threshold between 0.0 and 1.0.
///
/// If P(AI) >= threshold, the text is classified as AI-generated.
/// Lower thresholds are more sensitive (classify more as AI), higher thresholds are more conservative (classify more as Human).
/// This threshold is optimized for overall f1 score based on validation data and is used by default in prediction functions.
pub const CLASSIFICATION_THRESHOLD: f32 = {val};\n"
    );
    let mut file = fs::File::create(out_path).expect("Failed to write threshold.rs");
    file.write_all(threshold_const.as_bytes())
        .expect("Failed to write threshold.rs");
}
