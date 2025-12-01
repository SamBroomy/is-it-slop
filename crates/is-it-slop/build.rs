use std::{env, fs, io::Write, path::PathBuf};

fn main() {
    const DEFAULT_THRESHOLD: f32 = 0.5;
    // allow override (useful in CI or when building from other crates)
    let artifacts_dir = env::var("MODEL_ARTIFACTS_DIR").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("model_artifacts")
        },
        PathBuf::from,
    );

    let threshold_path = artifacts_dir
        .join(env!("CARGO_PKG_VERSION"))
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

    let threshold_const = format!("pub const CLASSIFICATION_THRESHOLD: f32 = {val};\n");
    let mut file = fs::File::create(out_path).expect("Failed to write threshold.rs");
    file.write_all(threshold_const.as_bytes())
        .expect("Failed to write threshold.rs");
}
