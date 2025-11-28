use std::{fs, io::Write};

fn main() {
    const THRESHOLD_PATH: &str = concat!(
        "../../model_artifacts/",
        env!("CARGO_PKG_VERSION"),
        "/classification_threshold.txt"
    );
    const OUT_PATH: &str = "src/model/threshold.rs";
    const DEFAULT_THRESHOLD: f64 = 0.5;
    let val = fs::read_to_string(THRESHOLD_PATH).map_or_else(|_| {
        println!("cargo:warning=Threshold file not found at {THRESHOLD_PATH}, using default 0.5");
        DEFAULT_THRESHOLD
    }, |contents| {
        let trimmed = contents.trim();
        trimmed.parse::<f64>().unwrap_or_else(|_| {
            println!(
                "cargo:warning=Could not parse threshold value in {THRESHOLD_PATH}, using default 0.5"
            );
            DEFAULT_THRESHOLD
        })
    });

    let threshold_const = format!("pub const CLASSIFICATION_THRESHOLD: f64 = {val};\n");
    let mut file = fs::File::create(OUT_PATH).expect("Failed to write threshold.rs");
    file.write_all(threshold_const.as_bytes())
        .expect("Failed to write threshold.rs");
}
