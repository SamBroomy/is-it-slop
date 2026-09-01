//! Smoke test against a real released split. Ignored by default because the parquets are not in
//! git; run with the data directory:
//!
//! ```sh
//! IS_IT_SLOP_DATA=/path/to/repo/data/3.0.0 cargo test -p is-it-slop-training -- --ignored
//! ```

use is_it_slop_training::data::read_split;

#[test]
#[ignore = "needs the released data/3.0.0 parquets"]
fn reads_a_real_split() {
    let dir = std::env::var("IS_IT_SLOP_DATA").expect("set IS_IT_SLOP_DATA to the split directory");
    let split = read_split(
        std::path::Path::new(&dir)
            .join("validation.parquet")
            .as_path(),
    )
    .expect("read validation split");

    assert_eq!(
        split.len(),
        95_009,
        "validation split row count from EXPERIMENTS.md"
    );
    let ai = split.labels.iter().filter(|&&l| l > 0.5).count();
    assert_eq!(ai, 57_506, "AI row count from the Python loader's log");
    assert!(split.texts.iter().take(100).any(|t| !t.trim().is_empty()));
}
