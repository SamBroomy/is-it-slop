//! Full-corpus vocabulary determinism against the frozen reference.
//!
//! Builds the v8192 vocabulary from the real training split and asserts every lookup row equals
//! the committed fixture (`fixtures/vocabulary.safetensors`, built from the same split). Like the
//! encoding-contract test, this freezes the ranking algorithm trained artifacts depend on —
//! counting, tie-breaking, and the no-trim-before-tokenize asymmetry — at full-corpus scale.
//! Ignored by default because the parquets are not in git:
//!
//! ```sh
//! IS_IT_SLOP_DATA=/path/to/repo/data/3.0.0 cargo test -p is-it-slop-training -- --ignored
//! ```

use std::path::Path;

use is_it_slop_training::data::{O200K_CEILING, Vocabulary, read_split};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
#[ignore = "needs the released data/3.0.0 parquets; tokenizes the full training split (~minutes)"]
fn vocabulary_build_is_deterministic_against_the_fixture() {
    let dir = std::env::var("IS_IT_SLOP_DATA").expect("set IS_IT_SLOP_DATA to the split directory");
    let split = read_split(Path::new(&dir).join("train.parquet").as_path()).expect("train split");

    // Python's build_vocab tokenizes the raw texts — no trim, unlike encoding.
    let built = Vocabulary::build(&split.texts, 8192);
    let shipped = Vocabulary::load(Path::new(&format!("{FIXTURES}/vocabulary.safetensors")))
        .expect("bundle vocabulary");

    let mut diffs = 0usize;
    let ceiling = u32::try_from(O200K_CEILING).expect("the o200k ceiling fits u32");
    for token in 0..ceiling {
        if built.row(token) != shipped.row(token) {
            diffs += 1;
        }
    }
    assert_eq!(
        diffs, 0,
        "{diffs} lookup rows differ from the frozen reference vocabulary"
    );
}
