//! Full-corpus vocabulary parity with the Python trainer.
//!
//! Builds the v8192 vocabulary from the real training split and asserts every lookup row equals
//! the one Python shipped in the bundle (`fixtures/vocabulary.safetensors`, built from the same
//! split). This proves the whole vocab path — shared tokenizer, counting, `Counter`-compatible
//! tie-breaking, and the no-trim-before-tokenize asymmetry of `build_vocab` — on real data.
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
fn builds_the_same_vocabulary_as_python() {
    let dir = std::env::var("IS_IT_SLOP_DATA").expect("set IS_IT_SLOP_DATA to the split directory");
    let split = read_split(Path::new(&dir).join("train.parquet").as_path()).expect("train split");

    // Python's build_vocab tokenizes the raw texts — no trim, unlike encoding.
    let built = Vocabulary::build(&split.texts, 8192);
    let shipped = Vocabulary::load(Path::new(&format!("{FIXTURES}/vocabulary.safetensors")))
        .expect("bundle vocabulary");

    let mut diffs = 0usize;
    for token in 0..O200K_CEILING as u32 {
        if built.row(token) != shipped.row(token) {
            diffs += 1;
        }
    }
    assert_eq!(
        diffs, 0,
        "{diffs} lookup rows differ from the Python-built vocabulary"
    );
}
