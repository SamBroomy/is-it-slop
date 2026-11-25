//! The encoding contract, pinned by golden vectors.
//!
//! Every trained artifact depends on exactly this encoding: a model plus its vocabulary only
//! work if text becomes the same grid of compact ids at training and inference time. This test
//! freezes that contract — raw fixture texts must encode to the committed grids bit-for-bit, so
//! any drift in trimming, tokenization, vocabulary application, or chunking fails CI instead of
//! silently skewing a shipped model. Changing the encoding deliberately means regenerating the
//! fixture (`scripts/export_encoder_fixture.py`) and versioning the artifact format.

use std::path::Path;

use is_it_slop_training::data::{DocumentEncoder, Vocabulary};
use safetensors::SafeTensors;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn encoding_contract_is_frozen() {
    let vocabulary = Vocabulary::load(Path::new(&format!("{FIXTURES}/vocabulary.safetensors")))
        .expect("fixture vocabulary");
    assert_eq!(vocabulary.vocab_size(), 8192);

    let texts: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(format!("{FIXTURES}/encoder_texts.json")).unwrap(),
    )
    .unwrap();

    let bytes = std::fs::read(format!("{FIXTURES}/encoder_grids.safetensors")).unwrap();
    let grids = SafeTensors::deserialize(&bytes).unwrap();
    let view = grids.tensor("ids").unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::I64);
    let expected: Vec<i64> = view
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect();
    let grid_len = view.shape()[1] * view.shape()[2];
    assert_eq!(texts.len(), view.shape()[0]);

    let encoder = DocumentEncoder::new(vocabulary);
    let encoded = encoder.encode_batch(&texts, &vec![0.0; texts.len()]);

    for (doc, (item, want)) in encoded
        .iter()
        .zip(expected.chunks_exact(grid_len))
        .enumerate()
    {
        let got: Vec<i64> = item.ids.iter().map(|&id| i64::from(id)).collect();
        assert_eq!(got, want, "document {doc} breaks the frozen encoding");
    }
}
