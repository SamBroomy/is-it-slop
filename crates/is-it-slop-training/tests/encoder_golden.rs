//! Golden-vector parity for the data layer.
//!
//! Encodes the fixture documents' raw texts with the trained bundle's vocabulary and asserts the
//! grids are bit-identical to what the Python pipeline produced for the same documents (the
//! `test-c192-k8-v8192` cache rows stored in `golden.safetensors`). Any drift in trimming,
//! tokenization, vocabulary application, or chunking fails here. Fixtures come from
//! `notebooks/export_burn_parity_fixture.py`.

use std::path::Path;

use is_it_slop_training::data::{DocumentEncoder, Vocabulary};
use safetensors::SafeTensors;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn encodes_identically_to_python() {
    let vocabulary = Vocabulary::load(Path::new(&format!("{FIXTURES}/vocabulary.safetensors")))
        .expect("bundle vocabulary");
    assert_eq!(vocabulary.vocab_size(), 8192);

    let texts: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(format!("{FIXTURES}/encoder_texts.json")).unwrap(),
    )
    .unwrap();

    let bytes = std::fs::read(format!("{FIXTURES}/golden.safetensors")).unwrap();
    let golden = SafeTensors::deserialize(&bytes).unwrap();
    let view = golden.tensor("ids_unmodified").unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::I64);
    let expected: Vec<i64> = view
        .data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
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
        assert_eq!(
            got, want,
            "document {doc} encodes differently from the Python pipeline"
        );
    }
}
