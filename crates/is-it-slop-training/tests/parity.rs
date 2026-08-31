//! Golden-vector parity with the `PyTorch` model.
//!
//! Loads the trained `c192-k8-v8192-d48-g2` bundle's weights into the Burn module and asserts its
//! outputs match what `PyTorch` produced for 16 real test documents. Any drift in masking, op order,
//! GELU/LayerNorm semantics, or weight remapping fails here rather than surfacing as silently
//! worse accuracy. Fixtures are produced by `notebooks/export_burn_parity_fixture.py`.

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Int, Tensor};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use is_it_slop_training::model::HierarchicalModelConfig;
use safetensors::SafeTensors;

type B = NdArray;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn f32_tensor(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    let view = st.tensor(name).unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn matches_pytorch_on_golden_vectors() {
    let device = NdArrayDevice::default();
    let config = HierarchicalModelConfig::new()
        .with_vocab_size(8192)
        .with_embed_dim(48);
    let mut model = config.init::<B>(&device);

    // PyTorch names -> Burn field paths. `trunk`/`doc_head` are nn.Sequential in Python, so their
    // layers arrive with numeric keys; the adapter handles linear transposes and norm renames.
    let mut store = SafetensorsStore::from_file(format!("{FIXTURES}/weights.safetensors"))
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"^trunk\.0\.", "trunk_in.")
        .with_key_remapping(r"^trunk\.2\.", "trunk_norm.")
        .with_key_remapping(r"^trunk\.4\.", "trunk_out.")
        .with_key_remapping(r"^doc_head\.0\.", "doc_norm.")
        .with_key_remapping(r"^doc_head\.2\.", "doc_head.");
    let result = model.load_from(&mut store).expect("weights load");
    assert!(
        result.is_success() && result.missing.is_empty(),
        "weight loading incomplete:\n{result}"
    );

    let bytes = std::fs::read(format!("{FIXTURES}/golden.safetensors")).unwrap();
    let golden = SafeTensors::deserialize(&bytes).unwrap();

    let ids_view = golden.tensor("ids").unwrap();
    assert_eq!(ids_view.dtype(), safetensors::Dtype::I64);
    let shape = ids_view.shape().to_vec();
    let ids: Vec<i64> = ids_view
        .data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let ids = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device)
        .reshape([shape[0], shape[1], shape[2]]);

    let out = model.forward(ids);

    // Raw logits are the sharp comparison; probabilities saturate near 0/1 and would hide drift.
    let expected_doc = f32_tensor(&golden, "doc_logits");
    let expected_chunk = f32_tensor(&golden, "chunk_logits");
    let expected_probs = f32_tensor(&golden, "doc_probs");
    let doc: Vec<f32> = out.doc_logits.into_data().to_vec().unwrap();
    let chunk: Vec<f32> = out.chunk_logits.into_data().to_vec().unwrap();

    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    };
    let doc_diff = max_diff(&doc, &expected_doc);
    let chunk_diff = max_diff(&chunk, &expected_chunk);
    assert!(
        doc_diff <= 1e-3,
        "doc logits drift {doc_diff} > 1e-3\nburn: {doc:?}\ntorch: {expected_doc:?}"
    );
    assert!(chunk_diff <= 1e-3, "chunk logits drift {chunk_diff} > 1e-3");

    let probs: Vec<f32> = doc.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let prob_diff = max_diff(&probs, &expected_probs);
    assert!(prob_diff <= 1e-5, "probability drift {prob_diff} > 1e-5");
}
