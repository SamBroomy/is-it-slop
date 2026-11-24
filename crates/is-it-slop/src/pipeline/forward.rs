//! One forward pass over a group of encoded documents.

use burn::tensor::{Device, Int, Tensor, activation::sigmoid};
use is_it_slop_model::HierarchicalSlopClassifier;

use super::UnifiedPrediction;
use crate::model::{CALIBRATION_TEMPERATURE, ENCODER, MODEL, device};

/// A function that scores one group of documents; see [`run`] for the contract.
pub type Forward = fn(&[&[u16]]) -> Vec<UnifiedPrediction>;

/// Score a group of documents in one forward pass on the primary backend.
///
/// Each document is its chunk ids, row-major `[chunks, chunk_size]`, already capped. The group
/// is padded to its longest member with pad-id chunks, which the model masks out of every
/// pooling step, so a document's result does not depend on what it was grouped with.
pub fn run(docs: &[&[u16]]) -> Vec<UnifiedPrediction> {
    run_with(&MODEL, &device(), docs)
}

/// [`run`] on the CPU backend's copy of the model.
#[cfg(accelerated)]
pub fn run_on_cpu(docs: &[&[u16]]) -> Vec<UnifiedPrediction> {
    run_with(&crate::model::CPU_MODEL, &crate::model::cpu_device(), docs)
}

fn run_with(
    model: &HierarchicalSlopClassifier,
    device: &Device,
    docs: &[&[u16]],
) -> Vec<UnifiedPrediction> {
    let chunk_size = ENCODER.chunk_size();
    let pad = ENCODER.pad_id();
    let batch = docs.len();
    let max_chunks = docs
        .iter()
        .map(|doc| doc.len() / chunk_size)
        .max()
        .expect("a group has at least one document");

    let mut ids = vec![i64::from(pad); batch * max_chunks * chunk_size];
    for (row, doc) in docs.iter().enumerate() {
        debug_assert_eq!(
            doc.len() % chunk_size,
            0,
            "a document's ids are whole chunks"
        );
        let start = row * max_chunks * chunk_size;
        for (slot, &id) in ids[start..start + doc.len()].iter_mut().zip(*doc) {
            *slot = i64::from(id);
        }
    }
    let ids = Tensor::<1, Int>::from_ints(ids.as_slice(), device)
        .reshape([batch, max_chunks, chunk_size]);

    let output = model.forward(ids);

    let doc_probs: Vec<f32> = sigmoid(output.doc_logits / f64::from(CALIBRATION_TEMPERATURE))
        .into_data()
        .try_to_vec()
        .expect("document probabilities");
    let chunk_probs: Vec<f32> = sigmoid(output.chunk_logits)
        .into_data()
        .try_to_vec()
        .expect("chunk probabilities");
    // Read the mask through int: GPU backends store bools as bytes, which `to_vec::<bool>` rejects.
    let chunk_mask: Vec<bool> = output
        .chunk_mask
        .int()
        .into_data()
        .iter::<i64>()
        .map(|real| real != 0)
        .collect();

    doc_probs
        .iter()
        .enumerate()
        .map(|(row, &doc_prob)| {
            let slots = row * max_chunks..(row + 1) * max_chunks;
            let real_chunk_probs: Vec<f32> = chunk_probs[slots.clone()]
                .iter()
                .zip(&chunk_mask[slots])
                .filter_map(|(&prob, &real)| real.then_some(prob))
                .collect();
            UnifiedPrediction::new(doc_prob, &real_chunk_probs)
        })
        .collect()
}
