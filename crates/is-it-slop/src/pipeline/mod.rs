mod classification;
mod prediction;

use std::sync::Mutex;

pub use classification::Classification;
use ort::{
    session::Session,
    value::{Tensor, Value},
};
pub use prediction::Prediction;
use sprs::CsMat;

use crate::model::PRE_PROCESSOR;

fn prepare_input_for_inference(
    input_vector: &CsMat<f32>,
) -> ort::Result<Value<ort::value::TensorValueType<f32>>> {
    let dense = input_vector.to_dense();
    let shape = dense.shape().to_vec();
    let data = dense.into_raw_vec_and_offset().0.into_boxed_slice();

    let input = Tensor::from_array((shape, data))?;
    Ok(input)
}

fn run_model_inference(
    session: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<ort::session::SessionOutputs<'_>> {
    let input_name = session.inputs[0].name.clone();
    session.run(ort::inputs![input_name => input])
}

/// Extracts class probabilities from model outputs
fn parse_model_outputs(outputs: &ort::session::SessionOutputs<'_>) -> ort::Result<Prediction> {
    // Second output: class probabilities (e.g., [{0: ..., 1: ...}])
    let probs_array = outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("valid 2d array");

    let first_row = probs_array.row(0);
    Ok([first_row[0], first_row[1]].into())
}

fn parse_model_outputs_batch(
    outputs: &ort::session::SessionOutputs<'_>,
) -> ort::Result<Vec<Prediction>> {
    let probs_array = outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("valid 2d array");

    Ok(probs_array
        .outer_iter()
        .map(|row| [row[0], row[1]].into())
        .collect())
}

fn run_inference_single(
    session: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<Prediction> {
    let outputs = run_model_inference(session, input)?;
    parse_model_outputs(&outputs)
}

fn run_inference_batch(
    session: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<Vec<Prediction>> {
    let outputs = run_model_inference(session, input)?;
    parse_model_outputs_batch(&outputs)
}

pub fn predict<T: AsRef<str> + Sync>(
    session: &Mutex<Session>,
    input: T,
) -> ort::Result<Prediction> {
    let input = prepare_input_for_inference(&PRE_PROCESSOR.transform(&[input]))?;
    {
        let mut session = session.lock().unwrap();
        run_inference_single(&mut session, input)
    }
}

pub fn predict_batch<T: AsRef<str> + Sync>(
    session: &Mutex<Session>,
    inputs: &[T],
) -> ort::Result<Vec<Prediction>> {
    let input = prepare_input_for_inference(&PRE_PROCESSOR.transform(inputs))?;
    {
        let mut session = session.lock().unwrap();
        run_inference_batch(&mut session, input)
    }
}
