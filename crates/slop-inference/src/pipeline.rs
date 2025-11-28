use std::sync::Mutex;

use ort::{
    session::Session,
    value::{Tensor, Value},
};

use crate::model::PRE_PROCESSOR;

pub fn predict(session: &Mutex<Session>, input: &str) -> ort::Result<Vec<f32>> {
    let input_vector = PRE_PROCESSOR.transform(&[input]);

    // Convert to f32 for ONNX (Precision might be f64 in the future)
    let input = input_vector.map(|x| *x as f32);

    let dense = input.to_owned().to_dense();
    let shape = dense.shape().to_vec();
    let data = dense.into_raw_vec_and_offset().0.into_boxed_slice();

    let input = Tensor::from_array((shape, data))?;
    //let input = TensorRef::from_array_view(&dense)?;

    {
        let mut session = session.lock().unwrap();
        run_inference(&mut session, input)
    }
}
#[allow(clippy::needless_pass_by_value)]
fn run_inference(
    session: &mut Session,
    input: Value<ort::value::TensorValueType<f32>>,
) -> ort::Result<Vec<f32>> {
    let input_name = session.inputs[0].name.clone();
    let outputs: ort::session::SessionOutputs<'_> =
        session.run(ort::inputs![input_name.as_str() => &input])?;
    // // First output: class labels (e.g., [1])
    // let labels = outputs[0]
    //     .try_extract_array::<i64>()?
    //     .into_dimensionality::<ndarray::Ix1>()
    //     .unwrap()
    //     .to_vec();

    // Second output: class probabilities (e.g., [{0: ..., 1: ...}])
    Ok(outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("valid 2d array")
        .outer_iter()
        .map(|row| row[1])
        .collect())

    // // Extract only the AI class probability (index 1)
    // // Since we have binary classification and probs sum to 1.0,
    // // we only need P(AI) as P(Human) = 1.0 - P(AI)
    // let ai_probs: Vec<f32> = probs.outer_iter().map(|row| row[1]).collect();

    // Ok((labels, ai_probs))
}

// TODO: Some helper methods to get class labels and probabilities separately
