use core::fmt;
use std::sync::Mutex;

use ort::{
    session::Session,
    value::{Tensor, Value},
};
use sprs::CsMat;

use crate::model::PRE_PROCESSOR;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Classification {
    Human,
    AI,
}

impl Classification {
    /// Returns true if this classification is Human
    #[must_use]
    pub fn is_human(&self) -> bool {
        matches!(self, Self::Human)
    }

    /// Returns true if this classification is AI
    #[must_use]
    pub fn is_ai(&self) -> bool {
        matches!(self, Self::AI)
    }
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Human => write!(f, "Human"),
            Self::AI => write!(f, "AI"),
        }
    }
}

impl From<Classification> for i64 {
    fn from(class: Classification) -> Self {
        match class {
            Classification::Human => 0,
            Classification::AI => 1,
        }
    }
}

/// Struct to hold prediction probabilities
/// 0: P(Human), 1: P(AI)
#[derive(Debug, Clone, Copy)]
pub struct Prediction(f32, f32);

impl Prediction {
    /// Create a new Prediction instance
    /// `human_prob` + `ai_prob` must equal 1.0
    fn new(human_prob: f32, ai_prob: f32) -> Self {
        debug_assert!(
            (human_prob + ai_prob - 1.0).abs() < f32::EPSILON,
            "Probabilities must sum to 1.0"
        );
        Self(human_prob, ai_prob)
    }

    #[must_use]
    pub fn human_probability(&self) -> f32 {
        self.0
    }

    #[must_use]
    pub fn ai_probability(&self) -> f32 {
        self.1
    }

    #[inline]
    #[must_use]
    pub fn classification(&self, threshold: f32) -> Classification {
        if self.1 >= threshold {
            Classification::AI
        } else {
            Classification::Human
        }
    }
}

impl fmt::Display for Prediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P(Human)={:.3}, P(AI)={:.3}", self.0, self.1)
    }
}

impl From<[f32; 2]> for Prediction {
    fn from(probs: [f32; 2]) -> Self {
        Self::new(probs[0], probs[1])
    }
}

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
