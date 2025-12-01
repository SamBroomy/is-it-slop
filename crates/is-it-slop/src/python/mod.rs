#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;

use crate::Prediction;

/// Result of a prediction containing probabilities and classification.
///
/// Attributes:
///     `human_probability` (float): Probability that the text is human-written (0.0 to 1.0)
///     `ai_probability` (float): Probability that the text is AI-generated (0.0 to 1.0)
///     classification (str): Classification label ("Human" or "AI")
#[pyclass]
#[derive(Debug, Clone)]
struct PredictionResult {
    #[pyo3(get)]
    human_probability: f32,
    #[pyo3(get)]
    ai_probability: f32,
    #[pyo3(get)]
    classification: String,
}

impl From<Prediction> for PredictionResult {
    fn from(pred: Prediction) -> Self {
        Self {
            human_probability: pred.human_probability(),
            ai_probability: pred.ai_probability(),
            classification: pred
                .classification(crate::CLASSIFICATION_THRESHOLD)
                .to_string(),
        }
    }
}

#[pymethods]
impl PredictionResult {
    fn __repr__(&self) -> String {
        format!(
            "PredictionResult(human={:.3}, ai={:.3}, class={})",
            self.human_probability, self.ai_probability, self.classification
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{} (AI: {:.1}%)",
            self.classification,
            self.ai_probability * 100.0
        )
    }
}

#[pyfunction]
#[pyo3(signature = (text, threshold=None))]
fn is_this_slop(py: Python<'_>, text: &str, threshold: Option<f32>) -> PyResult<PredictionResult> {
    py.detach(|| {
        let predictor = crate::Predictor::new();
        let predictor = if let Some(t) = threshold {
            predictor.with_threshold(t)
        } else {
            predictor
        };

        let prediction = predictor.predict(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {e}"))
        })?;

        let classification = prediction.classification(predictor.threshold());

        Ok(PredictionResult {
            human_probability: prediction.human_probability(),
            ai_probability: prediction.ai_probability(),
            classification: classification.to_string(),
        })
    })
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (texts, threshold=None))]
fn is_this_slop_batch(
    py: Python<'_>,
    texts: Vec<String>,
    threshold: Option<f32>,
) -> PyResult<Vec<PredictionResult>> {
    py.detach(|| {
        let predictor = crate::Predictor::new();
        let predictor = if let Some(t) = threshold {
            predictor.with_threshold(t)
        } else {
            predictor
        };

        let predictions = predictor.predict_batch(&texts).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Batch prediction failed: {e}"
            ))
        })?;

        let results = predictions
            .into_iter()
            .map(|pred| {
                let classification = pred.classification(predictor.threshold());
                PredictionResult {
                    human_probability: pred.human_probability(),
                    ai_probability: pred.ai_probability(),
                    classification: classification.to_string(),
                }
            })
            .collect();

        Ok(results)
    })
}

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_is_it_slop_rust_bindings")]
fn is_it_slop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CLASSIFICATION_THRESHOLD", crate::CLASSIFICATION_THRESHOLD)?;

    m.add_class::<PredictionResult>()?;
    m.add_function(wrap_pyfunction!(is_this_slop, m)?)?;
    m.add_function(wrap_pyfunction!(is_this_slop_batch, m)?)?;

    Ok(())
}
