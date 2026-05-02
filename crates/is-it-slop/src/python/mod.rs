//! Python bindings for AI text detection inference.
//!
//! This module exposes the inference pipeline to Python via `PyO3`.
//!
//! # Python API
//!
//! ```python
//! from is_it_slop import is_this_slop, is_this_slop_batch
//!
//! # Single prediction
//! result = is_this_slop("Some text to classify")
//! print(f"{result.classification}: {result.ai_probability:.2%}")
//!
//! # Batch prediction
//! results = is_this_slop_batch(["text 1", "text 2", "text 3"])
//! for r in results:
//!     print(f"{r.classification}: {r.ai_probability:.2%}")
//!
//! # Custom threshold
//! result = is_this_slop("Text", threshold=0.6)
//! ```
//!
//! # Return Type
//!
//! `PredictionResult` contains:
//! - `human_probability`: P(Human) in [0.0, 1.0]
//! - `ai_probability`: P(AI) in [0.0, 1.0]
//! - `classification`: "Human" or "AI"

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;

use crate::Prediction;

/// Python prediction result containing probabilities and classification.
///
/// Attributes:
///     `human_probability` (float): Probability that the text is human-written (0.0 to 1.0)
///     `ai_probability` (float): Probability that the text is AI-generated (0.0 to 1.0)
///     classification (str): Classification label ("Human" or "AI")
#[pyclass(from_py_object)]
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
            human_probability: prediction.prediction.human_probability(),
            ai_probability: prediction.prediction.ai_probability(),
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
                    human_probability: pred.prediction.human_probability(),
                    ai_probability: pred.prediction.ai_probability(),
                    classification: classification.to_string(),
                }
            })
            .collect();

        Ok(results)
    })
}

/// CLI entry point for Python console scripts.
///
/// Parses arguments from env::args() and runs the CLI.
/// Handles errors cleanly without Python tracebacks.
///
/// **Note:** GIL is automatically released for the entire function since we don't
/// take `py: Python<'_>` parameter. This allows optimal rayon parallelization.
#[cfg(feature = "cli")]
#[pyfunction]
fn cli_main() -> PyResult<()> {
    use clap::Parser;

    use crate::cli::{Cli, run};

    // Get argv from env::args(), skipping the Python interpreter path
    // env::args() = ["/path/to/python", "/path/to/is-it-slop", "arg1", ...]
    // skip(1)     = ["/path/to/is-it-slop", "arg1", ...]
    let argv: Vec<String> = std::env::args().skip(1).collect();

    // Parse CLI arguments - clap will use argv[0] as program name
    let cli = Cli::try_parse_from(&argv).map_err(|e| {
        // Print the error/help message (try_parse_from doesn't auto-print)
        let _ = e.print();
        // Exit with clap's exit code (0 for help, 2 for parse errors)
        PyErr::new::<pyo3::exceptions::PySystemExit, _>(e.exit_code())
    })?;

    // Run CLI (GIL already released automatically)
    if let Err(e) = run(&cli) {
        // Print error to stderr (like native Rust CLI does)
        eprintln!("Error: {e}");
        // Exit with code 1 (standard for runtime errors)
        return Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(1));
    }

    Ok(())
}

#[pymodule]
#[pyo3(name = "_is_it_slop_rust_bindings")]
fn is_it_slop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CLASSIFICATION_THRESHOLD", crate::CLASSIFICATION_THRESHOLD)?;
    m.add("MODEL_VERSION", crate::MODEL_VERSION)?;

    m.add_class::<PredictionResult>()?;
    m.add_function(wrap_pyfunction!(is_this_slop, m)?)?;
    m.add_function(wrap_pyfunction!(is_this_slop_batch, m)?)?;

    #[cfg(feature = "cli")]
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;

    Ok(())
}
