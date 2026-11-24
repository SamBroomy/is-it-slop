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
//! - `confidence`: calibrated margin in [0.0, 1.0]
//! - `num_chunks` / `chunk_dispersion`: per-chunk analysis

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{Predictor, Threshold, UnifiedPrediction, pipeline::PipelineError};

/// Python prediction result containing probabilities and classification.
///
/// Attributes:
///     `human_probability` (float): Probability that the text is human-written (0.0 to 1.0)
///     `ai_probability` (float): Probability that the text is AI-generated (0.0 to 1.0)
///     classification (str): Classification label ("Human" or "AI")
///     confidence (float): Calibrated margin from the coin-flip point (0.0 to 1.0)
///     `num_chunks` (int): Number of 192-token chunks the document produced
///     `chunk_dispersion` (float | None): Std dev of chunk AI probabilities; high values
///         suggest mixed human/AI content. None for single-chunk documents.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
struct PredictionResult {
    #[pyo3(get)]
    human_probability: f32,
    #[pyo3(get)]
    ai_probability: f32,
    #[pyo3(get)]
    classification: String,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    num_chunks: usize,
    #[pyo3(get)]
    chunk_dispersion: Option<f32>,
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

impl PredictionResult {
    fn new(prediction: &UnifiedPrediction, threshold: Threshold) -> Self {
        Self {
            human_probability: prediction.human_probability(),
            ai_probability: prediction.ai_probability(),
            classification: prediction.classification(threshold).to_string(),
            confidence: prediction.confidence(),
            num_chunks: prediction.num_chunks(),
            chunk_dispersion: prediction.chunk_dispersion(),
        }
    }
}

/// Build a predictor from an optional user-supplied threshold.
fn predictor(threshold: Option<f32>) -> PyResult<Predictor> {
    let predictor = Predictor::new();
    match threshold {
        None => Ok(predictor),
        Some(t) => Threshold::try_new(t)
            .map(|t| predictor.with_threshold(t))
            .map_err(|e| PyValueError::new_err(format!("Invalid threshold: {e}"))),
    }
}

fn pipeline_error(error: &PipelineError) -> PyErr {
    match error {
        PipelineError::EmptyInput => {
            PyValueError::new_err("Input text is empty. Please provide a non-empty string.")
        }
    }
}

#[pyfunction]
#[pyo3(signature = (text, threshold=None))]
fn is_this_slop(py: Python<'_>, text: &str, threshold: Option<f32>) -> PyResult<PredictionResult> {
    let predictor = predictor(threshold)?;
    py.detach(|| {
        let prediction = predictor.predict(text).map_err(|e| pipeline_error(&e))?;
        Ok(PredictionResult::new(&prediction, predictor.threshold()))
    })
}

/// All or nothing: any blank text raises `ValueError` for the whole batch.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (texts, threshold=None))]
fn is_this_slop_batch(
    py: Python<'_>,
    texts: Vec<String>,
    threshold: Option<f32>,
) -> PyResult<Vec<PredictionResult>> {
    let predictor = predictor(threshold)?;
    py.detach(|| {
        let predictions = predictor
            .predict_batch(&texts)
            .map_err(|e| pipeline_error(&e))?;
        Ok(predictions
            .iter()
            .map(|p| PredictionResult::new(p, predictor.threshold()))
            .collect())
    })
}

/// One entry per input text, in order; blank texts yield `None` instead of failing the batch.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (texts, threshold=None))]
fn is_this_slop_each(
    py: Python<'_>,
    texts: Vec<String>,
    threshold: Option<f32>,
) -> PyResult<Vec<Option<PredictionResult>>> {
    let predictor = predictor(threshold)?;
    Ok(py.detach(|| {
        predictor
            .predict_each(&texts)
            .into_iter()
            .map(|r| {
                r.ok()
                    .map(|p| PredictionResult::new(&p, predictor.threshold()))
            })
            .collect()
    }))
}

/// CLI entry point for Python console scripts.
///
/// Parses arguments from `env::args()` and runs the CLI.
/// Handles errors cleanly without Python tracebacks.
///
/// **Note:** GIL is automatically released for the entire function since we don't
/// take `py: Python<'_>` parameter. This allows optimal rayon parallelization.
#[cfg(feature = "cli")]
#[pyfunction]
fn cli_main() -> PyResult<()> {
    use clap::Parser;

    use crate::cli::{Cli, Commands, run};

    let argv: Vec<String> = std::env::args().skip(1).collect();

    let cli = Cli::try_parse_from(&argv).map_err(|e| {
        let _ = e.print();
        PyErr::new::<pyo3::exceptions::PySystemExit, _>(e.exit_code())
    })?;

    // Handle self subcommands for Python installations
    if let Some(command) = &cli.command {
        match command {
            Commands::Self_(namespace) => match namespace.command {
                crate::cli::SelfCommand::Update => {
                    eprintln!(
                        "Error: The 'self update' command is not available for Python installations."
                    );
                    eprintln!();
                    eprintln!("Please upgrade using your package manager:");
                    eprintln!("  pip install --upgrade is-it-slop");
                    eprintln!("  # or");
                    eprintln!("  uv tool upgrade is-it-slop");
                    return Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(1));
                }
            },
        }
    }

    match run(&cli) {
        Ok(0) => Ok(()),
        Ok(code) => Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(i32::from(
            code,
        ))),
        Err(e) => {
            eprintln!("Error: {e:#}");
            Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(1))
        }
    }
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
    m.add_function(wrap_pyfunction!(is_this_slop_each, m)?)?;

    #[cfg(feature = "cli")]
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;

    Ok(())
}
