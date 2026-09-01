//! Python bindings for the preprocessing pipeline.
//!
//! Exposes the Rust preprocessing components to Python via `PyO3`. Rust types are
//! prefixed with `Rust*` (e.g. `RustTextCleaner`); the Python package re-exports
//! them without the prefix.
//!
//! # Exposed Components
//!
//! - **`tokenize`**: text → BPE token IDs (tiktoken `o200k_base`)
//! - **`RustTokenChunker`**: token-based text chunking
//! - **`RustTextCleaner`**: two-stage text cleaning (universal + dataset artifacts)

use pyo3::prelude::*;

use crate::pre_processor::{
    TextCleaner, TokenChunker, text_cleaner_for_inference, text_cleaner_for_training,
    tokenize as tokenize_,
};

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RustTokenChunker {
    inner: TokenChunker,
}

#[pymethods]
impl RustTokenChunker {
    #[new]
    #[pyo3(signature = (chunk_size=150, overlap=15, min_chunk_size=30))]
    fn new(chunk_size: usize, overlap: usize, min_chunk_size: usize) -> Self {
        Self {
            inner: TokenChunker {
                chunk_size,
                overlap,
                min_chunk_size,
            },
        }
    }

    /// Chunk a single token sequence
    #[allow(clippy::needless_pass_by_value)]
    fn chunk(&self, tokens: Vec<u32>) -> Vec<Vec<u32>> {
        self.inner.chunk(&tokens)
    }

    /// Chunk multiple token sequences in parallel
    #[allow(clippy::needless_pass_by_value)]
    fn chunk_batch(&self, py: Python<'_>, token_sequences: Vec<Vec<u32>>) -> Vec<Vec<Vec<u32>>> {
        py.detach(|| self.inner.chunk_batch(&token_sequences))
    }

    fn to_dict(&self) -> std::collections::HashMap<String, usize> {
        std::collections::HashMap::from([
            ("chunk_size".to_string(), self.inner.chunk_size),
            ("overlap".to_string(), self.inner.overlap),
            ("min_chunk_size".to_string(), self.inner.min_chunk_size),
        ])
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum RustCleaningMode {
    Training,
    Inference,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct RustTextCleaner {
    inner: TextCleaner,
    mode: RustCleaningMode,
}

#[pymethods]
impl RustTextCleaner {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(mode: RustCleaningMode) -> Self {
        let inner = match mode {
            RustCleaningMode::Training => text_cleaner_for_training().clone(),
            RustCleaningMode::Inference => text_cleaner_for_inference().clone(),
        };

        Self { inner, mode }
    }

    /// Clean a single text string
    fn clean(&self, text: &str) -> String {
        self.inner.clean(text).to_string()
    }

    /// Clean multiple texts in parallel
    #[allow(clippy::needless_pass_by_value)]
    fn clean_batch(&self, python: Python<'_>, texts: Vec<String>) -> Vec<String> {
        python.detach(|| self.inner.clean_batch(&texts))
    }

    pub fn is_training_mode(&self) -> bool {
        matches!(self.mode, RustCleaningMode::Training)
    }
}

/// Encode text into BPE token IDs
/// This is a utility function to expose tokenization to Python.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tokenize(py: Python<'_>, text: Vec<String>) -> Vec<Vec<u32>> {
    py.detach(|| tokenize_(&text))
}

#[pymodule]
#[pyo3(name = "_is_it_slop_preprocessing_rust_bindings")]
fn is_it_slop_preprocessing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize pyo3_log to bridge Rust tracing to Python logging
    // This respects Python's logging level configuration
    let _ = pyo3_log::try_init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<RustTokenChunker>()?;

    m.add_class::<RustCleaningMode>()?;
    m.add_class::<RustTextCleaner>()?;

    m.add_function(wrap_pyfunction!(tokenize, m)?)?;

    Ok(())
}
