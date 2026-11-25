//! Python bindings for the training-time text cleaner and the tokenizer.
//!
//! The dataset pipeline cleans text with exactly the code the model is trained and evaluated with,
//! without depending on the inference crate. Internal: it is built into the
//! `is-it-slop-preprocessing` Python package and is not published to crates.io.

use pyo3::prelude::*;

/// Clean texts with the training-time cleaner (the one the shipped model's data went through).
///
/// A text that cleans to nothing comes back as an empty string.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn clean_for_training(py: Python<'_>, texts: Vec<String>) -> Vec<String> {
    py.detach(|| {
        is_it_slop_preprocessing::pre_processor::text_cleaner_for_training().clean_batch(&texts)
    })
}

/// Number of `o200k_base` tokens in each text.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn count_tokens(py: Python<'_>, texts: Vec<String>) -> Vec<usize> {
    py.detach(|| {
        is_it_slop_preprocessing::pre_processor::tokenize(&texts)
            .into_iter()
            .map(|tokens| tokens.len())
            .collect()
    })
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(clean_for_training, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    Ok(())
}
