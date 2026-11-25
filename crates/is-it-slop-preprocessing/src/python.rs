//! Python bindings: the training-time cleaner and the tokenizer, for the data pipeline.
//!
//! ```python
//! from is_it_slop_preprocessing import clean_for_training, count_tokens
//!
//! clean_for_training(["  Some   raw text "])
//! count_tokens(["Some raw text"])  # o200k_base tokens
//! ```
//!
//! Both take any iterable of strings (a list, a polars `Series`, a generator) and return a list.
//! Internal: not part of the supported public API.

use pyo3::prelude::*;

fn texts(iterable: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    iterable.try_iter()?.map(|item| item?.extract()).collect()
}

/// Clean texts with the training-time cleaner (the one the shipped model's data went through).
///
/// A text that cleans to nothing comes back as an empty string.
#[pyfunction]
fn clean_for_training(py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let texts = texts(iterable)?;
    Ok(py.detach(|| crate::pre_processor::text_cleaner_for_training().clean_batch(&texts)))
}

/// Number of `o200k_base` tokens in each text.
#[pyfunction]
fn count_tokens(py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let texts = texts(iterable)?;
    Ok(py.detach(|| {
        crate::pre_processor::tokenize(&texts)
            .into_iter()
            .map(|tokens| tokens.len())
            .collect()
    }))
}

#[pymodule]
#[pyo3(name = "is_it_slop_preprocessing")]
fn bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(clean_for_training, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    Ok(())
}
