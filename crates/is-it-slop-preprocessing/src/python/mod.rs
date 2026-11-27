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

#![expect(
    clippy::fn_params_excessive_bools,
    reason = "PyO3 signatures mirror Python keyword arguments at the language boundary"
)]

use pyo3::prelude::*;

/// Collect any Python iterable of strings into a Rust `Vec` in a single pass.
fn texts(iterable: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    iterable.try_iter()?.map(|item| item?.extract()).collect()
}

/// Clean texts with the training-time cleaner (the one the shipped model's data went through).
///
/// Markdown markers are stripped (the words stay) unless `strip_markdown` is false, which exists
/// only for comparison experiments. A text that cleans to nothing comes back as an empty string.
#[pyfunction]
#[pyo3(signature = (iterable, *, strip_markdown=true))]
fn clean_for_training(
    py: Python<'_>,
    iterable: &Bound<'_, PyAny>,
    strip_markdown: bool,
) -> PyResult<Vec<String>> {
    let texts = texts(iterable)?;
    Ok(py.detach(|| {
        let cleaner = if strip_markdown {
            crate::pre_processor::text_cleaner_for_training()
        } else {
            crate::pre_processor::text_cleaner_for_training_keeping_markdown()
        };
        cleaner.clean_batch(&texts)
    }))
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
#[pyo3(name = "_is_it_slop_preprocessing_rust_bindings")]
fn bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(clean_for_training, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    Ok(())
}
