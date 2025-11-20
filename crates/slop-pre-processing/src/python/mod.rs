use ahash::HashMap;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyTuple};
use serde::{Deserialize, Serialize};

use crate::pre_processor::{TfidfVectorizer, VectorizerParams};

/// A wrapper struct for VectorizerParams to expose it to Python.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RustVectorizerParams {
    #[pyo3(get)]
    ngram_range: (usize, usize),
    #[pyo3(get)]
    min_df: usize,
}

#[pymethods]
impl RustVectorizerParams {
    /// Creates a new RustVectorizerParams instance.
    #[new]
    fn new(ngram_range: (usize, usize), min_df: usize) -> Self {
        Self {
            ngram_range,
            min_df,
        }
    }

    /// Returns a string representation of the RustVectorizerParams.
    fn __repr__(&self) -> String {
        format!(
            "RustVectorizerParams(ngram_range=({}, {}), min_df={})",
            self.ngram_range.0, self.ngram_range.1, self.min_df
        )
    }

    /// Returns a detailed string representation of the RustVectorizerParams.
    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl Default for RustVectorizerParams {
    fn default() -> Self {
        Self {
            ngram_range: (3, 5),
            min_df: 10,
        }
    }
}

impl RustVectorizerParams {
    fn to_inner(&self) -> VectorizerParams {
        VectorizerParams::new(self.ngram_range.0..=self.ngram_range.1, self.min_df)
    }
}

/// A wrapper function around TfidfVectorizer to expose it to Python.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RustTfidfVectorizer {
    #[serde(flatten)]
    inner: TfidfVectorizer,
}

#[pymethods]
impl RustTfidfVectorizer {
    /// Fits the TfidfVectorizer to the provided texts with the given parameters.
    /// Returns a new instance of RustTfidfVectorizer.
    #[new]
    fn fit(py: Python<'_>, texts: Vec<String>, params: RustVectorizerParams) -> Self {
        py.detach(move || {
            let vectorizer = TfidfVectorizer::fit(texts.as_slice(), params.to_inner());
            Self { inner: vectorizer }
        })
    }

    /// Transforms the input texts and returns the TF-IDF matrix components.
    /// The returned tuple contains:
    /// - shape: (usize, usize) | (number of rows, number of columns)
    /// - data: np.ndarray of f64 | values of the non-zero entries
    /// - indices: np.ndarray of usize | column indices of the non-zero entries
    /// - indptr: np.ndarray of usize | index pointers to the start of each row
    fn transform<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyTuple>> {
        let tfidf_matrix: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> =
            py.detach(|| self.inner.transform(texts.as_slice()));
        let data = tfidf_matrix.data().to_pyarray(py);
        let indices = tfidf_matrix.indices().to_pyarray(py);
        let indptr = tfidf_matrix
            .indptr()
            .to_owned()
            .into_raw_storage()
            .to_pyarray(py);
        let shape = (tfidf_matrix.rows(), tfidf_matrix.cols());

        (shape, data, indices, indptr).into_pyobject(py)
    }

    /// Fits the vectorizer and transforms the input texts in one step.
    /// Returns a tuple of (vectorizer, tfidf_matrix_components).
    /// The tfidf_matrix_components is the same as returned by `transform`.
    #[staticmethod]
    pub fn fit_transform<'py>(
        py: Python<'py>,
        texts: Vec<String>,
        params: RustVectorizerParams,
    ) -> PyResult<(Self, Bound<'py, PyTuple>)> {
        let vectorizer = Self::fit(py, texts.clone(), params);
        let transform_result = vectorizer.transform(py, texts)?;
        Ok((vectorizer, transform_result))
    }

    /// Getter for the number of features (vocabulary size).
    #[getter]
    pub fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Getter for the vocabulary mapping (token to index).
    #[getter]
    pub fn vocabulary(&self) -> HashMap<String, usize> {
        self.inner.vocabulary()
    }

    /// Return a string representation of the RustTfidfVectorizer.
    fn __repr__(&self) -> String {
        format!(
            "RustTfidfVectorizer(num_features={}, vocabulary_size={})",
            self.num_features(),
            self.vocabulary().len()
        )
    }

    /// Return a detailed string representation of the RustTfidfVectorizer.
    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[pymodule]
#[pyo3(name = "_rust_bindings")]
fn slop_pre_processing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Python logging for Rust components
    pyo3_log::init();

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<RustVectorizerParams>()?;
    m.add_class::<RustTfidfVectorizer>()?;
    Ok(())
}
