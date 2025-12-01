use ahash::HashMap;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyTuple};

use crate::pre_processor::{TfidfVectorizer, VectorizerParams};
#[allow(clippy::unsafe_derive_deserialize)]
/// A wrapper struct for `VectorizerParams` to expose it to Python.
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
#[pyclass]
struct RustVectorizerParams {
    #[pyo3(get)]
    ngram_range: (usize, usize),
    #[pyo3(get)]
    min_df: f32,
    #[pyo3(get)]
    max_df: f32,
    #[pyo3(get)]
    sublinear_tf: bool,
}

#[pymethods]
impl RustVectorizerParams {
    /// Creates a new `RustVectorizerParams` instance.
    #[new]
    #[pyo3(signature = (ngram_range, min_df, max_df, sublinear_tf))]
    fn new(ngram_range: (usize, usize), min_df: f32, max_df: f32, sublinear_tf: bool) -> Self {
        Self {
            ngram_range,
            min_df,
            max_df,
            sublinear_tf,
        }
    }

    /// Returns a string representation of the `RustVectorizerParams`.
    fn __repr__(&self) -> String {
        format!(
            "RustVectorizerParams(ngram_range=({}, {}), min_df={}, max_df={}, sublinear_tf={})",
            self.ngram_range.0, self.ngram_range.1, self.min_df, self.max_df, self.sublinear_tf
        )
    }

    /// Returns a detailed string representation of the `RustVectorizerParams`.
    fn __str__(&self) -> String {
        format!("{self:#?}")
    }
}

impl Default for RustVectorizerParams {
    fn default() -> Self {
        Self {
            ngram_range: (3, 5),
            min_df: 10.0,
            max_df: 0.9,
            sublinear_tf: true,
        }
    }
}

impl RustVectorizerParams {
    fn to_inner(self) -> VectorizerParams {
        VectorizerParams::new(
            self.ngram_range.0..=self.ngram_range.1,
            self.min_df,
            self.max_df,
            self.sublinear_tf,
        )
    }
}

impl From<&VectorizerParams> for RustVectorizerParams {
    fn from(params: &VectorizerParams) -> Self {
        Self {
            ngram_range: params.ngram_range(),
            min_df: params.min_df(),
            max_df: params.max_df(),
            sublinear_tf: params.sublinear_tf(),
        }
    }
}
#[allow(clippy::unsafe_derive_deserialize)]
/// A wrapper function around `TfidfVectorizer` to expose it to Python.
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
#[pyclass]
struct RustTfidfVectorizer {
    #[serde(flatten)]
    inner: TfidfVectorizer,
}

#[pymethods]
impl RustTfidfVectorizer {
    /// Fits the `TfidfVectorizer` to the provided texts with the given parameters.
    /// Returns a new instance of `RustTfidfVectorizer`.
    #[new]
    pub fn fit(py: Python<'_>, texts: Vec<String>, params: RustVectorizerParams) -> Self {
        py.detach(move || {
            let vectorizer = TfidfVectorizer::fit(texts.as_slice(), params.to_inner());
            Self { inner: vectorizer }
        })
    }

    /// Transforms the input texts and returns the TF-IDF matrix components.
    /// The returned tuple contains:
    /// - shape: (usize, usize) | (number of rows, number of columns)
    /// - data: np.ndarray of f32 | values of the non-zero entries
    /// - indices: np.ndarray of usize | column indices of the non-zero entries
    /// - indptr: np.ndarray of usize | index pointers to the start of each row
    #[allow(clippy::needless_pass_by_value)]
    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let tfidf_matrix: sprs::CsMatBase<f32, usize, Vec<usize>, Vec<usize>, Vec<f32>> =
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
    /// Returns a tuple of (vectorizer, `tfidf_matrix_components`).
    /// The `tfidf_matrix_components` is the same as returned by `transform`.
    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    pub fn fit_transform(
        py: Python<'_>,
        texts: Vec<String>,
        params: RustVectorizerParams,
    ) -> PyResult<(Self, Bound<'_, PyTuple>)> {
        let (vectorizer, transform_result) =
            TfidfVectorizer::fit_transform(texts.as_slice(), params.to_inner());
        let vectorizer = Self { inner: vectorizer };
        let data = transform_result.data().to_pyarray(py);
        let indices = transform_result.indices().to_pyarray(py);
        let indptr = transform_result
            .indptr()
            .to_owned()
            .into_raw_storage()
            .to_pyarray(py);
        let shape = (transform_result.rows(), transform_result.cols());
        let transform_result = (shape, data, indices, indptr).into_pyobject(py)?;
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

    /// Getter for the vectorizer parameters.
    #[getter]
    pub fn params(&self) -> RustVectorizerParams {
        self.inner.params().into()
    }

    /// Return a string representation of the `RustTfidfVectorizer`.
    fn __repr__(&self) -> String {
        format!(
            "RustTfidfVectorizer(vocabulary_size={})",
            self.num_features(),
        )
    }

    /// Return a detailed string representation of the `RustTfidfVectorizer`.
    fn __str__(&self) -> String {
        format!("{self:#?}")
    }

    /// Serialize the vectorizer to bytes using bincode format.
    /// Returns a bytes object that can be saved to disk or passed to `from_bytes`.
    /// Return the inner vectorizer serialized as bytes so it is compatible with Rust side.
    #[cfg(feature = "bincode")]
    fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        py.detach(|| {
            self.inner.to_bytes().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize vectorizer: {e}"
                ))
            })
        })
    }

    /// Serialize the vectorizer to JSON string.
    /// Returns a JSON string that can be saved to disk or passed to `from_json`.
    /// Return the inner vectorizer serialized as JSON so it is compatible with Rust side.
    #[cfg(feature = "serde")]
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.inner.to_json().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize vectorizer to JSON: {e}"
                ))
            })
        })
    }

    /// Deserialize the vectorizer from bytes (bincode format).
    ///
    /// Args:
    ///     bytes: A bytes object containing the serialized vectorizer
    ///
    /// Returns:
    ///     A new `RustTfidfVectorizer` instance
    #[staticmethod]
    #[cfg(feature = "bincode")]
    fn from_bytes(py: Python<'_>, bytes: &[u8]) -> PyResult<Self> {
        py.detach(|| {
            let inner = TfidfVectorizer::from_bytes(bytes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to deserialize vectorizer from bytes: {e}"
                ))
            })?;
            Ok(Self { inner })
        })
    }

    /// Deserialize the vectorizer from JSON string.
    ///
    /// Args:
    ///     json: A JSON string containing the serialized vectorizer
    ///
    /// Returns:
    ///     A new `RustTfidfVectorizer` instance
    #[staticmethod]
    #[cfg(feature = "serde")]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        py.detach(|| {
            let vectorizer = TfidfVectorizer::from_json(json).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to deserialize vectorizer from JSON: {e}"
                ))
            })?;
            Ok(Self { inner: vectorizer })
        })
    }
}

#[pymodule]
#[pyo3(name = "_slop_pre_processing_rust_bindings")]
fn slop_pre_processing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Python logging for Rust components
    pyo3_log::init();

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<RustVectorizerParams>()?;
    m.add_class::<RustTfidfVectorizer>()?;
    Ok(())
}
