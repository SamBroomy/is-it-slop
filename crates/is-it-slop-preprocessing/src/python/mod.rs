//! Python bindings for the preprocessing pipeline.
//!
//! This module exposes Rust preprocessing components to Python via `PyO3`.
//! All Rust types are prefixed with `Rust*` (e.g., `RustTfidfVectorizer`), and
//! Python wrappers remove the prefix (e.g., `TfidfVectorizer` in Python).
//!
//! # Exposed Components
//!
//! - **`RustTfidfVectorizer`**: TF-IDF vectorization with token n-grams
//! - **`RustTokenChunker`**: Token-based text chunking
//! - **`RustTextCleaner`**: Two-stage text cleaning (universal + dataset artifacts)
//! - **`RustVectorizerParams`**: Configuration for vectorizers
//!
//! # Python Usage
//!
//! ```python
//! from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams
//!
//! # Create vectorizer
//! params = VectorizerParams(min_df=10, max_df=0.9, sublinear_tf=True)
//! vectorizer, X = TfidfVectorizer.fit_transform(texts, params)
//!
//! # Save to file
//! vectorizer.save("vectorizer.rkyv")
//! ```
//!
//! # CSR Matrix Format
//!
//! Sparse matrices are returned as tuples `(data, indices, indptr, shape)` compatible
//! with `scipy.sparse.csr_matrix`:
//! ```python
//! from scipy.sparse import csr_matrix
//! data, indices, indptr, shape = vectorizer.transform(texts)
//! X = csr_matrix((data, indices, indptr), shape=shape)
//! ```

use std::{fs, path::Path};

use ahash::HashMap;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyTuple};

use crate::pre_processor::{
    DEFAULT_MAX_NGRAM, DEFAULT_MIN_NGRAM, TextCleaner, TfidfVectorizer, TfidfVectorizerBuilder,
    TokenChunker, VectorizerParams, reverse_tokenize as reverse_tokenize_,
    text_cleaner_for_inference, text_cleaner_for_training, tokenize as tokenize_,
};

#[allow(clippy::unsafe_derive_deserialize)]
/// Python wrapper for [`VectorizerParams`].
///
/// Configuration parameters for TF-IDF vectorization exposed to Python.
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
#[pyclass(from_py_object)]
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
    #[pyo3(signature = (min_df, max_df, sublinear_tf))]
    fn new(min_df: f32, max_df: f32, sublinear_tf: bool) -> Self {
        Self {
            ngram_range: (DEFAULT_MIN_NGRAM, DEFAULT_MAX_NGRAM),
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
            // self.ngram_range.0..=self.ngram_range.1,
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
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
            py.detach(|| TfidfVectorizer::fit_transform(texts.as_slice(), params.to_inner()));
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

    /// Transform pre-tokenized sequences to TF-IDF matrix
    #[allow(clippy::needless_pass_by_value)]
    pub fn vectorize_from_tokens<'py>(
        &self,
        py: Python<'py>,
        token_sequences: Vec<Vec<u32>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let tfidf_matrix = py.detach(|| self.inner.vectorize_from_tokens(&token_sequences));

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

    /// Serialize the vectorizer to bytes using rkyv format.
    /// Returns a bytes object that can be saved to disk or passed to `from_bytes`.
    /// Return the inner vectorizer serialized as bytes so it is compatible with Rust side.
    #[cfg(feature = "rkyv")]
    fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        py.detach(|| {
            self.inner
                .to_bytes()
                // Converting the archived bytes to a Vec<u8> for Python compatibility, it allocates
                // and you should use `save` method for large models to avoid this allocation.
                .map(rkyv::util::AlignedVec::into_vec)
                .map_err(|e| {
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

    /// Save the vectorizer to a path based on the file extension.
    /// Supports .rkyv (rkyv) and .json (serde).
    #[cfg(any(feature = "rkyv", feature = "serde"))]
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        let extension = path.extension().and_then(|ext| ext.to_str());

        py.detach(|| match extension {
            Some("rkyv") => {
                #[cfg(feature = "rkyv")]
                {
                    let bytes = self.inner.to_bytes().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to serialize vectorizer: {e}"
                        ))
                    })?;
                    fs::write(path, bytes.as_slice()).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to write vectorizer to {}: {e}",
                            path.display()
                        ))
                    })
                }
                #[cfg(not(feature = "rkyv"))]
                {
                    let _ = path;
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "rkyv serialization is not enabled",
                    ))
                }
            }
            Some("json") => {
                #[cfg(feature = "serde")]
                {
                    let json = self.inner.to_json().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to serialize vectorizer to JSON: {e}"
                        ))
                    })?;
                    fs::write(path, json).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to write vectorizer to {}: {e}",
                            path.display()
                        ))
                    })
                }
                #[cfg(not(feature = "serde"))]
                {
                    let _ = path;
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "JSON serialization is not enabled",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "File extension must be .rkyv or .json",
            )),
        })
    }

    /// Deserialize the vectorizer from bytes (rkyv format).
    ///
    /// Args:
    ///     bytes: A bytes object containing the serialized vectorizer
    ///
    /// Returns:
    ///     A new `RustTfidfVectorizer` instance
    #[staticmethod]
    #[cfg(feature = "rkyv")]
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

    /// Load a vectorizer from a path based on the file extension.
    /// Supports .rkyv (rkyv) and .json (serde).
    #[staticmethod]
    #[cfg(any(feature = "rkyv", feature = "bincode", feature = "serde"))]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = Path::new(path);
        let extension = path.extension().and_then(|ext| ext.to_str());

        py.detach(|| match extension {
            Some("rkyv") => {
                #[cfg(feature = "rkyv")]
                {
                    let bytes = fs::read(path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to read vectorizer from {}: {e}",
                            path.display()
                        ))
                    })?;
                    let inner = TfidfVectorizer::from_bytes(&bytes).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to deserialize vectorizer from bytes: {e}"
                        ))
                    })?;
                    Ok(Self { inner })
                }
                #[cfg(not(feature = "rkyv"))]
                {
                    let _ = path;
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "rkyv serialization is not enabled",
                    ))
                }
            }
            Some("bin") => {
                #[cfg(feature = "bincode")]
                {
                    let bytes = fs::read(path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to read vectorizer from {}: {e}",
                            path.display()
                        ))
                    })?;
                    let inner = TfidfVectorizer::from_bincode_bytes(&bytes).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to deserialize vectorizer from bincode bytes: {e}"
                        ))
                    })?;
                    Ok(Self { inner })
                }
                #[cfg(not(feature = "bincode"))]
                {
                    let _ = path;
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "bincode serialization is not enabled",
                    ))
                }
            }
            Some("json") => {
                #[cfg(feature = "serde")]
                {
                    let json = fs::read_to_string(path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to read vectorizer from {}: {e}",
                            path.display()
                        ))
                    })?;
                    let inner = TfidfVectorizer::from_json(&json).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to deserialize vectorizer from JSON: {e}"
                        ))
                    })?;
                    Ok(Self { inner })
                }
                #[cfg(not(feature = "serde"))]
                {
                    let _ = path;
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "JSON serialization is not enabled",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "File extension must be .rkyv, .bin, or .json",
            )),
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
        python.detach(|| {
            self.inner.clean_batch(&texts)
            // texts
            //     .into_par_iter()
            //     .map(|text| self.inner.clean(text))
            //     .collect()
        })
    }

    pub fn is_training_mode(&self) -> bool {
        matches!(self.mode, RustCleaningMode::Training)
    }
}

/// Python wrapper for [`TfidfVectorizerBuilder`].
///
/// Supports sklearn-style incremental training with `partial_fit()` for large datasets.
#[pyclass]
#[derive(Debug)]
pub struct RustTfidfVectorizerBuilder {
    inner: Option<TfidfVectorizerBuilder>,
}

#[pymethods]
impl RustTfidfVectorizerBuilder {
    /// Create a new builder with the given parameters.
    #[new]
    fn new(params: RustVectorizerParams) -> Self {
        Self {
            inner: Some(TfidfVectorizerBuilder::new(params.to_inner())),
        }
    }

    /// Process a batch of texts, updating document frequencies.
    ///
    /// Can be called multiple times with different batches.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if called after `finalize()`.
    #[allow(clippy::needless_pass_by_value)]
    fn partial_fit(&mut self, py: Python<'_>, texts: Vec<String>) -> PyResult<()> {
        let builder = self.inner.as_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has been finalized. Create a new TfidfVectorizerBuilder to process more data.",
            )
        })?;

        py.detach(|| {
            builder.partial_fit(&texts);
        });
        Ok(())
    }

    /// Finalize the vectorizer and return a fitted `TfidfVectorizer`.
    ///
    /// After calling this method, the builder is consumed and cannot be used again.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if called more than once.
    fn finalize(&mut self, py: Python<'_>) -> PyResult<RustTfidfVectorizer> {
        let builder = self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been finalized. Each builder can only be finalized once.",
            )
        })?;

        let vectorizer = py.detach(|| RustTfidfVectorizer {
            inner: builder.finalize(),
        });

        Ok(vectorizer)
    }

    /// Get current number of documents processed.
    #[getter]
    fn total_docs(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, TfidfVectorizerBuilder::total_docs)
    }

    /// Get current vocabulary size (before filtering).
    #[getter]
    fn raw_vocab_size(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, TfidfVectorizerBuilder::raw_vocab_size)
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        self.inner.as_ref().map_or_else(
            || "RustTfidfVectorizerBuilder(finalized)".to_string(),
            |builder| {
                format!(
                    "RustTfidfVectorizerBuilder(total_docs={}, raw_vocab_size={})",
                    builder.total_docs(),
                    builder.raw_vocab_size()
                )
            },
        )
    }
}

/// Extract combined features (document + chunk) for a batch of documents.
///
/// # Arguments
/// * `full_texts` - List of full document texts
/// * `chunk_tokens_batch` - List of chunked token sequences (list of list of list of tokens)
///
/// # Returns
/// Numpy array of shape (`total_chunks`, 9) with combined features
///
/// # Example
/// ```python
/// from is_it_slop_preprocessing import extract_combined_batch, tokenize
///
/// texts = ["First document", "Second document"]
/// tokens = [tokenize(t) for t in texts]
/// # Assume we have chunks for each document
/// features = extract_combined_batch(texts, chunks)
/// ```
#[cfg(feature = "statistical-features")]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn rust_extract_combined_batch(
    py: Python<'_>,
    full_texts: Vec<String>,
    chunk_tokens_batch: Vec<Vec<Vec<u32>>>,
) -> Bound<'_, numpy::PyArray2<f32>> {
    use crate::pre_processor::extract_combined_batch;

    // Release GIL during computation
    let features = py.detach(|| extract_combined_batch(&full_texts, &chunk_tokens_batch));

    // Convert to numpy array with GIL
    features.to_pyarray(py)
}

/// Encode text into BPE token IDs
/// This is a utility function to expose tokenization to Python.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tokenize(py: Python<'_>, text: Vec<String>) -> Vec<Vec<u32>> {
    py.detach(|| tokenize_(&text))
}

/// Decode BPE token IDs back to text
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn reverse_tokenize(py: Python<'_>, tokens: Vec<u32>) -> String {
    py.detach(|| reverse_tokenize_(&tokens))
}

#[pymodule]
#[pyo3(name = "_is_it_slop_preprocessing_rust_bindings")]
fn is_it_slop_preprocessing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize pyo3_log to bridge Rust tracing to Python logging
    // This respects Python's logging level configuration
    let _ = pyo3_log::try_init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<RustVectorizerParams>()?;
    m.add_class::<RustTfidfVectorizer>()?;
    m.add_class::<RustTfidfVectorizerBuilder>()?;

    m.add_class::<RustTokenChunker>()?;

    m.add_class::<RustCleaningMode>()?;
    m.add_class::<RustTextCleaner>()?;

    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_tokenize, m)?)?;
    #[cfg(feature = "statistical-features")]
    m.add_function(wrap_pyfunction!(rust_extract_combined_batch, m)?)?;

    Ok(())
}
