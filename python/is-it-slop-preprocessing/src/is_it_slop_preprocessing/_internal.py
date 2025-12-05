"""Internal interface and types for the `is_it_slop_preprocessing` package.

This module provides the main interface to the Rust bindings for text vectorization
using TF-IDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ._is_it_slop_preprocessing_rust_bindings import RustTfidfVectorizer, RustVectorizerParams, __version__

__all__ = ["TfidfVectorizer", "VectorizerParams", "__version__"]


class VectorizerParams:
    """Parameters for configuring the text vectorizer.

    Both min_df and max_df can be specified as either:
    - A float in (0.0, 1.0) representing a proportion of documents
    - A float >= 1.0 representing an absolute document count

    Args:
        ngram_range: Tuple of (min_n, max_n) for n-gram range
        min_df: Minimum document frequency (proportion or count)
        max_df: Maximum document frequency (proportion or count)
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf))

    Examples:
        min_df=0.05  # Filter terms appearing in < 5% of documents
        min_df=10.0  # Filter terms appearing in < 10 documents
        max_df=0.9   # Filter terms appearing in > 90% of documents
        max_df=500.0 # Filter terms appearing in > 500 documents
        sublinear_tf=True  # Use log scaling for term frequency

    """

    __slots__ = ("_inner",)

    def __init__(
        self, *, ngram_range: tuple[int, int], min_df: float, max_df: float, sublinear_tf: bool = True
    ) -> None:
        self._inner = RustVectorizerParams(ngram_range, min_df, max_df, sublinear_tf)

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self._inner.ngram_range

    @property
    def min_df(self) -> float:
        return self._inner.min_df

    @property
    def max_df(self) -> float:
        return self._inner.max_df

    @property
    def sublinear_tf(self) -> bool:
        return self._inner.sublinear_tf

    def __repr__(self) -> str:
        return (
            f"VectorizerParams(ngram_range={self.ngram_range}, min_df={self.min_df}, "
            f"max_df={self.max_df}, sublinear_tf={self.sublinear_tf})"
        )

    def as_rust(self) -> RustVectorizerParams:
        """Return the underlying Rust object."""
        return self._inner


class TfidfVectorizer:
    """TF-IDF text vectorizer with Rust-backed implementation.

    This vectorizer is always fitted - you cannot create an unfitted instance.
    Use the static `fit()` method to create a fitted vectorizer from training texts.

    Examples
    --------
    >>> from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams
    >>> params = VectorizerParams(ngram_range=(3, 5), min_df=10)
    >>> vectorizer = TfidfVectorizer.fit(train_texts, params)
    >>> X_test = vectorizer.transform(test_texts)

    """

    __slots__ = ("_parameters", "_vectorizer")

    def __init__(self, params: VectorizerParams, rust_vectorizer: RustTfidfVectorizer) -> None:
        """Private constructor. Use TfidfVectorizer.fit() to create instances.

        Args:
            params: VectorizerParams instance containing vectorizer configuration.
            rust_vectorizer: Fitted RustTfidfVectorizer instance.

        """
        self._parameters = params
        self._vectorizer = rust_vectorizer

    @staticmethod
    def fit(texts: list[str] | NDArray[np.str_] | NDArray[np.object_], params: VectorizerParams) -> TfidfVectorizer:
        """Fit a new TF-IDF vectorizer to the provided texts.

        Args:
            texts: Training texts to fit the vectorizer.
            params: Vectorizer parameters.

        Returns:
            A fitted TfidfVectorizer instance.

        """
        validated_texts = TfidfVectorizer._validate_texts(texts)
        rust_vectorizer = RustTfidfVectorizer(validated_texts, params.as_rust())
        return TfidfVectorizer(params, rust_vectorizer)

    @staticmethod
    def fit_transform(
        texts: list[str] | NDArray[np.str_] | NDArray[np.object_], params: VectorizerParams
    ) -> tuple[TfidfVectorizer, csr_matrix]:
        """Fit a new TF-IDF vectorizer and transform the texts in one optimized step.

        This is more efficient than calling fit() followed by transform() because
        it only computes n-grams once instead of twice.

        Args:
            texts: Training texts to fit the vectorizer and transform.
            params: Vectorizer parameters.

        Returns:
            A tuple of (fitted_vectorizer, transformed_matrix).

        """
        validated_texts = TfidfVectorizer._validate_texts(texts)
        rust_vectorizer, transform_result = RustTfidfVectorizer.fit_transform(validated_texts, params.as_rust())

        shape: tuple[int, int]
        data: NDArray[np.float32]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = transform_result  # type: ignore[assignment]
        transformed_matrix = csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

        vectorizer = TfidfVectorizer(params, rust_vectorizer)
        return vectorizer, transformed_matrix

    @staticmethod
    def _validate_texts(texts: list[str] | NDArray[np.str_] | NDArray[np.object_]) -> list[str]:
        """Validate the input texts for fitting or transforming.

        Validated here so we dont pass invalid data to the Rust side.

        Returns:
            The texts as a list of strings.

        """
        if isinstance(texts, np.ndarray):
            if texts.dtype.kind == "O":
                texts = texts.astype(str)
            if texts.dtype.kind not in {"U", "S"}:
                msg = "NumPy array must have dtype 'str' or 'unicode'."
                raise TypeError(msg)
            if texts.ndim != 1:
                msg = "Input NumPy array must be 1-dimensional."
                raise ValueError(msg)
            return texts.tolist()

        if not isinstance(texts, list):
            msg = "Input must be a list of strings or a 1D NumPy array of strings."
            raise TypeError(msg)

        if not all(isinstance(t, str) for t in texts):
            msg = "All elements in the input list must be strings."
            raise TypeError(msg)

        return texts

    def transform(self, texts: list[str] | NDArray[np.str_]) -> csr_matrix:
        """Transform new texts into TF-IDF feature vectors.

        Args:
            texts: Texts to transform.

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        """
        validated_texts = self._validate_texts(texts)

        shape: tuple[int, int]
        data: NDArray[np.float32]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = self._vectorizer.transform(validated_texts)  # type: ignore[assignment]
        return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

    @property
    def params(self) -> VectorizerParams:
        """Return the vectorizer parameters."""
        return self._parameters

    @property
    def num_features(self) -> int:
        """Return the number of features (vocabulary size) of the fitted vectorizer."""
        return self._vectorizer.num_features

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary of the fitted vectorizer as a mapping of terms to indices."""
        return self._vectorizer.vocabulary

    def __getstate__(self) -> dict[str, bytes]:
        """Get state for pickling.

        Returns:
            Dictionary containing the serialized vectorizer.

        """
        return {"vectorizer_bytes": bytes(self._vectorizer.to_bytes())}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        """Set state from unpickling.

        Args:
            state: Dictionary containing the serialized vectorizer.

        """
        rust_vectorizer_bytes: bytes = state["vectorizer_bytes"]
        self._vectorizer = RustTfidfVectorizer.from_bytes(rust_vectorizer_bytes)
        self._parameters = VectorizerParams(
            ngram_range=self._vectorizer.params.ngram_range,
            min_df=self._vectorizer.params.min_df,
            max_df=self._vectorizer.params.max_df,
            sublinear_tf=self._vectorizer.params.sublinear_tf,
        )

    def save(self, path: str | Path) -> None:
        """Save raw bincode bytes for direct Rust consumption.

        This method saves the vectorizer as raw bincode bytes without JSON wrapping,
        which can be loaded directly in Rust using TfidfVectorizer::from_bytes().

        Args:
            path: File path to save the raw bincode bytes.

        """
        path = Path(path)

        if path.suffix == ".json":
            path.write_text(self._vectorizer.to_json(), encoding="utf-8")
        elif path.suffix == ".bin":
            path.write_bytes(bytes(self._vectorizer.to_bytes()))
        else:
            msg = "File extension must be either .json or .bin"
            raise ValueError(msg)

    @classmethod
    def load(cls, path: str | Path) -> TfidfVectorizer:
        """Load a fitted vectorizer from raw bincode bytes.

        Args:
            path: File path to load the raw bincode bytes from.

        Returns:
            A loaded TfidfVectorizer instance.

        Raises:
            FileNotFoundError: If the file does not exist.

        """
        path = Path(path)

        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        if path.suffix == ".json":
            rust_vectorizer = RustTfidfVectorizer.from_json(path.read_text(encoding="utf-8"))
        elif path.suffix == ".bin":
            rust_vectorizer = RustTfidfVectorizer.from_bytes(path.read_bytes())
        else:
            msg = "File extension must be either .json or .bin"
            raise ValueError(msg)
        params = VectorizerParams(
            ngram_range=rust_vectorizer.params.ngram_range,
            min_df=rust_vectorizer.params.min_df,
            max_df=rust_vectorizer.params.max_df,
            sublinear_tf=rust_vectorizer.params.sublinear_tf,
        )
        return cls(params, rust_vectorizer)

    def __repr__(self) -> str:
        return (
            f"TfidfVectorizer(ngram_range={self._parameters.ngram_range}, "
            f"min_df={self._parameters.min_df}, max_df={self._parameters.max_df}, "
            f"num_features={self.num_features})"
        )

    def __str__(self) -> str:
        return self._vectorizer.__str__()
