"""Internal interface and types for the `slop_pre_processing` package.

This module provides the main interface to the Rust bindings for text vectorization
using TF-IDF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ._rust_bindings import RustTfidfVectorizer, RustVectorizerParams, __version__  # type: ignore[import]  # noqa: F401


class VectorizerParams:
    """Parameters for configuring the text vectorizer."""

    def __init__(self, ngram_range: tuple[int, int], min_df: int) -> None:
        self._inner = RustVectorizerParams(ngram_range, min_df)

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self._inner.ngram_range

    @property
    def min_df(self) -> int:
        return self._inner.min_df

    def __repr__(self) -> str:
        return f"VectorizerParams(ngram_range={self.ngram_range}, min_df={self.min_df})"

    def as_rust(self) -> RustVectorizerParams:
        """Return the underlying Rust object."""
        return self._inner


class TfidfVectorizer:
    """TF-IDF text vectorizer.

    This class provides methods to fit a TF-IDF vectorizer to a corpus of text
    documents and to transform new documents into TF-IDF feature vectors.
    """

    def __init__(self, vectorizer: RustTfidfVectorizer, *, fitted: bool) -> None:
        """Internal constructor. Use `fit` class method to create an instance."""
        self._vectorizer: RustTfidfVectorizer = vectorizer
        self._fitted: bool = fitted

    @staticmethod
    def _validate_texts(texts: list[str] | NDArray[np.str_]) -> None:
        """Validate the input texts for fitting or transforming.

        Validated here so we dont pass invalid data to the Rust side.
        """
        if isinstance(texts, np.ndarray):
            if texts.dtype.kind not in {"U", "S"}:
                msg = "NumPy array must have dtype 'str' or 'unicode'."
                raise TypeError(msg)
            if texts.ndim != 1:
                msg = "Input NumPy array must be 1-dimensional."
                raise ValueError(msg)

        elif not isinstance(texts, list):
            msg = "Input must be a list of strings or a 1D NumPy array of strings."
            raise TypeError(msg)

        elif not all(isinstance(t, str) for t in texts):
            msg = "All elements in the input list must be strings."
            raise TypeError(msg)

    @classmethod
    def fit(cls, texts: list[str] | NDArray[np.str_], params: VectorizerParams) -> Self:
        """Fit the TF-IDF vectorizer to the provided texts.

        Args:
            texts: A list or 1D NumPy array of text documents to fit the vectorizer.
            params: An instance of `VectorizerParams` specifying the vectorizer configuration.

        Returns:
            An instance of `TfidfVectorizer` fitted to the provided texts.

        """
        cls._validate_texts(texts)
        # Implicitly calls fit on the Rust side when creating the RustTfidfVectorizer
        vectorizer = RustTfidfVectorizer(texts, params.as_rust())
        return cls(vectorizer, fitted=True)

    def transform(self, texts: list[str] | NDArray[np.str_]) -> csr_matrix:
        """Transform new texts into TF-IDF feature vectors.

        Args:
            texts: A list or 1D NumPy array of text documents to transform.

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        """
        if not self._fitted:
            msg = "The vectorizer must be fitted before calling transform."
            raise RuntimeError(msg)

        self._validate_texts(texts)

        shape: tuple[int, int]
        data: NDArray[np.float64]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = self._vectorizer.transform(texts)  # type: ignore[assignment]
        # Build the scipy sparse matrix
        return csr_matrix((data, indices, indptr), shape=shape)

    def num_features(self) -> int:
        """Return the number of features (vocabulary size) of the fitted vectorizer."""
        if not self._fitted:
            msg = "The vectorizer must be fitted before accessing num_features."
            raise RuntimeError(msg)
        return self._vectorizer.num_features

    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary of the fitted vectorizer as a list of terms."""
        if not self._fitted:
            msg = "The vectorizer must be fitted before accessing vocabulary."
            raise RuntimeError(msg)
        return self._vectorizer.vocabulary

    def __repr__(self) -> str:
        return f"TfidfVectorizer({'fitted' if self._fitted else 'unfitted'}, vectorizer={self._vectorizer.__repr__()})"

    def __str__(self) -> str:
        return self._vectorizer.__str__()
