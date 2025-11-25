"""Internal interface and types for the `slop_pre_processing` package.

This module provides the main interface to the Rust bindings for text vectorization
using TF-IDF.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

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


class TfidfVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF text vectorizer compatible with scikit-learn.

    This class provides methods to fit a TF-IDF vectorizer to a corpus of text
    documents and to transform new documents into TF-IDF feature vectors.

    The vectorizer uses Rust implementation for fast tokenization and vectorization.

    Parameters
    ----------
    ngram_range : tuple[int, int], default=(3, 5)
        The range of n-gram sizes to extract. Note: these are token n-grams
        (sequences of consecutive tokens), not character or word n-grams.

    min_df : int, default=10
        Minimum document frequency for a token to be included in the vocabulary.
        Tokens that appear in fewer than `min_df` documents will be ignored.

    Examples
    --------
    >>> from slop_pre_processing import TfidfVectorizer
    >>> vectorizer = TfidfVectorizer(ngram_range=(3, 5), min_df=10)
    >>> X_train = vectorizer.fit_transform(train_texts)
    >>> X_test = vectorizer.transform(test_texts)

    """

    def __init__(self, ngram_range: tuple[int, int] = (3, 5), min_df: int = 10) -> None:
        """Initialize the TF-IDF vectorizer with parameters.

        Args:
            ngram_range: The range of n-gram sizes to extract (token n-grams).
            min_df: Minimum document frequency threshold.

        """
        self._parameters: VectorizerParams = VectorizerParams(ngram_range, min_df)
        self._vectorizer: RustTfidfVectorizer | None = None
        self._fitted: bool = False

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

    def fit(self, X: list[str] | NDArray[np.str_], _y: Any = None) -> Self:  # noqa: ANN401, N803
        """Fit the TF-IDF vectorizer to the provided texts.

        Args:
            X: A list or 1D NumPy array of text documents to fit the vectorizer.
            _y: Ignored. Present for sklearn API compatibility.

        Returns:
            self: The fitted vectorizer instance.

        """
        texts = self._validate_texts(X)

        # Implicitly calls fit on the Rust side when creating the RustTfidfVectorizer
        self._vectorizer = RustTfidfVectorizer(texts, self._parameters.as_rust())
        self._fitted = True
        return self

    def transform(self, X: list[str] | NDArray[np.str_]) -> csr_matrix:  # noqa: N803
        """Transform new texts into TF-IDF feature vectors.

        Args:
            X: A list or 1D NumPy array of text documents to transform.

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.

        """
        if not self._fitted or self._vectorizer is None:
            msg = "The vectorizer must be fitted before calling transform. Call fit() first."
            raise RuntimeError(msg)

        texts = self._validate_texts(X)

        shape: tuple[int, int]
        data: NDArray[np.float64]
        indices: NDArray[np.uintp]
        indptr: NDArray[np.uintp]

        shape, data, indices, indptr = self._vectorizer.transform(texts)  # type: ignore[assignment]
        # Build the scipy sparse matrix
        return csr_matrix((data, indices, indptr), shape=shape)

    def fit_transform(self, X: Any, y: Any = None, **_fit_params) -> Any:  # noqa: ANN003, ANN401, N803
        """Fit the vectorizer and transform the input texts in one step.

        Args:
            X: Input data (list of strings, 1D NumPy array, or compatible type).
            y: Ignored. Present for sklearn API compatibility.
            **fit_params: Additional fit parameters (ignored).

        Returns:
            A SciPy CSR sparse matrix containing the TF-IDF feature vectors.

        """
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT001, FBT002
        """Get parameters for this estimator.

        Args:
            deep: Ignored (no nested estimators).

        Returns:
            Dictionary of parameter names to values.

        """
        return {"ngram_range": self._parameters.ngram_range, "min_df": self._parameters.min_df}

    def set_params(self, **params: Any) -> Self:  # noqa: ANN401
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: The estimator instance.

        Raises:
            ValueError: If an invalid parameter is provided.

        """
        valid_params = {"ngram_range", "min_df"}
        for key, value in params.items():
            if key not in valid_params:
                msg = f"Invalid parameter '{key}'. Valid parameters are: {valid_params}"
                raise ValueError(msg)
            setattr(self, key, value)
        return self

    @property
    def fitted(self) -> bool:
        """Return whether the vectorizer has been fitted."""
        return self._fitted

    @property
    def num_features(self) -> int:
        """Return the number of features (vocabulary size) of the fitted vectorizer."""
        if not self._fitted or self._vectorizer is None:
            msg = "The vectorizer must be fitted before accessing num_features."
            raise RuntimeError(msg)
        return self._vectorizer.num_features

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary of the fitted vectorizer as a mapping of terms to indices."""
        if not self._fitted or self._vectorizer is None:
            msg = "The vectorizer must be fitted before accessing vocabulary."
            raise RuntimeError(msg)
        return self._vectorizer.vocabulary

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling.

        Returns:
            Dictionary containing the vectorizer state.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.

        """
        if not self._fitted or self._vectorizer is None:
            msg = "Cannot pickle an unfitted vectorizer."
            raise RuntimeError(msg)

        return {
            "ngram_range": self._parameters.ngram_range,
            "min_df": self._parameters.min_df,
            "vectorizer_bytes": bytes(self._vectorizer.to_bytes()),
            "fitted": self._fitted,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state from unpickling.

        Args:
            state: Dictionary containing the vectorizer state.

        """
        ngram_range: tuple[int, int] = state["ngram_range"]
        min_df: int = state["min_df"]
        fitted: bool = state["fitted"]
        rust_vectorizer_bytes: bytes = state["vectorizer_bytes"]
        rust_vectorizer: RustTfidfVectorizer = RustTfidfVectorizer.from_bytes(list(rust_vectorizer_bytes))
        self._parameters = VectorizerParams(ngram_range, min_df)
        self._fitted = fitted
        self._vectorizer = rust_vectorizer

    def save(self, path: str | Path, serialization_format: str = "bincode") -> None:
        """Save the fitted vectorizer to disk.

        Args:
            path: File path to save the vectorizer.
            serialization_format: Serialization format, either 'bincode' (binary, compact) or 'json' (human-readable).

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
            ValueError: If an invalid format is specified.

        """
        if not self._fitted or self._vectorizer is None:
            msg = "Cannot save an unfitted vectorizer."
            raise RuntimeError(msg)

        path = Path(path)

        # Create a wrapper dict with params and vectorizer data
        save_data = {
            "ngram_range": self._parameters.ngram_range,
            "min_df": self._parameters.min_df,
            "vectorizer_format": serialization_format,
        }

        if serialization_format == "bincode":
            save_data["vectorizer_bytes"] = bytes(self._vectorizer.to_bytes())
            # Save as JSON wrapper with base64-encoded bincode inside
            save_data["vectorizer_bytes"] = base64.b64encode(save_data["vectorizer_bytes"]).decode("ascii")
            path.write_text(json.dumps(save_data), encoding="utf-8")
        elif serialization_format == "json":
            save_data["vectorizer_json"] = self._vectorizer.to_json()
            # Save as JSON wrapper with nested JSON
            path.write_text(json.dumps(save_data), encoding="utf-8")
        else:
            msg = f"Invalid format '{serialization_format}'. Must be 'bincode' or 'json'."
            raise ValueError(msg)

    def save_raw_bincode(self, path: str | Path) -> None:
        """Save raw bincode bytes for direct Rust consumption.

        This method saves the vectorizer as raw bincode bytes without JSON wrapping,
        which can be loaded directly in Rust using TfidfVectorizer::from_bytes().

        Args:
            path: File path to save the raw bincode bytes.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.

        """
        if not self._fitted or self._vectorizer is None:
            msg = "Cannot save an unfitted vectorizer."
            raise RuntimeError(msg)

        path = Path(path)
        vectorizer_bytes = bytes(self._vectorizer.to_bytes())
        path.write_bytes(vectorizer_bytes)

    @classmethod
    def load(cls, path: str | Path, serialization_format: str = "bincode") -> Self:
        """Load a fitted vectorizer from disk.

        Args:
            path: File path to load the vectorizer from.
            serialization_format: Serialization format, either 'bincode' (binary, compact) or 'json' (human-readable).

        Returns:
            A loaded TfidfVectorizer instance.

        Raises:
            ValueError: If an invalid format is specified.
            FileNotFoundError: If the file does not exist.

        """
        path = Path(path)

        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Load the wrapper JSON
        json_str = path.read_text(encoding="utf-8")
        save_data = json.loads(json_str)

        # Extract params
        ngram_range = tuple(save_data["ngram_range"])
        min_df = save_data["min_df"]
        stored_format = save_data.get("vectorizer_format", serialization_format)

        # Load the Rust vectorizer based on stored format
        if stored_format == "bincode":
            if "vectorizer_bytes" not in save_data:
                msg = "Invalid save file: missing vectorizer_bytes"
                raise ValueError(msg)
            # Decode base64-encoded bincode data
            vectorizer_bytes = base64.b64decode(save_data["vectorizer_bytes"])
            rust_vectorizer = RustTfidfVectorizer.from_bytes(list(vectorizer_bytes))
        elif stored_format == "json":
            if "vectorizer_json" not in save_data:
                msg = "Invalid save file: missing vectorizer_json"
                raise ValueError(msg)
            rust_vectorizer = RustTfidfVectorizer.from_json(save_data["vectorizer_json"])
        else:
            msg = f"Invalid format in save file: '{stored_format}'. Must be 'bincode' or 'json'."
            raise ValueError(msg)

        # Create instance with correct params
        instance = cls(ngram_range=ngram_range, min_df=min_df)
        instance._vectorizer = rust_vectorizer
        instance._fitted = True
        return instance

    @classmethod
    def load_raw_bincode(cls, path: str | Path, ngram_range: tuple[int, int], min_df: int) -> Self:
        path = Path(path)

        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        vectorizer_bytes = path.read_bytes()
        rust_vectorizer = RustTfidfVectorizer.from_bytes(list(vectorizer_bytes))

        # Create instance with correct params
        instance = cls(ngram_range=ngram_range, min_df=min_df)
        instance._vectorizer = rust_vectorizer
        instance._fitted = True
        return instance

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:  # noqa: N803
        fitted_str = "fitted" if self._fitted else "unfitted"
        return f"TfidfVectorizer(ngram_range={self._parameters.ngram_range}, min_df={self._parameters.min_df}, {fitted_str})"

    def __str__(self) -> str:
        if self._vectorizer is not None:
            return self._vectorizer.__str__()
        return self.__repr__()
