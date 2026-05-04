import sys
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import polars as pl

# from is_it_slop import MODEL_VERSION
from semver import Version

SEED: Final[int] = 42
MODEL_VERSION = "2.1.0"
RETRAINED_MODEL_VERSION = Version.parse(MODEL_VERSION)

# Get from command line argument or set to True to force retraining
RETRAIN_VECTORIZER = "--force-retrain-vectorizer" in sys.argv or True
if RETRAIN_VECTORIZER:
    RETRAINED_MODEL_VERSION = RETRAINED_MODEL_VERSION.bump_minor()
if "--bump-major" in sys.argv:
    RETRAINED_MODEL_VERSION = RETRAINED_MODEL_VERSION.bump_major()

ROOT_DIR = Path(__file__).parent.parent.resolve()

PLOT_DIR = ROOT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


DATA_DIR = ROOT_DIR / "data" / str(RETRAINED_MODEL_VERSION)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "curated_dataset.parquet"
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

df = pl.scan_parquet(DATA_PATH)
df_train = pl.scan_parquet(TRAIN_PATH)
df_test = pl.scan_parquet(TEST_PATH)
df_validation = pl.scan_parquet(VALIDATION_PATH)


MODEL_DIR = ROOT_DIR / "crates" / "is-it-slop" / "model_artifacts" / str(RETRAINED_MODEL_VERSION)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


VECTORIZER_JSON_PATH = MODEL_DIR / "tfidf_vectorizer.json"
VECTORIZER_BIN_PATH = MODEL_DIR / "tfidf_vectorizer.rkyv"
MODEL_ONNX_PATH = MODEL_DIR / "slop-classifier.onnx"
CLASSIFICATION_THRESHOLD_PATH = MODEL_DIR / "classification_threshold.txt"
CHUNK_CLASSIFICATION_THRESHOLD_PATH = MODEL_DIR / "chunk_classification_threshold.txt"
CHUNKER_CONFIG_PATH = MODEL_DIR / "token_chunker_config.json"


class ProbabilisticClassifier(Protocol):
    """Base protocol for classifiers with predict_proba."""

    def predict_proba(self, X) -> np.ndarray: ...  # noqa: ANN001
    def predict(self, X) -> np.ndarray: ...  # noqa: ANN001
    def fit(self, X, y) -> "ProbabilisticClassifier": ...  # noqa: ANN001


class LinearClassifier(ProbabilisticClassifier, Protocol):
    """Protocol for linear classifiers with coefficients."""

    @property
    def coef_(self) -> np.ndarray: ...


__all__ = [
    "CHUNKER_CONFIG_PATH",
    "CHUNK_CLASSIFICATION_THRESHOLD_PATH",
    "CLASSIFICATION_THRESHOLD_PATH",
    "DATA_PATH",
    "MODEL_DIR",
    "MODEL_ONNX_PATH",
    "PLOT_DIR",
    "RETRAINED_MODEL_VERSION",
    "RETRAIN_VECTORIZER",
    "SEED",
    "TEST_PATH",
    "TRAIN_PATH",
    "VALIDATION_PATH",
    "VECTORIZER_BIN_PATH",
    "VECTORIZER_JSON_PATH",
    "LinearClassifier",
    "ProbabilisticClassifier",
    "df",
    "df_test",
    "df_train",
    "df_validation",
]
