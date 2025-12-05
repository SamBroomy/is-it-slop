import sys
from pathlib import Path

import polars as pl
from is_it_slop import MODEL_VERSION
from semver import Version

RETRAINED_MODEL_VERSION = Version.parse(MODEL_VERSION)

# Get from command line argument or set to True to force retraining
RETRAIN_VECTORIZER = "--force-retrain-vectorizer" in sys.argv or False
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
TRAIN_PATH = DATA_DIR / "curated_dataset_train.parquet"
TEST_PATH = DATA_DIR / "curated_dataset_test.parquet"

df = pl.scan_parquet(DATA_PATH)
df_train = pl.scan_parquet(TRAIN_PATH)
df_test = pl.scan_parquet(TEST_PATH)


MODEL_DIR = ROOT_DIR / "crates" / "is-it-slop" / "model_artifacts" / str(RETRAINED_MODEL_VERSION)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


VECTORIZER_JSON_PATH = MODEL_DIR / "tfidf_vectorizer.json"
VECTORIZER_BIN_PATH = MODEL_DIR / "tfidf_vectorizer.bin"
MODEL_ONNX_PATH = MODEL_DIR / "slop-classifier.onnx"
CLASSIFICATION_THRESHOLD_PATH = MODEL_DIR / "classification_threshold.txt"


__all__ = [
    "CLASSIFICATION_THRESHOLD_PATH",
    "DATA_PATH",
    "MODEL_DIR",
    "MODEL_ONNX_PATH",
    "PLOT_DIR",
    "RETRAINED_MODEL_VERSION",
    "RETRAIN_VECTORIZER",
    "TEST_PATH",
    "TRAIN_PATH",
    "VECTORIZER_BIN_PATH",
    "VECTORIZER_JSON_PATH",
    "df",
    "df_test",
    "df_train",
]
