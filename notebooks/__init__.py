"""Shared constants for the data pipeline (`dataset_curation.py`).

Importing this module has a side effect: it creates `data/<version>/`, where the curated dataset and
its train/test/validation splits are written.
"""

import sys
from pathlib import Path
from typing import Final

from semver import Version

SEED: Final[int] = 42
MODEL_VERSION = "3.0.0"
RETRAINED_MODEL_VERSION = Version.parse(MODEL_VERSION)

# Every run produces a new dataset version: minor by default, major with --bump-major.
RETRAINED_MODEL_VERSION = RETRAINED_MODEL_VERSION.bump_minor()
if "--bump-major" in sys.argv:
    RETRAINED_MODEL_VERSION = RETRAINED_MODEL_VERSION.bump_major()

ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR / "data" / str(RETRAINED_MODEL_VERSION)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "curated_dataset.parquet"
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

__all__ = [
    "DATA_DIR",
    "DATA_PATH",
    "RETRAINED_MODEL_VERSION",
    "SEED",
    "TEST_PATH",
    "TRAIN_PATH",
    "VALIDATION_PATH",
]
