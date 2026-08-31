#!/usr/bin/env python
"""Export the encoding-contract fixture for the Rust training crate.

The Rust `DocumentEncoder` is pinned by golden vectors: raw texts, the vocabulary, and the exact
grids they must encode to (`crates/is-it-slop-training/tests/encoder_golden.rs`). The grids came
from the original Python pipeline's cache, which defined the encoding every trained artifact
depends on; the fixture's job now is to freeze that contract, not to track Python. Changing the
encoding deliberately means regenerating this fixture and versioning the artifact format.

    uv run --with safetensors python notebooks/export_encoder_fixture.py

Reads the v8192 cache and bundle; the outputs are committed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from safetensors.numpy import save_file

ROOT = Path(__file__).parent.parent.resolve()
# Caches and bundles live in the main checkout; a worktree can point back at it.
ARTIFACTS = Path(os.environ.get("ARTIFACTS_ROOT", ROOT)).resolve()
BUNDLE = ARTIFACTS / "model_artifacts" / "hierarchical" / "c192-k8-v8192-d48-g2-wd0.01-p6-cos"
CACHE = ARTIFACTS / "data" / "3.0.0" / "hierarchical_cache" / "test-c192-k8-v8192.npy"
OUT = ROOT / "crates" / "is-it-slop-training" / "tests" / "fixtures"

N_DOCS = 16
MAX_CHUNKS = 8
CHUNK_SIZE = 192


def main() -> None:
    feats = np.memmap(CACHE, dtype=np.uint16, mode="r").reshape(-1, MAX_CHUNKS, CHUNK_SIZE)
    idx = np.linspace(0, len(feats) - 1, N_DOCS, dtype=np.int64)
    grids = np.asarray(feats[idx], dtype=np.int64).copy()

    texts = (
        pl.scan_parquet(ARTIFACTS / "data" / "3.0.0" / "test.parquet")
        .select("text")
        .collect()["text"]
        .to_list()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "encoder_texts.json").write_text(
        json.dumps([texts[i] for i in idx], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    save_file({"ids": grids}, OUT / "encoder_grids.safetensors")
    shutil.copyfile(BUNDLE / "token_vocabulary.safetensors", OUT / "vocabulary.safetensors")
    print(f"fixture -> {OUT} ({N_DOCS} docs)")


if __name__ == "__main__":
    main()
