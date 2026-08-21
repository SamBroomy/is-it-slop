#!/usr/bin/env python
"""Break the production-vs-Magika accuracy gap down by document length and chunking behaviour.

Answers where the two models actually differ, rather than which headline number is bigger. Reads the
document-level probabilities saved by `compare_models.py`, so it needs no model inference.

    uv run python notebooks/analyse_comparison.py
"""

from __future__ import annotations

# ruff: file-ignore[assert]
import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl
from is_it_slop_preprocessing import TokenChunker, tokenize
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data" / "3.0.0"
PROBS_DIR = ROOT_DIR / "model_artifacts" / "magika_style"

LENGTH_EDGES = (0, 50, 100, 150, 200, 300, 500, 1000, 2000, np.iinfo(np.int64).max)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probs", type=Path, default=PROBS_DIR / "test-probs-tokens.npz")
    p.add_argument("--prod-threshold", type=float, default=0.3954)
    p.add_argument("--magika-threshold", type=float, default=0.2054)
    p.add_argument("--min-bucket", type=int, default=20, help="skip buckets thinner than this")
    args = p.parse_args()

    d = np.load(args.probs, allow_pickle=False)
    prod, magika, y = d["production"], d["magika"], d["label"]

    texts = pl.scan_parquet(DATA_DIR / "test.parquet").select("text").collect().to_series().to_list()
    assert len(texts) == len(y), f"{len(texts)} texts vs {len(y)} labels"

    tokens = tokenize(texts)
    chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)
    chunked = chunker.chunk_batch(tokens)

    n_tokens = np.array([len(t) for t in tokens])
    n_chunks = np.array([len(c) for c in chunked])
    # The emit loop walks one step past its own plan, so the final chunk is sometimes wholly
    # contained in its predecessor and the document tail is counted twice by the weighted mean.
    redundant = np.array([len(c) > 1 and set(c[-1]) <= set(c[-2]) for c in chunked])

    prod_hit = (prod >= args.prod_threshold).astype(np.int8) == y
    magika_hit = (magika >= args.magika_threshold).astype(np.int8) == y

    def report(rows: list[dict], title: str) -> None:
        logger.info(f"{title}\n{pl.DataFrame(rows)}")

    rows = []
    for lo, hi in itertools.pairwise(LENGTH_EDGES):
        mask = (n_tokens >= lo) & (n_tokens < hi)
        if mask.sum() < args.min_bucket:
            continue
        rows.append({
            "tokens": f"{lo}-{'inf' if hi == LENGTH_EDGES[-1] else hi}",
            "n": int(mask.sum()),
            "chunks": round(float(n_chunks[mask].mean()), 2),
            "production": round(float(prod_hit[mask].mean()), 4),
            "magika": round(float(magika_hit[mask].mean()), 4),
            "delta": round(float(magika_hit[mask].mean() - prod_hit[mask].mean()), 4),
        })
    report(rows, "accuracy by document length")

    rows = [
        {
            "group": name,
            "n": int(mask.sum()),
            "production": round(float(prod_hit[mask].mean()), 4),
            "magika": round(float(magika_hit[mask].mean()), 4),
            "delta": round(float(magika_hit[mask].mean() - prod_hit[mask].mean()), 4),
        }
        for name, mask in [("single chunk", n_chunks == 1), ("multi chunk", n_chunks > 1)]
    ]
    report(rows, "accuracy by chunk count")

    # Length-controlled, because whether a redundant tail appears is itself decided by length.
    rows = []
    for lo, hi in [(150, 200), (200, 300), (300, 500), (500, 1000), (1000, 2000)]:
        band = (n_tokens >= lo) & (n_tokens < hi) & (n_chunks > 1)
        red, clean = band & redundant, band & ~redundant
        if red.sum() < 50 or clean.sum() < 50:
            continue
        rows.append({
            "tokens": f"{lo}-{hi}",
            "n_redundant": int(red.sum()),
            "acc_redundant": round(float(prod_hit[red].mean()), 4),
            "n_clean": int(clean.sum()),
            "acc_clean": round(float(prod_hit[clean].mean()), 4),
            "delta": round(float(prod_hit[red].mean() - prod_hit[clean].mean()), 4),
        })
    report(rows, "production accuracy: redundant trailing chunk vs clean tiling, within length bands")


if __name__ == "__main__":
    main()
