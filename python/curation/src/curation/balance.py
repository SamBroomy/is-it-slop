"""Balance formatting between the classes, so a model cannot separate them on layout alone.

Authors of AI text and of human text differ in how they format (paragraph breaks, markdown, dashes, length) for
reasons that are often not about writing: which collection a text came from, how it was scraped. When one class
fills a combination of those features far more than the other, a classifier can score well by learning the
combination. This step compares the two classes inside every combination ("cell") of five coarse features and
keeps the minority class whole while limiting the majority to `ratio` times as many rows. Rows are ranked by a
hash of the text, so the choice is random but reproducible, and a row's fate does not depend on row order.

Small cells (under `min_cell` rows) are left alone, and a cell that only one class occupies loses all its rows:
those rows are exactly the ones a feature gives away.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .stats import FORMAT_FEATURES

LENGTH_BREAKS = (400, 800, 1600, 3200)
CELL_FEATURES = ("markdown", "paragraph_breaks", "non_ascii", "em_dash")


def balance(df: pl.DataFrame, *, ratio: float, seed: int, min_cell: int = 60) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Drop majority-class rows from over-filled cells. Returns the rows kept and a report of what was dropped.

    Each split is balanced on its own, because each split is evaluated on its own. Rows in splits other than
    train, test and validation (the holdout) must not be passed in.
    """
    length_bin = pl.sum_horizontal(pl.col("text").str.len_chars() >= b for b in LENGTH_BREAKS)
    cell = ["split", "_len", *(f"_{f}" for f in CELL_FEATURES)]
    marked = df.with_columns(
        _len=length_bin, **{f"_{f}": FORMAT_FEATURES[f] for f in CELL_FEATURES}, _h=pl.col("text").hash(seed)
    )
    human = (pl.col("label") == 0).sum().over(cell)
    ai = (pl.col("label") == 1).sum().over(cell)
    allowed = (pl.min_horizontal(human, ai) * ratio).ceil()
    rank = pl.int_range(pl.len()).over([*cell, "label"], order_by=["_h", "doc_id"])
    scored = marked.with_columns(_keep=(human + ai < min_cell) | (rank < allowed))
    kept = scored.filter(pl.col("_keep")).drop("_keep", "_h", "_len", *(f"_{f}" for f in CELL_FEATURES))

    removed = scored.filter(~pl.col("_keep"))
    report = {
        "ratio": ratio,
        "min_cell": min_cell,
        "cells": [*CELL_FEATURES, "length"],
        "dropped": {
            split: {
                "human": int(((removed["split"] == split) & (removed["label"] == 0)).sum()),
                "ai": int(((removed["split"] == split) & (removed["label"] == 1)).sum()),
            }
            for split in df["split"].cast(pl.Utf8).unique().sort()
        },
        "dropped_by_dataset": dict(
            removed
            .group_by("dataset")
            .len()
            .with_columns(pl.col("dataset").cast(pl.Utf8))
            .sort("len", descending=True)
            .iter_rows()
        ),
    }
    return kept, report
