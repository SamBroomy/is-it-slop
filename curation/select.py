"""Choose which rows of a source go to which split, without holding the text.

Pass one hashes each row (cheap, text is dropped straight away) and ranks rows within each label;
quotas then slice that ranking into train / test / validation. Because ranks follow a hash, any
slice is a random sample, the choice is reproducible, and rows sharing a `group` rank together so a
document's variants stay in one split. Pass two (in `build`) fetches the text of only the picked rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .config import Source

SPLITS = ("train", "test", "validation")


def key_rows(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
    """One row per document: stable `doc` hash of its text, `label`, and the ranking hash `h`.

    `h` hashes the `group` when there is one (so variants rank together) and the text otherwise.
    """
    text_hash = pl.col("text").hash(seed)
    group = pl.col("group")
    return lf.select(
        pl.col("label"), doc=text_hash, h=pl.when(group.is_not_null()).then(group.hash(seed)).otherwise(text_hash)
    )


def assign_splits(keys: pl.DataFrame, source: Source, test_fraction: float) -> pl.DataFrame:
    """Pick rows for each split by quota. Returns `doc`, `label`, `split` for the picked rows only.

    Exact duplicate texts inside the source are collapsed first. Primary rows (train + test) are
    taken before validation rows, and the two never overlap.
    """
    keys = keys.unique(subset=["doc", "label"], keep="first", maintain_order=True)
    available = dict(keys.group_by("label").len().iter_rows())
    primary = source.primary.resolve(available)
    remaining = {label: n - primary.get(label, 0) for label, n in available.items()}
    validation = source.validation.resolve(remaining)

    plan = pl.DataFrame(
        [
            {
                "label": label,
                "n_test": round(test_fraction * primary.get(label, 0)),
                "n_primary": primary.get(label, 0),
                "n_validation": validation.get(label, 0),
            }
            for label in available
        ],
        schema={"label": keys.schema["label"], "n_test": pl.Int64, "n_primary": pl.Int64, "n_validation": pl.Int64},
    )
    rank = pl.int_range(pl.len()).over("label")
    n_train = pl.col("n_primary") - pl.col("n_test")
    return (
        keys
        .sort(["label", "h", "doc"])
        .with_columns(rank=rank)
        .join(plan, on="label")
        .with_columns(
            split=pl
            .when(pl.col("rank") < n_train)
            .then(pl.lit("train"))
            .when(pl.col("rank") < pl.col("n_primary"))
            .then(pl.lit("test"))
            .when(pl.col("rank") < pl.col("n_primary") + pl.col("n_validation"))
            .then(pl.lit("validation"))
        )
        .filter(pl.col("split").is_not_null())
        .select("doc", "label", "split")
    )
