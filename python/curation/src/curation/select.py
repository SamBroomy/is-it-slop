"""Choose which rows of a source go to which split, without holding the text.

Pass one hashes each row (cheap, text is dropped straight away) and ranks rows within each label;
quotas then slice that ranking into train / test / validation. Because ranks follow a hash, any
slice is a random sample and the choice is reproducible. Rows sharing a `group` rank together, so a
document's variants stay in one split except where a quota boundary falls inside a group (at most
one group per label per boundary; the overlap guards remove any leakage that causes). Pass two (in `build`) fetches the text of only the picked rows.
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

    A text that appears more than once in the source is collapsed to one row ranked by its smallest
    `h`, so the result never depends on the order rows arrive in. Primary rows (train + test) are
    taken before validation rows, and the two never overlap.
    """
    keys = keys.group_by("doc", "label").agg(pl.col("h").min())
    available = dict(keys.group_by("label").len().iter_rows())
    primary = source.primary.resolve(available)
    validation = source.validation.resolve({label: n - primary.get(label, 0) for label, n in available.items()})

    plan = pl.DataFrame(
        {
            "label": list(available),
            "n_primary": [primary.get(label, 0) for label in available],
            "n_validation": [validation.get(label, 0) for label in available],
        },
        schema={"label": keys.schema["label"], "n_primary": pl.Int64, "n_validation": pl.Int64},
    ).with_columns(n_test=(pl.col("n_primary") * test_fraction).round().cast(pl.Int64))
    rank = pl.int_range(pl.len()).over("label", order_by=["h", "doc"])
    n_train = pl.col("n_primary") - pl.col("n_test")
    return (
        keys
        .join(plan, on="label")
        .with_columns(rank=rank)
        .select(
            "doc",
            "label",
            split=pl
            .when(pl.col("rank") < n_train)
            .then(pl.lit("train"))
            .when(pl.col("rank") < pl.col("n_primary"))
            .then(pl.lit("test"))
            .when(pl.col("rank") < pl.col("n_primary") + pl.col("n_validation"))
            .then(pl.lit("validation")),
        )
        .drop_nulls("split")
    )
