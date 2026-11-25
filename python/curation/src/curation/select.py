"""Choose which rows of a source go to which split, without holding the text.

Pass one hashes each row (cheap, text is dropped straight away) and ranks rows within each label;
the fractions then slice that ranking into train / test / validation. Because ranks follow a hash, any
slice is a random sample and the choice is reproducible. Rows sharing a `group` rank together, so a
document's variants stay in one split except where a slice boundary falls inside a group (at most
one group per label per boundary; the overlap guards remove any leakage that causes). Pass two (in `build`) fetches the text of only the picked rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .config import Source

SPLITS = ("train", "test", "validation", "holdout")
PROPORTIONAL = SPLITS[:3]


def key_rows(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
    """One row per document: stable `doc` hash of its text, `label`, and the ranking hash `h`.

    `h` hashes the `group` when there is one (so variants rank together) and the text otherwise.
    """
    text_hash = pl.col("text").hash(seed)
    group = pl.col("group")
    return lf.select(
        pl.col("label"), doc=text_hash, h=pl.when(group.is_not_null()).then(group.hash(seed)).otherwise(text_hash)
    )


def assign_splits(keys: pl.DataFrame, source: Source, fractions: tuple[float, float, float]) -> pl.DataFrame:
    """Pick rows for each split. Returns `doc`, `label`, `split` for the picked rows only.

    A holdout source puts every row in `holdout`. Any other source takes `source.take` rows per
    label and cuts them by `fractions` (train, test, validation), so every split has the same mix of
    sources: one random ranking per label, sliced three ways.

    A text that appears more than once in the source is collapsed to one row ranked by its smallest
    `h`, so the result never depends on the order rows arrive in.
    """
    keys = keys.group_by("doc", "label").agg(pl.col("h").min())
    if source.holdout:
        return keys.select("doc", "label", split=pl.lit("holdout"))
    available = dict(keys.group_by("label").len().iter_rows())
    take = source.take.resolve(available)
    train, test, _ = fractions

    plan = pl.DataFrame(
        {"label": list(available), "n_take": [take.get(label, 0) for label in available]},
        schema={"label": keys.schema["label"], "n_take": pl.Int64},
    ).with_columns(
        n_train=(pl.col("n_take") * train).round().cast(pl.Int64),
        n_test=(pl.col("n_take") * test).round().cast(pl.Int64),
    )
    rank = pl.int_range(pl.len()).over("label", order_by=["h", "doc"])
    n_train = pl.col("n_train")
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
            .when(pl.col("rank") < n_train + pl.col("n_test"))
            .then(pl.lit("test"))
            .when(pl.col("rank") < pl.col("n_take"))
            .then(pl.lit("validation")),
        )
        .drop_nulls("split")
    )
