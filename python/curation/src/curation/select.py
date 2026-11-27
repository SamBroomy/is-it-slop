"""Choose which rows of a source go to which split, without holding the text.

Pass one hashes each row (cheap, text is dropped straight away) and ranks rows within each label;
the fractions then slice that ranking into train / test / validation. Because ranks follow a hash, any
slice is a random sample and the choice is reproducible. Rows sharing a `group` rank together, so a
document's variants stay in one split except where a slice boundary falls inside a group (at most
one group per cell per boundary; the overlap guards remove any leakage that causes). Pass two (in `build`)
fetches the text of only the picked rows.

The cut into train / test / validation is stratified: it is made inside every cell of
(label, domain, generator), so each cell is split 70/15/15 and no split lacks a domain or a generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .config import Source

SPLITS = ("train", "test", "validation", "holdout")
PROPORTIONAL = SPLITS[:3]
CELL = ["label", "domain", "generator"]
RANK = ["h", "doc"]  # the total order every ranking uses, so the result never depends on row order


def key_rows(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
    """One row per document: stable `doc` hash of its text, `label`, `domain`, `generator`, and the ranking hash `h`.

    `h` hashes the `group` when there is one (so variants rank together) and the text otherwise.
    """
    text_hash = pl.col("text").hash(seed)
    group = pl.col("group")
    return lf.select(
        pl.col("label", "domain", "generator"),
        doc=text_hash,
        h=pl.when(group.is_not_null()).then(group.hash(seed)).otherwise(text_hash),
    )


def _first(keys: pl.DataFrame, per_label: dict[int, int]) -> pl.DataFrame:
    """Keep the first `per_label[label]` rows of each label in rank order."""
    counts = pl.DataFrame(
        {"label": list(per_label), "n": list(per_label.values())}, schema={"label": keys.schema["label"], "n": pl.Int64}
    )
    rank = pl.int_range(pl.len()).over("label", order_by=RANK)
    return keys.join(counts, on="label").filter(rank < pl.col("n")).drop("n")


def _held_out_generators(keys: pl.DataFrame, source: Source) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split `keys` into the rows that go to the holdout and the rest, for `holdout_generators`.

    Every row of a listed generator is withheld from training: up to `holdout_cap` of each goes to the
    holdout and the others are dropped (they are the same generator, so they may not train either).
    As many human rows of the same source go to the holdout with them, so the holdout block has both classes.
    """
    held = pl.col("generator").is_in(list(source.holdout_generators)).fill_null(value=False)
    ai = keys.filter(held)
    if source.holdout_cap:
        ai = ai.filter(pl.int_range(pl.len()).over("generator", order_by=RANK) < source.holdout_cap)
    rest = keys.filter(~held)
    humans = _first(rest.filter(pl.col("label") == 0), {0: ai.height})
    chosen = pl.concat([ai, humans])
    return chosen, rest.join(humans.select("doc", "label"), on=["doc", "label"], how="anti")


def assign_splits(keys: pl.DataFrame, source: Source, fractions: tuple[float, float, float]) -> pl.DataFrame:
    """Pick rows for each split. Returns `doc`, `label`, `split` for the picked rows only.

    A holdout source puts the rows it takes (all of them, unless `take` limits it) in `holdout`.
    Any other source first sets aside its `holdout_generators` rows, then takes `source.take` rows per
    label and cuts them by `fractions` (train, test, validation) inside every (label, domain,
    generator) cell, so every split has the same mix of domains and generators.

    A text that appears more than once in the source is collapsed to one row ranked by its smallest
    `h`, so the result never depends on the order rows arrive in.
    """
    keys = keys.group_by("doc", "label").agg(pl.col("h").min(), pl.col("domain").min(), pl.col("generator").min())
    available = dict(keys.group_by("label").len().iter_rows())
    if source.holdout:
        every = source.take.resolve(available) if source.take.is_set else available
        return _first(keys, every).select("doc", "label", split=pl.lit("holdout"))

    held = keys.clear()
    if source.holdout_generators:
        held, keys = _held_out_generators(keys, source)
        available = dict(keys.group_by("label").len().iter_rows())
    take = source.take.resolve(available)
    train, test, _ = fractions

    chosen = _first(keys, take)
    rank = pl.int_range(pl.len()).over(CELL, order_by=RANK)
    position = (rank + 0.5) / pl.len().over(CELL)  # where the row sits in its cell, 0 to 1
    cut = chosen.select(
        "doc",
        "label",
        split=pl
        .when(position < train)
        .then(pl.lit("train"))
        .when(position < train + test)
        .then(pl.lit("test"))
        .otherwise(pl.lit("validation")),
    )
    return pl.concat([held.select("doc", "label", split=pl.lit("holdout")), cut])
