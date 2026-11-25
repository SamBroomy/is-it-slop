"""Clean the picked rows, then remove duplicates and anything that leaks between splits.

Every comparison is a join on 64-bit hashes of the text, never a Python collection of strings.
"""

from __future__ import annotations

import polars as pl

from .select import SPLITS

# Lower-cased with whitespace collapsed: "Same  text" and "same text" are one document.
NORMALISED = pl.col("text").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Run the Rust training cleaner over `text` and drop rows that clean to nothing."""
    from is_it_slop.preprocessing import clean_for_training

    return df.with_columns(pl.Series("text", clean_for_training(df["text"]), dtype=pl.Utf8)).filter(
        pl.col("text").str.len_chars() > 0
    )


def add_keys(df: pl.DataFrame, seed: int, prefix_chars: int) -> pl.DataFrame:
    """Add `doc_id` (exact text), `norm` (normalised text) and `prefix` (start of the normalised text)."""
    norm = NORMALISED
    return df.with_columns(
        doc_id=pl.col("text").hash(seed),
        norm=norm.hash(seed),
        prefix=norm.str.slice(0, prefix_chars).hash(seed) if prefix_chars else pl.lit(0, dtype=pl.UInt64),
    )


def apply_guards(df: pl.DataFrame, *, prefix_chars: int) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    """Remove label conflicts, duplicates and cross-split overlap. Returns the frame and a report.

    The report maps each guard to how many rows it removed per split, so the effect is visible:

    * `label_conflict`: texts that appear with both labels (neither label can be trusted),
    * `duplicate`: an exact text already present, keeping the earliest split
      (train, then test, then validation),
    * `shared_normalised`: test or validation rows whose lower-cased, whitespace-collapsed text
      already occurs in an earlier split,
    * `shared_prefix`: test or validation rows whose first `prefix_chars` characters (normalised)
      match an earlier split: near-duplicates such as re-wrapped or re-cased copies.

    Variants of one document that sit in the *same* split are kept on purpose (RAID's attack
    variants, for example, are useful robustness training); only overlap between splits is a leak.
    """
    report: dict[str, dict[str, int]] = {}

    def removed(name: str, before: pl.DataFrame, after: pl.DataFrame) -> None:
        counts = dict(before.group_by("split").len().iter_rows())
        kept = dict(after.group_by("split").len().iter_rows())
        report[name] = {s: counts.get(s, 0) - kept.get(s, 0) for s in SPLITS}

    after = df.filter(pl.col("label").n_unique().over("norm") == 1)
    removed("label_conflict", df, after)
    df = after

    # Earliest split first, then a total order, so which copy survives never depends on row order.
    split_rank = pl.col("split").cast(pl.Enum(SPLITS)).to_physical()
    tie_breakers = [c for c in ("doc_id", "dataset", "generator", "domain", "attack") if c in df.columns]
    ordered = df.sort(split_rank, *tie_breakers, nulls_last=True, maintain_order=True)
    after = ordered.unique(subset="doc_id", keep="first", maintain_order=True)
    removed("duplicate", df, after)
    df = after

    # A later split may not repeat what an earlier one holds; repeats inside one split are kept.
    for name, key in (("shared_normalised", "norm"), ("shared_prefix", "prefix")):
        if key == "prefix" and not prefix_chars:
            continue
        after = df.filter(split_rank == split_rank.min().over(key))
        removed(name, df, after)
        df = after

    return df, report
