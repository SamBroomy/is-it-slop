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

    conflicts = df.group_by("norm").agg(pl.col("label").n_unique().alias("labels")).filter(pl.col("labels") > 1)
    after = df.join(conflicts.select("norm"), on="norm", how="anti")
    removed("label_conflict", df, after)
    df = after

    order = pl.col("split").replace_strict({s: i for i, s in enumerate(SPLITS)}, return_dtype=pl.Int8)
    after = df.sort(order, "doc_id").unique(subset="doc_id", keep="first", maintain_order=True)
    removed("duplicate", df, after)
    df = after

    earlier = {"test": ["train"], "validation": ["train", "test"]}
    for name, key in (("shared_normalised", "norm"), ("shared_prefix", "prefix")):
        if key == "prefix" and not prefix_chars:
            continue
        kept = [df.filter(pl.col("split") == "train")]
        for split, against in earlier.items():
            part = df.filter(pl.col("split") == split)
            seen = df.filter(pl.col("split").is_in(against)).select(key).unique()
            kept.append(part.join(seen, on=key, how="anti"))
        after = pl.concat(kept)
        removed(name, df, after)
        df = after

    return df, report
