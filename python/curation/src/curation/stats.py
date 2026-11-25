"""Dataset statistics for the metadata file, computed with polars expressions."""

from __future__ import annotations

from typing import Any

import polars as pl

MARKDOWN_PATTERNS = {
    "heading": r"(?m)^#+\s+",
    "bold": r"\*\*[^*]+\*\*",
    # Single-asterisk emphasis. No lookaround: polars' regex engine does not support it.
    "italic": r"(?:^|[^*])\*[^*\s][^*\n]*\*(?:[^*]|$)",
    "code_block": r"```",
    "inline_code": r"`[^`]+`",
    "list_item": r"(?m)^\s*[-*+]\s+",
    "numbered_list": r"(?m)^\s*\d+\.\s+",
    "blockquote": r"(?m)^>\s+",
}

HUMAN = pl.col("label") == 0
AI = pl.col("label") == 1


def composition(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Count rows per source dataset: total, human and AI."""
    counts = (
        df
        .group_by(pl.col("dataset").cast(pl.Utf8))
        .agg(count=pl.len(), human=HUMAN.sum(), ai=AI.sum())
        .sort("dataset")
        .to_dicts()
    )
    return {row.pop("dataset"): row for row in counts}


def class_balance(df: pl.DataFrame) -> dict[str, float]:
    """Give the share of human and AI rows."""
    return df.select(human=HUMAN.mean().round(4), ai=AI.mean().round(4)).row(0, named=True)


def summarise(values: pl.Expr) -> list[pl.Expr]:
    """Describe a numeric column: mean, standard deviation, extremes and two percentiles."""
    return [
        values.mean().round(2).alias("mean"),
        values.std().round(2).alias("std"),
        values.min().alias("min"),
        values.max().alias("max"),
        values.quantile(0.5).cast(pl.Int64).alias("p50"),
        values.quantile(0.95).cast(pl.Int64).alias("p95"),
    ]


def length_stats(df: pl.DataFrame, *, seed: int, sample: int = 50_000) -> dict[str, dict[str, Any]]:
    """Summarise text length in characters and in o200k tokens (tokens on a sample, for speed)."""
    from is_it_slop.preprocessing import count_tokens

    if df.is_empty():
        return {}
    texts = df["text"] if len(df) <= sample else df.sample(sample, seed=seed)["text"]
    tokens = pl.DataFrame({"tokens": count_tokens(texts)})
    return {
        "text_lengths_chars": df.select(summarise(pl.col("text").str.len_chars())).row(0, named=True),
        "token_counts": tokens.select(summarise(pl.col("tokens"))).row(0, named=True),
    }


def markdown_bias(df: pl.DataFrame) -> dict[str, Any]:
    """Compare how often AI and human text use markdown, overall and per pattern."""
    matches = [pl.col("text").str.count_matches(pattern).alias(name) for name, pattern in MARKDOWN_PATTERNS.items()]
    by_label = (
        df
        .select("label", *matches)
        .group_by("label")
        .agg(
            share=pl.any_horizontal(pl.col(name) > 0 for name in MARKDOWN_PATTERNS).mean(),
            **{name: pl.col(name).mean() for name in MARKDOWN_PATTERNS},
        )
    )
    rows = {row["label"]: row for row in by_label.to_dicts()}
    if not (0 in rows and 1 in rows):
        return {}
    human, ai = rows[0], rows[1]
    return {
        "ai_pct": round(ai["share"] * 100, 2),
        "human_pct": round(human["share"] * 100, 2),
        "ratio": round(ai["share"] / human["share"], 2) if human["share"] else 0.0,
        "pattern_breakdown": {
            name: {"ai": round(ai[name], 4), "human": round(human[name], 4)} for name in MARKDOWN_PATTERNS
        },
    }
