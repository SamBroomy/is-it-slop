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
LABELS = {0: "human", 1: "ai"}


def composition(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Rows per source dataset: total, human and AI."""
    counts = df.group_by("dataset", "label").len().sort("dataset", "label")
    out: dict[str, dict[str, int]] = {}
    for dataset, label, n in counts.iter_rows():
        entry = out.setdefault(str(dataset), {"count": 0, "human": 0, "ai": 0})
        entry["count"] += n
        entry[LABELS[label]] += n
    return out


def class_balance(df: pl.DataFrame) -> dict[str, float]:
    """Share of human and AI rows."""
    total = len(df)
    return {LABELS[label]: round(n / total, 4) for label, n in df.group_by("label").len().sort("label").iter_rows()}


def length_stats(df: pl.DataFrame, *, seed: int, sample: int = 50_000) -> dict[str, dict[str, float]]:
    """Summary of text length in characters and in o200k tokens (tokens on a sample for speed)."""
    from is_it_slop.preprocessing import count_tokens

    if df.is_empty():
        return {}

    def summary(values: pl.Series) -> dict[str, float]:
        return {
            "mean": round(float(values.mean()), 2),
            "std": round(float(values.std() or 0.0), 2),
            "min": int(values.min()),
            "max": int(values.max()),
            "p50": int(values.quantile(0.5)),
            "p95": int(values.quantile(0.95)),
        }

    texts = df["text"] if len(df) <= sample else df.sample(sample, seed=seed)["text"]
    return {
        "text_lengths_chars": summary(df["text"].str.len_chars()),
        "token_counts": summary(pl.Series(count_tokens(texts))),
    }


def markdown_bias(df: pl.DataFrame) -> dict[str, Any]:
    """How much more often AI text uses markdown than human text, overall and per pattern."""
    counts = {name: pl.col("text").str.count_matches(pattern) for name, pattern in MARKDOWN_PATTERNS.items()}
    per_row = df.select("label", **counts).with_columns(any_markdown=pl.any_horizontal(pl.exclude("label") > 0))
    by_label = per_row.group_by("label").agg(
        pl.len().alias("rows"),
        pl.col("any_markdown").mean().alias("share"),
        *[pl.col(name).mean().alias(name) for name in MARKDOWN_PATTERNS],
    )
    rows = {LABELS[row["label"]]: row for row in by_label.iter_rows(named=True)}
    ai, human = rows.get("ai"), rows.get("human")
    if not ai or not human:
        return {}
    ai_pct, human_pct = ai["share"] * 100, human["share"] * 100
    return {
        "ai_pct": round(ai_pct, 2),
        "human_pct": round(human_pct, 2),
        "ratio": round(ai_pct / human_pct, 2) if human_pct else 0.0,
        "pattern_breakdown": {
            name: {"ai": round(ai[name], 4), "human": round(human[name], 4)} for name in MARKDOWN_PATTERNS
        },
    }
