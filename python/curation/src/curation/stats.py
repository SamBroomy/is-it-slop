"""Dataset statistics for the metadata file, computed with polars expressions."""

from __future__ import annotations

from typing import Any

import polars as pl
from is_it_slop_preprocessing import count_tokens

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


# --------------------------------------------------------------------------------------------
# Formatting leaks: features that say "AI" without being about the writing
# --------------------------------------------------------------------------------------------

_TEXT = pl.col("text")
FORMAT_FEATURES = {
    "paragraph_breaks": _TEXT.str.contains("\n\n"),
    "markdown": _TEXT.str.contains(r"(?m)(^#+\s|\*\*[^*]+\*\*|^\s*[-*\u2022]\s|^\s*\d+[.)]\s|^\s*\|.*\|)"),
    "em_dash": _TEXT.str.contains("\u2014"),
    "curly_quotes": _TEXT.str.contains(r"[\u2018\u2019\u201c\u201d]"),
    "non_ascii": _TEXT.str.contains(r"[^\x00-\x7F]"),
    "spaced_punctuation": _TEXT.str.contains(r"\w [.,;:!?](?:\s|$)|\w n't\b|\w 's\b"),
    "starts_lowercase": _TEXT.str.contains(r"^[a-z]"),
    "ends_with_punctuation": _TEXT.str.contains(r"[.!?\"')\]]$"),
    "url": _TEXT.str.contains(r"https?://"),
    "html_tag": _TEXT.str.contains(r"<[a-zA-Z/][^>]*>"),
}
BOTH = (0, 1)  # the two labels
LEAK_GAP = 0.35  # a within-source human/AI difference in rate this large is flagged


def format_leaks(df: pl.DataFrame, *, min_rows: int = 200) -> dict[str, Any]:
    """Measure how much label signal formatting alone carries.

    `overall` gives each feature's rate among human and AI rows and its AUC as a one-feature
    classifier (0.5 means no signal, 1.0 means it alone separates the classes). `within_source`
    lists, for every source that has both classes, the features whose rate differs by at least
    `LEAK_GAP` between human and AI text: inside one source the writing is the only difference that
    should exist, so a large gap is a collection artefact the model could learn instead of style.
    """
    features = df.select("label", "dataset", **{name: expr.cast(pl.Float64) for name, expr in FORMAT_FEATURES.items()})
    by_label = features.group_by("label").agg(pl.col(name).mean() for name in FORMAT_FEATURES).sort("label")
    rates = {row["label"]: row for row in by_label.to_dicts()}
    overall: dict[str, Any] = {}
    if 0 in rates and 1 in rates:
        for name in FORMAT_FEATURES:
            human, ai = rates[0][name], rates[1][name]
            overall[name] = {"human": round(human, 4), "ai": round(ai, 4), "auc": round(0.5 + (ai - human) / 2, 4)}
        lengths = df.select("label", n=_TEXT.str.len_chars()).with_columns(r=pl.col("n").rank("average"))
        n_ai = int((lengths["label"] == 1).sum())
        n_human = len(lengths) - n_ai
        rank_sum = float(lengths.filter(pl.col("label") == 1)["r"].sum())
        overall["length"] = {"auc": round((rank_sum - n_ai * (n_ai + 1) / 2) / (n_ai * n_human), 4)}

    per_source = (
        features
        .with_columns(pl.col("dataset").cast(pl.Utf8))
        .group_by("dataset", "label")
        .agg(pl.len().alias("n"), *(pl.col(name).mean() for name in FORMAT_FEATURES))
    )
    within: dict[str, dict[str, list[float]]] = {}
    for dataset in per_source["dataset"].unique().sort():
        rows = {r["label"]: r for r in per_source.filter(pl.col("dataset") == dataset).to_dicts()}
        if len(rows) < len(BOTH) or min(r["n"] for r in rows.values()) < min_rows:
            continue
        flagged = {
            name: [round(rows[0][name], 3), round(rows[1][name], 3)]
            for name in FORMAT_FEATURES
            if abs(rows[1][name] - rows[0][name]) >= LEAK_GAP
        }
        if flagged:
            within[dataset] = flagged
    return {"overall": overall, "within_source": within}
