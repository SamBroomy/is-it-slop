"""Per-document style features, stored beside the text so they can be analysed or modelled later.

These describe *how* a text is written, not which words it uses: length, sentence and word shape,
punctuation habits, markdown, and how varied the vocabulary is. They are computed on the text as
collected (after the source's repairs, before the model's cleaner), so markdown the cleaner removes
is still counted: that is the point, because markdown is something to measure, not something the
model should learn from.

Counts are per 100 words (or per 100 characters where noted) so documents of different length compare.
Every feature is a Float32 column named `f_<name>`.
"""

from __future__ import annotations

import polars as pl

_T = pl.col("text")
_CHARS = _T.str.len_chars().clip(lower_bound=1)
_WORDS = _T.str.count_matches(r"\b\w+\b").clip(lower_bound=1)
_SENTENCES = _T.str.count_matches(r"[.!?]+(?:\s|$)").clip(lower_bound=1)


def _per_100_words(pattern: str) -> pl.Expr:
    return _T.str.count_matches(pattern) / _WORDS * 100


def _per_100_chars(pattern: str) -> pl.Expr:
    return _T.str.count_matches(pattern) / _CHARS * 100


FEATURES: dict[str, pl.Expr] = {
    # Length and shape
    "log_chars": _CHARS.log(),
    "log_words": _WORDS.log(),
    "words_per_sentence": _WORDS / _SENTENCES,
    "avg_word_len": _CHARS / _WORDS,
    "type_token": _T.str.to_lowercase().str.extract_all(r"\b\w+\b").list.n_unique() / _WORDS,
    "paragraphs": _T.str.count_matches(r"\n\n") + 1,
    "paragraph_breaks": _per_100_words(r"\n\n"),
    "newlines": _per_100_words(r"\n"),
    # Markdown (counted here, removed by the cleaner)
    "md_heading": _per_100_words(r"(?m)^#+\s"),
    "md_bold": _per_100_words(r"\*\*[^*]+\*\*"),
    "md_list": _per_100_words(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s"),
    "md_code": _per_100_words(r"`"),
    "md_table": _per_100_words(r"(?m)^\s*\|"),
    # Punctuation habits
    "em_dash": _per_100_words(r"\u2014"),
    "en_dash": _per_100_words(r"\u2013"),
    "curly_quotes": _per_100_words(r"[\u2018\u2019\u201c\u201d]"),
    "non_ascii": _per_100_chars(r"[^\x00-\x7f]"),
    "commas": _per_100_words(","),
    "semicolons": _per_100_words(";"),
    "colons": _per_100_words(":"),
    "parens": _per_100_words(r"\("),
    "quotes": _per_100_words('"'),
    "exclamations": _per_100_words("!"),
    "questions": _per_100_words(r"\?"),
    "ellipses": _per_100_words(r"\.\.\.|\u2026"),
    # Character mix
    "digits": _per_100_chars(r"\d"),
    "upper": _per_100_chars(r"[A-Z]"),
    # Voice
    "first_person": _per_100_words(r"(?i)\b(?:i|my|me|i'm|i've|i'd|i'll)\b"),
    "second_person": _per_100_words(r"(?i)\b(?:you|your|you're|you'll)\b"),
    "contractions": _per_100_words(r"\w'(?:s|t|re|ve|ll|d|m)\b"),
    "urls": _per_100_words(r"https?://"),
    # Edges
    "starts_lowercase": _T.str.contains(r"^[a-z]").cast(pl.Float64),
    "ends_with_punct": _T.str.contains(r"[.!?\"')\]]$").cast(pl.Float64),
}
COLUMNS = tuple(f"f_{name}" for name in FEATURES)


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append the `f_*` Float32 columns, computed from `text` as it is now."""
    return df.with_columns(expr.cast(pl.Float32).alias(f"f_{name}") for name, expr in FEATURES.items())
