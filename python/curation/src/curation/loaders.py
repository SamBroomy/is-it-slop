"""Loaders: read a source into the common schema, lazily.

Every loader returns a `LazyFrame`; `standardise` then maps it onto the schema the rest of the
pipeline relies on: `text`, `label`, `group`, `generator`, `domain`, `attack`, `dataset`.

Sources are plain files in a Hugging Face (or Kaggle) dataset repo, read straight into polars:
parquet, csv and ndjson are scanned lazily, and JSON arrays are streamed one file at a time.
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import kagglehub
import polars as pl
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from polars.io.plugins import register_io_source

from .config import AI, HUMAN, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

COLUMNS = ("text", "label", "group", "generator", "domain", "attack", "dataset")

LOADERS: dict[str, Callable[[Source], pl.LazyFrame]] = {}


def loader(name: str) -> Callable[[Callable[[Source], pl.LazyFrame]], Callable[[Source], pl.LazyFrame]]:
    """Register a loader under the name `sources.toml` uses."""

    def register(fn: Callable[[Source], pl.LazyFrame]) -> Callable[[Source], pl.LazyFrame]:
        LOADERS[name] = fn
        return fn

    return register


# --------------------------------------------------------------------------------------------
# Getting files
# --------------------------------------------------------------------------------------------


@functools.cache
def repo_files(repo: str) -> tuple[str, ...]:
    """Every file in a Hugging Face dataset repo."""
    return tuple(HfApi().list_repo_files(repo, repo_type="dataset"))


def download(repo: str, filename: str) -> Path:
    """Download one file of a dataset repo (cached by `huggingface_hub`) and return its path."""
    try:  # a cached file is used as is: no request to the Hub at all
        return Path(hf_hub_download(repo, filename, repo_type="dataset", local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def expand(repo: str, patterns: list[str]) -> list[str]:
    """Resolve file names and `*`/`?`/`[...]` patterns against a repo; every pattern must match."""
    names = repo_files(repo)
    found: list[str] = []
    for pattern in patterns:
        matches = sorted(fnmatch.filter(names, pattern))
        if not matches:
            msg = f"{repo}: no file matches {pattern!r}"
            raise FileNotFoundError(msg)
        found.extend(matches)
    return found


# --------------------------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------------------------


def lazy_from_frames(frames: Callable[[], Iterator[pl.DataFrame]], schema: Callable[[], pl.Schema]) -> pl.LazyFrame:
    """Scan frames lazily as they are produced, so only one is in memory at a time.

    Projection, predicate and row-limit pushdown are honoured; columns are cast to `schema` so that
    files whose inferred types drift (an all-null column, say) still concatenate.
    """

    def source(
        with_columns: list[str] | None, predicate: pl.Expr | None, n_rows: int | None, _hint: int | None
    ) -> Iterator[pl.DataFrame]:
        declared = schema()
        remaining = n_rows
        for frame in frames():
            df = frame
            if remaining is not None:
                if remaining <= 0:
                    return
                df = df.head(remaining)
                remaining -= len(df)
            if predicate is not None:
                df = df.filter(predicate)
            columns = with_columns if with_columns is not None else list(declared)
            df = df.select(
                pl.col(c).cast(declared[c]) if c in df.columns else pl.lit(None, declared[c]).alias(c) for c in columns
            )
            if len(df):
                yield df

    return register_io_source(source, schema=schema, is_pure=True)


@functools.cache
def _inferred_schema(path: Path) -> pl.Schema:
    return pl.read_json(path).schema


def scan_json(paths: list[Path], columns: list[str] | None = None) -> pl.LazyFrame:
    """Scan JSON-array files one at a time (polars cannot scan them lazily, and they can be huge).

    Given `columns`, only those are parsed, as text: nested columns the source does not use (whose
    structure can differ from file to file) are skipped, which is also much faster and lighter.
    Without it every column is inferred from the file.
    """
    schema = pl.Schema(dict.fromkeys(columns, pl.String)) if columns else None
    return lazy_from_frames(
        lambda: (pl.read_json(p, schema=schema) for p in paths), lambda: schema or _inferred_schema(paths[0])
    )


def scan_files(paths: list[Path], columns: list[str] | None = None) -> pl.LazyFrame:
    """Scan data files of one kind (or a mix) into a single lazy frame; `columns` applies to JSON."""
    if all(p.suffix == ".json" for p in paths):
        return scan_json(paths, columns)
    scans = []
    for path in paths:
        name = path.name
        if name.endswith(".parquet"):
            scans.append(pl.scan_parquet(path))
        elif name.endswith(".csv"):
            scans.append(pl.scan_csv(path, infer_schema_length=10_000))
        elif name.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
            scans.append(pl.scan_ndjson(path, infer_schema_length=10_000))
        elif name.endswith(".json"):
            scans.append(scan_json([path], columns))
        else:
            msg = f"don't know how to read {path}"
            raise ValueError(msg)
    return pl.concat(scans, how="diagonal")


@loader("files")
def files(source: Source) -> pl.LazyFrame:
    """Read files of a Hugging Face dataset repo. Params: `repo` and `files`.

    `files` is a list of names or patterns. It can also be a table of `name = names`; each name
    then labels its files in the `dataset` column (as `<repo>/<name>`), the way subsets of one
    repo are told apart.
    """
    repo = source.params["repo"]
    tail = repo.rsplit("/", maxsplit=1)[-1]
    spec = source.params["files"]
    groups = spec.items() if isinstance(spec, dict) else [(None, spec)]
    frames = []
    for name, patterns in groups:
        names = expand(repo, [patterns] if isinstance(patterns, str) else patterns)
        lf = scan_files([download(repo, n) for n in names], source.referenced_columns())
        frames.append(lf.with_columns(dataset=pl.lit(f"{tail}/{name}")) if name else lf)
    return pl.concat(frames, how="diagonal")


@loader("kaggle")
def kaggle(source: Source) -> pl.LazyFrame:
    """Read a file of a Kaggle dataset (needs `KAGGLE_API_TOKEN`). Params: `handle`, `file`."""
    directory = Path(kagglehub.dataset_download(source.params["handle"]))
    return scan_files([directory / source.params["file"]])


@loader("pairs")
def pairs(source: Source) -> pl.LazyFrame:
    """Read datasets with one `human` and one `ai` column per row (a continuation of the same prompt).

    Each pair becomes two rows that share a `group`, so both sides always land in the same split.
    """
    return (
        files(source)
        .with_row_index("pair")
        .unpivot(index="pair", on=["human", "ai"], variable_name="side", value_name="text")
        .select(
            "text",
            label=pl.when(pl.col("side") == "ai").then(AI).otherwise(HUMAN).cast(pl.Int8),
            group=pl.col("pair").cast(pl.Utf8),
        )
    )


# --------------------------------------------------------------------------------------------
# Chat arena datasets: the assistant turns are the AI text
# --------------------------------------------------------------------------------------------

_STRING = r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*\""""
_NUMPY = re.compile(rf"(?P<string>{_STRING})|(?P<array>\barray\()|(?P<dtype>,\s*dtype=object\))", re.DOTALL)
_BRACES = re.compile(rf"(?P<string>{_STRING})|(?P<open>\{{)|(?P<close>\}})", re.DOTALL)


def _unwrap_numpy(text: str) -> str:
    """Rewrite numpy's `array([...], dtype=object)` as a plain literal, leaving string contents alone."""

    def replace(match: re.Match[str]) -> str:
        if match.group("string") is not None:
            return match.group("string")
        return "(" if match.group("array") else ")"

    return _NUMPY.sub(replace, text)


def _top_level_dicts(text: str) -> list[str]:
    """Find the `{...}` objects at depth 0 of a repr, honouring quotes."""
    found: list[str] = []
    depth = 0
    start = 0
    for match in _BRACES.finditer(text):
        if match.group("open"):
            if depth == 0:
                start = match.start()
            depth += 1
        elif match.group("close"):
            depth -= 1
            if depth == 0:
                found.append(text[start : match.end()])
    return found


def parse_conversation(raw: str | None) -> list[dict[str, str]]:
    """Parse the numpy-repr conversation string some arena exports use into `{role, content}` turns.

    `content` is the concatenated text of the turn's text parts. Unparseable input gives `[]`.
    """
    if not raw:
        return []
    turns: list[dict[str, str]] = []
    for chunk in _top_level_dicts(raw):
        try:
            turn: Any = ast.literal_eval(_unwrap_numpy(chunk))
        except (ValueError, SyntaxError):
            continue
        content = turn.get("content")
        if isinstance(content, tuple | list):
            parts = content[0] if len(content) == 1 and isinstance(content[0], list) else content
            texts = (p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text" and p.get("text"))
            content = "\n".join(texts)
        turns.append({"role": str(turn.get("role", "")), "content": content or ""})
    return turns


_TURNS = pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8}))


def assistant_texts(lf: pl.LazyFrame, conversation: str, roles: tuple[str, ...] = ("assistant",)) -> pl.LazyFrame:
    """Explode `conversation` to one row per turn by `roles` (the assistant by default), from any message layout.

    Handles messages whose `content` is a string, or a list of `{type, text}` parts, or whose
    whole conversation is a stringified numpy repr. Keeps the other columns of `lf`.
    """
    if lf.collect_schema()[conversation] == pl.Utf8:
        lf = lf.with_columns(pl.col(conversation).map_elements(parse_conversation, return_dtype=_TURNS))
    turns = lf.explode(conversation).unnest(conversation).filter(pl.col("role").is_in(list(roles)))
    if isinstance(turns.collect_schema()["content"], pl.List):
        parts = pl.col("content").list.eval(
            pl.element().struct.field("text").filter(pl.element().struct.field("type") == "text")
        )
        return turns.with_columns(text=parts.list.join("\n"))
    return turns.with_columns(text=pl.col("content"))


@loader("arena")
def arena(source: Source) -> pl.LazyFrame:
    """Read two-model chat arenas: every assistant reply is an AI text, tagged with the model that wrote it.

    Params: `repo`, `files` and `sides` (a list of `[conversation_column, model_column]` pairs).
    Both replies to one prompt share a `group`.
    """
    base = files(source).with_row_index("convo")
    sides = []
    for conversation, model in source.params["sides"]:
        side = base.select("convo", conversation, pl.col(model).alias("generator"))
        sides.append(
            assistant_texts(side, conversation).select(
                "text",
                pl.lit(AI, dtype=pl.Int8).alias("label"),
                pl.col("convo").cast(pl.Utf8).alias("group"),
                "generator",
            )
        )
    return pl.concat(sides)


@loader("chat")
def chat(source: Source) -> pl.LazyFrame:
    """Read one-model chat logs: every turn by `roles` (default: the assistant) is an AI text.

    Params: `repo`, `files`, `conversation` (the column of turns), and optionally `roles`,
    `model_column` (the model that wrote it) and `where` (a table of `column = allowed values`
    applied to whole conversations, e.g. language or toxicity). Turns of one conversation share a `group`.
    """
    lf = files(source)
    for column, values in (source.params.get("where") or {}).items():
        lf = lf.filter(pl.col(column).is_in(values))
    conversation = source.params["conversation"]
    model = source.params.get("model_column")
    base = lf.with_row_index("convo").select(
        "convo", conversation, *([pl.col(model).alias("generator")] if model else [])
    )
    turns = assistant_texts(base, conversation, tuple(source.params.get("roles", ["assistant"])))
    return turns.select(
        "text",
        pl.lit(AI, dtype=pl.Int8).alias("label"),
        pl.col("convo").cast(pl.Utf8).alias("group"),
        *(["generator"] if model else []),
    )


@loader("answers")
def answers(source: Source) -> pl.LazyFrame:
    """Read question-and-answer corpora with a list of human answers and a list of AI answers per row.

    Params: `repo`, `files`, `human` and `ai` (the list columns). Every answer to one question
    shares a `group`, so a question's human and AI answers always land in the same split.
    """
    base = files(source).with_row_index("_row")
    extra = ["dataset"] if "dataset" in base.collect_schema() else []
    sides = [
        base
        .select("_row", *extra, pl.col(column).alias("text"))
        .explode("text")
        .select(
            "text", pl.lit(label, dtype=pl.Int8).alias("label"), pl.col("_row").cast(pl.Utf8).alias("group"), *extra
        )
        for label, column in ((HUMAN, source.params["human"]), (AI, source.params["ai"]))
    ]
    return pl.concat(sides)


_HTML_TO_TEXT = ((r"(?i)</p>\s*", "\n\n"), (r"(?i)<br\s*/?>", "\n"), (r"(?i)<li[^>]*>", "\n- "), (r"<[^>]+>", ""))


@loader("stackexchange")
def stackexchange(source: Source) -> pl.LazyFrame:
    """Read Stack Exchange questions and keep the best-scored answer of each, as human text.

    Params: `repo`, `sites` (folder names under `data/`), `before` (only questions asked before this
    date, `YYYY/MM/DD`: keep it before ChatGPT so the answers are certainly human) and `min_score`.
    Answers with code blocks are dropped (this is for prose); HTML becomes plain text with paragraph breaks.
    """
    repo = source.params["repo"]
    frames = [
        pl
        .scan_parquet(download(repo, name))
        .select("qid", "date", "answers")
        .with_columns(site=pl.lit(site.split(".", maxsplit=1)[0]))
        for site in source.params["sites"]
        for name in expand(repo, [f"data/{site}/*.parquet"])
    ]
    answers_ = pl.col("answers")
    scores = answers_.list.eval(pl.element().struct.field("pm_score"))
    best = answers_.list.get(scores.list.arg_max(), null_on_oob=True)
    text = pl.col("best").struct.field("text")
    for pattern, replacement in _HTML_TO_TEXT:
        text = text.str.replace_all(pattern, replacement)
    return (
        pl
        .concat(frames)
        .filter(pl.col("date") < source.params["before"])
        .with_columns(best=best)
        .filter(pl.col("best").struct.field("pm_score") >= source.params.get("min_score", 1))
        .filter(~pl.col("best").struct.field("text").str.contains("<pre"))
        .select(
            text.alias("text"),
            pl.lit(HUMAN, dtype=pl.Int8).alias("label"),
            pl.col("qid").cast(pl.Utf8).alias("group"),
            "site",
        )
    )


# --------------------------------------------------------------------------------------------
# Repairs for how a source was collected
# --------------------------------------------------------------------------------------------

_DETOKENISE = (
    (r"\s+([.,;:!?%])", "${1}"),
    (r"([(\[{])\s+", "${1}"),
    (r"\s+([)\]}])", "${1}"),
    (r"\s+(n't)\b", "${1}"),
    (r"\s+('(?:s|re|m|ll|ve|d))\b", "${1}"),
    (r"``\s*", '"'),
    (r"\s*''", '"'),
)


def unwrap_lines(text: pl.Expr) -> pl.Expr:
    """Join hard-wrapped lines: a single newline becomes a space, a paragraph break stays."""
    return (
        text.str
        .replace_all("\n\n", "\u0001", literal=True)
        .str.replace_all(r"\s*\n\s*", " ")
        .str.replace_all("\u0001", "\n\n", literal=True)
    )


def detokenise(text: pl.Expr) -> pl.Expr:
    """Undo pre-tokenised spacing ("it 's", "word .", "( Reuters )") and PTB-style quotes."""
    for pattern, replacement in _DETOKENISE:
        text = text.str.replace_all(pattern, replacement)
    return text


def normalise_breaks(text: pl.Expr) -> pl.Expr:
    """Make every newline, or run of blank space around one, a single paragraph break."""
    return text.str.replace_all(r"[ \t]*\n\s*", "\n\n")


def ascii_punctuation(text: pl.Expr) -> pl.Expr:
    """Straight quotes, hyphens for en/em dashes, three dots for an ellipsis, plain spaces."""
    return (
        ascii_quotes(text)
        .str.replace_all(r"[\u2013\u2014\u2012\u2015]", "-")
        .str.replace_all("\u2026", "...", literal=True)
        .str.replace_all(r"[\u00a0\u2009\u202f]", " ")
    )


def ascii_quotes(text: pl.Expr) -> pl.Expr:
    """Turn curly quotes into straight ones."""
    return text.str.replace_all(r"[\u2018\u2019]", "'").str.replace_all(r"[\u201c\u201d]", '"')


_PARTIAL_START = r"(?s)^[a-z].*?[.!?][\"')\]]*\s+([A-Z])"
_PARTIAL_END = r"(?s)^(.*[.!?][\"')\]]*)\s+[^.!?]*$"


def trim_fragments(text: pl.Expr, where: str) -> pl.Expr:
    """Cut a fragment that begins (`start`) or ends (`end`) mid-sentence; `both` does both."""
    if where in {"start", "both"}:
        text = text.str.replace(_PARTIAL_START, "${1}")
    if where in {"end", "both"}:
        text = text.str.replace(_PARTIAL_END, "${1}")
    return text


def has_whole_ends(where: str) -> pl.Expr:
    """Select rows whose text no longer begins and/or ends mid-sentence, as `trim_fragments` asks."""
    keep = pl.lit(True)  # ruff: ignore[boolean-positional-value-in-call]
    if where in {"start", "both"}:
        keep &= ~pl.col("text").str.contains(r"^[a-z]")
    if where in {"end", "both"}:
        keep &= pl.col("text").str.contains(r"[.!?\"')\]]$")
    return keep


def strip_markdown(text: pl.Expr) -> pl.Expr:
    """Remove the markdown that the Rust cleaner leaves behind: markers go, the words stay."""
    steps = (
        (r"(?m)^[ \t]*```[A-Za-z0-9_+-]*[ \t]*$\n?", ""),  # code fences
        (r"`([^`\n]+)`", "${1}"),  # inline code
        (r"(?m)^[ \t]*#{1,6}[ \t]+", ""),  # headings
        (r"(?m)^[ \t]*>[ \t]?", ""),  # block quotes
        (r"(?m)^[ \t]*\|?(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*$\n?", ""),  # table separator rows
        (r"(?m)^([ \t]*)\|[ \t]*", "${1}"),  # table row starts
        (r"[ \t]*\|[ \t]*", " "),  # table cell bars
        (r"(?m)^[ \t]*(?:[-*\u2022+]|\d+[.)])[ \t]+", ""),  # list markers
        (r"\*\*\*([^*\n]+)\*\*\*", "${1}"),
        (r"\*\*([^*\n]+)\*\*", "${1}"),  # bold
        (r"__([^_\n]+)__", "${1}"),
        (r"(?:^|[ \t(])\*([^*\s][^*\n]*?)\*", " ${1}"),  # italics
        (r"\n{3,}", "\n\n"),
    )
    for pattern, replacement in steps:
        text = text.str.replace_all(pattern, replacement)
    return text.str.strip_chars()


def flatten_paragraphs(text: pl.Expr) -> pl.Expr:
    """Collapse every run of whitespace, newlines included, to a single space."""
    return text.str.replace_all(r"\s+", " ")


# --------------------------------------------------------------------------------------------
# Schema normalisation
# --------------------------------------------------------------------------------------------


def _label_expression(source: Source) -> pl.Expr:
    """Build the label (0 human, 1 AI) the source's configuration defines."""
    if source.label in {HUMAN, AI}:
        return pl.lit(source.label, dtype=pl.Int8)
    if source.human_value is not None and source.label_column:
        return pl.when(pl.col(source.label_column) == source.human_value).then(HUMAN).otherwise(AI).cast(pl.Int8)
    return pl.col(source.label_column or "label").cast(pl.Int8)


def _text_expression(source: Source, schema: list[str]) -> pl.Expr:
    """Build the text column with the source's collection repairs applied."""
    text = pl.col(source.text_column if source.text_column in schema else "text").cast(pl.Utf8)
    if source.strip_think:
        text = text.str.replace(r"(?s)^\s*<think>.*?</think>\s*", "")
    if source.unwrap_lines:
        text = unwrap_lines(text)
    if source.remove_pattern:  # before detokenise: removing a character can leave a gap behind
        text = text.str.replace_all(source.remove_pattern, "")
    if source.detokenise:
        text = detokenise(text)
    if source.ascii_punctuation:
        text = ascii_punctuation(text)
    elif source.ascii_quotes:
        text = ascii_quotes(text)
    if source.normalise_breaks:
        text = normalise_breaks(text)
    if source.flatten_paragraphs:
        text = flatten_paragraphs(text)
    if source.trim_fragments:
        text = trim_fragments(text, source.trim_fragments)
    return text.str.strip_chars()


def standardise(lf: pl.LazyFrame, source: Source) -> pl.LazyFrame:
    """Map a loader's output onto `COLUMNS`, applying the source's configured column choices."""
    schema = lf.collect_schema().names()

    for column, values in (source.params.get("keep") or {}).items():
        lf = lf.filter(pl.col(column).is_in(values))

    def column_or_null(name: str | None, regex: str | None = None) -> pl.Expr:
        if not (name and name in schema):
            return pl.lit(None, dtype=pl.Utf8)
        column = pl.col(name).cast(pl.Utf8)
        return column.str.extract(regex, 1) if regex else column

    if source.generator:
        generator = pl.lit(source.generator)
    elif source.generator_column or "generator" not in schema:
        generator = column_or_null(source.generator_column, source.generator_regex)
    else:
        generator = pl.col("generator").cast(pl.Utf8)  # supplied by the loader (arena)

    out = (
        lf
        .select(
            text=_text_expression(source, schema),
            label=_label_expression(source),
            group=pl.col("group").cast(pl.Utf8) if "group" in schema else column_or_null(source.group_column),
            generator=generator,
            domain=pl.lit(source.domain)
            if source.domain
            else column_or_null(source.domain_column, source.domain_regex),
            attack=column_or_null(source.attack_column),
            dataset=pl.col("dataset") if "dataset" in schema else pl.lit(source.name),
        )
        # A generator only makes sense for AI text; human rows in mixed sources must not carry one.
        .with_columns(generator=pl.when(pl.col("label") == AI).then(pl.col("generator")))
        .filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > max(source.min_chars, 1) - 1))
    )
    if source.drop_pattern:
        out = out.filter(~pl.col("text").str.contains(source.drop_pattern))
    if source.strip_think:
        out = out.filter(~pl.col("text").str.contains("<think>"))  # a trace that never closed
    if source.trim_fragments:
        out = out.filter(has_whole_ends(source.trim_fragments))  # rows where no sentence boundary could be found go
    return out


def load_source(source: Source, limit: int | None = None) -> pl.LazyFrame:
    """Load one source in the common schema; `limit` keeps only the first rows (for quick dev runs)."""
    try:
        read = LOADERS[source.loader]
    except KeyError:
        msg = f"source {source.name!r}: unknown loader {source.loader!r} (have {sorted(LOADERS)})"
        raise ValueError(msg) from None
    lf = read(source)
    if limit:
        lf = lf.head(limit)
    return standardise(lf, source)


def missing_credentials(requires: str | None) -> str | None:
    """Say why a source's credentials are missing, or return `None` when it can be read."""
    if requires == "hf_token":
        return None if get_token() else "gated on Hugging Face: set HF_TOKEN (in .env or the environment)"
    if requires == "kaggle":
        have = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
        have = have or Path("~/.kaggle/kaggle.json").expanduser().exists()
        return None if have else "needs Kaggle credentials: set KAGGLE_API_TOKEN (in .env or the environment)"
    return None
