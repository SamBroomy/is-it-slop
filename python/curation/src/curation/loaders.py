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

import polars as pl
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
    from huggingface_hub import HfApi

    return tuple(HfApi().list_repo_files(repo, repo_type="dataset"))


def download(repo: str, filename: str) -> Path:
    """Download one file of a dataset repo (cached by `huggingface_hub`) and return its path."""
    from huggingface_hub import hf_hub_download

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
        match path.suffix:
            case ".parquet":
                scans.append(pl.scan_parquet(path))
            case ".csv":
                scans.append(pl.scan_csv(path, infer_schema_length=10_000))
            case ".jsonl":
                scans.append(pl.scan_ndjson(path))
            case ".json":
                scans.append(scan_json([path], columns))
            case _:
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
    import kagglehub

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


def assistant_texts(lf: pl.LazyFrame, conversation: str) -> pl.LazyFrame:
    """Explode `conversation` to one row per assistant turn, from any of the arena message layouts.

    Handles messages whose `content` is a string, or a list of `{type, text}` parts, or whose
    whole conversation is a stringified numpy repr. Keeps the other columns of `lf`.
    """
    if lf.collect_schema()[conversation] == pl.Utf8:
        lf = lf.with_columns(pl.col(conversation).map_elements(parse_conversation, return_dtype=_TURNS))
    turns = lf.explode(conversation).unnest(conversation).filter(pl.col("role") == "assistant")
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


# --------------------------------------------------------------------------------------------
# Schema normalisation
# --------------------------------------------------------------------------------------------


def standardise(lf: pl.LazyFrame, source: Source) -> pl.LazyFrame:
    """Map a loader's output onto `COLUMNS`, applying the source's configured column choices."""
    schema = lf.collect_schema().names()

    for column, values in (source.params.get("keep") or {}).items():
        lf = lf.filter(pl.col(column).is_in(values))

    def column_or_null(name: str | None) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name and name in schema else pl.lit(None, dtype=pl.Utf8)

    if source.label in {HUMAN, AI}:
        label = pl.lit(source.label, dtype=pl.Int8)
    elif source.human_value is not None and source.label_column:
        label = pl.when(pl.col(source.label_column) == source.human_value).then(HUMAN).otherwise(AI).cast(pl.Int8)
    elif source.label_column:
        label = pl.col(source.label_column).cast(pl.Int8)
    else:
        label = pl.col("label").cast(pl.Int8)

    if source.generator:
        generator = pl.lit(source.generator)
    elif source.generator_column or "generator" not in schema:
        generator = column_or_null(source.generator_column)
    else:
        generator = pl.col("generator").cast(pl.Utf8)  # supplied by the loader (arena)

    text_name = source.text_column if source.text_column in schema else "text"
    return (
        lf
        .select(
            text=pl.col(text_name).cast(pl.Utf8).str.strip_chars(),
            label=label,
            group=pl.col("group").cast(pl.Utf8) if "group" in schema else column_or_null(source.group_column),
            generator=generator,
            domain=pl.lit(source.domain) if source.domain else column_or_null(source.domain_column),
            attack=column_or_null(source.attack_column),
            dataset=pl.col("dataset") if "dataset" in schema else pl.lit(source.name),
        )
        # A generator only makes sense for AI text; human rows in mixed sources must not carry one.
        .with_columns(generator=pl.when(pl.col("label") == AI).then(pl.col("generator")))
        .filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))
    )


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
        from huggingface_hub import get_token

        return None if get_token() else "gated on Hugging Face: set HF_TOKEN (in .env or the environment)"
    if requires == "kaggle":
        have = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
        have = have or Path("~/.kaggle/kaggle.json").expanduser().exists()
        return None if have else "needs Kaggle credentials: set KAGGLE_API_TOKEN (in .env or the environment)"
    return None
