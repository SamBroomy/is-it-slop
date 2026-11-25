"""Loaders: read a source into the common schema, lazily.

Every loader returns a `LazyFrame`; `standardise` then maps it onto the schema the rest of the
pipeline relies on: `text`, `label`, `group`, `generator`, `domain`, `attack`, `dataset`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.io.plugins import register_io_source

from .config import AI, HUMAN, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pyarrow as pa

COLUMNS = ("text", "label", "group", "generator", "domain", "attack", "dataset")

Loader = "Callable[[Source], pl.LazyFrame]"
LOADERS: dict[str, Callable[[Source], pl.LazyFrame]] = {}


def loader(name: str) -> Callable[[Callable[[Source], pl.LazyFrame]], Callable[[Source], pl.LazyFrame]]:
    """Register a loader under the name `sources.toml` uses."""

    def register(fn: Callable[[Source], pl.LazyFrame]) -> Callable[[Source], pl.LazyFrame]:
        LOADERS[name] = fn
        return fn

    return register


# --------------------------------------------------------------------------------------------
# Generic readers
# --------------------------------------------------------------------------------------------


def _scan(path: str) -> pl.LazyFrame:
    if path.endswith(".parquet"):
        return pl.scan_parquet(path)
    if path.endswith(".csv"):
        return pl.scan_csv(path, infer_schema_length=10_000)
    if path.endswith((".json", ".jsonl")):
        return pl.read_json(path).lazy() if path.endswith(".json") else pl.scan_ndjson(path)
    msg = f"don't know how to read {path}"
    raise ValueError(msg)


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def arrow_scan(table: pa.Table, batch_size: int = 20_000) -> pl.LazyFrame:
    """Scan an Arrow table lazily, converting `batch_size` rows at a time.

    `pl.from_arrow` copies every string of the table, which for a multi-gigabyte news corpus is more
    memory than the machine has. Converting batch by batch keeps the footprint at one batch, and
    projection, predicate and row-limit pushdown are honoured so polars can still optimise around it.
    """
    schema = pl.from_arrow(table.schema.empty_table()).schema

    def source(
        with_columns: list[str] | None, predicate: pl.Expr | None, n_rows: int | None, _hint: int | None
    ) -> Iterator[pl.DataFrame]:
        remaining = n_rows
        for batch in table.to_batches(max_chunksize=batch_size):
            if remaining is not None and remaining <= 0:
                return
            df = pl.from_arrow(batch)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= len(df)
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if len(df):
                yield df

    return register_io_source(source, schema=schema)


@loader("hf_files")
def hf_files(source: Source) -> pl.LazyFrame:
    """Specific data files of a dataset repo (csv, parquet, json), scanned lazily.

    Params: `repo`, `files` (list of paths inside the repo).
    """
    repo = source.params["repo"]
    paths = [_download(repo, f) for f in source.params["files"]]
    return pl.concat([_scan(p) for p in paths], how="diagonal")


@loader("hf_dataset")
def hf_dataset(source: Source) -> pl.LazyFrame:
    """Load a dataset through the `datasets` library, every split concatenated (zero-copy from arrow).

    Params: `repo`, and `subset` or `subsets` (a list; each becomes its own `dataset` value).
    """
    from datasets import load_dataset

    repo = source.params["repo"]
    tail = repo.rsplit("/", maxsplit=1)[-1]
    subsets = source.params.get("subsets") or [source.params.get("subset")]
    frames = []
    for subset in subsets:
        splits = load_dataset(repo, subset) if subset else load_dataset(repo)
        table = pl.concat([arrow_scan(split.data.table) for split in splits.values()], how="diagonal")
        label = f"{tail}/{subset}" if subset and len(subsets) > 1 else None
        frames.append(table.with_columns(dataset=pl.lit(label)) if label else table)
    return pl.concat(frames, how="diagonal")


@loader("kaggle")
def kaggle(source: Source) -> pl.LazyFrame:
    """Load a file of a Kaggle dataset (needs Kaggle credentials). Params: `handle`, `file`."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    return kagglehub.dataset_load(KaggleDatasetAdapter.POLARS, source.params["handle"], source.params["file"]).lazy()


@loader("pairs")
def pairs(source: Source) -> pl.LazyFrame:
    """Datasets with one `human` and one `ai` column per row (a continuation of the same prompt).

    Each pair becomes two rows that share a `group`, so both sides always land in the same split.
    """
    lf = hf_dataset(source).with_row_index("pair")
    return (
        lf
        .select("pair", "human", "ai")
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
    for match in re.finditer(rf"(?P<string>{_STRING})|(?P<open>\{{)|(?P<close>\}})", text, re.DOTALL):
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
    import ast

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
            content = "\n".join(
                p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            )
        turns.append({"role": str(turn.get("role", "")), "content": content or ""})
    return turns


_TURNS = pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8}))


def _turns_from_strings(series: pl.Series) -> pl.Series:
    return pl.Series(series.name, [parse_conversation(s) for s in series], dtype=_TURNS)


def assistant_texts(lf: pl.LazyFrame, conversation: str) -> pl.LazyFrame:
    """One row per assistant turn of `conversation`, from any of the arena message layouts.

    Handles messages whose `content` is a string, or a list of `{type, text}` parts, or whose
    whole conversation is a stringified numpy repr. Keeps the other columns of `lf`.
    """
    schema = lf.collect_schema()
    column = pl.col(conversation)
    if schema[conversation] == pl.Utf8:
        lf = lf.with_columns(column.map_batches(_turns_from_strings, return_dtype=_TURNS))
    turns = lf.explode(conversation).unnest(conversation).filter(pl.col("role") == "assistant")
    content_type = turns.collect_schema()["content"]
    if isinstance(content_type, pl.List):
        text = (
            pl
            .col("content")
            .list.eval(pl.element().struct.field("text").filter(pl.element().struct.field("type") == "text"))
            .list.join("\n")
        )
        return turns.with_columns(text=text)
    return turns.with_columns(text=pl.col("content"))


@loader("arena")
def arena(source: Source) -> pl.LazyFrame:
    """Two-model chat arenas: every assistant reply is an AI text, tagged with the model that wrote it.

    Params: `repo`, `files`, `sides` (list of `[conversation_column, model_column]` pairs),
    optional `column_aliases`. Both replies to one prompt share a `group`.
    """
    base = hf_files(source).with_row_index("convo")
    out = []
    for conversation, model in source.params["sides"]:
        side = base.select("convo", conversation, pl.col(model).alias("generator"))
        out.append(
            assistant_texts(side, conversation).select(
                "text",
                pl.lit(AI, dtype=pl.Int8).alias("label"),
                pl.col("convo").cast(pl.Utf8).alias("group"),
                "generator",
            )
        )
    return pl.concat(out)


# --------------------------------------------------------------------------------------------
# Schema normalisation
# --------------------------------------------------------------------------------------------


def standardise(lf: pl.LazyFrame, source: Source) -> pl.LazyFrame:
    """Map a loader's output onto `COLUMNS`, applying the source's configured column choices."""
    schema = lf.collect_schema().names()

    keep = source.params.get("keep")
    if keep:
        for column, values in keep.items():
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
