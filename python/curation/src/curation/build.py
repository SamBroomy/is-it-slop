"""Build the train / test / validation splits from the configured sources."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import is_it_slop_preprocessing
import polars as pl
from loguru import logger

from . import stats
from .balance import balance
from .features import COLUMNS as FEATURE_COLUMNS
from .features import add_features
from .guards import add_keys, apply_guards, clean
from .loaders import load_source, missing_credentials
from .select import PROPORTIONAL, SPLITS, assign_splits, key_rows

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config, Source

DISABLED = "disabled in sources.toml"

OUTPUT_COLUMNS = ("text", "label", "dataset", "generator", "domain", "attack", "doc_id", "split", *FEATURE_COLUMNS)


def pick_source(source: Source, config: Config, limit: int | None) -> pl.DataFrame:
    """Load one source and return only the rows its quotas select, cleaned, with a `split` column."""
    lf = load_source(source, limit)
    keys = key_rows(lf, config.seed).collect(engine="streaming")
    chosen = assign_splits(keys, source, config.fractions)
    picked = (
        lf
        .with_columns(doc=pl.col("text").hash(config.seed))
        .join(chosen.lazy(), on=["doc", "label"], how="inner")
        .collect(engine="streaming")
        # The streaming engine does not preserve row order, so "first" would be arbitrary when one
        # text appears twice with different metadata: sort on a total order, then keep the first.
        .sort("doc", "label", "dataset", "generator", "domain", "attack", "group", nulls_last=True, maintain_order=True)
        .unique(subset=["doc", "label"], keep="first", maintain_order=True)
        .drop("doc")
    )
    # Style features are measured on the text as collected, so the markdown the cleaner removes is still counted.
    return clean(add_features(picked), strip_markdown=config.strip_markdown, trim=source.trim_fragments)


def _balance(df: pl.DataFrame, config: Config) -> tuple[pl.DataFrame, dict[str, Any] | None]:
    """Balance formatting between the classes in train, test and validation when `config.balance` asks; the holdout is untouched."""
    if config.balance <= 0:
        return df, None
    is_proportional = pl.col("split").cast(pl.Utf8).is_in(PROPORTIONAL)
    balanced, report = balance(df.filter(is_proportional), ratio=config.balance, seed=config.seed)
    logger.info("balance {}: dropped {}", config.balance, report["dropped"])
    return pl.concat([balanced, df.filter(~is_proportional)]), report


def build(
    config: Config,
    out_dir: Path,
    *,
    version: str,
    only: set[str] | None = None,
    limit: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Run the whole pipeline, write the splits and `dataset_metadata.json`, return the metadata."""
    frames: list[pl.DataFrame] = []
    skipped: dict[str, str] = {}
    selected: dict[str, dict[str, int]] = {}
    todo = [source for source in config.sources if not only or source.name in only]
    for number, source in enumerate(todo, 1):
        if not source.enabled:
            skipped[source.name] = DISABLED
            continue
        if reason := missing_credentials(source.requires):
            if strict:
                msg = f"{source.name}: {reason}"
                raise RuntimeError(msg)
            skipped[source.name] = reason
            logger.warning("[{}/{}] skipping {}: {}", number, len(todo), source.name, reason)
            continue
        logger.info("[{}/{}] {}", number, len(todo), source.name)
        started = time.monotonic()
        picked = pick_source(source, config, limit)
        selected[source.name] = {s: int((picked["split"] == s).sum()) for s in SPLITS}
        counts = selected[source.name]
        shown = " / ".join(f"{counts[s]:,} {s}" for s in SPLITS if counts[s])
        logger.info("    picked {} in {:.0f}s", shown, time.monotonic() - started)
        frames.append(picked)
    if not frames:
        msg = "no source produced any rows"
        raise RuntimeError(msg)

    df = pl.concat(frames, how="diagonal")
    del frames
    df, guard_report = apply_guards(add_keys(df, config.seed, config.prefix_chars), prefix_chars=config.prefix_chars)
    df = df.with_columns(pl.col("dataset", "split").cast(pl.Categorical))
    df, balance_report = _balance(df, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        part = df.filter(pl.col("split") == split).select(OUTPUT_COLUMNS)
        if split not in PROPORTIONAL and part.is_empty():
            continue  # no holdout sources, no holdout file
        splits[split] = part.sample(fraction=1.0, shuffle=True, seed=config.seed)
        splits[split].write_parquet(out_dir / f"{split}.parquet", compression="zstd")

    metadata = describe(config, version, splits, guard_report, selected, skipped, limit, balance_report)
    (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.success("wrote {}", out_dir)
    if missing := {name: why for name, why in skipped.items() if why != DISABLED}:
        logger.warning(
            "This dataset is INCOMPLETE: {} source(s) were skipped. Set the credentials to get all of it:\n{}",
            len(missing),
            "\n".join(f"  - {name}: {why}" for name, why in missing.items()),
        )
    return metadata


def describe(
    config: Config,
    version: str,
    splits: dict[str, pl.DataFrame],
    guard_report: dict[str, dict[str, int]],
    selected: dict[str, dict[str, int]],
    skipped: dict[str, str],
    limit: int | None,
    balance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata file: provenance of the run plus what is in each split."""
    return {
        "dataset_version": version,
        "created_timestamp": datetime.now(UTC).isoformat(),
        "preprocessing_version": is_it_slop_preprocessing.__version__,
        "polars_version": pl.__version__,
        "seed": config.seed,
        "strip_markdown": config.strip_markdown,
        "balance": balance_report,
        "config_sha256": config.digest,
        "limit_per_source": limit,
        "sample_counts": {**{s: len(d) for s, d in splits.items()}, "total": sum(len(d) for d in splits.values())},
        "dataset_composition": {s: stats.composition(d) for s, d in splits.items()},
        "class_balance": {s: stats.class_balance(d) for s, d in splits.items()},
        "statistics": {s: stats.length_stats(d, seed=config.seed) for s, d in splits.items()},
        "markdown_bias": stats.markdown_bias(splits["train"]),
        "format_leaks": stats.format_leaks(splits["train"]),
        "selected_before_guards": selected,
        "guards_removed": guard_report,
        "skipped_sources": skipped,
    }
