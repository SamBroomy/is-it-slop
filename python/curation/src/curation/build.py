"""Build the train / test / validation splits from the configured sources."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from . import stats
from .guards import add_keys, apply_guards, clean
from .loaders import load_source, missing_credentials
from .select import SPLITS, assign_splits, key_rows

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config, Source

log = logging.getLogger("curation")

OUTPUT_COLUMNS = ("text", "label", "dataset", "generator", "domain", "attack", "doc_id")


def pick_source(source: Source, config: Config, limit: int | None) -> pl.DataFrame:
    """Load one source and return only the rows its quotas select, cleaned, with a `split` column."""
    lf = load_source(source, limit)
    keys = key_rows(lf, config.seed).collect(engine="streaming")
    chosen = assign_splits(keys, source, config.test_fraction)
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
    return clean(picked)


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
    for source in config.sources:
        if only and source.name not in only:
            continue
        if not source.enabled:
            skipped[source.name] = "disabled in sources.toml"
            continue
        if reason := missing_credentials(source.requires):
            if strict:
                msg = f"{source.name}: {reason}"
                raise RuntimeError(msg)
            skipped[source.name] = reason
            log.warning("skipping %s: %s", source.name, reason)
            continue
        log.info("loading %s", source.name)
        picked = pick_source(source, config, limit)
        selected[source.name] = {s: int((picked["split"] == s).sum()) for s in SPLITS}
        log.info("  %s -> %s", source.name, selected[source.name])
        frames.append(picked)
    if not frames:
        msg = "no source produced any rows"
        raise RuntimeError(msg)

    df = pl.concat(frames, how="diagonal")
    del frames
    df, guard_report = apply_guards(add_keys(df, config.seed, config.prefix_chars), prefix_chars=config.prefix_chars)
    df = df.with_columns(pl.col("dataset").cast(pl.Categorical))

    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        part = df.filter(pl.col("split") == split).select(OUTPUT_COLUMNS)
        splits[split] = part.sample(fraction=1.0, shuffle=True, seed=config.seed)
        splits[split].write_parquet(out_dir / f"{split}.parquet", compression="zstd")

    metadata = describe(config, version, splits, guard_report, selected, skipped, limit)
    (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir)
    return metadata


def describe(
    config: Config,
    version: str,
    splits: dict[str, pl.DataFrame],
    guard_report: dict[str, dict[str, int]],
    selected: dict[str, dict[str, int]],
    skipped: dict[str, str],
    limit: int | None,
) -> dict[str, Any]:
    """Build the metadata file: provenance of the run plus what is in each split."""
    import is_it_slop

    return {
        "dataset_version": version,
        "created_timestamp": datetime.now(UTC).isoformat(),
        "preprocessing_version": is_it_slop.__version__,
        "polars_version": pl.__version__,
        "seed": config.seed,
        "config_sha256": config.digest,
        "limit_per_source": limit,
        "sample_counts": {**{s: len(d) for s, d in splits.items()}, "total": sum(len(d) for d in splits.values())},
        "dataset_composition": {s: stats.composition(d) for s, d in splits.items()},
        "class_balance": {s: stats.class_balance(d) for s, d in splits.items()},
        "statistics": {s: stats.length_stats(d, seed=config.seed) for s, d in splits.items()},
        "markdown_bias": stats.markdown_bias(splits["train"]),
        "selected_before_guards": selected,
        "guards_removed": guard_report,
        "skipped_sources": skipped,
    }
