"""Command line: `python -m curation build --version 4.0.0`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .build import build
from .config import load_config
from .memory import watch

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="curation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build train/test/validation parquet files")
    b.add_argument("--version", required=True, help="dataset version; output goes to data/<version>/")
    b.add_argument("--config", type=Path, default=Path(__file__).with_name("sources.toml"))
    b.add_argument("--out", type=Path, help="output directory (default data/<version>)")
    b.add_argument("--only", help="comma-separated source names to build (for testing)")
    b.add_argument("--limit-per-source", type=int, help="read only the first N rows of every source (quick dev runs)")
    b.add_argument("--max-memory-gb", type=float, default=16.0, help="stop if the process uses more RAM than this")
    b.add_argument("--strict", action="store_true", help="fail instead of skipping sources that need credentials")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "datasets", "filelock", "urllib3", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")  # HF_TOKEN etc.; never overrides variables already set
    watch(args.max_memory_gb)
    config = load_config(args.config)
    only = set(args.only.split(",")) if args.only else None
    if only and (unknown := only - {s.name for s in config.sources}):
        parser.error(f"unknown sources: {sorted(unknown)}")
    metadata = build(
        config,
        args.out or ROOT / "data" / args.version,
        version=args.version,
        only=only,
        limit=args.limit_per_source,
        strict=args.strict,
    )
    counts = metadata["sample_counts"]
    print(f"train={counts['train']:,} test={counts['test']:,} validation={counts['validation']:,}")
    if metadata["skipped_sources"]:
        print("skipped:", metadata["skipped_sources"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
