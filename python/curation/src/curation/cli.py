"""Command line for the dataset pipeline.

curation build --version 4.0.0        build data/4.0.0/{train,test,validation}.parquet
curation sources                      list the configured sources and whether they can be read
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from .build import build
from .config import load_config
from .loaders import missing_credentials

DEFAULT_CONFIG = resources.files("curation") / "sources.toml"


def parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    root = argparse.ArgumentParser(prog="curation", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    root.add_argument("--config", type=Path, help="sources file (default: the one packaged with curation)")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("build", help="build the train/test/validation parquet files")
    run.add_argument("--version", required=True, help="dataset version, used as the output folder name")
    run.add_argument("--data-dir", type=Path, default=Path("data"), help="where versions are written (default: ./data)")
    run.add_argument("--only", help="comma-separated source names to build")
    run.add_argument("--limit-per-source", type=int, help="read only the first N rows of every source (quick runs)")
    run.add_argument(
        "--keep-markdown",
        action="store_true",
        help="do not strip markdown (comparison experiments only: the shipped cleaner strips it)",
    )
    run.add_argument(
        "--balance",
        type=float,
        metavar="RATIO",
        help="in every formatting cell keep at most RATIO majority-class rows per minority-class row (try 2)",
    )
    run.add_argument(
        "--strict", action="store_true", help="fail instead of skipping sources whose credentials are missing"
    )

    commands.add_parser("sources", help="list the configured sources")
    return root


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the process exit code."""
    args = parser().parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> <level>{message}</level>")
    load_dotenv(find_dotenv(usecwd=True))  # HF_TOKEN, KAGGLE_API_TOKEN; never overrides the environment

    config = load_config(args.config or DEFAULT_CONFIG)
    if args.command == "build" and args.balance:
        config = replace(config, balance=args.balance)
    if args.command == "build" and args.keep_markdown:
        config = replace(config, strip_markdown=False)
    if args.command == "sources":
        for source in config.sources:
            status = "disabled" if not source.enabled else missing_credentials(source.requires) or "ok"
            print(f"{source.name:55} {source.loader:8} {status}")
        return 0

    only = set(args.only.split(",")) if args.only else None
    if only and (unknown := only - {s.name for s in config.sources}):
        print(f"unknown sources: {sorted(unknown)}", file=sys.stderr)
        return 2
    metadata = build(
        config,
        args.data_dir / args.version,
        version=args.version,
        only=only,
        limit=args.limit_per_source,
        strict=args.strict,
    )
    counts = metadata["sample_counts"]
    print(f"train={counts['train']:,} test={counts['test']:,} validation={counts['validation']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
