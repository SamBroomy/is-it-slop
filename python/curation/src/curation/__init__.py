"""Dataset curation: turn many public human/AI text sources into train, test and validation splits.

    just curation::build 4.0.0

What it does, in order:

1. Each source in `sources.toml` is loaded lazily into one common schema (`text`, `label`, `group`,
   `generator`, `domain`, `attack`). Label 0 is human, 1 is AI.
2. Rows are picked *before* any text is cleaned or held in memory: a cheap first pass computes a
   stable hash per row and ranks rows within each label, then quotas slice that ranking into
   train / test / validation. Rows that share a `group` (variants of one source document) hash
   identically, so they land in the same split.
3. Only the picked rows are materialised and cleaned with the Rust training cleaner.
4. Duplicates and cross-split overlap (exact, normalised and shared-prefix) are removed with joins.
5. The splits and a metadata file (composition, balance, statistics, what each guard removed) are
   written.

Everything is deterministic for a given `sources.toml`, seed and pinned polars version.
"""

from __future__ import annotations

__all__: list[str] = []
