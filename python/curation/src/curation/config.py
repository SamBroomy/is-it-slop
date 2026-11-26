"""Configuration: the `sources.toml` schema and the quota arithmetic."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from pathlib import Path

HUMAN, AI = 0, 1
_TOLERANCE = 1e-9
ALL = "all"


@dataclass(frozen=True)
class Quota:
    """How many rows of a source go to one split.

    Written in TOML as:

    * `"all"`: every available row,
    * an integer: that many rows in total, split across labels in proportion to what exists,
    * `{ each = n }`: up to `n` rows of every label that exists,
    * `{ 0 = n, 1 = "all" }`: an explicit amount per label (labels not listed get none).
    """

    total: int | None = None
    each: int | None = None
    per_label: dict[int, int | str] = field(default_factory=dict)
    everything: bool = False

    @classmethod
    def parse(cls, value: Any) -> Quota:  # ruff: ignore[any-type]
        """Build a quota from its TOML value; `None` means no rows."""
        if value is None:
            return cls()
        if value == ALL:
            return cls(everything=True)
        if isinstance(value, bool):
            msg = f"invalid quota {value!r}"
            raise TypeError(msg)
        if isinstance(value, int):
            return cls(total=value)
        if isinstance(value, dict):
            if "each" in value:
                return cls(each=int(value["each"]))
            per_label = {int(k): v for k, v in value.items()}
            if not per_label or not set(per_label) <= {HUMAN, AI}:
                msg = f"a per-label quota needs keys 0 and/or 1, got {value!r}"
                raise ValueError(msg)
            return cls(per_label=per_label)
        msg = f"invalid quota {value!r}"
        raise TypeError(msg)

    def resolve(self, available: dict[int, int]) -> dict[int, int]:
        """Rows to take per label given how many rows each label has; never more than available."""
        if self.everything:
            return dict(available)
        if self.each is not None:
            return {label: min(self.each, n) for label, n in available.items()}
        if self.total is not None:
            population = sum(available.values())
            if population == 0:
                return dict.fromkeys(available, 0)
            take = min(self.total, population)
            shares = {label: round(take * n / population) for label, n in available.items()}
            # Rounding can overshoot by a row; trim from the largest label.
            while sum(shares.values()) > take:
                shares[max(shares, key=shares.__getitem__)] -= 1
            return {label: min(n, available[label]) for label, n in shares.items()}
        out = dict.fromkeys(available, 0)
        for label, amount in self.per_label.items():
            if label in available:
                out[label] = available[label] if amount == ALL else min(int(amount), available[label])
        return out


@dataclass(frozen=True)
class Source:
    """One dataset: where it comes from, how to read it, how much of it to use and how to repair it."""

    name: str
    loader: str
    take: Quota = field(default_factory=Quota)
    holdout: bool = False
    holdout_each: int = 0  # reserve this many rows per label (at most 10%) for the holdout; the rest is split as usual
    label: int | None = None
    label_column: str | None = None
    human_value: str | int | None = None
    generator: str | None = None
    generator_column: str | None = None
    generator_regex: str | None = None
    domain: str | None = None
    domain_column: str | None = None
    domain_regex: str | None = None
    attack_column: str | None = None
    group_column: str | None = None
    text_column: str = "text"
    requires: str | None = None
    enabled: bool = True
    # Repairs for how a source was collected (not for what its authors wrote):
    unwrap_lines: bool = False  # join hard-wrapped lines, keep paragraph breaks
    detokenise: bool = False  # undo pre-tokenised spacing ("it 's", "word .") and PTB quotes
    flatten_paragraphs: bool = False  # collapse every run of whitespace, newlines included, to one space
    remove_pattern: str | None = None  # delete everything matching this regex (boilerplate characters)
    normalise_breaks: bool = (
        False  # every newline (run) becomes one paragraph break (sources that mark paragraphs with \n)
    )
    ascii_punctuation: bool = False  # curly quotes, en/em dashes and ellipses to ASCII (when only one side keeps them)
    ascii_quotes: bool = False  # curly quotes to straight ones (when only one side of a source uses them)
    trim_fragments: str | None = None  # "start", "end" or "both": cut text that begins or ends mid-sentence
    strip_think: bool = (
        False  # drop a leading <think>...</think> reasoning trace (rows with an unterminated one are dropped)
    )
    drop_pattern: str | None = None  # drop rows whose text matches this regex (contamination filter)
    min_chars: int = 0  # drop rows shorter than this after repairs
    params: dict[str, Any] = field(default_factory=dict)

    def referenced_columns(self) -> list[str]:
        """List the columns of the raw files that this source's configuration reads (sorted, unique)."""
        named = [
            self.text_column,
            self.label_column,
            self.generator_column,
            self.domain_column,
            self.attack_column,
            self.group_column,
            *self.params.get("keep", {}),
        ]
        return sorted({name for name in named if name})

    def validate(self) -> None:
        """Reject sources whose label cannot be determined or whose size is unspecified."""
        has_label = (
            self.label in {HUMAN, AI}
            or self.label_column is not None
            or self.loader in {"pairs", "arena", "chat", "answers", "stackexchange"}
        )
        if not has_label:
            msg = f"source {self.name!r} needs `label` or `label_column`"
            raise ValueError(msg)
        if not self.holdout and self.take == Quota():
            msg = f"source {self.name!r} needs `take` (or `holdout = true`)"
            raise ValueError(msg)


@dataclass(frozen=True)
class Config:
    """Pipeline-wide settings plus the sources."""

    seed: int
    fractions: tuple[float, float, float]  # train, test, validation
    prefix_chars: int
    sources: tuple[Source, ...]
    digest: str
    balance: float = (
        0.0  # keep at most this many majority-class rows per minority-class row in each formatting cell (0 = off)
    )
    strip_markdown: bool = False  # remove what markdown leaves behind in every source (an experiment)


_SOURCE_KEYS = {f.name for f in fields(Source) if f.name != "params"} | {"notes"}  # keys that are not `params`  # fmt: skip
_REMOVED_KEYS = {"primary": "take", "validation": "[settings] train/test/validation fractions"}


def load_config(path: Path | Traversable) -> Config:
    """Read and validate a sources file."""
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode())
    settings = raw.get("settings", {})
    train, test, validation = (
        float(settings.get(k, d)) for k, d in (("train", 0.70), ("test", 0.15), ("validation", 0.15))
    )
    if abs(train + test + validation - 1.0) > _TOLERANCE:
        msg = f"[settings] train + test + validation must be 1, got {train + test + validation}"
        raise ValueError(msg)
    sources = []
    for entry in raw.get("source", []):
        if removed := _REMOVED_KEYS.keys() & entry.keys():
            key = min(removed)
            msg = f"source {entry.get('name')!r}: `{key}` was replaced by {_REMOVED_KEYS[key]}"
            raise ValueError(msg)
        known = {k: v for k, v in entry.items() if k in _SOURCE_KEYS and k != "notes"}
        known["take"] = Quota.parse(entry.get("take"))
        known["params"] = {k: v for k, v in entry.items() if k not in _SOURCE_KEYS}
        source = Source(**known)
        source.validate()
        sources.append(source)
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        msg = "duplicate source names in config"
        raise ValueError(msg)
    return Config(
        seed=int(settings.get("seed", 42)),
        fractions=(train, test, validation),
        prefix_chars=int(settings.get("prefix_chars", 200)),
        strip_markdown=bool(settings.get("strip_markdown", False)),
        balance=float(settings.get("balance", 0.0)),
        sources=tuple(sources),
        digest=hashlib.sha256(raw_bytes).hexdigest(),
    )
