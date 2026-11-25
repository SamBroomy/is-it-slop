"""Tests for the dataset curation pipeline. Offline: sources are synthetic frames."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from curation.build import build
from curation.config import Config, Quota, Source
from curation.guards import add_keys, apply_guards
from curation.loaders import lazy_from_frames, parse_conversation, scan_files, standardise
from curation.select import assign_splits, key_rows

from curation import loaders

SEED = 42


def source(
    name: str = "s",
    *,
    primary: Quota | str | int | dict | None = None,
    validation: Quota | str | int | dict | None = None,
    **fields: Any,  # ruff: ignore[any-type]
) -> Source:
    """Build a source with sensible defaults for tests."""

    def quota(value: Quota | str | int | dict | None) -> Quota:
        return value if isinstance(value, Quota) else Quota.parse(value)

    fields = {"loader": "frame", "label_column": "label", **fields}
    return Source(name=name, primary=quota(primary), validation=quota(validation), **fields)


def frame(n_per_label: int, *, prefix: str = "doc", groups: int | None = None) -> pl.DataFrame:
    """`n_per_label` rows of each label with distinct texts; optional group ids shared across labels."""
    rows = [
        {
            "text": f"{prefix} {label} number {i} " + "lorem ipsum " * 5,
            "label": label,
            "group": str(i % groups) if groups else None,
        }
        for label in (0, 1)
        for i in range(n_per_label)
    ]
    return pl.DataFrame(rows, schema={"text": pl.Utf8, "label": pl.Int8, "group": pl.Utf8})


# ---------------------------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------------------------


def test_quota_parsing_and_resolution() -> None:
    available = {0: 100, 1: 40}
    assert Quota.parse("all").resolve(available) == {0: 100, 1: 40}
    assert Quota.parse({"each": 50}).resolve(available) == {0: 50, 1: 40}
    assert Quota.parse({"0": 10, "1": "all"}).resolve(available) == {0: 10, 1: 40}
    assert Quota.parse(None).resolve(available) == {0: 0, 1: 0}
    total = Quota.parse(70).resolve(available)
    assert sum(total.values()) == 70
    assert total[0] > total[1], "a total quota is shared in proportion to what exists"
    assert sum(Quota.parse(10_000).resolve(available).values()) == 140, "never more than exists"


@pytest.mark.parametrize("bad", [True, 1.5, {"2": 5}, {}])
def test_quota_rejects_nonsense(bad: object) -> None:
    with pytest.raises((TypeError, ValueError), match=r"quota|per-label"):
        Quota.parse(bad)


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------


def pick(df: pl.DataFrame, src: Source, test_fraction: float = 0.2) -> pl.DataFrame:
    return assign_splits(key_rows(standardise(df.lazy(), src), SEED).collect(), src, test_fraction)


def test_splits_have_exact_sizes_and_never_overlap() -> None:
    src = source(primary=Quota.parse({"each": 50}), validation=Quota.parse({"each": 20}))
    chosen = pick(frame(200), src)
    counts = {(label, split): n for label, split, n in chosen.group_by("label", "split").len().iter_rows()}
    for label in (0, 1):
        assert counts[label, "train"] == 40
        assert counts[label, "test"] == 10
        assert counts[label, "validation"] == 20
    assert chosen.select("doc").is_duplicated().sum() == 0, "a document is in exactly one split"


def test_selection_is_deterministic_and_row_order_independent() -> None:
    src = source(primary=Quota.parse(80), validation=Quota.parse(20))
    df = frame(100)
    first = pick(df, src).sort("doc")
    shuffled = pick(df.sample(fraction=1.0, shuffle=True, seed=7), src).sort("doc")
    assert first.equals(shuffled)


def test_selection_ignores_arrival_order_for_duplicate_texts() -> None:
    # The same text twice with different groups (so different ranking hashes): which row arrives
    # first must not change what is picked.
    rows = [
        {"text": f"duplicated text number {i} " + "x" * 20, "label": label, "group": f"g{i}-{copy}"}
        for i in range(200)
        for label in (0, 1)
        for copy in range(2)
    ]
    df = pl.DataFrame(rows, schema={"text": pl.Utf8, "label": pl.Int8, "group": pl.Utf8})
    src = source(primary={"each": 100}, validation={"each": 30})
    reference = pick(df, src).sort("doc", "label")
    for seed in (1, 2, 3):
        assert reference.equals(pick(df.sample(fraction=1.0, shuffle=True, seed=seed), src).sort("doc", "label"))


def test_variants_of_one_document_share_a_split() -> None:
    # 30 groups x 10 variants per label; both labels share group ids, as RAID's human/AI rows do.
    rows = [
        {"text": f"g{g} label{label} variant{v} " + "x" * 20, "label": label, "group": str(g)}
        for g in range(30)
        for label in (0, 1)
        for v in range(10)
    ]
    df = pl.DataFrame(rows, schema={"text": pl.Utf8, "label": pl.Int8, "group": pl.Utf8})
    src = source(primary="all")
    chosen = pick(df, src)
    texts = standardise(df.lazy(), src).with_columns(doc=pl.col("text").hash(SEED)).collect()
    joined = texts.join(chosen, on=["doc", "label"])
    per_group = joined.group_by("group").agg(pl.col("split").n_unique().alias("splits"))
    # At most the groups at the two slice boundaries per label can straddle; the rest are intact.
    assert (per_group["splits"] > 1).sum() <= 4
    assert (per_group["splits"] == 1).sum() >= 26


def test_exact_duplicates_inside_a_source_collapse() -> None:
    df = pl.concat([frame(10), frame(10)])
    chosen = pick(df, source(primary="all"))
    assert len(chosen) == 20


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------


def guarded(rows: list[tuple[str, int, str]], prefix_chars: int = 0) -> tuple[pl.DataFrame, dict]:
    df = pl.DataFrame(rows, schema={"text": pl.Utf8, "label": pl.Int8, "split": pl.Utf8}, orient="row")
    return apply_guards(add_keys(df, SEED, prefix_chars), prefix_chars=prefix_chars)


def test_guards_drop_cross_split_normalised_duplicates_keeping_the_earliest_split() -> None:
    out, report = guarded([
        ("The cat sat.", 0, "train"),
        ("the   CAT sat.", 0, "test"),
        ("Another document entirely.", 1, "validation"),
    ])
    assert sorted(out["split"].to_list()) == ["train", "validation"]
    assert report["shared_normalised"]["test"] == 1


def test_guards_keep_variants_of_a_document_within_one_split() -> None:
    out, report = guarded([
        ("Original wording of the text.", 0, "train"),
        ("ORIGINAL   wording of the TEXT.", 0, "train"),  # same document, e.g. a RAID attack variant
        ("Original wording of the text.", 0, "train"),  # an exact copy is still a duplicate
    ])
    assert len(out) == 2
    assert report["duplicate"]["train"] == 1
    assert report["shared_normalised"]["train"] == 0


def test_guards_drop_texts_with_conflicting_labels() -> None:
    out, report = guarded([
        ("Same words here.", 0, "train"),
        ("Same words here.", 1, "train"),
        ("Fine on its own.", 1, "train"),
    ])
    assert out["text"].to_list() == ["Fine on its own."]
    assert report["label_conflict"]["train"] == 2


def test_prefix_guard_removes_near_duplicates_across_splits_only() -> None:
    shared = "a long shared opening sentence that is clearly the same document " * 3
    out, report = guarded(
        [
            (shared + "ending one", 0, "train"),
            (shared + "ending two", 0, "test"),
            (shared + "ending three", 0, "train"),
            ("unrelated text", 1, "validation"),
        ],
        prefix_chars=100,
    )
    assert sorted(out["split"].to_list()) == ["train", "train", "validation"]
    assert report["shared_prefix"]["test"] == 1


# ---------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------

ARENA_REPR = (
    "[{'role': 'user', 'content': array([{'type': 'text', 'text': 'array(1) {not json} it\\'s', "
    "'image': None, 'mimeType': None}],\n       dtype=object)}\n"
    " {'role': 'assistant', 'content': array([{'type': 'text', 'text': 'First part', 'image': None, "
    "'mimeType': None},\n       {'type': 'text', 'text': 'Second } part', 'image': None, 'mimeType': None}],\n"
    "       dtype=object)}]"
)


def test_stringified_arena_conversation_is_parsed_without_eval() -> None:
    turns = parse_conversation(ARENA_REPR)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "array(1) {not json} it's"
    assert turns[1]["content"] == "First part\nSecond } part"
    assert parse_conversation("not a conversation") == []
    assert parse_conversation(None) == []


def test_assistant_texts_handles_every_message_layout() -> None:
    plain = pl.DataFrame({"messages": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]})
    parts = pl.DataFrame({
        "messages": [
            [{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "image", "text": None}]}]
        ]
    })
    string = pl.DataFrame({"messages": [ARENA_REPR]})
    assert loaders.assistant_texts(plain.lazy(), "messages").collect()["text"].to_list() == ["hello"]
    assert loaders.assistant_texts(parts.lazy(), "messages").collect()["text"].to_list() == ["a"]
    assert loaders.assistant_texts(string.lazy(), "messages").collect()["text"].to_list() == [
        "First part\nSecond } part"
    ]


def test_lazy_from_frames_streams_and_honours_pushdown() -> None:
    chunks = [
        pl.DataFrame({"id": list(range(i, i + 7)), "text": [f"t{n}" for n in range(i, i + 7)]}) for i in range(0, 70, 7)
    ]
    lf = lazy_from_frames(lambda: iter(chunks), lambda: chunks[0].schema)

    assert lf.collect().equals(pl.concat(chunks))
    assert lf.select("id").collect().columns == ["id"]
    assert lf.filter(pl.col("id") % 10 == 0).collect()["id"].to_list() == list(range(0, 70, 10))
    assert lf.head(20).collect()["id"].to_list() == list(range(20))
    assert lf.filter(pl.col("id") > 60).select("text").collect()["text"].to_list() == [f"t{i}" for i in range(61, 70)]


def test_scan_files_reads_every_format(tmp_path) -> None:  # ruff: ignore[missing-type-function-argument]
    df = pl.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    df.write_parquet(tmp_path / "x.parquet")
    df.write_csv(tmp_path / "x.csv")
    df.write_ndjson(tmp_path / "x.jsonl")
    (tmp_path / "x.json").write_text(df.write_json())
    (tmp_path / "y.json").write_text(pl.DataFrame({"text": ["c"], "label": [1], "extra": [1.5]}).write_json())

    for name in ("x.parquet", "x.csv", "x.jsonl", "x.json"):
        assert scan_files([tmp_path / name]).collect().sort("text").equals(df), name
    both = scan_files([tmp_path / "x.json", tmp_path / "y.json"], ["text"]).collect()
    assert both["text"].to_list() == ["a", "b", "c"], "JSON files are read one after another"
    assert both.columns == ["text"], "only the requested columns are parsed"

    # A column whose structure differs between files must not stop the columns we do use.
    (tmp_path / "z1.json").write_text('[{"text": "p", "meta": {"a": 1}}]')
    (tmp_path / "z2.json").write_text('[{"text": "q", "meta": {"b": "x", "c": [1]}}]')
    assert scan_files([tmp_path / "z1.json", tmp_path / "z2.json"], ["text"]).collect()["text"].to_list() == ["p", "q"]


def test_files_loader_resolves_patterns_and_labels_subsets(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # ruff: ignore[missing-type-function-argument]
    names = ("data/a-1.csv", "data/a-2.csv", "b.csv", "README.md")
    for name in names[:3]:
        (tmp_path / name.replace("/", "_")).write_text("text,label\nx,0\n")
    monkeypatch.setattr(loaders, "repo_files", lambda _repo: names)
    monkeypatch.setattr(loaders, "download", lambda _repo, name: tmp_path / name.replace("/", "_"))

    assert loaders.expand("o/r", ["data/a-*.csv", "b.csv"]) == ["data/a-1.csv", "data/a-2.csv", "b.csv"]
    with pytest.raises(FileNotFoundError, match="nothing"):
        loaders.expand("o/r", ["nothing*"])

    plain = source(loader="files", params={"repo": "o/r", "files": ["data/a-*.csv"]})
    assert loaders.files(plain).collect().height == 2
    labelled = source(loader="files", params={"repo": "o/r", "files": {"first": "data/a-1.csv", "second": "b.csv"}})
    assert loaders.files(labelled).collect()["dataset"].to_list() == ["r/first", "r/second"]


def test_standardise_keeps_a_generator_the_loader_supplies() -> None:
    raw = pl.DataFrame({"text": ["an ai reply"], "label": [1], "generator": ["gpt-4o"]})
    out = standardise(raw.lazy(), source(label_column=None, label=1)).collect()
    assert out["generator"].to_list() == ["gpt-4o"]


def test_standardise_maps_labels_generators_and_blanks() -> None:
    raw = pl.DataFrame({
        "generation": ["human words", "ai words", "   ", None],
        "model": ["human", "gpt4", "gpt4", "gpt4"],
        "domain": ["news", "news", "news", "news"],
    })
    src = source(
        text_column="generation", label_column="model", human_value="human",
        generator_column="model", domain_column="domain",
    )  # fmt: skip
    out = standardise(raw.lazy(), src).collect()
    assert out["text"].to_list() == ["human words", "ai words"], "blank and null texts are dropped"
    assert out["label"].to_list() == [0, 1]
    assert out["generator"].to_list() == [None, "gpt4"], "only AI rows carry a generator"


# ---------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> Config:
    """Two synthetic sources registered under a `frame` loader."""
    frames = {"mixed": frame(300, prefix="mixed"), "ai_only": frame(150, prefix="ai").filter(pl.col("label") == 1)}
    monkeypatch.setitem(loaders.LOADERS, "frame", lambda src: frames[src.name].lazy())
    sources = (
        source("mixed", primary=Quota.parse({"each": 100}), validation=Quota.parse({"each": 30})),
        source("ai_only", label=1, primary=Quota.parse(60), generator="some-model"),
    )
    return Config(seed=SEED, test_fraction=0.2, prefix_chars=0, sources=sources, digest="test")


def test_build_writes_disjoint_splits_with_provenance(synthetic: Config, tmp_path) -> None:  # ruff: ignore[missing-type-function-argument]
    meta = build(synthetic, tmp_path, version="0.0.1")
    train, test, val = (pl.read_parquet(tmp_path / f"{s}.parquet") for s in ("train", "test", "validation"))

    assert (len(train), len(test), len(val)) == (
        meta["sample_counts"]["train"], meta["sample_counts"]["test"], meta["sample_counts"]["validation"],
    )  # fmt: skip
    assert {"text", "label", "dataset", "generator", "domain", "attack", "doc_id"} <= set(train.columns)
    ids = [set(d["doc_id"].to_list()) for d in (train, test, val)]
    assert not ids[0] & ids[1], "train and test share no document"
    assert not ids[0] & ids[2], "train and validation share no document"
    assert not ids[1] & ids[2], "test and validation share no document"
    assert set(train["dataset"].cast(pl.Utf8).unique()) == {"mixed", "ai_only"}
    assert train.filter(pl.col("dataset") == "ai_only")["generator"].unique().to_list() == ["some-model"]
    assert train.filter(pl.col("label") == 0)["generator"].null_count() == (train["label"] == 0).sum()
    assert meta["config_sha256"] == "test"
    assert meta["markdown_bias"] == {} or "ratio" in meta["markdown_bias"]


def test_build_is_reproducible(synthetic: Config, tmp_path) -> None:  # ruff: ignore[missing-type-function-argument]
    build(synthetic, tmp_path / "a", version="0.0.1")
    build(synthetic, tmp_path / "b", version="0.0.1")
    for split in ("train", "test", "validation"):
        a = pl.read_parquet(tmp_path / "a" / f"{split}.parquet")
        b = pl.read_parquet(tmp_path / "b" / f"{split}.parquet")
        assert a.equals(b)


def test_sources_needing_credentials_are_skipped_not_fatal(synthetic: Config, tmp_path, monkeypatch) -> None:  # ruff: ignore[missing-type-function-argument]
    import dataclasses

    gated = dataclasses.replace(synthetic.sources[0], name="gated", requires="hf_token")
    config = dataclasses.replace(synthetic, sources=(*synthetic.sources, gated))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    meta = build(config, tmp_path, version="0.0.1")
    assert "gated" in meta["skipped_sources"]
    with pytest.raises(RuntimeError, match="gated"):
        build(config, tmp_path / "strict", version="0.0.1", strict=True)
