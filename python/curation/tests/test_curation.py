"""Tests for the dataset curation pipeline. Offline: sources are synthetic frames."""

from __future__ import annotations

from pathlib import Path  # ruff: ignore[typing-only-standard-library-import]
from typing import Any

import polars as pl
import pytest
from curation.balance import balance
from curation.build import build
from curation.config import Config, Quota, Source
from curation.guards import add_keys, apply_guards, clean
from curation.loaders import (
    ascii_quotes,
    detokenise,
    flatten_paragraphs,
    lazy_from_frames,
    parse_conversation,
    scan_files,
    standardise,
    strip_markdown,
    trim_fragments,
    unwrap_lines,
)
from curation.select import assign_splits, key_rows
from curation.stats import format_leaks

from curation import loaders

SEED = 42


def source(
    name: str = "s",
    *,
    take: Quota | str | int | dict | None = "all",
    **fields: Any,  # ruff: ignore[any-type]
) -> Source:
    """Build a source with sensible defaults for tests."""
    quota = take if isinstance(take, Quota) else Quota.parse(take)
    fields = {"loader": "frame", "label_column": "label", **fields}
    known = {k: v for k, v in fields.items() if k in Source.__dataclass_fields__ and k != "params"}
    params = {k: v for k, v in fields.items() if k not in Source.__dataclass_fields__ or k == "params"}
    params = {**params.pop("params", {}), **params}
    return Source(name=name, take=quota, params=params, **known)


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


FRACTIONS = (0.7, 0.15, 0.15)


def pick(df: pl.DataFrame, src: Source, fractions: tuple[float, float, float] = FRACTIONS) -> pl.DataFrame:
    return assign_splits(key_rows(standardise(df.lazy(), src), SEED).collect(), src, fractions)


def test_splits_have_exact_sizes_and_never_overlap() -> None:
    src = source(take={"each": 100})
    chosen = pick(frame(200), src)
    counts = {(label, split): n for label, split, n in chosen.group_by("label", "split").len().iter_rows()}
    for label in (0, 1):
        assert counts[label, "train"] == 70
        assert counts[label, "test"] == 15
        assert counts[label, "validation"] == 15
    assert chosen.select("doc").is_duplicated().sum() == 0, "a document is in exactly one split"


def test_selection_is_deterministic_and_row_order_independent() -> None:
    src = source(take=100)
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
    src = source(take={"each": 130})
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
    src = source(take="all")
    chosen = pick(df, src)
    texts = standardise(df.lazy(), src).with_columns(doc=pl.col("text").hash(SEED)).collect()
    joined = texts.join(chosen, on=["doc", "label"])
    per_group = joined.group_by("group").agg(pl.col("split").n_unique().alias("splits"))
    # At most the groups at the two slice boundaries per label can straddle; the rest are intact.
    assert (per_group["splits"] > 1).sum() <= 4
    assert (per_group["splits"] == 1).sum() >= 26


def test_exact_duplicates_inside_a_source_collapse() -> None:
    df = pl.concat([frame(10), frame(10)])
    chosen = pick(df, source(take="all"))
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
        source("mixed", take=Quota.parse({"each": 130})),
        source("ai_only", label=1, take=Quota.parse(60), generator="some-model"),
    )
    return Config(seed=SEED, fractions=(0.7, 0.15, 0.15), prefix_chars=0, sources=sources, digest="test")


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


# --------------------------------------------------------------------------------------------
# Holdout, repairs and the newer loaders
# --------------------------------------------------------------------------------------------


def test_holdout_sources_put_every_row_in_the_holdout_split() -> None:
    chosen = pick(frame(30), source(holdout=True, take=None))
    assert set(chosen["split"].unique()) == {"holdout"}
    assert len(chosen) == 60


def test_a_holdout_row_that_repeats_training_text_is_dropped() -> None:
    rows = [
        ("A shared passage that appears in training and in the holdout.", 1, "train"),
        ("A shared passage that appears in training and in the holdout.", 1, "holdout"),
        ("Only the holdout has this one.", 1, "holdout"),
    ]
    out, report = apply_guards(
        add_keys(pl.DataFrame(rows, schema=["text", "label", "split"], orient="row"), SEED, 0), prefix_chars=0
    )
    assert sorted(out["split"].to_list()) == ["holdout", "train"]
    assert report["duplicate"]["holdout"] == 1


def test_unwrap_lines_joins_hard_wraps_but_keeps_paragraphs() -> None:
    text = "A fully differential\ncalculation is\npresented.\n\nSecond paragraph\nwraps too."
    out = pl.select(unwrap_lines(pl.lit(text))).item()
    assert out == "A fully differential calculation is presented.\n\nSecond paragraph wraps too."


def test_detokenise_undoes_pretokenised_spacing_and_ptb_quotes() -> None:
    text = "It has all the makings ( see below ) , but you ca n't buy it . She said ``No '' and it 's done ."
    out = pl.select(detokenise(pl.lit(text))).item()
    assert out == "It has all the makings (see below), but you can't buy it. She said \"No\" and it's done."


def test_flatten_paragraphs_collapses_all_whitespace() -> None:
    assert pl.select(flatten_paragraphs(pl.lit("One.\n\nTwo.\n  Three."))).item() == "One. Two. Three."


def test_standardise_applies_repairs_and_filters() -> None:
    df = pl.DataFrame({
        "text": ["Hello there, friend of mine.", "As an AI language model, I cannot.", "tiny"],
        "label": [0, 0, 0],
    })
    out = standardise(df.lazy(), source(drop_pattern=r"(?i)language model", min_chars=10)).collect()
    assert out["text"].to_list() == ["Hello there, friend of mine."]


def test_standardise_extracts_generator_and_domain_with_a_regex() -> None:
    df = pl.DataFrame({
        "text": ["human text", "machine text"],
        "label": [1, 0],  # this source labels human text 1
        "src": ["cmv_human", "wp_machine_topical_text-davinci-002"],
    })
    src = source(
        human_value=1, generator_column="src", generator_regex="_machine_(?:continuation|topical|specified)_(.+)$",
        domain_column="src", domain_regex="^(.+?)_(?:human|machine)",
    )  # fmt: skip
    out = standardise(df.lazy(), src).sort("text").collect()
    assert out["label"].to_list() == [0, 1]  # "human text" first: MAGE's label 1 is human
    assert out["generator"].to_list() == [None, "text-davinci-002"]
    assert out["domain"].to_list() == ["cmv", "wp"]


def test_chat_loader_takes_chosen_roles_and_filters_conversations(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8}))
    raw = pl.DataFrame(
        {
            "language": ["English", "French"],
            "model": ["m1", "m2"],
            "conversation": [
                [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
                [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
            ],
        },
        schema={"language": pl.Utf8, "model": pl.Utf8, "conversation": turns},
    )
    monkeypatch.setattr(loaders, "files", lambda _s: raw.lazy())
    base: dict[str, Any] = {"conversation": "conversation", "model_column": "model", "where": {"language": ["English"]}}
    both = loaders.chat(
        source(loader="chat", label=1, label_column=None, roles=["user", "assistant"], **base)
    ).collect()
    assert sorted(both["text"].to_list()) == ["a1", "q1"]
    only = loaders.chat(source(loader="chat", label=1, label_column=None, **base)).collect()
    assert only["text"].to_list() == ["a1"]
    assert only["generator"].to_list() == ["m1"]


def test_answers_loader_explodes_human_and_ai_lists_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pl.DataFrame({"human_answers": [["h1", "h2"], ["h3"]], "chatgpt_answers": [["c1"], ["c2", "c3"]]})
    monkeypatch.setattr(loaders, "files", lambda _s: raw.lazy())
    out = loaders.answers(
        source(loader="answers", label_column=None, human="human_answers", ai="chatgpt_answers")
    ).collect()
    assert out.group_by("label").len().sort("label")["len"].to_list() == [3, 3]
    first = out.filter(pl.col("group") == "0")
    assert sorted(first["text"].to_list()) == ["c1", "h1", "h2"]


def test_stackexchange_loader_keeps_the_best_pre_cutoff_answer_as_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer = pl.Struct({"text": pl.Utf8, "pm_score": pl.Float64})
    rows = pl.DataFrame(
        {
            "qid": [1, 2, 3, 4],
            "date": ["2020/01/01", "2023/05/05", "2021/02/02", "2021/03/03"],
            "answers": [
                [
                    {"text": "<p>Low.</p>", "pm_score": 1.0},
                    {"text": "<p>Best &amp; <b>bold</b>.</p><p>Second paragraph.</p>", "pm_score": 4.0},
                ],
                [{"text": "<p>Too late.</p>", "pm_score": 9.0}],
                [{"text": "<pre><code>x = 1</code></pre>", "pm_score": 5.0}],
                [],
            ],
        },
        schema={"qid": pl.Int64, "date": pl.Utf8, "answers": pl.List(answer)},
    )
    rows.write_parquet(tmp_path / "x.parquet")
    monkeypatch.setattr(loaders, "expand", lambda _repo, _patterns: ["x"])
    monkeypatch.setattr(loaders, "download", lambda _repo, _name: tmp_path / "x.parquet")
    src = source(
        loader="stackexchange", label_column=None, repo="r", sites=["writing.stackexchange.com"], before="2022/12/01"
    )
    out = loaders.stackexchange(src).collect()
    assert [t.strip() for t in out["text"]] == ["Best &amp; bold.\n\nSecond paragraph."]
    assert out["site"].to_list() == ["writing"]
    assert out["group"].to_list() == ["1"]


def test_format_leaks_flags_a_within_source_formatting_gap() -> None:
    n = 400
    human = [f"Human text number {i} without breaks." for i in range(n)]
    ai = [f"AI text number {i} with.\n\nA second paragraph." for i in range(n)]
    df = pl.DataFrame({"text": human + ai, "label": [0] * n + [1] * n, "dataset": ["leaky"] * (2 * n)})
    report = format_leaks(df)
    assert report["overall"]["paragraph_breaks"]["auc"] == pytest.approx(1.0)
    assert report["within_source"]["leaky"]["paragraph_breaks"] == [0.0, 1.0]
    assert report["overall"]["length"]["auc"] > 0.5


def test_strip_think_removes_the_reasoning_trace_and_drops_unterminated_ones() -> None:
    df = pl.DataFrame({
        "text": [
            "<think>\nplanning the story\n</think>\nThe real story begins.",
            "No trace here.",
            "<think> never closes",
        ],
        "label": [1, 1, 1],
    })
    out = standardise(df.lazy(), source(strip_think=True)).collect()
    assert sorted(out["text"].to_list()) == ["No trace here.", "The real story begins."]


def test_trim_fragments_cuts_partial_sentences_at_either_end() -> None:
    text = "ing over the hill. The second sentence is whole. And a third that is cut off mid"
    both = pl.select(trim_fragments(pl.lit(text), "both")).item()
    assert both == "The second sentence is whole."
    assert pl.select(trim_fragments(pl.lit(text), "start")).item().startswith("The second")
    assert pl.select(trim_fragments(pl.lit(text), "end")).item().endswith("whole.")
    whole = "A complete text. It ends properly."
    assert pl.select(trim_fragments(pl.lit(whole), "both")).item() == whole


def test_standardise_drops_fragments_it_cannot_repair() -> None:
    df = pl.DataFrame({"text": ["no sentence boundary here at all", "Good start. Then more."], "label": [0, 0]})
    out = standardise(df.lazy(), source(trim_fragments="both")).collect()
    assert out["text"].to_list() == ["Good start. Then more."]


def test_ascii_quotes() -> None:
    assert pl.select(ascii_quotes(pl.lit("\u201cIt\u2019s fine,\u201d she said."))).item() == '"It\'s fine," she said.'


def test_strip_markdown_removes_markers_and_keeps_the_words() -> None:
    text = "# Title\n\n**Bold** and *italic* and `code`.\n\n- first\n- second\n1. numbered\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = pl.select(strip_markdown(pl.lit(text))).item()
    for marker in ("#", "**", "`", "|", "---"):
        assert marker not in out
    for word in ("Title", "Bold", "italic", "code", "first", "second", "numbered"):
        assert word in out
    assert "- first" not in out
    assert "5 * 3 = 15" in pl.select(strip_markdown(pl.lit("math: 5 * 3 = 15"))).item()


def test_clean_trims_a_fragment_the_cleaner_leaves_behind() -> None:
    df = pl.DataFrame({"text": ["leftover words here. A whole sentence follows. And another one."], "label": [1]})
    assert clean(df, trim="start")["text"].to_list() == ["A whole sentence follows. And another one."]
    assert clean(df)["text"].to_list() == ["leftover words here. A whole sentence follows. And another one."]


def test_remove_pattern_deletes_boilerplate_characters() -> None:
    df = pl.DataFrame({"text": ["Read more \u00bb the story \u2022 continues."], "label": [0]})
    out = standardise(df.lazy(), source(remove_pattern="[\u00bb\u2022]")).collect()
    assert out["text"].to_list() == ["Read more  the story  continues."]


def test_normalise_breaks_and_ascii_punctuation() -> None:
    df = pl.DataFrame({"text": ["One\u2014two \u201cquoted\u201d\u2026\nNext paragraph.\n\n\nLast."], "label": [0]})
    out = standardise(df.lazy(), source(normalise_breaks=True, ascii_punctuation=True)).collect()
    assert out["text"].to_list() == ['One-two "quoted"...\n\nNext paragraph.\n\nLast.']


def test_holdout_each_reserves_rows_per_label() -> None:
    keys = pl.DataFrame({"doc": list(range(200)), "label": [0, 1] * 100, "h": list(range(200))}).cast({
        "doc": pl.UInt64,
        "h": pl.UInt64,
        "label": pl.Int8,
    })
    out = assign_splits(keys, source(holdout_each=5, take="all"), (0.7, 0.15, 0.15))
    counts = out.group_by("split", "label").len()
    held = counts.filter(pl.col("split") == "holdout")
    assert sorted(held["len"].to_list()) == [5, 5]
    assert out.height == 200


def test_balance_limits_the_majority_class_in_a_cell_and_keeps_small_cells() -> None:
    long_text = "word " * 300
    rows = (
        [{"text": f"# Title\n\n{long_text}{i}", "label": 1, "split": "train", "dataset": "a"} for i in range(300)]
        + [{"text": f"# Title\n\n{long_text}x{i}", "label": 0, "split": "train", "dataset": "b"} for i in range(100)]
        + [{"text": f"{long_text}{i}", "label": 0, "split": "train", "dataset": "b"} for i in range(40)]
    )
    df = pl.DataFrame(rows).with_row_index("doc_id")
    kept, report = balance(df, ratio=1.5, seed=1)
    cell = kept.filter(pl.col("text").str.starts_with("#"))
    assert (cell["label"] == 0).sum() == 100  # the minority class is kept whole
    assert (cell["label"] == 1).sum() == 150  # the majority class is cut to 1.5 x
    assert (kept["label"] == 0).sum() == 140  # the 40 plain rows sit in a small cell and are untouched
    assert report["dropped"]["train"] == {"human": 0, "ai": 150}
