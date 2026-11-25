"""Smoke tests for the bindings; the cleaner and tokenizer themselves are tested in Rust."""

from __future__ import annotations

from is_it_slop_preprocessing import __version__, clean_for_training, count_tokens


def test_clean_keeps_order_and_length() -> None:
    texts = ["  Hello   world  ", "", "Second text."]
    cleaned = clean_for_training(texts)
    assert len(cleaned) == len(texts)
    assert not cleaned[1]
    assert "Hello" in cleaned[0]


def test_clean_accepts_any_iterable() -> None:
    assert clean_for_training(iter(["a b c"])) == clean_for_training(["a b c"])


def test_count_tokens_is_positive_for_text_and_zero_for_empty() -> None:
    assert count_tokens(["The quick brown fox", ""]) == [4, 0]


def test_version_is_exposed() -> None:
    assert __version__
