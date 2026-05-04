import pickle
from pathlib import Path

import numpy as np
import pytest
from is_it_slop_preprocessing import TfidfVectorizer


def test_save_load_rkyv(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str], tmp_path: Path) -> None:
    path = tmp_path / "vectorizer.rkyv"
    fitted_vectorizer.save(path)
    loaded = TfidfVectorizer.load(path)

    X1 = fitted_vectorizer.transform(sample_texts)
    X2 = loaded.transform(sample_texts)
    np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())


def test_save_load_json(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str], tmp_path: Path) -> None:
    path = tmp_path / "vectorizer.json"
    fitted_vectorizer.save(path)
    loaded = TfidfVectorizer.load(path)

    X1 = fitted_vectorizer.transform(sample_texts)
    X2 = loaded.transform(sample_texts)
    np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())


def test_pickle(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str]) -> None:
    pickled = pickle.dumps(fitted_vectorizer)
    loaded = pickle.loads(pickled)

    X1 = fitted_vectorizer.transform(sample_texts)
    X2 = loaded.transform(sample_texts)
    np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())


def test_vocabulary_preserved(fitted_vectorizer: TfidfVectorizer, tmp_path: Path) -> None:
    path = tmp_path / "vectorizer.rkyv"
    fitted_vectorizer.save(path)
    loaded = TfidfVectorizer.load(path)

    assert fitted_vectorizer.vocabulary == loaded.vocabulary


def test_invalid_extension(fitted_vectorizer: TfidfVectorizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fitted_vectorizer.save(tmp_path / "test.txt")


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TfidfVectorizer.load(tmp_path / "missing.rkyv")
