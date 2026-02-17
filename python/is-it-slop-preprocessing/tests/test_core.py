import numpy as np
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams
from scipy.sparse import issparse


def test_params_creation() -> None:
    params = VectorizerParams(min_df=10.0, max_df=0.9, sublinear_tf=True)
    assert params.min_df == 10.0
    assert params.ngram_range == (2, 4)


def test_fit(sample_texts, default_params) -> None:
    vectorizer = TfidfVectorizer.fit(sample_texts, default_params)
    assert vectorizer.num_features > 0


def test_transform(fitted_vectorizer, sample_texts) -> None:
    X = fitted_vectorizer.transform(sample_texts)
    assert issparse(X)
    assert X.shape[0] == len(sample_texts)


def test_fit_transform_equivalence(sample_texts, default_params) -> None:
    _v1, X1 = TfidfVectorizer.fit_transform(sample_texts, default_params)
    v2 = TfidfVectorizer.fit(sample_texts, default_params)
    X2 = v2.transform(sample_texts)

    np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())


def test_l2_normalization(fitted_vectorizer, sample_texts) -> None:
    X = fitted_vectorizer.transform(sample_texts)
    norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
    np.testing.assert_array_almost_equal(norms, np.ones(len(sample_texts)))


def test_vocabulary(fitted_vectorizer) -> None:
    vocab = fitted_vectorizer.vocabulary
    assert isinstance(vocab, dict)
    assert all(isinstance(k, str) for k in vocab)


def test_numpy_array_input() -> None:
    texts = np.array(["text1", "text2"])
    params = VectorizerParams(min_df=1.0, max_df=1.0)
    vectorizer = TfidfVectorizer.fit(texts, params)
    assert vectorizer.num_features > 0


def test_determinism() -> None:
    texts = ["test"]
    params = VectorizerParams(min_df=1.0, max_df=1.0)

    _, X1 = TfidfVectorizer.fit_transform(texts, params)
    _, X2 = TfidfVectorizer.fit_transform(texts, params)

    assert (X1 != X2).nnz == 0
