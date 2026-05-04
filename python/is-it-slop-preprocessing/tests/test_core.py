import numpy as np
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, VectorizerParams, tokenize
from scipy.sparse import issparse


def test_params_creation() -> None:
    params = VectorizerParams(min_df=10.0, max_df=0.9, sublinear_tf=True)
    assert params.min_df == 10.0
    assert params.ngram_range == (2, 4)


def test_fit(sample_texts: list[str], default_params: VectorizerParams) -> None:
    vectorizer = TfidfVectorizer.fit(sample_texts, default_params)
    assert vectorizer.num_features > 0


def test_transform(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str]) -> None:
    X = fitted_vectorizer.transform(sample_texts)
    assert issparse(X)
    assert X.shape[0] == len(sample_texts)  # type: ignore[union-attr]


def test_fit_transform_equivalence(sample_texts: list[str], default_params: VectorizerParams) -> None:
    _v1, X1 = TfidfVectorizer.fit_transform(sample_texts, default_params)
    v2 = TfidfVectorizer.fit(sample_texts, default_params)
    X2 = v2.transform(sample_texts)

    np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())


def test_l2_normalization(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str]) -> None:
    X = fitted_vectorizer.transform(sample_texts)
    norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1  # type: ignore[union-attr]
    np.testing.assert_array_almost_equal(norms, np.ones(len(sample_texts)))


def test_vocabulary(fitted_vectorizer: TfidfVectorizer) -> None:
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

    assert (X1 != X2).nnz == 0  # type: ignore[union-attr]


def test_vectorize_from_tokens(fitted_vectorizer: TfidfVectorizer, sample_texts: list[str]) -> None:
    tokens = tokenize(sample_texts)
    X_from_tokens = fitted_vectorizer.vectorize_from_tokens(tokens)
    X_from_texts = fitted_vectorizer.transform(sample_texts)

    assert X_from_tokens.shape == X_from_texts.shape
    np.testing.assert_array_almost_equal(X_from_tokens.toarray(), X_from_texts.toarray())


def test_chunk_batch_correctness() -> None:
    chunker = TokenChunker(chunk_size=100, overlap=10, min_chunk_size=30)
    tokens_seq = [list(range(250)), list(range(50)), []]

    batched = chunker.chunk_batch(tokens_seq)
    sequential = [chunker.chunk(ts) for ts in tokens_seq]

    assert len(batched) == len(sequential)
    for i, (b, s) in enumerate(zip(batched, sequential, strict=True)):
        assert b == s, f"chunk_batch and chunk disagree at index {i}"


def test_tokenize_basic() -> None:
    texts = ["hello world", "test document"]
    result = tokenize(texts)

    assert len(result) == 2
    assert all(isinstance(tokens, list) for tokens in result)
    assert all(len(tokens) > 0 for tokens in result)
    assert all(isinstance(tok, int) for tokens in result for tok in tokens)

    result_single = tokenize(["hello world"])
    assert len(result_single) == 1


def test_chunker_to_dict() -> None:
    chunker = TokenChunker(chunk_size=200, overlap=20, min_chunk_size=50)
    d = chunker.to_dict()

    assert d == {"chunk_size": 200, "overlap": 20, "min_chunk_size": 50}


def test_vectorizer_params_match_fitted() -> None:
    params = VectorizerParams(min_df=5.0, max_df=0.8, sublinear_tf=True)
    v = TfidfVectorizer.fit(["test"], params)

    assert v.params.min_df == 5.0
    assert abs(v.params.max_df - 0.8) < 1e-6
    assert v.params.sublinear_tf is True
