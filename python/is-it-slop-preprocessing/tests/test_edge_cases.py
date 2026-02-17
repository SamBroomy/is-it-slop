import pytest
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams


def test_empty_string() -> None:
    texts = ["normal", "", "text"]
    params = VectorizerParams(min_df=1.0, max_df=1.0)
    _, X = TfidfVectorizer.fit_transform(texts, params)
    assert X[1].nnz == 0


def test_unicode() -> None:
    texts = ["こんにちは world", "Hello 世界", "你好 test"]
    params = VectorizerParams(min_df=1.0, max_df=1.0)
    vectorizer = TfidfVectorizer.fit(texts, params)
    assert vectorizer.num_features > 0


def test_min_df_filtering() -> None:
    texts = ["a b", "b c", "c d"]
    params1 = VectorizerParams(min_df=1.0, max_df=1.0)
    params2 = VectorizerParams(min_df=2.0, max_df=1.0)

    v1 = TfidfVectorizer.fit(texts, params1)
    v2 = TfidfVectorizer.fit(texts, params2)

    assert v1.num_features >= v2.num_features


def test_sublinear_tf_effect() -> None:
    texts = ["a " * 100, "b"]
    params_linear = VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=False)
    params_sub = VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=True)

    _, X1 = TfidfVectorizer.fit_transform(texts, params_linear)
    _, X2 = TfidfVectorizer.fit_transform(texts, params_sub)

    assert X1[0].max() != X2[0].max()


def test_oov_handling() -> None:
    train = ["known words"]
    test = ["completely different unknown"]

    params = VectorizerParams(min_df=1.0, max_df=1.0)
    vectorizer = TfidfVectorizer.fit(train, params)
    X = vectorizer.transform(test)

    # Should not crash, may have sparse/zero rows
    assert X.shape[0] == 1


def test_invalid_input() -> None:
    with pytest.raises(TypeError):
        TfidfVectorizer.fit("not a list", VectorizerParams(1.0, 1.0))
