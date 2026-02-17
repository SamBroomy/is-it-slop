import pytest
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams


@pytest.fixture
def sample_texts():
    return ["hello world", "test document", "another sample"]


@pytest.fixture
def default_params():
    return VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=False)


@pytest.fixture
def fitted_vectorizer(sample_texts, default_params):
    return TfidfVectorizer.fit(sample_texts, default_params)
