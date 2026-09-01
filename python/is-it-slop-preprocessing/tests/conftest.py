import pytest


@pytest.fixture
def sample_texts() -> list[str]:
    return ["hello world", "test document", "another sample"]
