"""Pytest configuration for is-it-slop inference tests."""

import pytest


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing inference."""
    return [
        "This is a short human-written text.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can detect patterns in text data.",
        "Natural language processing enables computers to understand human language.",
        "Artificial intelligence has transformed many industries in recent years.",
    ]


@pytest.fixture(scope="session")
def long_text() -> str:
    """Long text sample for testing chunking behavior."""
    return (
        """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language, in particular how to program computers to process and analyze large
    amounts of natural language data. The goal is a computer capable of understanding
    the contents of documents, including the contextual nuances of the language within them.
    The technology can then accurately extract information and insights contained in the
    documents as well as categorize and organize the documents themselves.

    Challenges in natural language processing frequently involve speech recognition,
    natural-language understanding, and natural-language generation. Modern NLP algorithms
    are based on machine learning, especially statistical machine learning. The paradigm
    of machine learning is different from that of most prior attempts at language processing.

    Prior implementations of language-processing tasks typically involved the direct hand
    coding of large sets of rules. The machine-learning paradigm calls instead for using
    statistical inference to automatically learn such rules through the analysis of large
    corpora of typical real-world examples.
    """
        * 5
    )  # Repeat to create longer text
