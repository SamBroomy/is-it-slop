import numpy as np
from is_it_slop_preprocessing import CleaningMode, TextCleaner, TokenChunker, tokenize


def test_tokenize_basic() -> None:
    texts = ["hello world", "test document"]
    result = tokenize(texts)

    assert len(result) == 2
    assert all(isinstance(tokens, list) for tokens in result)
    assert all(len(tokens) > 0 for tokens in result)
    assert all(isinstance(tok, int) for tokens in result for tok in tokens)

    result_single = tokenize(["hello world"])
    assert len(result_single) == 1


def test_tokenize_empty_string() -> None:
    result = tokenize(["normal", "", "text"])
    assert len(result) == 3
    assert result[1] == []


def test_tokenize_unicode() -> None:
    texts = ["こんにちは world", "Hello 世界", "你好 test"]
    result = tokenize(texts)
    assert all(len(tokens) > 0 for tokens in result)


def test_tokenize_numpy_array_input() -> None:
    texts = np.array(["text1", "text2"])
    result = tokenize(texts)
    assert len(result) == 2


def test_chunk_batch_correctness() -> None:
    chunker = TokenChunker(chunk_size=100, overlap=10, min_chunk_size=30)
    tokens_seq = [list(range(250)), list(range(50)), []]

    batched = chunker.chunk_batch(tokens_seq)
    sequential = [chunker.chunk(ts) for ts in tokens_seq]

    assert len(batched) == len(sequential)
    for i, (b, s) in enumerate(zip(batched, sequential, strict=True)):
        assert b == s, f"chunk_batch and chunk disagree at index {i}"


def test_chunker_behavior() -> None:
    chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)

    tokens = list(range(300))
    chunks = chunker.chunk(tokens)

    assert len(chunks) > 1
    assert chunks[0][0] == 0
    assert chunks[-1][-1] == 299
    for chunk in chunks:
        assert isinstance(chunk, list)
        assert all(isinstance(t, int) for t in chunk)


def test_chunker_to_dict() -> None:
    chunker = TokenChunker(chunk_size=200, overlap=20, min_chunk_size=50)
    d = chunker.to_dict()

    assert d == {"chunk_size": 200, "overlap": 20, "min_chunk_size": 50}


def test_cleaner_training_vs_inference_mode() -> None:
    text_with_artifacts = "Test text [1] with citation and WASHINGTON — dateline"

    # Training mode: removes dataset artifacts
    training_cleaner = TextCleaner(CleaningMode.TRAINING)
    cleaned_train = training_cleaner.clean(text_with_artifacts)

    assert "[1]" not in cleaned_train
    assert "WASHINGTON —" not in cleaned_train

    # Inference mode: keeps dataset-specific patterns, only universal cleanup
    inference_cleaner = TextCleaner(CleaningMode.INFERENCE)
    cleaned_infer = inference_cleaner.clean(text_with_artifacts)
    assert "citation" in cleaned_infer


def test_cleaner_batch(sample_texts: list[str]) -> None:
    cleaner = TextCleaner(CleaningMode.INFERENCE)
    cleaned = cleaner.clean_batch(sample_texts)
    assert len(cleaned) == len(sample_texts)
    assert all(isinstance(t, str) for t in cleaned)
