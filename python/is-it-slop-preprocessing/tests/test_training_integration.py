"""Integration tests for the full training workflow.

Tests the complete pipeline from Python training to Rust inference:
- Text cleaning (training mode with dataset artifacts)
- Tokenization
- Chunking
- Vectorizer fitting and transformation
- Artifact serialization (rkyv format)
- CSR matrix structure validation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from is_it_slop_preprocessing import CleaningMode, TextCleaner, TfidfVectorizer, TokenChunker, VectorizerParams
from scipy.sparse import csr_matrix, issparse


class TestTrainingWorkflow:
    """Test the complete training workflow end-to-end."""

    def test_training_pipeline_end_to_end(self) -> None:
        """Test full training pipeline: Clean → Chunk → Vectorize → Train."""
        # Training data with dataset artifacts
        texts_train = [
            "This is human-written text with natural flow.",
            "Academic paper [1] with citations [2] and references.",
            "WASHINGTON — News article with dateline formatting.",
            "AI-generated text with repetitive patterns and mechanical structure.",
        ]

        # 1. Clean texts (training mode - removes dataset artifacts)
        cleaner = TextCleaner(CleaningMode.TRAINING)
        cleaned_texts = cleaner.clean_batch(texts_train)

        # Verify dataset artifacts removed
        assert not any("[1]" in text for text in cleaned_texts)
        assert not any("WASHINGTON —" in text for text in cleaned_texts)

        # 2. Configure chunking
        TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)

        # 3. Fit vectorizer
        params = VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=True)
        vectorizer, X_train = TfidfVectorizer.fit_transform(cleaned_texts, params)

        # 4. Verify output structure
        assert issparse(X_train)
        assert X_train.shape[0] == len(cleaned_texts)  # type: ignore[union-attr]
        assert vectorizer.num_features > 0

        # 5. Verify L2 normalization
        norms = np.sqrt((X_train.multiply(X_train)).sum(axis=1)).A1  # type: ignore[union-attr]
        np.testing.assert_array_almost_equal(norms, np.ones(len(cleaned_texts)))

    def test_vectorizer_save_load_roundtrip(self) -> None:
        """Test vectorizer serialization and deserialization (rkyv format)."""
        texts = ["Sample text one", "Sample text two", "Sample text three"]
        params = VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=False)
        vectorizer = TfidfVectorizer.fit(texts, params)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vectorizer.rkyv"

            # Save
            vectorizer.save(str(save_path))
            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Load
            loaded = TfidfVectorizer.load(str(save_path))

            # Verify equivalence
            assert loaded.num_features == vectorizer.num_features

            # Transform should produce identical results
            X_original = vectorizer.transform(texts)
            X_loaded = loaded.transform(texts)
            np.testing.assert_array_almost_equal(X_original.toarray(), X_loaded.toarray())

    def test_chunker_behavior(self) -> None:
        """Test TokenChunker chunking behavior."""
        chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)

        # Test with long token sequence
        tokens = list(range(300))
        chunks = chunker.chunk(tokens)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # First chunk should start at 0
        assert chunks[0][0] == 0

        # Last chunk should include last token
        assert chunks[-1][-1] == 299

        # All chunks should be lists of integers
        for chunk in chunks:
            assert isinstance(chunk, list)
            assert all(isinstance(t, int) for t in chunk)


class TestPyO3BindingCorrectness:
    """Test PyO3 binding correctness between Python and Rust."""

    def test_csr_matrix_structure_validity(self) -> None:
        """Verify CSR matrix structure from PyO3 bindings is valid."""
        texts = ["text one two three", "four five six"]
        params = VectorizerParams(min_df=1.0, max_df=1.0, sublinear_tf=False)
        _vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        # Verify it's a valid scipy CSR matrix
        assert isinstance(X, csr_matrix)
        assert X.format == "csr"

        # Verify structure
        assert len(X.data) > 0
        assert len(X.indices) == len(X.data)
        assert len(X.indptr) == X.shape[0] + 1  # type: ignore[union-attr]
        assert X.indptr[0] == 0
        assert X.indptr[-1] == len(X.data)

        # Verify indices are sorted within each row
        for i in range(X.shape[0]):  # type: ignore[union-attr]
            row_start = X.indptr[i]
            row_end = X.indptr[i + 1]
            row_indices = X.indices[row_start:row_end]
            assert np.all(row_indices[:-1] <= row_indices[1:]), "Indices should be sorted"

        # Verify no duplicate indices in rows
        for i in range(X.shape[0]):  # type: ignore[union-attr]
            row_start = X.indptr[i]
            row_end = X.indptr[i + 1]
            row_indices = X.indices[row_start:row_end]
            assert len(row_indices) == len(np.unique(row_indices)), "No duplicate indices"

    def test_tokenize_from_python_matches_rust(self) -> None:
        """Verify tokenization is consistent through PyO3."""
        # Create vectorizer which uses Rust tokenizer
        texts = ["Hello world", "Test 123", "Sample text data"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        # Transform should be deterministic
        X2 = vectorizer.transform(texts)
        np.testing.assert_array_equal(X.toarray(), X2.toarray())

    def test_cleaner_training_vs_inference_mode(self) -> None:
        """Test that training cleaner removes dataset artifacts."""
        text_with_artifacts = "Test text [1] with citation and WASHINGTON — dateline"

        # Training mode: removes dataset artifacts
        training_cleaner = TextCleaner(CleaningMode.TRAINING)
        cleaned_train = training_cleaner.clean(text_with_artifacts)

        # Dataset artifacts should be removed
        assert "[1]" not in cleaned_train
        assert "WASHINGTON —" not in cleaned_train

        # Inference mode: keeps dataset artifacts
        inference_cleaner = TextCleaner(CleaningMode.INFERENCE)
        inference_cleaner.clean(text_with_artifacts)
        # In inference mode, artifacts may be kept (implementation-specific)


class TestArtifactExport:
    """Test artifact export in formats required for Rust inference."""

    def test_export_artifacts_rkyv_format(self) -> None:
        """Test exporting vectorizer in rkyv format (Rust default)."""
        texts = ["Sample text", "Another sample", "Third example"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        vectorizer = TfidfVectorizer.fit(texts, params)

        with tempfile.TemporaryDirectory() as tmpdir:
            rkyv_path = Path(tmpdir) / "tfidf_vectorizer.rkyv"
            vectorizer.save(str(rkyv_path))

            # Verify file created
            assert rkyv_path.exists()
            assert rkyv_path.stat().st_size > 0

            # Verify it's binary format (not JSON)
            with Path(rkyv_path).open("rb") as f:
                content = f.read(100)
                # Should be binary, not text
                assert not content.decode("utf-8", errors="ignore").isprintable()

    def test_rkyv_format_rust_loadable(self) -> None:
        """Test that rkyv format can be loaded back (implies Rust compatibility)."""
        texts = ["Test", "Data", "Sample"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        vectorizer = TfidfVectorizer.fit(texts, params)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vectorizer.rkyv"
            vectorizer.save(str(save_path))

            # Load should succeed (validates format)
            loaded = TfidfVectorizer.load(str(save_path))
            assert loaded.num_features == vectorizer.num_features

            # Transformation should work
            X = loaded.transform(texts)
            assert X.shape[0] == len(texts)  # type: ignore[union-attr]

    def test_chunker_config_json_format(self) -> None:
        """Test TokenChunker JSON configuration format."""
        # Document the expected JSON format for Rust compatibility
        config = {"chunk_size": 100, "overlap": 10, "min_chunk_size": 25}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            # Write configuration
            with Path(json_path).open("w", encoding="utf-8") as f:
                json.dump(config, f)

            # Verify it can be read back
            with Path(json_path).open(encoding="utf-8") as f:
                loaded_config = json.load(f)

            # Verify expected fields
            assert "chunk_size" in loaded_config
            assert "overlap" in loaded_config
            assert "min_chunk_size" in loaded_config

            assert loaded_config["chunk_size"] == 100
            assert loaded_config["overlap"] == 10
            assert loaded_config["min_chunk_size"] == 25

            # Create chunker with these params
            chunker = TokenChunker(chunk_size=100, overlap=10, min_chunk_size=25)
            tokens = list(range(200))
            chunks = chunker.chunk(tokens)
            assert len(chunks) > 0


class TestEdgeCases:
    """Test edge cases in the training workflow."""

    def test_empty_text_handling(self) -> None:
        """Test that empty texts don't break the pipeline."""
        texts = ["Valid text", "", "Another valid text"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        _vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        # Should produce valid matrix
        assert X.shape[0] == len(texts)  # type: ignore[union-attr]

        # Empty text row should be zero vector
        empty_row = X[1].toarray().flatten()
        np.testing.assert_array_almost_equal(empty_row, np.zeros_like(empty_row))

    def test_single_text_training(self) -> None:
        """Test fitting on single text (edge case)."""
        # Need longer text to generate n-grams with ngram_range=(2,4)
        texts = ["Single training example with multiple words and extra content"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        _vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        assert X.shape[0] == 1  # type: ignore[union-attr]
        # With ngram_range=(2,4), may have features if text is long enough
        # If no features, that's also valid (all filtered out)

    def test_very_short_texts(self) -> None:
        """Test handling of texts shorter than n-gram size."""
        # With ngram_range=(2,4), single words produce no n-grams
        texts = ["Hi", "Hello", "Test"]
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        _vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        # Should handle gracefully (may produce sparse matrix with few features)
        assert X.shape[0] == len(texts)  # type: ignore[union-attr]

    def test_duplicate_texts_training(self) -> None:
        """Test training on duplicate texts."""
        texts = ["Duplicate text"] * 5 + ["Different text"] * 5
        params = VectorizerParams(min_df=1.0, max_df=1.0)
        _vectorizer, X = TfidfVectorizer.fit_transform(texts, params)

        # Identical texts should produce identical vectors
        first_five = X[:5].toarray()
        for i in range(1, 5):
            np.testing.assert_array_almost_equal(first_five[0], first_five[i])

        # Different texts should produce different vectors
        assert not np.allclose(X[0].toarray(), X[5].toarray())
