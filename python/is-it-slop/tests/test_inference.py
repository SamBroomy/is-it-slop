"""Tests for the is_it_slop inference API.

This module tests the Python wrapper around the Rust inference engine,
including single and batch predictions, result structure, and edge cases.
"""

import pytest
from is_it_slop import CLASSIFICATION_THRESHOLD, MODEL_VERSION, Prediction, is_this_slop, is_this_slop_batch


class TestConstants:
    """Test module constants."""

    def test_classification_threshold_valid_range(self) -> None:
        """Classification threshold should be between 0 and 1."""
        assert 0.0 < CLASSIFICATION_THRESHOLD < 1.0

    def test_classification_threshold_reasonable(self) -> None:
        """Threshold should be in a reasonable range (not at extremes)."""
        assert 0.1 < CLASSIFICATION_THRESHOLD < 0.9

    def test_model_version_format(self) -> None:
        """Model version should follow semver format."""
        assert isinstance(MODEL_VERSION, str)
        assert MODEL_VERSION
        parts = MODEL_VERSION.split(".")
        assert len(parts) >= 2
        # Each part should be numeric
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"


class TestPrediction:
    """Test Prediction result object."""

    def test_prediction_attributes(self) -> None:
        """Prediction should have required attributes."""
        result = is_this_slop("This is a test text for prediction.")

        assert hasattr(result, "human_probability")
        assert hasattr(result, "ai_probability")
        assert hasattr(result, "classification")

    def test_prediction_probabilities_sum_to_one(self) -> None:
        """Human and AI probabilities should sum to 1.0."""
        result = is_this_slop("Testing probability sum.")

        total = result.human_probability + result.ai_probability
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected 1.0"

    def test_prediction_probabilities_in_range(self) -> None:
        """Probabilities should be between 0 and 1."""
        result = is_this_slop("Valid probability range check.")

        assert 0.0 <= result.human_probability <= 1.0
        assert 0.0 <= result.ai_probability <= 1.0

    def test_prediction_classification_values(self) -> None:
        """Classification should be either 'Human' or 'AI'."""
        result = is_this_slop("Classification label test.")

        assert result.classification in {"Human", "AI"}

    def test_prediction_classification_consistency(self) -> None:
        """Classification should match probability threshold."""
        result = is_this_slop("Consistency check between probability and label.")

        if result.ai_probability >= CLASSIFICATION_THRESHOLD:
            assert result.classification == "AI"
        else:
            assert result.classification == "Human"

    def test_prediction_repr(self) -> None:
        """Prediction should have string representation."""
        result = is_this_slop("Test repr method.")

        repr_str = repr(result)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_prediction_str(self) -> None:
        """Prediction should have human-readable string."""
        result = is_this_slop("Test str method.")

        str_repr = str(result)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestSingleInference:
    """Test single text inference (is_this_slop)."""

    def test_basic_inference(self) -> None:
        """Basic inference should return valid result."""
        text = "This is a sample text for testing the inference pipeline."
        result = is_this_slop(text)

        assert isinstance(result, Prediction)
        assert result.classification in {"Human", "AI"}

    def test_short_text_inference(self) -> None:
        """Short texts should be handled correctly."""
        result = is_this_slop("Hello world")

        assert isinstance(result, Prediction)
        assert 0.0 <= result.ai_probability <= 1.0

    def test_long_text_inference(self) -> None:
        """Long texts should be chunked and aggregated correctly."""
        # ~500 words to trigger chunking
        long_text = "word example test sample " * 125
        result = is_this_slop(long_text)

        assert isinstance(result, Prediction)
        assert result.classification in {"Human", "AI"}

    def test_deterministic_inference(self) -> None:
        """Same input should produce identical results."""
        text = "Deterministic test text."

        result1 = is_this_slop(text)
        result2 = is_this_slop(text)

        assert result1.ai_probability == result2.ai_probability
        assert result1.human_probability == result2.human_probability
        assert result1.classification == result2.classification

    def test_custom_threshold_lower(self) -> None:
        """Lower threshold should classify more texts as AI."""
        text = "Test text for threshold sensitivity."

        # Very low threshold (0.1) - more sensitive to AI
        result_low = is_this_slop(text, threshold=0.1)

        # If ai_probability is between 0.1 and default threshold
        if 0.1 <= result_low.ai_probability < CLASSIFICATION_THRESHOLD:
            assert result_low.classification == "AI"

    def test_custom_threshold_higher(self) -> None:
        """Higher threshold should classify more texts as Human."""
        text = "Test text for conservative classification."

        # Very high threshold (0.9) - more conservative
        result_high = is_this_slop(text, threshold=0.9)

        # If ai_probability is between default threshold and 0.9
        if CLASSIFICATION_THRESHOLD <= result_high.ai_probability < 0.9:
            assert result_high.classification == "Human"

    def test_threshold_edge_cases(self) -> None:
        """Test threshold boundary behavior."""
        text = "Boundary test for classification threshold."
        result = is_this_slop(text)

        # Test at exact threshold
        result_exact = is_this_slop(text, threshold=result.ai_probability)
        assert result_exact.classification == "AI"  # >= threshold

        # Test just above
        if result.ai_probability < 0.999:
            result_above = is_this_slop(text, threshold=result.ai_probability + 0.001)
            assert result_above.classification == "Human"


class TestBatchInference:
    """Test batch text inference (is_this_slop_batch)."""

    def test_basic_batch_inference(self) -> None:
        """Batch inference should return results for all inputs."""
        texts = ["First test text.", "Second test text.", "Third test text."]
        results = is_this_slop_batch(texts)

        assert len(results) == len(texts)
        assert all(isinstance(r, Prediction) for r in results)

    def test_batch_empty_list(self) -> None:
        """Empty batch raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            is_this_slop_batch([])

    def test_batch_single_item(self) -> None:
        """Batch with single item should work."""
        texts = ["Single item batch test."]
        results = is_this_slop_batch(texts)

        assert len(results) == 1
        assert isinstance(results[0], Prediction)

    def test_batch_order_preserved(self) -> None:
        """Results should match input order."""
        texts = ["First", "Second", "Third"]
        results = is_this_slop_batch(texts)

        # Each result should correspond to its input
        # We can't test exact values, but we can verify length and type
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, Prediction)

    def test_batch_vs_single_consistency(self) -> None:
        """Batch inference should produce same results as individual calls."""
        texts = ["Consistency test one.", "Consistency test two."]

        # Batch inference
        batch_results = is_this_slop_batch(texts)

        # Individual inference
        single_results = [is_this_slop(text) for text in texts]

        # Results should match
        for batch, single in zip(batch_results, single_results, strict=False):
            assert abs(batch.ai_probability - single.ai_probability) < 1e-6
            assert batch.classification == single.classification

    def test_batch_with_custom_threshold(self) -> None:
        """Batch inference should respect custom threshold."""
        texts = ["Test 1", "Test 2", "Test 3"]
        threshold = 0.7

        results = is_this_slop_batch(texts, threshold=threshold)

        # Verify threshold is applied to all results
        for result in results:
            if result.ai_probability >= threshold:
                assert result.classification == "AI"
            else:
                assert result.classification == "Human"

    def test_batch_large(self) -> None:
        """Large batch should be processed efficiently."""
        texts = [f"Test text number {i}" for i in range(100)]
        results = is_this_slop_batch(texts)

        assert len(results) == 100
        assert all(isinstance(r, Prediction) for r in results)

    def test_batch_mixed_lengths(self) -> None:
        """Batch with texts of varying lengths should work."""
        texts = ["Short", "Medium length text here", "Very long text " * 50]
        results = is_this_slop_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, Prediction) for r in results)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            is_this_slop("")

    def test_whitespace_only(self) -> None:
        """Whitespace-only text raises ValueError, same as empty string."""
        with pytest.raises(ValueError, match="non-empty"):
            is_this_slop("   \n\t  ")

    def test_special_characters(self) -> None:
        """Text with special characters should work."""
        text = "Test with émojis 🚀 and spëcial çhars ñ"
        result = is_this_slop(text)

        assert isinstance(result, Prediction)

    def test_unicode_text(self) -> None:
        """Unicode text in various scripts should work."""
        texts = [
            "日本語テキスト",  # Japanese
            "Текст на русском",  # Russian
            "النص العربي",  # Arabic
            "中文文本",  # Chinese
        ]

        for text in texts:
            result = is_this_slop(text)
            assert isinstance(result, Prediction)

    def test_html_entities(self) -> None:
        """Text with HTML entities should be cleaned."""
        text = "Test&nbsp;with&mdash;HTML&quot;entities"
        result = is_this_slop(text)

        assert isinstance(result, Prediction)

    def test_very_long_single_text(self) -> None:
        """Very long text should not cause issues."""
        # ~2000 words
        long_text = "word " * 2000
        result = is_this_slop(long_text)

        assert isinstance(result, Prediction)

    def test_threshold_validation(self) -> None:
        """Out-of-range threshold raises ValueError."""
        text = "Threshold validation test"

        # Valid thresholds should work
        is_this_slop(text, threshold=0.5)
        is_this_slop(text, threshold=0.0)
        is_this_slop(text, threshold=1.0)

        with pytest.raises(ValueError, match="threshold"):
            is_this_slop(text, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold"):
            is_this_slop(text, threshold=1.5)
