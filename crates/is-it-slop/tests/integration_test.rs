//! Integration tests for the is-it-slop inference pipeline.
//!
//! These tests validate the full end-to-end pipeline from text input to prediction,
//! including text cleaning, tokenization, chunking, vectorization, ONNX inference,
//! and aggregation.

use is_it_slop::{AggregationMethod, Classification, MODEL_VERSION, Predictor, Threshold};

/// Test basic single-text prediction
#[test]
fn test_predict_single_text() {
    let text = "This is a test text for the inference pipeline.";
    let predictor = Predictor::new();
    let result = predictor.predict(text);

    assert!(result.is_ok(), "Prediction should succeed");
    let prediction = result.unwrap();

    // Verify probabilities are valid
    assert!(
        prediction.prediction.ai_probability() >= 0.0,
        "AI probability should be non-negative"
    );
    assert!(
        prediction.prediction.ai_probability() <= 1.0,
        "AI probability should be at most 1.0"
    );

    // Verify probabilities sum to 1.0
    let total = prediction.prediction.ai_probability() + prediction.prediction.human_probability();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "Probabilities should sum to 1.0, got {total}"
    );

    // Verify classification is valid (using default threshold)
    let classification = predictor.classify(text).unwrap();
    assert!(
        matches!(classification, Classification::Human | Classification::AI),
        "Classification should be Human or AI"
    );

    // Verify chunk predictions exist
    assert!(
        !prediction.chunk_predictions.is_empty(),
        "Should have at least one chunk prediction"
    );

    // Verify chunk agreement is in valid range
    assert!(
        prediction.chunk_agreement >= 0.5 && prediction.chunk_agreement <= 1.0,
        "Chunk agreement should be in [0.5, 1.0], got {}",
        prediction.chunk_agreement
    );
}

/// Test prediction with short text (single chunk)
#[test]
#[allow(clippy::float_cmp)]
fn test_predict_short_text() {
    let short_text = "Hello world";
    let predictor = Predictor::new();
    let result = predictor.predict(short_text);

    assert!(result.is_ok(), "Short text prediction should succeed");
    let prediction = result.unwrap();

    // Short text should produce single chunk
    assert_eq!(
        prediction.chunk_predictions.len(),
        1,
        "Short text should produce exactly one chunk"
    );
    // Single chunk should have perfect agreement
    assert_eq!(
        prediction.chunk_agreement, 1.0f32,
        "Single chunk should have perfect agreement"
    );
}

/// Test prediction with long text (multiple chunks)
#[test]
fn test_predict_long_text() {
    // Create text long enough to produce multiple chunks (~500 words)
    let long_text = "word example test sample ".repeat(125);
    let predictor = Predictor::new();
    let result = predictor.predict(&long_text);

    assert!(result.is_ok(), "Long text prediction should succeed");
    let prediction = result.unwrap();

    // Long text should produce multiple chunks
    assert!(
        prediction.chunk_predictions.len() > 1,
        "Long text should produce multiple chunks, got {}",
        prediction.chunk_predictions.len()
    );

    // Verify all chunk predictions are valid
    for (i, chunk_pred) in prediction.chunk_predictions.iter().enumerate() {
        assert!(
            chunk_pred.ai_probability() >= 0.0 && chunk_pred.ai_probability() <= 1.0,
            "Chunk {i} has invalid probability: {}",
            chunk_pred.ai_probability()
        );
    }

    // Chunk agreement should be less than 1.0 for multiple chunks (unless perfectly aligned)
    assert!(
        prediction.chunk_agreement >= 0.5 && prediction.chunk_agreement <= 1.0,
        "Chunk agreement should be in valid range, got {}",
        prediction.chunk_agreement
    );
}

/// Test batch prediction
#[test]
fn test_predict_batch() {
    let texts = vec!["First test text.", "Second test text.", "Third test text."];
    let predictor = Predictor::new();
    let results = predictor.predict_batch(&texts);

    assert!(results.is_ok(), "Batch prediction should succeed");
    let predictions = results.unwrap();

    // Should return one prediction per input
    assert_eq!(
        predictions.len(),
        texts.len(),
        "Should have one prediction per input text"
    );

    // Verify all predictions are valid
    for (i, prediction) in predictions.iter().enumerate() {
        assert!(
            prediction.prediction.ai_probability() >= 0.0
                && prediction.prediction.ai_probability() <= 1.0,
            "Prediction {i} has invalid probability"
        );
        assert!(
            !prediction.chunk_predictions.is_empty(),
            "Prediction {i} should have chunk predictions"
        );
    }
}

/// Test batch prediction with empty input
#[test]
fn test_predict_batch_empty() {
    let texts: Vec<&str> = vec![];
    let predictor = Predictor::new();
    let results = predictor.predict_batch(&texts);

    // Empty batch may either succeed with empty results or fail gracefully
    match results {
        Ok(predictions) => {
            assert_eq!(predictions.len(), 0, "Empty batch should return no results");
        }
        Err(e) => {
            // Error is acceptable for empty batch
            assert!(!e.to_string().is_empty(), "Error should be descriptive");
        }
    }
}

/// Test prediction determinism
#[test]
#[allow(clippy::float_cmp)]
fn test_predict_deterministic() {
    let text = "Deterministic prediction test.";
    let predictor = Predictor::new();

    let result1 = predictor.predict(text).unwrap();
    let result2 = predictor.predict(text).unwrap();

    // Same input should produce identical results
    assert_eq!(
        result1.prediction.ai_probability(),
        result2.prediction.ai_probability(),
        "Predictions should be deterministic"
    );
    assert_eq!(
        result1.classification(predictor.threshold()),
        result2.classification(predictor.threshold()),
        "Classifications should be deterministic"
    );
    assert_eq!(
        result1.chunk_predictions.len(),
        result2.chunk_predictions.len(),
        "Chunk counts should be deterministic"
    );
}

/// Test different aggregation methods
#[test]
fn test_aggregation_methods() {
    // Use text long enough to produce multiple chunks
    let text = "word example test sample ".repeat(50);

    let result_mean = Predictor::new()
        .with_aggregation_method(AggregationMethod::Mean)
        .predict(&text)
        .unwrap();
    let result_max = Predictor::new()
        .with_aggregation_method(AggregationMethod::Max)
        .predict(&text)
        .unwrap();
    let result_weighted = Predictor::new().predict(&text).unwrap();

    // All should succeed and produce valid results
    assert!(result_mean.prediction.ai_probability() >= 0.0);
    assert!(result_max.prediction.ai_probability() >= 0.0);
    assert!(result_weighted.prediction.ai_probability() >= 0.0);

    // If chunks have different predictions, aggregation methods may differ
    // All should have the same chunk predictions (aggregation happens after)
    assert_eq!(
        result_mean.chunk_predictions.len(),
        result_max.chunk_predictions.len(),
        "Same input should produce same chunk count"
    );
}

/// Test batch vs single consistency
#[test]
#[allow(clippy::float_cmp)]
fn test_batch_vs_single_consistency() {
    let texts = vec!["Test one.", "Test two."];
    let predictor = Predictor::new();

    // Batch prediction
    let batch_results = predictor.predict_batch(&texts).unwrap();

    // Individual predictions
    let single_results: Vec<_> = texts
        .iter()
        .map(|&text| predictor.predict(text).unwrap())
        .collect();

    // Results should match
    assert_eq!(batch_results.len(), single_results.len());
    let threshold = predictor.threshold();
    for (batch, single) in batch_results.iter().zip(single_results.iter()) {
        let diff = (batch.prediction.ai_probability() - single.prediction.ai_probability()).abs();
        assert!(
            diff < 1e-6,
            "Batch and single predictions should match (diff = {diff:e})"
        );
        assert_eq!(
            batch.classification(threshold),
            single.classification(threshold),
            "Batch and single classifications should match"
        );
    }
}

/// Test prediction with special characters
#[test]
fn test_predict_special_characters() {
    let text = "Test with émojis 🚀 and spëcial çhars ñ";
    let predictor = Predictor::new();
    let result = predictor.predict(text);

    assert!(
        result.is_ok(),
        "Prediction with special characters should succeed"
    );
}

/// Test prediction with unicode text
#[test]
fn test_predict_unicode() {
    let texts = vec![
        "日本語テキスト",   // Japanese
        "Текст на русском", // Russian
        "النص العربي",      // Arabic
        "中文文本",         // Chinese
    ];
    let predictor = Predictor::new();

    for text in texts {
        let result = predictor.predict(text);
        assert!(
            result.is_ok(),
            "Unicode text prediction should succeed for: {text}"
        );
    }
}

/// Test prediction with whitespace-only text
#[test]
fn test_predict_whitespace_only() {
    let text = "   \n\t  ";
    let predictor = Predictor::new();
    let result = predictor.predict(text);

    assert!(result.is_err(), "Whitespace-only text should be rejected");
}

/// Test prediction with HTML entities
#[test]
fn test_predict_html_entities() {
    let text = "Test&nbsp;with&mdash;HTML&quot;entities";
    let predictor = Predictor::new();
    let result = predictor.predict(text);

    assert!(
        result.is_ok(),
        "HTML entities should be cleaned and handled"
    );
}

/// Test large batch performance
#[test]
fn test_predict_large_batch() {
    let texts: Vec<String> = (0..100).map(|i| format!("Test text number {i}")).collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let predictor = Predictor::new();

    let result = predictor.predict_batch(&text_refs);

    assert!(result.is_ok(), "Large batch should succeed");
    let predictions = result.unwrap();

    assert_eq!(predictions.len(), 100, "Should predict all 100 texts");

    // All predictions should be valid
    for prediction in &predictions {
        assert!(prediction.prediction.ai_probability() >= 0.0);
        assert!(prediction.prediction.ai_probability() <= 1.0);
    }
}

/// Test classification using classify method
#[test]
fn test_classify_method() {
    let text = "Test classification method.";
    let predictor = Predictor::new();
    let result = predictor.classify(text);

    assert!(result.is_ok(), "Classification should succeed");
    let classification = result.unwrap();

    assert!(
        matches!(classification, Classification::Human | Classification::AI),
        "Classification should be Human or AI"
    );
}

/// Test model version is available
#[test]
fn test_model_version_available() {
    assert!(!MODEL_VERSION.is_empty(), "Model version should be set");
    assert!(
        MODEL_VERSION.contains('.'),
        "Model version should be in semver format"
    );
}

/// Test confidence metrics in predictions
#[test]
fn test_confidence_metrics() {
    let text = "Test confidence metrics calculation.";
    let predictor = Predictor::new();
    let result = predictor.predict(text).unwrap();

    // Confidence should be affected by entropy and chunk agreement
    // Just verify it's in valid range
    assert!(
        result.chunk_agreement >= 0.5 && result.chunk_agreement <= 1.0,
        "Chunk agreement should be in [0.5, 1.0]"
    );
}

/// Test chunking behavior with boundary cases
#[test]
fn test_chunking_boundaries() {
    // Test with text around chunking boundaries
    // Default chunk size is 150 tokens, overlap is 15

    // ~150 words (may be ~150 tokens depending on tokenization)
    let boundary_text = "word ".repeat(150);
    let predictor = Predictor::new();
    let result = predictor.predict(&boundary_text);

    assert!(result.is_ok(), "Boundary text should be handled");
    let prediction = result.unwrap();

    // Should produce at least one chunk
    assert!(!prediction.chunk_predictions.is_empty());
}

/// Test empty text handling
#[test]
fn test_predict_empty_text() {
    let predictor = Predictor::new();
    let result = predictor.predict("");

    // Empty text should be handled gracefully
    // May return a valid prediction or error depending on implementation
    match result {
        Ok(prediction) => {
            // If it succeeds, verify it's valid
            assert!(prediction.prediction.ai_probability() >= 0.0);
            assert!(prediction.prediction.ai_probability() <= 1.0);
        }
        Err(e) => {
            // If it errors, error should be meaningful
            let error_msg = e.to_string();
            assert!(!error_msg.is_empty(), "Error message should be descriptive");
        }
    }
}

/// Test mixed-length batch processing
#[test]
fn test_predict_batch_mixed_lengths() {
    let long_text = "Long text ".repeat(100);
    let texts = vec![
        "Short",
        "Medium length text here with more words",
        long_text.as_str(),
    ];
    let predictor = Predictor::new();

    let result = predictor.predict_batch(&texts);

    assert!(result.is_ok(), "Mixed-length batch should succeed");
    let predictions = result.unwrap();

    assert_eq!(predictions.len(), 3, "Should predict all three texts");

    // Verify chunk counts vary with text length
    let short_chunks = predictions[0].chunk_predictions.len();
    let long_chunks = predictions[2].chunk_predictions.len();

    // Long text should have more chunks than short text
    assert!(
        long_chunks >= short_chunks,
        "Longer text should have at least as many chunks"
    );
}

/// Test custom threshold
#[test]
fn test_custom_threshold() {
    let text = "Test custom threshold.";

    // Create predictors with different thresholds
    let predictor_low = Predictor::new().with_threshold(Threshold::new(0.1));
    let predictor_high = Predictor::new().with_threshold(Threshold::new(0.9));

    let class_low = predictor_low.classify(text).unwrap();
    let class_high = predictor_high.classify(text).unwrap();

    // Both should return valid classifications
    assert!(matches!(
        class_low,
        Classification::Human | Classification::AI
    ));
    assert!(matches!(
        class_high,
        Classification::Human | Classification::AI
    ));
}

/// Test batch classification
#[test]
fn test_batch_classification() {
    let texts = vec!["Text 1", "Text 2", "Text 3"];
    let predictor = Predictor::new();

    let results = predictor.classify_batch(&texts);

    assert!(results.is_ok(), "Batch classification should succeed");
    let classifications = results.unwrap();

    assert_eq!(classifications.len(), 3);
    for classification in classifications {
        assert!(matches!(
            classification,
            Classification::Human | Classification::AI
        ));
    }
}
