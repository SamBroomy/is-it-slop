mod model;
mod pipeline;

pub use model::CLASSIFICATION_THRESHOLD;

use crate::model::MODEL;

/// Get [P(Human), P(AI)] probabilities for a single text
pub fn predict_probabilities(text: impl AsRef<str>) -> ort::Result<[f32; 2]> {
    let ai_prob = pipeline::predict(&MODEL, text.as_ref())?[0];
    Ok([1.0 - ai_prob, ai_prob])
}

/// Get probabilities for multiple texts
pub fn predict_probabilities_batch(texts: &[&str]) -> ort::Result<Vec<[f32; 2]>> {
    texts
        .iter()
        .map(|&text| predict_probabilities(text))
        .collect()
}

/// Get class (0=Human, 1=AI) using built-in CLASSIFICATION_THRESHOLD
pub fn predict_class(text: impl AsRef<str>) -> ort::Result<i64> {
    let ai_prob = pipeline::predict(&MODEL, text.as_ref())?[0];
    Ok(i64::from(ai_prob >= CLASSIFICATION_THRESHOLD as f32))
}

/// Get class with custom threshold
pub fn predict_class_with_threshold(text: &str, threshold: f32) -> ort::Result<i64> {
    let ai_prob = pipeline::predict(&MODEL, text)?[0];
    Ok(i64::from(ai_prob >= threshold))
}

/// Batch class predictions with built-in threshold
pub fn predict_class_batch(texts: &[&str]) -> ort::Result<Vec<i64>> {
    texts.iter().map(|&text| predict_class(text)).collect()
}

/// Batch class predictions with custom threshold
pub fn predict_class_batch_with_threshold(texts: &[&str], threshold: f32) -> ort::Result<Vec<i64>> {
    texts
        .iter()
        .map(|&text| predict_class_with_threshold(text, threshold))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TEXT: &str = "This is a sample text for testing the inference pipeline.";

    #[test]
    fn test_predict_probabilities() {
        let probs = predict_probabilities(TEST_TEXT).unwrap();
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to 1.0 (within floating point tolerance)
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-5);
        // Each probability should be between 0 and 1
        assert!(probs[0] >= 0.0 && probs[0] <= 1.0);
        assert!(probs[1] >= 0.0 && probs[1] <= 1.0);
    }

    #[test]
    fn test_predict_class() {
        let class = predict_class(TEST_TEXT).unwrap();
        // Class should be 0 or 1
        assert!(class == 0 || class == 1);
    }

    #[test]
    fn test_predict_class_with_threshold() {
        // Test with very low threshold (everything should be AI)
        let class_low = predict_class_with_threshold(TEST_TEXT, 0.0).unwrap();
        assert_eq!(class_low, 1);

        // Test with very high threshold (everything should be human)
        let class_high = predict_class_with_threshold(TEST_TEXT, 1.0).unwrap();
        assert_eq!(class_high, 0);
    }

    #[test]
    fn test_batch_predictions() {
        let texts = vec!["Text 1", "Text 2", "Text 3"];

        // Test probability batch
        let probs = predict_probabilities_batch(&texts).unwrap();
        assert_eq!(probs.len(), 3);
        for prob_pair in &probs {
            assert!((prob_pair[0] + prob_pair[1] - 1.0).abs() < 1e-5);
        }

        // Test class batch
        let classes = predict_class_batch(&texts).unwrap();
        assert_eq!(classes.len(), 3);
        for &class in &classes {
            assert!(class == 0 || class == 1);
        }
    }
}
