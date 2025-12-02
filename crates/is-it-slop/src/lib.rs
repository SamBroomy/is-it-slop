#[cfg(feature = "python")]
mod python;

#[cfg(feature = "cli")]
pub mod cli;

mod model;
mod pipeline;

use anyhow::Context;
pub use model::CLASSIFICATION_THRESHOLD;
pub use pipeline::{Classification, Prediction};

use crate::model::MODEL;

/// Builder struct for configuring and running predictions.
///
/// Use `Predictor::new()` to create with default threshold, or chain
/// `.with_threshold()` to customize.
///
/// # Examples
///
/// ```rust
/// use is_it_slop::Predictor;
///
/// // Use default threshold
/// let predictor = Predictor::new();
/// let prediction = predictor.predict("some text")?;
///
/// // Custom threshold
/// let predictor = Predictor::new().with_threshold(0.7);
/// let class = predictor.classify("some text")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Predictor {
    threshold: f32,
}

impl Predictor {
    /// Create a new predictor with the default classification threshold.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: CLASSIFICATION_THRESHOLD,
        }
    }

    /// Set a custom classification threshold.
    ///
    /// The threshold determines the AI probability cutoff for classification:
    /// - If P(AI) >= threshold: classified as AI
    /// - If P(AI) < threshold: classified as Human
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Get the current threshold value.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Predict probabilities for a single text.
    ///
    /// Returns a `Prediction` containing P(Human) and P(AI).
    pub fn predict<T: AsRef<str>>(&self, text: T) -> anyhow::Result<Prediction> {
        pipeline::predict(&MODEL, text.as_ref())
            .with_context(|| "Failed to predict probabilities for the given text")
    }

    /// Predict probabilities for multiple texts.
    ///
    /// Returns a vector of `Prediction` values, one for each input text.
    pub fn predict_batch<T: AsRef<str> + Sync>(
        &self,
        texts: &[T],
    ) -> anyhow::Result<Vec<Prediction>> {
        pipeline::predict_batch(&MODEL, texts)
            .with_context(|| "Failed to predict probabilities for the given texts")
    }

    /// Classify a single text using the configured threshold.
    ///
    /// Returns `Classification::Human` or `Classification::AI`.
    pub fn classify<T: AsRef<str>>(&self, text: T) -> anyhow::Result<Classification> {
        self.predict(text)
            .map(|pred| pred.classification(self.threshold))
    }

    /// Classify multiple texts using the configured threshold.
    ///
    /// Returns a vector of classifications, one for each input text.
    pub fn classify_batch<T: AsRef<str> + Sync>(
        &self,
        texts: &[T],
    ) -> anyhow::Result<Vec<Classification>> {
        self.predict_batch(texts).map(|preds| {
            preds
                .into_iter()
                .map(|pred| pred.classification(self.threshold))
                .collect()
        })
    }
}

impl Default for Predictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_probabilities() {
        let predictor = Predictor::new();
        let prediction = predictor
            .predict("This is a test text")
            .expect("Prediction should succeed");

        assert!(prediction.human_probability() >= 0.0);
        assert!(prediction.human_probability() <= 1.0);
        assert!(prediction.ai_probability() >= 0.0);
        assert!(prediction.ai_probability() <= 1.0);
        assert!((prediction.human_probability() + prediction.ai_probability() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_predict_class() {
        let predictor = Predictor::new();
        let class = predictor
            .classify("This is a test text")
            .expect("Classification should succeed");

        assert!(matches!(class, Classification::Human | Classification::AI));
    }

    #[test]
    fn test_predict_class_with_threshold() {
        let predictor = Predictor::new().with_threshold(0.99);
        let class = predictor
            .classify("This is a test text")
            .expect("Classification should succeed");

        assert!(matches!(class, Classification::Human | Classification::AI));
    }

    #[test]
    fn test_batch_predictions() {
        let predictor = Predictor::new();
        let texts = vec!["Text 1", "Text 2", "Text 3"];

        let predictions = predictor
            .predict_batch(&texts)
            .expect("Batch prediction should succeed");

        assert_eq!(predictions.len(), 3);
        for pred in predictions {
            assert!(pred.human_probability() >= 0.0);
            assert!(pred.human_probability() <= 1.0);
            assert!(pred.ai_probability() >= 0.0);
            assert!(pred.ai_probability() <= 1.0);
        }
    }

    #[test]
    fn test_threshold_accessor() {
        let predictor = Predictor::new().with_threshold(0.75);
        assert!((predictor.threshold() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_threshold() {
        let predictor = Predictor::new();
        assert!((predictor.threshold() - CLASSIFICATION_THRESHOLD).abs() < f32::EPSILON);
    }
}
