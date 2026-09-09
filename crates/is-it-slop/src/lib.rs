//! # is-it-slop
//!
//! A fast and accurate AI text detector built with Rust.
//!
//! Supports multiple platforms including macOS (Apple Silicon), Linux, and Windows.
//!
//! This crate provides tools to classify whether text was written by AI or a human,
//! using a hierarchical token-level classifier run natively with Burn — no runtime
//! dependencies, the whole model ships inside the binary.
//!
//! ## Quick Start
//!
//! ```rust
//! use is_it_slop::{Predictor, pipeline::PipelineError};
//!
//! let predictor = Predictor::new();
//!
//! // Get raw probabilities
//! let result = predictor.predict("Some text to analyze")?;
//! println!(
//!     "AI probability: {:.2}%",
//!     result.prediction.ai_probability() * 100.0
//! );
//!
//! // Or get a classification directly
//! let class = predictor.classify("Some text to analyze")?;
//! println!("Classification: {:?}", class);
//! # Ok::<(), PipelineError>(())
//! ```
//!
//! ## Custom Threshold
//!
//! ```rust
//! use is_it_slop::{Predictor, Threshold, pipeline::PipelineError};
//!
//! // Use a custom threshold (default is [`CLASSIFICATION_THRESHOLD`])
//! let predictor = Predictor::new().with_threshold(Threshold::new(0.7));
//! let class = predictor.classify("Some text")?;
//! # Ok::<(), PipelineError>(())
//! ```
//!
//! ## Batch Processing
//!
//! ```rust
//! use is_it_slop::{Predictor, pipeline::PipelineError};
//!
//! let predictor = Predictor::new();
//! let texts = vec!["First text", "Second text", "Third text"];
//! let predictions = predictor.predict_batch(&texts)?;
//! # Ok::<(), PipelineError>(())
//! ```

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "jni")]
mod kotlin;

pub mod model;
pub mod pipeline;

use std::{fmt, ops::Deref, result::Result};

use is_it_slop_preprocessing::as_strs;
pub use model::MODEL_VERSION;
pub use pipeline::{Classification, Prediction, UnifiedPrediction};

use crate::model::CLASSIFICATION_THRESHOLD;

/// A value that is not a classification threshold.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ThresholdError {
    /// Outside `[0.0, 1.0]`.
    #[error("threshold must be in [0.0, 1.0], got {0}")]
    OutOfRange(f32),
    /// Text that is not a number.
    #[error("threshold is not a number: {0}")]
    Parse(#[from] std::num::ParseFloatError),
}

/// A validated classification threshold in the range `[0.0, 1.0]`.
///
/// Text with a predicted AI probability >= the threshold is classified as AI;
/// anything below is classified as Human.
///
/// Construct with [`Threshold::new`] (panics on invalid input) or
/// [`Threshold::try_new`] (returns `Err` on invalid input). Also constructible
/// via [`TryFrom`] for `f32`, `f64`, and `&str`.
///
/// The default value is [`CLASSIFICATION_THRESHOLD`].
///
/// # Examples
///
/// ```rust
/// use is_it_slop::Threshold;
///
/// let t = Threshold::new(0.7);
///
/// // Fallible construction
/// let t = Threshold::try_new(0.7).unwrap();
/// let err = Threshold::try_new(1.5);
/// assert!(err.is_err());
///
/// // From numeric types
/// let t = Threshold::try_from(0.7_f32).unwrap();
/// let t = Threshold::try_from(0.7_f64).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Threshold(f32);

impl Threshold {
    /// Create a `Threshold`, panicking if `value` is outside `[0.0, 1.0]`.
    ///
    /// Use this when the value is a compile-time constant or programmer-controlled.
    /// For user-supplied values, prefer [`Threshold::try_new`].
    ///
    /// # Panics
    ///
    /// Panics if `value` is not in the range `[0.0, 1.0]`.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self::try_new(value).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a `Threshold`, returning `Err` if `value` is outside `[0.0, 1.0]`.
    ///
    /// Use this when the value comes from user input or configuration.
    pub fn try_new(value: f32) -> Result<Self, ThresholdError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ThresholdError::OutOfRange(value))
        }
    }

    /// Get the default classification threshold, fitted on calibrated validation probabilities.
    /// The default value is [`CLASSIFICATION_THRESHOLD`].
    const fn classification_threshold() -> Self {
        Self(CLASSIFICATION_THRESHOLD)
    }
}
impl TryFrom<f64> for Threshold {
    type Error = ThresholdError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::try_new(value as f32)
    }
}

impl TryFrom<f32> for Threshold {
    type Error = ThresholdError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl TryFrom<&str> for Threshold {
    type Error = ThresholdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_new(value.parse::<f32>()?)
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self::classification_threshold()
    }
}

impl fmt::Display for Threshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for Threshold {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Builder struct for configuring and running predictions.
///
/// Use `Predictor::new()` to create with default threshold, or chain
/// `.with_threshold()` to customize.
///
/// # Examples
///
/// ```rust
/// use is_it_slop::{Predictor, Threshold, pipeline::PipelineError};
///
/// // Use default threshold
/// let predictor = Predictor::new();
/// let prediction = predictor.predict("some text")?;
///
/// // Custom threshold
/// let predictor = Predictor::new().with_threshold(Threshold::new(0.7));
/// let class = predictor.classify("some text")?;
/// # Ok::<(), PipelineError>(())
/// ```
pub struct Predictor {
    threshold: Threshold,
}

impl Predictor {
    /// Create a new predictor with the default classification threshold.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: Threshold::default(),
        }
    }

    /// Set a custom classification threshold.
    ///
    /// The threshold determines the AI probability cutoff for classification:
    /// - If P(AI) >= threshold: classified as AI
    /// - If P(AI) < threshold: classified as Human
    #[must_use]
    pub fn with_threshold(mut self, threshold: Threshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// Get the current threshold value.
    #[must_use]
    pub fn threshold(&self) -> Threshold {
        self.threshold
    }

    /// Predict probabilities for a single text.
    ///
    /// Returns a `Prediction` containing P(Human) and P(AI).
    pub fn predict<T: AsRef<str>>(&self, text: T) -> pipeline::Result<UnifiedPrediction> {
        pipeline::predict(text.as_ref())
    }

    /// Predict probabilities for multiple texts, all or nothing.
    ///
    /// Fails if the slice is empty or any text is blank; see [`Predictor::predict_each`] for
    /// per-text results.
    pub fn predict_batch<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> pipeline::Result<Vec<UnifiedPrediction>> {
        pipeline::predict_batch(&as_strs(texts))
    }

    /// Predict each text in one forward pass, with a result per text.
    ///
    /// A blank text fails on its own instead of failing the batch.
    pub fn predict_each<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> Vec<pipeline::Result<UnifiedPrediction>> {
        pipeline::predict_each(&as_strs(texts))
    }

    /// [`Predictor::predict_each`] under an explicit scheduling [`pipeline::Strategy`].
    #[must_use]
    pub fn predict_each_with<T: AsRef<str>>(
        &self,
        texts: &[T],
        strategy: pipeline::Strategy,
    ) -> Vec<pipeline::Result<UnifiedPrediction>> {
        pipeline::predict_each_with(&as_strs(texts), strategy)
    }

    /// The text behind each chunk prediction of [`Predictor::predict`] for `text`, in order.
    #[must_use]
    pub fn chunk_texts<T: AsRef<str>>(&self, text: T) -> Vec<String> {
        pipeline::chunk_texts(text.as_ref())
    }

    /// Classify a single text using the configured threshold.
    ///
    /// Returns `Classification::Human` or `Classification::AI`.
    pub fn classify<T: AsRef<str>>(&self, text: T) -> pipeline::Result<Classification> {
        pipeline::predict(text.as_ref()).map(|pred| pred.classification(self.threshold))
    }

    /// Classify multiple texts using the configured threshold.
    ///
    /// Returns a vector of classifications, one for each input text.
    pub fn classify_batch<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> pipeline::Result<Vec<Classification>> {
        pipeline::predict_batch(&as_strs(texts)).map(|preds| {
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
        let predictor = Predictor::default();
        let prediction = predictor
            .predict("This is a test text")
            .expect("Prediction should succeed");
        assert!(prediction.chunk_predictions.iter().all(|p| {
            p.human_probability() >= 0.0
                && p.human_probability() <= 1.0
                && p.ai_probability() >= 0.0
                && p.ai_probability() <= 1.0
                && (p.human_probability() + p.ai_probability() - 1.0).abs() < 0.001
        }));

        assert!(prediction.prediction.human_probability() >= 0.0);
        assert!(prediction.prediction.human_probability() <= 1.0);
        assert!(prediction.prediction.ai_probability() >= 0.0);
        assert!(prediction.prediction.ai_probability() <= 1.0);
        assert!(
            (prediction.prediction.human_probability() + prediction.prediction.ai_probability()
                - 1.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_predict_class() {
        let predictor = Predictor::default();
        let class = predictor
            .classify("This is a test text")
            .expect("Classification should succeed");

        assert!(matches!(class, Classification::Human | Classification::AI));
    }

    #[test]
    fn test_predict_class_with_threshold() {
        let predictor = Predictor::new().with_threshold(Threshold::new(0.99));
        let class = predictor
            .classify("This is a test text")
            .expect("Classification should succeed");

        assert!(matches!(class, Classification::Human | Classification::AI));
    }

    #[test]
    fn test_batch_predictions() {
        let predictor = Predictor::default();
        let texts = vec!["Text 1", "Text 2", "Text 3"];

        let predictions = predictor
            .predict_batch(&texts)
            .expect("Batch prediction should succeed");

        assert_eq!(predictions.len(), 3);
        for pred in predictions {
            assert!(pred.prediction.human_probability() >= 0.0);
            assert!(pred.prediction.human_probability() <= 1.0);
            assert!(pred.prediction.ai_probability() >= 0.0);
            assert!(pred.prediction.ai_probability() <= 1.0);
        }
    }

    #[test]
    fn test_threshold_accessor() {
        let predictor = Predictor::new().with_threshold(0.50.try_into().unwrap());
        assert!((*predictor.threshold() - 0.50).abs() < f32::EPSILON);
        let predictor = predictor.with_threshold(0.75.try_into().unwrap());
        assert!((*predictor.threshold() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_threshold() {
        let predictor = Predictor::default();
        assert!((*predictor.threshold() - CLASSIFICATION_THRESHOLD).abs() < f32::EPSILON);
    }

    #[test]
    fn test_classify_batch() {
        let predictor = Predictor::default();
        let texts = vec!["Text A", "Text B", "Text C"];
        let classes = predictor
            .classify_batch(&texts)
            .expect("Batch classification should succeed");

        assert_eq!(classes.len(), 3);
        for class in classes {
            assert!(matches!(class, Classification::Human | Classification::AI));
        }
    }

    #[test]
    fn test_default_predictor_equals_new() {
        let default = Predictor::default();
        let new = Predictor::new();
        assert!((*default.threshold() - *new.threshold()).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "threshold must be in [0.0, 1.0]")]
    fn test_with_threshold_rejects_negative() {
        let _ = Threshold::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "threshold must be in [0.0, 1.0]")]
    fn test_with_threshold_rejects_over_one() {
        let _ = Threshold::new(1.5);
    }

    #[test]
    fn test_with_threshold_accepts_edge_values() {
        let p = Predictor::new().with_threshold(Threshold::new(0.0));
        assert!((*p.threshold() - 0.0).abs() < f32::EPSILON);

        let p = Predictor::new().with_threshold(Threshold::new(1.0));
        assert!((*p.threshold() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_threshold_try_new_valid() {
        assert!(Threshold::try_new(0.0).is_ok());
        assert!(Threshold::try_new(0.5).is_ok());
        assert!(Threshold::try_new(1.0).is_ok());
    }

    #[test]
    fn test_threshold_try_new_invalid() {
        assert!(Threshold::try_new(-0.1).is_err());
        assert!(Threshold::try_new(1.1).is_err());
        let err = Threshold::try_new(-0.5).unwrap_err();
        assert_eq!(err, ThresholdError::OutOfRange(-0.5));
    }

    #[test]
    fn test_threshold_try_from() {
        assert!(Threshold::try_from(0.5_f32).is_ok());
        assert!(Threshold::try_from(1.5_f32).is_err());
        assert!(Threshold::try_from(0.5_f64).is_ok());
        assert!(Threshold::try_from(-0.1_f64).is_err());
    }

    #[test]
    fn test_predict_rejects_empty() {
        let predictor = Predictor::new();
        assert!(predictor.predict("").is_err());
        assert!(predictor.predict("   \n\t  ").is_err());
    }

    #[test]
    fn test_predict_batch_rejects_empty() {
        let predictor = Predictor::new();
        let empty: &[&str] = &[];
        assert!(predictor.predict_batch(empty).is_err());
    }
}
