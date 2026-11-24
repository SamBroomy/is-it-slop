//! Prediction types.
//!
//! - [`Prediction`]: single prediction with P(Human) and P(AI) probabilities
//! - [`UnifiedPrediction`]: document-level prediction plus per-chunk analysis
//! - [`ConfidenceMetrics`]: the calibrated confidence signals
//!
//! # The decision contract
//!
//! The document probability is `sigmoid(doc_logit / T)` with the temperature `T` fitted on the
//! validation split; the classification threshold was fitted on those calibrated probabilities.
//! The network's document head *is* the chunk aggregation — it was trained end to end to combine
//! chunk representations, so nothing outside the model recombines chunk scores.
//!
//! Per-chunk probabilities (`sigmoid(chunk_logit)`, uncalibrated) are exposed for inspection:
//! which parts of a document look AI-written, and how much the chunks disagree
//! ([`UnifiedPrediction::chunk_dispersion`], the mixed-content indicator).

use core::fmt;

use super::Classification;
use crate::Threshold;

/// Single prediction result containing probabilities for both classes.
///
/// Contains P(Human) and P(AI), which always sum to 1.0.
///
/// # Examples
///
/// ```rust
/// use is_it_slop::{Predictor, pipeline::PipelineError};
///
/// let predictor = Predictor::new();
/// let result = predictor.predict("some text")?;
///
/// println!(
///     "Human: {:.2}%",
///     result.prediction.human_probability() * 100.0
/// );
/// println!("AI: {:.2}%", result.prediction.ai_probability() * 100.0);
/// # Ok::<(), PipelineError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction(f32, f32);

impl Prediction {
    /// Create a new `Prediction` instance.
    ///
    /// # Panics
    /// Panics in debug mode if `human_prob` + `ai_prob` does not equal 1.0.
    pub(crate) fn new(human_prob: f32, ai_prob: f32) -> Self {
        debug_assert!(
            (human_prob + ai_prob - 1.0).abs() < 1e-6,
            "Probabilities must sum to 1.0, got {} + {} = {}",
            human_prob,
            ai_prob,
            human_prob + ai_prob
        );
        Self(human_prob, ai_prob)
    }

    /// Create a prediction from AI probability (P(Human) = 1 - P(AI)).
    #[must_use]
    pub fn from_ai_probability(ai_prob: f32) -> Self {
        let human_prob = 1.0 - ai_prob;
        Self::new(human_prob, ai_prob)
    }

    /// Get the probability that the text is human-written (0.0 to 1.0).
    #[must_use]
    pub fn human_probability(&self) -> f32 {
        self.0
    }

    /// Get the probability that the text is AI-generated (0.0 to 1.0).
    #[must_use]
    pub fn ai_probability(&self) -> f32 {
        self.1
    }

    /// Get the binary classification using the given threshold.
    ///
    /// Returns [`Classification::AI`] if `ai_probability >= threshold`,
    /// otherwise returns [`Classification::Human`].
    #[inline]
    #[must_use]
    pub fn classification(&self, threshold: Threshold) -> Classification {
        if self.1 >= *threshold {
            Classification::AI
        } else {
            Classification::Human
        }
    }

    /// Get probabilities as an array: [P(Human), P(AI)]
    #[must_use]
    pub fn probabilities(&self) -> [f32; 2] {
        [self.0, self.1]
    }
}

impl fmt::Display for Prediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P(Human)={:.3}, P(AI)={:.3}", self.0, self.1)
    }
}

impl From<[f32; 2]> for Prediction {
    fn from(probs: [f32; 2]) -> Self {
        Self::new(probs[0], probs[1])
    }
}

/// The calibrated confidence signals for a prediction.
///
/// Both were validated against the model's actual errors on the validation split: the calibrated
/// margin ranks errors best (documents the model gets wrong sit near 0), with chunk dispersion
/// as the secondary indicator for mixed human/AI content.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceMetrics {
    /// The calibrated margin: how far the calibrated probability sits from the 0.5 coin-flip
    /// point, scaled to 0.0-1.0. This is the primary confidence number.
    pub confidence: f32,

    /// Standard deviation of the per-chunk AI probabilities over real chunks.
    /// High dispersion suggests mixed human/AI content. `None` for single-chunk documents.
    pub chunk_dispersion: Option<f32>,
}

/// Summary of the per-chunk analysis.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkInfo {
    /// Number of real (non-padding) chunks the document produced.
    pub num_chunks: usize,
    /// Standard deviation of chunk AI probabilities (None for single chunk).
    pub chunk_dispersion: Option<f32>,
}

/// Document-level prediction with per-chunk analysis.
///
/// The document probability comes straight from the network's document head (calibrated by the
/// shipped temperature); the chunk probabilities are the auxiliary chunk head's view of each
/// 192-token chunk, useful for locating AI-looking passages and measuring mixed content.
#[derive(Debug, Clone)]
pub struct UnifiedPrediction {
    /// Calibrated prediction for the full document.
    pub prediction: Prediction,
    /// Per-chunk predictions (real chunks only, uncalibrated).
    pub chunk_predictions: Vec<Prediction>,
}

impl UnifiedPrediction {
    /// Build from the calibrated document probability and the real chunks' AI probabilities.
    #[must_use]
    pub fn new(doc_ai_probability: f32, chunk_ai_probabilities: &[f32]) -> Self {
        Self {
            prediction: Prediction::from_ai_probability(doc_ai_probability),
            chunk_predictions: chunk_ai_probabilities
                .iter()
                .map(|&p| Prediction::from_ai_probability(p))
                .collect(),
        }
    }

    /// Calibrated probability that the document is AI-generated, in `[0.0, 1.0]`.
    #[must_use]
    pub fn ai_probability(&self) -> f32 {
        self.prediction.ai_probability()
    }

    /// Calibrated probability that the document is human-written, in `[0.0, 1.0]`.
    #[must_use]
    pub fn human_probability(&self) -> f32 {
        self.prediction.human_probability()
    }

    /// Number of chunks the document was split into.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.chunk_predictions.len()
    }

    /// The calibrated margin: distance of the calibrated probability from 0.5, scaled to
    /// 0.0-1.0. Near 0 means the model is guessing; near 1 means it is certain either way.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        (self.prediction.ai_probability() - 0.5).abs() * 2.0
    }

    /// Standard deviation of chunk AI probabilities over real chunks.
    ///
    /// High dispersion is the mixed-content signal: some chunks look AI, others human.
    /// `None` for single-chunk documents.
    #[must_use]
    pub fn chunk_dispersion(&self) -> Option<f32> {
        if self.chunk_predictions.len() < 2 {
            return None;
        }
        let probs: Vec<f32> = self
            .chunk_predictions
            .iter()
            .map(Prediction::ai_probability)
            .collect();
        let mean = probs.iter().sum::<f32>() / probs.len() as f32;
        let variance = probs.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / probs.len() as f32;
        Some(variance.sqrt())
    }

    /// Get the confidence signals.
    #[must_use]
    pub fn confidence_metrics(&self) -> ConfidenceMetrics {
        ConfidenceMetrics {
            confidence: self.confidence(),
            chunk_dispersion: self.chunk_dispersion(),
        }
    }

    /// Get chunk information summary.
    #[must_use]
    pub fn chunk_info(&self) -> ChunkInfo {
        ChunkInfo {
            num_chunks: self.chunk_predictions.len(),
            chunk_dispersion: self.chunk_dispersion(),
        }
    }

    /// Get the binary classification using the given threshold.
    #[must_use]
    pub fn classification(&self, threshold: Threshold) -> Classification {
        self.prediction.classification(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_classification() {
        let threshold = Threshold::new(0.5);

        let pred_ai = Prediction::from_ai_probability(0.7);
        assert!(matches!(
            pred_ai.classification(threshold),
            Classification::AI
        ));

        let pred_human = Prediction::from_ai_probability(0.3);
        assert!(matches!(
            pred_human.classification(threshold),
            Classification::Human
        ));

        // Exactly at threshold should classify as AI (>= threshold)
        let pred_threshold = Prediction::from_ai_probability(0.5);
        assert!(matches!(
            pred_threshold.classification(threshold),
            Classification::AI
        ));
    }

    #[test]
    fn test_confidence_is_calibrated_margin() {
        let sure_ai = UnifiedPrediction::new(0.95, &[0.9]);
        assert!((sure_ai.confidence() - 0.9).abs() < 1e-6);

        let sure_human = UnifiedPrediction::new(0.05, &[0.1]);
        assert!((sure_human.confidence() - 0.9).abs() < 1e-6);

        let coin_flip = UnifiedPrediction::new(0.5, &[0.5]);
        assert!(coin_flip.confidence() < 1e-6);
    }

    #[test]
    fn test_chunk_dispersion_single_chunk_none() {
        let single = UnifiedPrediction::new(0.8, &[0.8]);
        assert!(single.chunk_dispersion().is_none());
        assert!(single.chunk_info().chunk_dispersion.is_none());
        assert_eq!(single.chunk_info().num_chunks, 1);
    }

    #[test]
    fn test_chunk_dispersion_uniform_chunks_zero() {
        let uniform = UnifiedPrediction::new(0.8, &[0.7, 0.7, 0.7]);
        assert!(uniform.chunk_dispersion().unwrap() < 1e-6);
    }

    #[test]
    fn test_chunk_dispersion_mixed_content_high() {
        let mixed = UnifiedPrediction::new(0.6, &[0.05, 0.95, 0.1, 0.9]);
        let dispersion = mixed.chunk_dispersion().unwrap();
        assert!(
            dispersion > 0.4,
            "chunks split between classes should disperse widely, got {dispersion}"
        );

        let consistent = UnifiedPrediction::new(0.9, &[0.85, 0.9, 0.95, 0.88]);
        assert!(consistent.chunk_dispersion().unwrap() < 0.05);
    }

    #[test]
    fn test_confidence_metrics_shape() {
        let pred = UnifiedPrediction::new(0.9, &[0.8, 0.9]);
        let metrics = pred.confidence_metrics();
        assert!((metrics.confidence - 0.8).abs() < 1e-6);
        assert!(metrics.chunk_dispersion.is_some());
    }
}
