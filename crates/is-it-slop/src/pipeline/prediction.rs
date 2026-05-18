//! Prediction types and aggregation strategies.
//!
//! This module defines the core prediction types used throughout the pipeline:
//!
//! - [`Prediction`]: Single prediction with P(Human) and P(AI) probabilities
//! - [`UnifiedPrediction`]: Document-level prediction combining multiple chunk predictions
//! - [`AggregationMethod`]: Strategy for combining chunk predictions
//! - [`ConfidenceMetrics`]: Multi-dimensional confidence assessment
//!
//! # Chunked Prediction Flow
//!
//! For long documents (>150 tokens):
//! 1. Split into overlapping chunks
//! 2. Generate [`Prediction`] for each chunk
//! 3. Aggregate using [`AggregationMethod`] (Mean/Max/WeightedMean)
//! 4. Calculate chunk agreement (how consistently chunks agree on classification)
//! 5. Return [`UnifiedPrediction`] with aggregated result and metadata
//!
//! # Aggregation Strategies
//!
//! - **Mean**: Simple average of chunk probabilities
//!   - Best for: Balanced view of document
//!   - Downside: Outlier chunks diluted
//!
//! - **Max**: Most suspicious chunk wins
//!   - Best for: Detecting mixed human/AI content
//!   - Downside: Single outlier can dominate
//!
//! - **`WeightedMean`** (default): Weight by distance from chunk threshold
//!   - Best for: Emphasizing high-confidence chunks
//!   - Downside: More complex to interpret
//!
//! # Example
//!
//! ```rust
//! use is_it_slop::pipeline::{AggregationMethod, Prediction, UnifiedPrediction};
//!
//! // Chunk predictions
//! let chunks = vec![
//!     Prediction::from_ai_probability(0.3), // Low AI probability
//!     Prediction::from_ai_probability(0.7), // High AI probability
//! ];
//!
//! // Aggregate with weighted mean
//! let method = AggregationMethod::default();
//! let result = UnifiedPrediction::new(chunks, method);
//!
//! println!(
//!     "Document AI probability: {:.2}",
//!     result.prediction.ai_probability()
//! );
//! println!("Chunk agreement: {:.2}", result.chunk_agreement);
//! ```

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
#[derive(Debug, Clone, Copy)]
pub struct Prediction(f32, f32);

impl Prediction {
    /// Create a new `Prediction` instance.
    ///
    /// # Panics
    /// Panics in debug mode if `human_prob` + `ai_prob` does not equal 1.0.
    pub(super) fn new(human_prob: f32, ai_prob: f32) -> Self {
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
    ///
    /// This is a convenience constructor for tests and examples.
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

    /// Calculate model's confidence in its prediction.
    ///
    /// This is the probability assigned to the predicted class:
    /// - If predicting AI (prob ≥ threshold): confidence = P(AI)
    /// - If predicting Human (prob < threshold): confidence = P(Human)
    ///
    /// Range: 0.5-1.0 (binary classification, predicted class always ≥ 0.5)
    #[must_use]
    pub fn model_confidence(&self, threshold: Threshold) -> f32 {
        let ai_prob = self.ai_probability();
        if ai_prob >= *threshold {
            ai_prob // Confidence in AI prediction
        } else {
            1.0 - ai_prob // Confidence in Human prediction
        }
    }

    /// Calculate distance from decision threshold.
    ///
    /// Measures how far the prediction is from the decision boundary.
    /// Range: 0.0-1.0 (0.0 = at threshold, 1.0 = at extreme)
    #[must_use]
    pub fn threshold_distance(&self, threshold: Threshold) -> f32 {
        let ai_prob = self.ai_probability();
        let distance = (ai_prob - *threshold).abs();
        let max_distance = threshold.max(1.0 - *threshold);
        distance / max_distance
    }

    /// Shannon entropy of the probability distribution
    ///
    /// Lower entropy = more confident
    /// - H=0.0: Perfect certainty ([1.0, 0.0] or [0.0, 1.0])
    /// - H=1.0: Maximum uncertainty ([0.5, 0.5])
    #[must_use]
    pub fn entropy(&self) -> f32 {
        let p_human = self.human_probability();
        let p_ai = self.ai_probability();

        let h_human = if p_human > 0.0 {
            -p_human * p_human.log2()
        } else {
            0.0
        };
        let h_ai = if p_ai > 0.0 { -p_ai * p_ai.log2() } else { 0.0 };

        h_human + h_ai
    }

    /// Calculate entropy-based confidence.
    ///
    /// Uses Shannon entropy to measure uncertainty in the probability distribution.
    /// Range: 0.0-1.0 (0.0 = maximum uncertainty, 1.0 = perfect certainty)
    #[must_use]
    pub fn entropy_confidence(&self) -> f32 {
        1.0 - self.entropy()
    }

    /// Get comprehensive confidence metrics.
    ///
    /// Returns all confidence measures:
    /// - Model confidence (probability of predicted class)
    /// - Threshold distance (how far from decision boundary)
    /// - Entropy confidence (information-theoretic uncertainty)
    /// - Overall confidence (weighted combination)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use is_it_slop::{Predictor, Threshold, pipeline::PipelineError};
    ///
    /// let predictor = Predictor::new();
    /// let result = predictor.predict("some text")?;
    /// let metrics = result.prediction.confidence_metrics(Threshold::default());
    ///
    /// println!("Model confidence: {:.1}%", metrics.model_confidence * 100.0);
    /// println!(
    ///     "Threshold distance: {:.1}%",
    ///     metrics.threshold_distance * 100.0
    /// );
    /// println!(
    ///     "Entropy confidence: {:.1}%",
    ///     metrics.entropy_confidence * 100.0
    /// );
    /// println!("Overall: {:.1}%", metrics.overall * 100.0);
    /// # Ok::<(), PipelineError>(())
    /// ```
    #[must_use]
    pub fn confidence_metrics(&self, threshold: Threshold) -> ConfidenceMetrics {
        let model_confidence = self.model_confidence(threshold);
        let threshold_distance = self.threshold_distance(threshold);
        let entropy_confidence = self.entropy_confidence();

        // Weighted combination: prioritize model confidence (it's calibrated)
        // but penalize if near threshold or high entropy
        let overall =
            model_confidence * 0.5 + threshold_distance * 0.25 + entropy_confidence * 0.25;

        ConfidenceMetrics {
            model_confidence,
            threshold_distance,
            entropy_confidence,
            overall,
            sample_reliability: 1.0, // Single prediction - no sample size penalty
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

/// Strategy for combining chunk predictions into a document-level prediction.
///
/// Different aggregation methods make different trade-offs:
///
/// # Variants
///
/// ## Mean
/// Simple average of all chunk probabilities.
/// - **Use when**: You want a balanced view of the entire document
/// - **Strengths**: Simple, interpretable, treats all chunks equally
/// - **Weaknesses**: Outlier chunks get diluted, may miss localized AI content
///
/// ## Max
/// Takes the maximum AI probability across all chunks (most suspicious chunk wins).
/// - **Use when**: Detecting mixed human/AI content or AI insertions
/// - **Strengths**: Sensitive to any AI-generated segments
/// - **Weaknesses**: Single outlier can dominate, prone to false positives
///
/// ## WeightedMean(threshold)
/// Weights chunks by their distance from the threshold—chunks with probabilities
/// far from the threshold (high confidence) contribute more to the final result.
/// - **Use when**: You want to emphasize high-confidence chunks (default)
/// - **Strengths**: Balances sensitivity and specificity, reduces outlier impact
/// - **Weaknesses**: More complex, requires tuning threshold parameter
///
/// # Example
///
/// ```rust
/// use is_it_slop::{
///     Threshold,
///     pipeline::{AggregationMethod, Prediction, UnifiedPrediction},
/// };
///
/// let chunks = vec![
///     Prediction::from_ai_probability(0.2), // Low confidence AI
///     Prediction::from_ai_probability(0.9), // High confidence AI
/// ];
///
/// // Mean: (0.2 + 0.9) / 2 = 0.55
/// let mean = AggregationMethod::Mean.aggregate_predictions(&chunks);
/// assert!((mean.ai_probability() - 0.55).abs() < 0.01);
///
/// // Max: max(0.2, 0.9) = 0.9
/// let max = AggregationMethod::Max.aggregate_predictions(&chunks);
/// assert!((max.ai_probability() - 0.9).abs() < 0.01);
///
/// // WeightedMean: weights high-confidence chunks more
/// let weighted =
///     AggregationMethod::WeightedMean(Threshold::default()).aggregate_predictions(&chunks);
/// // Result will be closer to 0.9 than 0.55 (high-confidence chunk weighted more)
/// ```
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Simple average of all chunk probabilities
    Mean,
    /// Maximum probability across chunks (most suspicious chunk)
    Max,
    /// Weighted average based on distance from threshold
    WeightedMean(Threshold),
}

impl AggregationMethod {
    /// Aggregate chunk predictions into a single document prediction
    pub fn aggregate_predictions(&self, chunk_predictions: &[Prediction]) -> Prediction {
        debug_assert!(!chunk_predictions.is_empty(), "No predictions to aggregate");
        if chunk_predictions.len() == 1 {
            // No aggregation needed
            return chunk_predictions[0];
        }

        let ai_probs = chunk_predictions
            .iter()
            .map(Prediction::ai_probability)
            .collect::<Vec<_>>();

        let aggregated_ai_prob = match self {
            Self::Mean => ai_probs.iter().sum::<f32>() / ai_probs.len() as f32,
            Self::Max => *ai_probs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.5),
            Self::WeightedMean(threshold) => {
                // Weights based on distance from threshold
                let weights = ai_probs
                    .iter()
                    .map(|&p| (p - **threshold).abs())
                    .collect::<Vec<_>>();
                let total_weight: f32 = weights.iter().sum();
                if total_weight == 0.0 {
                    0.5 // Neutral if all weights are zero
                } else {
                    ai_probs
                        .iter()
                        .zip(weights.iter())
                        .map(|(&p, &w)| p * w)
                        .sum::<f32>()
                        / total_weight
                }
            }
        };
        Prediction::from_ai_probability(aggregated_ai_prob)
    }

    fn calculate_chunk_agreement(self, chunk_predictions: &[Prediction]) -> f32 {
        if matches!(self, Self::Max) || chunk_predictions.len() < 2 {
            return 1.0; // Agreement not meaningful for max or single chunk
        }
        let threshold = match self {
            Self::WeightedMean(t) => t,
            _ => Threshold::chunk_classification_threshold(),
        };

        let ai_chunks = chunk_predictions
            .iter()
            .filter(|p| matches!(p.classification(threshold), Classification::AI))
            .count();

        let ai_ratio = ai_chunks as f32 / chunk_predictions.len() as f32;

        // We want max agreement at extremes (0 or 1), min at 0.5
        0.5 + (ai_ratio - 0.5).abs()
    }
}

impl Default for AggregationMethod {
    fn default() -> Self {
        Self::WeightedMean(Threshold::chunk_classification_threshold())
    }
}

/// Comprehensive confidence metrics for a prediction
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceMetrics {
    /// Model's confidence in its prediction (probability of predicted class)
    /// Range: 0.5-1.0 (binary classification, so predicted class always ≥ 0.5)
    pub model_confidence: f32,

    /// Distance from decision threshold (how far from boundary)
    /// Range: 0.0-1.0 (0.0 = at threshold, 1.0 = at extreme)
    pub threshold_distance: f32,

    /// Entropy-based confidence (information-theoretic uncertainty)
    /// Range: 0.0-1.0 (0.0 = maximum uncertainty, 1.0 = perfect certainty)
    pub entropy_confidence: f32,

    /// Overall confidence score (weighted combination of above)
    /// Range: 0.0-1.0
    pub overall: f32,

    /// Sample reliability (statistical adequacy of sample size)
    /// Range: 0.0-1.0 (accounts for chunk count, token count, and variance)
    pub sample_reliability: f32,
}

impl ConfidenceMetrics {
    /// Get overall confidence as a percentage (0-100)
    #[must_use]
    pub fn overall_percent(&self) -> f32 {
        self.overall * 100.0
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkInfo {
    /// Number of chunks processed
    pub num_chunks: usize,
    /// Agreement score across chunks (0.5-1.0)
    pub chunk_agreement: f32,
    /// Standard deviation of chunk AI probabilities (None for single chunk)
    pub chunk_std_dev: Option<f32>,
    /// Token count for single-chunk documents (None for multi-chunk)
    pub token_count: Option<usize>,
}
/// Extended prediction with chunk-level analysis
/// Document-level prediction combining multiple chunk predictions.
///
/// For texts longer than the chunk size (150 tokens), the document is split into
/// overlapping chunks, each producing a [`Prediction`]. This struct aggregates
/// those chunk predictions into a final document-level result.
///
/// # Fields
///
/// - `prediction`: Final aggregated probability for the document
/// - `chunk_predictions`: Individual predictions for each chunk (for debugging/analysis)
/// - `aggregation_method`: Strategy used to combine chunks
/// - `chunk_agreement`: How consistently chunks agree (0.5-1.0)
///   - 1.0 = perfect agreement (all chunks same classification)
///   - 0.5 = maximum disagreement (evenly split)
///
/// # Confidence Interpretation
///
/// High-confidence predictions have:
/// - Aggregated probability far from threshold (e.g., 0.1 or 0.9)
/// - High chunk agreement (>0.8)
/// - Low entropy in aggregated prediction
///
/// Low-confidence predictions may indicate:
/// - Mixed human/AI content
/// - Borderline cases near decision boundary
/// - Inconsistent writing style across document
pub struct UnifiedPrediction {
    /// Aggregated prediction for the full document
    pub prediction: Prediction,
    /// Individual predictions for each chunk
    pub chunk_predictions: Vec<Prediction>,
    /// Aggregation method used
    pub aggregation_method: AggregationMethod,
    /// Agreement score across chunks (0.5-1.0)
    pub chunk_agreement: f32,
    /// Token count for single-chunk documents (None for multi-chunk)
    pub single_chunk_token_count: Option<usize>,
}

impl UnifiedPrediction {
    /// Create a new detailed prediction from chunk predictions
    #[must_use]
    pub fn new(chunk_predictions: Vec<Prediction>, aggregation_method: AggregationMethod) -> Self {
        assert!(
            !chunk_predictions.is_empty(),
            "No chunk predictions provided"
        );

        let prediction = aggregation_method.aggregate_predictions(&chunk_predictions);
        let chunk_agreement = aggregation_method.calculate_chunk_agreement(&chunk_predictions);

        Self {
            prediction,
            chunk_predictions,
            aggregation_method,
            chunk_agreement,
            single_chunk_token_count: None,
        }
    }

    /// Get confidence metrics accounting for chunk agreement.
    ///
    /// Combines model confidence with inter-chunk agreement:
    /// - High agreement → confidence boosted
    /// - Low agreement (e.g., 50/50 split) → confidence penalized
    #[must_use]
    pub fn confidence_metrics(&self, threshold: Threshold) -> ConfidenceMetrics {
        let mut metrics = self.prediction.confidence_metrics(threshold);

        // Adjust overall confidence based on chunk agreement
        // chunk_agreement = 1.0 (all chunks agree) → no penalty
        // chunk_agreement = 0.5 (50/50 split)     → reduce confidence
        metrics.overall *= self.chunk_agreement;

        // Calculate sample reliability and fold into overall
        let chunk_std_dev = self.calculate_chunk_std_dev();
        metrics.sample_reliability = calculate_sample_reliability(
            self.chunk_predictions.len(),
            self.single_chunk_token_count,
            chunk_std_dev,
        );
        metrics.overall *= metrics.sample_reliability;

        metrics
    }

    #[must_use]
    /// Get chunk information summary.
    pub fn chunk_info(&self) -> ChunkInfo {
        ChunkInfo {
            num_chunks: self.chunk_predictions.len(),
            chunk_agreement: self.chunk_agreement,
            chunk_std_dev: self.calculate_chunk_std_dev(),
            token_count: self.single_chunk_token_count,
        }
    }

    #[must_use]
    /// Get the binary classification using the given threshold.
    pub fn classification(&self, threshold: Threshold) -> Classification {
        self.prediction.classification(threshold)
    }

    /// Calculate the standard deviation of chunk AI probabilities.
    /// Returns None for single-chunk documents.
    fn calculate_chunk_std_dev(&self) -> Option<f32> {
        if self.chunk_predictions.len() < 2 {
            return None;
        }

        let ai_probs: Vec<f32> = self
            .chunk_predictions
            .iter()
            .map(Prediction::ai_probability)
            .collect();

        let mean = ai_probs.iter().sum::<f32>() / ai_probs.len() as f32;
        let variance =
            ai_probs.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / ai_probs.len() as f32;

        Some(variance.sqrt()) // Standard deviation
    }
}

/// Calculate sample adequacy based on chunk count and token count.
/// For single-chunk documents, applies compound penalty based on token count.
/// For multi-chunk documents, uses exponential curve based on chunk count.
fn sample_adequacy(num_chunks: usize, token_count: Option<usize>) -> f32 {
    if num_chunks == 1 {
        // Single chunk: apply token-level penalty
        if let Some(tokens) = token_count {
            let chunk_adequacy = 1.0_f32 - (-1.0_f32 / 2.0_f32).exp(); // ~0.39 base for n=1
            let token_adequacy = (tokens as f32 / 150.0_f32).min(1.0_f32);
            return chunk_adequacy * token_adequacy;
        }
    }

    // Multi-chunk: only chunk count matters
    let n = num_chunks as f32;
    1.0_f32 - (-n / 2.0_f32).exp()
}

/// Calculate variance penalty based on chunk prediction spread.
/// Uses exponential decay so penalty smoothly approaches 0 without a hard cutoff.
/// Returns 1.0 for single chunks (no variance), decreases for high variance.
fn variance_penalty(chunk_std_dev: Option<f32>) -> f32 {
    chunk_std_dev.map_or(1.0, |std_dev| (-std_dev * 3.0).exp())
}

/// Calculate overall sample reliability score.
/// Combines sample adequacy (chunk/token count) with variance penalty.
fn calculate_sample_reliability(
    num_chunks: usize,
    token_count: Option<usize>,
    chunk_std_dev: Option<f32>,
) -> f32 {
    sample_adequacy(num_chunks, token_count) * variance_penalty(chunk_std_dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic aggregation strategies

    #[test]
    fn test_aggregation_mean_basic() {
        let chunks = vec![
            Prediction::from_ai_probability(0.3),
            Prediction::from_ai_probability(0.5),
            Prediction::from_ai_probability(0.7),
        ];
        let agg = AggregationMethod::Mean;
        let result = agg.aggregate_predictions(&chunks);
        assert!((result.ai_probability() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_aggregation_max_selects_highest() {
        let chunks = vec![
            Prediction::from_ai_probability(0.3),
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.5),
        ];
        let agg = AggregationMethod::Max;
        let result = agg.aggregate_predictions(&chunks);
        assert!((result.ai_probability() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_aggregation_weighted_mean_confidence_based() {
        // Test that chunks farther from threshold get more weight
        let threshold = Threshold(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.3),  // distance = 0.2
            Prediction::from_ai_probability(0.49), // distance = 0.01 (near threshold, low weight)
            Prediction::from_ai_probability(0.9),  // distance = 0.4 (high weight)
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let result = agg.aggregate_predictions(&chunks);

        // Result should be weighted toward 0.9 and 0.3, with 0.49 having minimal influence
        // Expected: (0.3 * 0.2 + 0.49 * 0.01 + 0.9 * 0.4) / (0.2 + 0.01 + 0.4)
        // = (0.06 + 0.0049 + 0.36) / 0.61 = 0.4249 / 0.61 ≈ 0.697
        let expected: f32 = (0.3 * 0.2 + 0.49 * 0.01 + 0.9 * 0.4) / (0.2 + 0.01 + 0.4);
        assert!((result.ai_probability() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_aggregation_single_chunk_returns_same() {
        let chunks = vec![Prediction::from_ai_probability(0.75)];

        let result_mean = AggregationMethod::Mean.aggregate_predictions(&chunks);
        assert!((result_mean.ai_probability() - 0.75).abs() < 1e-6);

        let result_max = AggregationMethod::Max.aggregate_predictions(&chunks);
        assert!((result_max.ai_probability() - 0.75).abs() < 1e-6);

        let result_weighted =
            AggregationMethod::WeightedMean(Threshold(0.5)).aggregate_predictions(&chunks);
        assert!((result_weighted.ai_probability() - 0.75).abs() < 1e-6);
    }

    // Chunk agreement calculation

    #[test]
    fn test_chunk_agreement_perfect_unanimous() {
        let threshold = Threshold::new(0.5);
        let chunks_all_ai = vec![
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.8),
            Prediction::from_ai_probability(0.7),
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let agreement = agg.calculate_chunk_agreement(&chunks_all_ai);
        assert!(
            (agreement - 1.0).abs() < 1e-6,
            "All AI chunks should have agreement=1.0"
        );

        let chunks_all_human = vec![
            Prediction::from_ai_probability(0.1),
            Prediction::from_ai_probability(0.2),
            Prediction::from_ai_probability(0.3),
        ];
        let agreement = agg.calculate_chunk_agreement(&chunks_all_human);
        assert!(
            (agreement - 1.0).abs() < 1e-6,
            "All Human chunks should have agreement=1.0"
        );
    }

    #[test]
    fn test_chunk_agreement_split_fifty_fifty() {
        let threshold = Threshold::new(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.7), // AI
            Prediction::from_ai_probability(0.3), // Human
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let agreement = agg.calculate_chunk_agreement(&chunks);
        // ai_ratio = 1/2 = 0.5
        // agreement = 0.5 + |0.5 - 0.5| = 0.5
        assert!(
            (agreement - 0.5).abs() < 1e-6,
            "50/50 split should have agreement=0.5"
        );
    }

    #[test]
    fn test_chunk_agreement_weighted_majority() {
        let threshold = Threshold::new(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.8), // AI
            Prediction::from_ai_probability(0.7), // AI
            Prediction::from_ai_probability(0.3), // Human
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let agreement = agg.calculate_chunk_agreement(&chunks);
        // ai_ratio = 2/3
        // agreement = 0.5 + |2/3 - 0.5| = 0.5 + 0.1667 ≈ 0.667
        let ai_ratio: f32 = 2.0 / 3.0;
        let expected: f32 = 0.5 + (ai_ratio - 0.5).abs();
        assert!((agreement - expected).abs() < 1e-3);
    }

    #[test]
    fn test_chunk_agreement_max_method_always_one() {
        let chunks = vec![
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.1),
        ];
        let agg = AggregationMethod::Max;
        let agreement = agg.calculate_chunk_agreement(&chunks);
        assert!(
            (agreement - 1.0).abs() < 1e-6,
            "Max method should always return agreement=1.0"
        );
    }

    #[test]
    fn test_chunk_agreement_single_chunk() {
        let chunks = vec![Prediction::from_ai_probability(0.7)];
        let agg = AggregationMethod::WeightedMean(Threshold(0.5));
        let agreement = agg.calculate_chunk_agreement(&chunks);
        assert!(
            (agreement - 1.0).abs() < 1e-6,
            "Single chunk should have agreement=1.0"
        );
    }

    // Confidence metrics

    #[test]
    fn test_confidence_metrics_high_certainty() {
        let pred = Prediction::from_ai_probability(0.95);
        let threshold = Threshold::new(0.5);
        let metrics = pred.confidence_metrics(threshold);

        // High certainty prediction
        assert!(
            metrics.model_confidence > 0.9,
            "Model confidence should be high"
        );
        assert!(
            metrics.threshold_distance > 0.8,
            "Should be far from threshold"
        );
        assert!(
            metrics.entropy_confidence > 0.7,
            "Entropy confidence should be high"
        );
        assert!(metrics.overall > 0.8, "Overall confidence should be high");
    }

    #[test]
    fn test_confidence_metrics_near_threshold() {
        let pred = Prediction::from_ai_probability(0.51);
        let threshold = Threshold::new(0.5);
        let metrics = pred.confidence_metrics(threshold);

        // Near threshold = low confidence
        assert!(metrics.model_confidence > 0.5 && metrics.model_confidence < 0.6);
        assert!(
            metrics.threshold_distance < 0.1,
            "Should be close to threshold"
        );
        assert!(
            metrics.overall < 0.6,
            "Overall confidence should be low near threshold"
        );
    }

    #[test]
    fn test_entropy_calculation_extremes() {
        // Perfect certainty: P(AI) = 1.0
        let pred_certain = Prediction::from_ai_probability(1.0);
        assert!(
            (pred_certain.entropy() - 0.0).abs() < 1e-6,
            "Perfect certainty should have entropy=0"
        );
        assert!((pred_certain.entropy_confidence() - 1.0).abs() < 1e-6);

        // Perfect certainty: P(Human) = 1.0
        let pred_certain_human = Prediction::from_ai_probability(0.0);
        assert!((pred_certain_human.entropy() - 0.0).abs() < 1e-6);
        assert!((pred_certain_human.entropy_confidence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_calculation_uniform() {
        // Maximum uncertainty: P(AI) = P(Human) = 0.5
        let pred_uncertain = Prediction::from_ai_probability(0.5);
        // Entropy = -0.5 * log2(0.5) - 0.5 * log2(0.5) = -0.5 * (-1) - 0.5 * (-1) = 1.0
        assert!(
            (pred_uncertain.entropy() - 1.0).abs() < 1e-6,
            "Uniform distribution should have entropy=1.0"
        );
        assert!((pred_uncertain.entropy_confidence() - 0.0).abs() < 1e-6);
    }

    // UnifiedPrediction integration

    #[test]
    fn test_unified_prediction_confidence_with_agreement() {
        let threshold = Threshold::new(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.8),
            Prediction::from_ai_probability(0.85),
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let unified = UnifiedPrediction::new(chunks, agg);

        let metrics = unified.confidence_metrics(threshold);

        // High agreement (all chunks agree) boosts confidence, but
        // overall now folds in sample_reliability (n=3 chunks)
        assert!(unified.chunk_agreement > 0.9, "Agreement should be high");
        assert!(
            metrics.overall > 0.4,
            "Overall confidence should account for sample reliability"
        );
        assert!(
            metrics.sample_reliability > 0.6,
            "Sample reliability should be high with multiple agreeing chunks"
        );
    }

    #[test]
    fn test_unified_prediction_low_agreement_penalty() {
        let threshold = Threshold::new(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.9), // AI
            Prediction::from_ai_probability(0.1), // Human
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let unified = UnifiedPrediction::new(chunks, agg);

        let metrics = unified.confidence_metrics(threshold);

        // Low agreement (50/50 split) should penalize confidence heavily
        assert!(
            (unified.chunk_agreement - 0.5).abs() < 1e-6,
            "Agreement should be 0.5"
        );
        assert!(
            metrics.overall < 0.1,
            "Overall confidence should be severely penalized for disagreement"
        );
    }

    #[test]
    fn test_unified_prediction_chunk_info() {
        let chunks = vec![
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.8),
        ];
        let agg = AggregationMethod::Mean;
        let unified = UnifiedPrediction::new(chunks.clone(), agg);

        let info = unified.chunk_info();
        assert_eq!(info.num_chunks, 2);
        assert!(info.chunk_agreement >= 0.5 && info.chunk_agreement <= 1.0);
    }

    // Edge cases

    #[test]
    #[should_panic(expected = "No predictions to aggregate")]
    fn test_aggregation_empty_chunks_panics() {
        let chunks = vec![];
        let agg = AggregationMethod::Mean;
        let _ = agg.aggregate_predictions(&chunks);
    }

    #[test]
    fn test_aggregation_all_same_probability() {
        let chunks = vec![
            Prediction::from_ai_probability(0.6),
            Prediction::from_ai_probability(0.6),
            Prediction::from_ai_probability(0.6),
        ];

        let result_mean = AggregationMethod::Mean.aggregate_predictions(&chunks);
        assert!((result_mean.ai_probability() - 0.6).abs() < 1e-6);

        let result_max = AggregationMethod::Max.aggregate_predictions(&chunks);
        assert!((result_max.ai_probability() - 0.6).abs() < 1e-6);

        let result_weighted =
            AggregationMethod::WeightedMean(Threshold::new(0.5)).aggregate_predictions(&chunks);
        assert!((result_weighted.ai_probability() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_all_at_threshold() {
        // Edge case: all chunks exactly at threshold (all weights = 0)
        let threshold = Threshold::new(0.5);
        let chunks = vec![
            Prediction::from_ai_probability(0.5),
            Prediction::from_ai_probability(0.5),
        ];
        let agg = AggregationMethod::WeightedMean(threshold);
        let result = agg.aggregate_predictions(&chunks);
        // When all weights are 0, should return 0.5 (neutral)
        assert!((result.ai_probability() - 0.5).abs() < 1e-6);
    }

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
    fn test_threshold_distance_calculation() {
        let threshold = Threshold::new(0.5);

        // At threshold: distance = 0
        let pred_at = Prediction::from_ai_probability(0.5);
        assert!((pred_at.threshold_distance(threshold) - 0.0).abs() < 1e-6);

        // At extreme (1.0): distance = 1
        let pred_max = Prediction::from_ai_probability(1.0);
        assert!((pred_max.threshold_distance(threshold) - 1.0).abs() < 1e-6);

        // At extreme (0.0): distance = 1
        let pred_min = Prediction::from_ai_probability(0.0);
        assert!((pred_min.threshold_distance(threshold) - 1.0).abs() < 1e-6);

        // Halfway between threshold and max
        let pred_mid = Prediction::from_ai_probability(0.75);
        assert!((pred_mid.threshold_distance(threshold) - 0.5).abs() < 1e-6);
    }

    // Sample Reliability Tests

    #[test]
    fn test_sample_adequacy_multi_chunk() {
        assert!((sample_adequacy(2, None) - 0.63).abs() < 0.1);
        assert!((sample_adequacy(3, None) - 0.78).abs() < 0.1);
        assert!((sample_adequacy(5, None) - 0.92).abs() < 0.05);
    }

    #[test]
    fn test_sample_adequacy_single_chunk_with_tokens() {
        // Very short: 5 tokens
        assert!(sample_adequacy(1, Some(5)) < 0.05); // ~1%

        // Short: 30 tokens (minimum)
        assert!((sample_adequacy(1, Some(30)) - 0.08).abs() < 0.05); // ~8%

        // Medium: 75 tokens
        assert!((sample_adequacy(1, Some(75)) - 0.20).abs() < 0.05); // ~20%

        // Full chunk: 150 tokens
        assert!((sample_adequacy(1, Some(150)) - 0.39).abs() < 0.05); // ~39%
    }

    #[test]
    fn test_chunk_std_dev_calculation() {
        let chunks = vec![
            Prediction::from_ai_probability(0.8),
            Prediction::from_ai_probability(0.85),
            Prediction::from_ai_probability(0.9),
        ];
        let unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        let std_dev = unified.calculate_chunk_std_dev();
        assert!(std_dev.is_some());
        assert!(std_dev.unwrap() < 0.1);
    }

    #[test]
    fn test_sample_reliability_single_chunk_no_token_count() {
        let chunks = vec![Prediction::from_ai_probability(0.9)];
        let unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        // No token count set - should use base adequacy only
        let metrics = unified.confidence_metrics(Threshold::default());
        assert!((metrics.sample_reliability - 0.39).abs() < 0.1);
    }

    #[test]
    fn test_sample_reliability_single_chunk_very_short() {
        let chunks = vec![Prediction::from_ai_probability(0.9)];
        let mut unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        unified.single_chunk_token_count = Some(5); // Very short
        let metrics = unified.confidence_metrics(Threshold::default());
        assert!(metrics.sample_reliability < 0.05); // Very low reliability
    }

    #[test]
    fn test_sample_reliability_single_chunk_full_size() {
        let chunks = vec![Prediction::from_ai_probability(0.9)];
        let mut unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        unified.single_chunk_token_count = Some(150); // Full chunk
        let metrics = unified.confidence_metrics(Threshold::default());
        assert!((metrics.sample_reliability - 0.39).abs() < 0.1); // Moderate reliability
    }

    #[test]
    fn test_sample_reliability_multi_chunk_low_variance() {
        let chunks = vec![
            Prediction::from_ai_probability(0.88),
            Prediction::from_ai_probability(0.90),
            Prediction::from_ai_probability(0.92),
            Prediction::from_ai_probability(0.89),
            Prediction::from_ai_probability(0.91),
        ];
        let unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        let metrics = unified.confidence_metrics(Threshold::default());
        assert!(metrics.sample_reliability > 0.7); // High reliability
    }

    #[test]
    fn test_sample_reliability_multi_chunk_high_variance() {
        let chunks = vec![
            Prediction::from_ai_probability(0.2),
            Prediction::from_ai_probability(0.9),
            Prediction::from_ai_probability(0.3),
            Prediction::from_ai_probability(0.8),
            Prediction::from_ai_probability(0.4),
        ];
        let unified = UnifiedPrediction::new(chunks, AggregationMethod::Mean);
        let metrics = unified.confidence_metrics(Threshold::default());
        assert!(metrics.sample_reliability < 0.5); // Low reliability due to variance
    }
}
