use core::fmt;

use super::Classification;

/// Prediction result containing probabilities for both classes.
///
/// Contains P(Human) and P(AI), which always sum to 1.0.
///
/// # Examples
///
/// ```rust
/// use is_it_slop::Predictor;
///
/// let predictor = Predictor::new();
/// let prediction = predictor.predict("some text")?;
///
/// println!("Human: {:.2}%", prediction.human_probability() * 100.0);
/// println!("AI: {:.2}%", prediction.ai_probability() * 100.0);
/// # Ok::<(), anyhow::Error>(())
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
            (human_prob + ai_prob - 1.0).abs() < f32::EPSILON,
            "Probabilities must sum to 1.0"
        );
        Self(human_prob, ai_prob)
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
    pub fn classification(&self, threshold: f32) -> Classification {
        if self.1 >= threshold {
            Classification::AI
        } else {
            Classification::Human
        }
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
