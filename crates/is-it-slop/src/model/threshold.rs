// This file is auto-generated and updated on build by build.rs

/// Default classification threshold between 0.0 and 1.0.
///
/// If P(AI) >= threshold, the text is classified as AI-generated.
/// Lower thresholds are more sensitive (classify more as AI), higher thresholds are more conservative (classify more as Human).
/// This threshold is optimized for overall f1 score based on validation data and is used by default in prediction functions.
pub const CLASSIFICATION_THRESHOLD: f32 = 0.42344016;
