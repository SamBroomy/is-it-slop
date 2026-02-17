//! Vectorizer configuration parameters.
//!
//! This module defines [`VectorizerParams`], which controls:
//! - N-gram range (e.g., 2-4 for bigrams through 4-grams)
//! - Vocabulary filtering (`min_df` and `max_df` thresholds)
//! - Term frequency scaling (sublinear TF option)
//!
//! # Parameter Guidelines
//!
//! - **`min_df`**: Filter rare terms to reduce vocabulary size and noise
//!   - Proportion (0.0-1.0): Term must appear in at least X% of documents
//!   - Absolute (≥1.0): Term must appear in at least X documents
//!   - Default: 10.0 (10 documents minimum)
//!
//! - **`max_df`**: Filter common terms that appear in most documents
//!   - Proportion (0.0-1.0): Term must appear in at most X% of documents
//!   - Absolute (>1.0): Term must appear in at most X documents
//!   - Default: 1.0 (100% - no filtering)
//!
//! - **`sublinear_tf`**: Apply log scaling to term frequencies
//!   - `tf → log(tf + 1)`
//!   - Reduces impact of terms repeated many times
//!   - Default: false

/// Default minimum n-gram size.
pub const DEFAULT_MIN_NGRAM: usize = 2;

/// Default maximum n-gram size.
///
/// **Important**: This value is also used to optimize storage for n-gram keys as they are stored in
/// a fixed-size format (u128) that can hold up to 4 tokens (4 * 32 bits = 128 bits). If you need
/// n-grams larger than 4, you will need to change the storage format and increase this constant
/// accordingly, which may involve more complex logic for handling variable-length n-grams and could
/// impact performance. If you frequently use n-gram ranges larger than this, consider increasing
/// this constant and recompiling for better performance. If you frequently use n-gram ranges larger
/// than this, consider increasing this constant and recompiling for better performance.
pub const DEFAULT_MAX_NGRAM: usize = 4;

/// Configuration parameters for text vectorization.
///
/// Controls n-gram extraction, vocabulary filtering, and term frequency scaling.
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct VectorizerParams {
    ngram_range: Vec<usize>,
    /// Minimum document frequency for filtering vocabulary.
    /// - If `min_df` is in (0.0, 1.0), it's a proportion of documents
    /// - If `min_df` >= 1.0, it's an absolute document count
    min_df: f32,
    /// Maximum document frequency for filtering vocabulary.
    /// - If `max_df` is in (0.0, 1.0], it's a proportion of documents
    /// - If `max_df` > 1.0, it's an absolute document count
    max_df: f32,
    /// Apply sublinear tf scaling: replace term frequency `tf` with `1 + log(tf)`.
    /// This reduces the impact of terms that occur many times in a document.
    sublinear_tf: bool,
}

impl VectorizerParams {
    /// Create new vectorizer parameters.
    ///
    /// # Arguments
    /// * `ngram_range` - Range of n-gram sizes (e.g., `3..=5` for trigrams to 5-grams)
    /// * `min_df` - Minimum document frequency (proportion or count)
    /// * `max_df` - Maximum document frequency (proportion or count)
    /// * `sublinear_tf` - Whether to apply log scaling to term frequencies
    ///
    /// # Panics
    /// Panics if `min_df` or `max_df` are not positive, or if `ngram_range` is empty.
    #[must_use]
    pub fn new(
        // ngram_range: impl Into<RangeInclusive<usize>>,
        min_df: f32,
        max_df: f32,
        sublinear_tf: bool,
    ) -> Self {
        let ngram_range = DEFAULT_MIN_NGRAM..=DEFAULT_MAX_NGRAM;
        let n_sizes = ngram_range.collect::<Vec<_>>();
        assert!(
            !n_sizes.is_empty(),
            "ngram_range must contain at least one value"
        );
        assert!(
            min_df > 0.0,
            "min_df must be positive (proportion in (0.0, 1.0) or absolute count >= 1.0)"
        );
        assert!(
            max_df > 0.0,
            "max_df must be positive (proportion in (0.0, 1.0] or absolute count > 1.0)"
        );
        Self {
            ngram_range: n_sizes,
            min_df,
            max_df,
            sublinear_tf,
        }
    }

    /// Get all n-gram sizes as a slice.
    #[must_use]
    pub fn ngram_counts(&self) -> &[usize] {
        &self.ngram_range
    }

    /// Get the n-gram range as a tuple `(min, max)`.
    #[must_use]
    pub fn ngram_range(&self) -> (usize, usize) {
        (
            *self.ngram_range.first().expect("ngram_range is not empty"),
            *self.ngram_range.last().expect("ngram_range is not empty"),
        )
    }

    /// Get the minimum document frequency threshold.
    #[must_use]
    pub fn min_df(&self) -> f32 {
        self.min_df
    }

    /// Get the maximum document frequency threshold.
    #[must_use]
    pub fn max_df(&self) -> f32 {
        self.max_df
    }

    /// Get whether sublinear TF scaling is enabled.
    #[must_use]
    pub fn sublinear_tf(&self) -> bool {
        self.sublinear_tf
    }
}
impl Default for VectorizerParams {
    fn default() -> Self {
        Self {
            ngram_range: (DEFAULT_MIN_NGRAM..=DEFAULT_MAX_NGRAM).collect(),
            min_df: 10.0,
            max_df: 1.0,
            sublinear_tf: false,
        }
    }
}

// impl From<((usize, usize), f32, f32, bool)> for VectorizerParams {
//     fn from(value: ((usize, usize), f32, f32, bool)) -> Self {
//         Self::new(value.0.0..=value.0.1, value.1, value.2, value.3)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_default_params() {
        let params = VectorizerParams::default();
        assert_eq!(params.ngram_range(), (2, 4));
        assert_eq!(params.min_df(), 10.0);
        assert_eq!(params.max_df(), 1.0);
        assert!(!params.sublinear_tf());
    }

    #[test]
    #[should_panic(expected = "min_df must be positive")]
    fn test_invalid_min_df() {
        let _ = VectorizerParams::new(0.0, 1.0, false);
    }

    #[test]
    fn test_ngram_counts() {
        let params = VectorizerParams::default();
        assert_eq!(params.ngram_counts(), &[2, 3, 4]);
    }
}
