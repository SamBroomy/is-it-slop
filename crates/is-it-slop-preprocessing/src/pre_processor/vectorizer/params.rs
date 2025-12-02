use std::ops::RangeInclusive;

/// Default minimum n-gram size.
pub const DEFAULT_MIN_NGRAM: usize = 2;

/// Default maximum n-gram size.
///
/// **Important**: This value is also used to optimize `SmallVec` storage for n-gram keys.
/// If you frequently use n-gram ranges larger than this, consider increasing this constant
/// and recompiling for better performance.
pub const DEFAULT_MAX_NGRAM: usize = 4;

/// Configuration parameters for text vectorization.
///
/// Controls n-gram extraction, vocabulary filtering, and term frequency scaling.
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
    pub fn new(
        ngram_range: impl Into<RangeInclusive<usize>>,
        min_df: f32,
        max_df: f32,
        sublinear_tf: bool,
    ) -> Self {
        let n_sizes = ngram_range.into().collect::<Vec<_>>();
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
            ngram_range: vec![DEFAULT_MIN_NGRAM, DEFAULT_MAX_NGRAM],
            min_df: 10.0,
            max_df: 1.0,
            sublinear_tf: false,
        }
    }
}

impl From<((usize, usize), f32, f32, bool)> for VectorizerParams {
    fn from(value: ((usize, usize), f32, f32, bool)) -> Self {
        Self::new(value.0.0..=value.0.1, value.1, value.2, value.3)
    }
}
