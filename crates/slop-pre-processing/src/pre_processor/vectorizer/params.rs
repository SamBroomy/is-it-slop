use std::ops::RangeInclusive;

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct VectorizerParams {
    ngram_range: Vec<usize>,
    /// Minimum document frequency for filtering vocabulary.
    /// - If `min_df` is in (0.0, 1.0), it's a proportion of documents
    /// - If `min_df` >= 1.0, it's an absolute document count
    min_df: f64,
    /// Maximum document frequency for filtering vocabulary.
    /// - If `max_df` is in (0.0, 1.0], it's a proportion of documents
    /// - If `max_df` > 1.0, it's an absolute document count
    max_df: f64,
    /// Apply sublinear tf scaling: replace term frequency `tf` with `1 + log(tf)`.
    /// This reduces the impact of terms that occur many times in a document.
    sublinear_tf: bool,
}

impl VectorizerParams {
    pub fn new(
        ngram_range: impl Into<RangeInclusive<usize>>,
        min_df: f64,
        max_df: f64,
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

    #[must_use]
    pub fn ngram_counts(&self) -> &[usize] {
        &self.ngram_range
    }

    #[must_use]
    pub fn ngram_range(&self) -> (usize, usize) {
        (
            *self.ngram_range.first().expect("ngram_range is not empty"),
            *self.ngram_range.last().expect("ngram_range is not empty"),
        )
    }

    #[must_use]
    pub fn min_df(&self) -> f64 {
        self.min_df
    }

    #[must_use]
    pub fn max_df(&self) -> f64 {
        self.max_df
    }

    #[must_use]
    pub fn sublinear_tf(&self) -> bool {
        self.sublinear_tf
    }
}
impl Default for VectorizerParams {
    fn default() -> Self {
        Self {
            ngram_range: vec![2, 4],
            min_df: 10.0,
            max_df: 1.0,
            sublinear_tf: false,
        }
    }
}

impl From<((usize, usize), f64, f64, bool)> for VectorizerParams {
    fn from(value: ((usize, usize), f64, f64, bool)) -> Self {
        Self::new(value.0.0..=value.0.1, value.1, value.2, value.3)
    }
}
