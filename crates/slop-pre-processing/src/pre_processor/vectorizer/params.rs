use std::ops::RangeInclusive;

#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct VectorizerParams {
    ngram_range: Vec<usize>,
    min_df: usize,
}
impl VectorizerParams {
    pub fn new(ngram_range: impl Into<RangeInclusive<usize>>, min_df: usize) -> Self {
        let n_sizes = ngram_range.into().collect::<Vec<_>>();
        assert!(
            !n_sizes.is_empty(),
            "ngram_range must contain at least one value"
        );
        Self {
            ngram_range: n_sizes,
            min_df,
        }
    }

    #[must_use]
    pub fn ngram_range(&self) -> &[usize] {
        &self.ngram_range
    }

    #[must_use]
    pub fn min_df(&self) -> usize {
        self.min_df
    }
}
impl Default for VectorizerParams {
    fn default() -> Self {
        Self {
            ngram_range: vec![3, 5],
            min_df: 10,
        }
    }
}

impl From<((usize, usize), usize)> for VectorizerParams {
    fn from(value: ((usize, usize), usize)) -> Self {
        Self::new(value.0.0..=value.0.1, value.1)
    }
}
