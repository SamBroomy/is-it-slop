use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorizerParams {
    ngram_range: Vec<usize>,
    min_df: usize,
}
impl VectorizerParams {
    pub fn new(ngram_range: impl Into<RangeInclusive<usize>>, min_df: usize) -> Self {
        let n_sizes = ngram_range.into().collect();
        Self {
            ngram_range: n_sizes,
            min_df,
        }
    }

    pub fn ngram_range(&self) -> &[usize] {
        &self.ngram_range
    }

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
