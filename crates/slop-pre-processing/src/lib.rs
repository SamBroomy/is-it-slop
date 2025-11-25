#[cfg(feature = "python")]
mod python;

/// SET THIS TO MATCH the max `ngram_range` once we know what parameters we'll use for max efficiency
const NGRAM_CONST_KEY: usize = 8;

mod data_loader;
pub mod pre_processor;
