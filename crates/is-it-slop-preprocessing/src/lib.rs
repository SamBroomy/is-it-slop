#[cfg(feature = "python")]
mod python;

/// IMPORTANT! SET THIS TO MATCH the max `ngram_range` once we know what parameters we'll use for lowest most efficient memory usage
const NGRAM_CONST_KEY: usize = 4;

pub mod pre_processor;
