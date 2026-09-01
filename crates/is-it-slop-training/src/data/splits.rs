//! Reading the released dataset splits.
//!
//! The splits are the parquets under `data/<version>/` (`train`, `validation`, `test`), already
//! cleaned during curation. Only `text` and `label` are read; the `dataset` source column exists
//! for `CutMix` partner selection, which is not ported.

use std::path::Path;

use polars::prelude::*;

use super::DataError;

/// One split's documents.
pub struct SplitData {
    /// Raw document texts.
    pub texts: Vec<String>,
    /// 1.0 for AI, 0.0 for human.
    pub labels: Vec<f32>,
}

impl SplitData {
    /// Number of documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// Read a split parquet (e.g. `data/3.0.0/train.parquet`).
pub fn read_split(path: &Path) -> Result<SplitData, DataError> {
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;

    // Null cells fail loudly: the released splits are fully populated, so a null means the file
    // is not what this loader thinks it is.
    let texts: Vec<String> = df
        .column("text")?
        .str()?
        .into_iter()
        .map(|t| t.map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| DataError::Contract(format!("null text cell in {}", path.display())))?;
    let labels: Vec<f32> = df
        .column("label")?
        .cast(&DataType::Float32)?
        .f32()?
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| DataError::Contract(format!("null label cell in {}", path.display())))?;

    if texts.len() != labels.len() || texts.is_empty() {
        return Err(DataError::Contract(format!(
            "split {} has {} texts and {} labels",
            path.display(),
            texts.len(),
            labels.len()
        )));
    }
    tracing::info!(
        split = %path.display(),
        rows = texts.len(),
        ai = labels.iter().filter(|&&l| l > 0.5).count(),
        "split loaded"
    );
    Ok(SplitData { texts, labels })
}
