//! Reading the released dataset splits.
//!
//! The splits are the parquets under `data/<version>/` (`train`, `validation`, `test`), already
//! cleaned during curation. Only `text` and `label` are read; the `dataset` source column exists
//! for `CutMix` partner selection, which is not ported.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use polars::prelude::*;

/// One split's documents.
#[derive(Debug)]
pub struct SplitData {
    /// Raw document texts.
    pub texts: Vec<String>,
    /// 1.0 for AI, 0.0 for human.
    pub labels: Vec<f32>,
}

impl SplitData {
    /// Number of documents.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// Read a split parquet (e.g. `data/3.0.0/train.parquet`).
///
/// # Errors
///
/// Returns an error if the file cannot be read or decoded as parquet, or if a required column is
/// missing or has the wrong type.
pub fn read_split(path: &Path) -> Result<SplitData> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // polars' own `PolarsContext` also has `with_context`, so name anyhow's explicitly.
    let df = Context::with_context(ParquetReader::new(file).finish(), || {
        format!("reading parquet {}", path.display())
    })?;

    // Null cells fail loudly: the released splits are fully populated, so a null means the file
    // is not what this loader thinks it is.
    let texts: Vec<String> = df
        .column("text")?
        .str()?
        .iter()
        .map(|t| t.map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| anyhow!("null text cell in {}", path.display()))?;
    let labels: Vec<f32> = df
        .column("label")?
        .cast(&DataType::Float32)?
        .f32()?
        .iter()
        .collect::<Option<_>>()
        .ok_or_else(|| anyhow!("null label cell in {}", path.display()))?;

    ensure!(
        texts.len() == labels.len() && !texts.is_empty(),
        "split {} has {} texts and {} labels",
        path.display(),
        texts.len(),
        labels.len()
    );
    tracing::info!(
        split = %path.display(),
        rows = texts.len(),
        ai = labels.iter().filter(|&&l| l > 0.5).count(),
        "split loaded"
    );
    Ok(SplitData { texts, labels })
}
