//! Reading the released dataset splits.
//!
//! The splits are the parquets under `data/<version>/` (`train`, `validation`, `test`), already
//! cleaned during curation. Only `text` and `label` are read; the `dataset` source column exists
//! for `CutMix` partner selection, which is not ported.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use arrow_array::{Array, Int8Array, LargeStringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

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
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("opening parquet {}", path.display()))?
        .build()
        .with_context(|| format!("reading parquet {}", path.display()))?;

    let mut texts = Vec::new();
    let mut labels = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("reading parquet {}", path.display()))?;

        // Null cells fail loudly: the released splits are fully populated, so a null means the
        // file is not what this loader thinks it is.
        let text_column = batch
            .column_by_name("text")
            .ok_or_else(|| anyhow!("missing `text` column in {}", path.display()))?
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| {
                anyhow!(
                    "`text` column is not a large utf8 string in {}",
                    path.display()
                )
            })?;
        let label_column = batch
            .column_by_name("label")
            .ok_or_else(|| anyhow!("missing `label` column in {}", path.display()))?
            .as_any()
            .downcast_ref::<Int8Array>()
            .ok_or_else(|| anyhow!("`label` column is not an int8 in {}", path.display()))?;

        texts.extend(
            text_column
                .iter()
                .map(|text| text.map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow!("null text cell in {}", path.display()))?,
        );
        labels.extend(
            label_column
                .iter()
                .map(|label| label.map(f32::from))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow!("null label cell in {}", path.display()))?,
        );
    }

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
