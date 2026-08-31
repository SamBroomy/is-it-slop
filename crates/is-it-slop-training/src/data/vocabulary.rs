//! Frequency-ranked token vocabulary: raw o200k ids to compact embedding rows.
//!
//! Port of `build_vocab` in `notebooks/hierarchical_train.py`. Folding ids modulo a bound would
//! make two *common* tokens share one embedding row at random; ranking by frequency confines
//! collisions to the rare tail, which all shares a single OOV row.

use std::collections::HashMap;
use std::path::Path;

use is_it_slop_preprocessing::pre_processor::tokenize;

use super::DataError;

/// `o200k_base` ids fit comfortably below this; the lookup table covers the whole space.
pub const O200K_CEILING: usize = 201_000;

/// Maps raw o200k token ids to compact rows: `0..vocab_size` for the ranked tokens, `vocab_size`
/// (the OOV row) for everything else. Row `vocab_size + 1` is padding, written by the encoder.
pub struct Vocabulary {
    /// Indexed by raw token id; values are compact rows (rank or OOV).
    lut: Vec<u16>,
    vocab_size: usize,
}

impl Vocabulary {
    /// Rank the top `vocab_size` tokens of a corpus by frequency.
    ///
    /// Derive the vocabulary from the training split only; deriving it from test would leak.
    pub fn build<T: AsRef<str> + Sync>(texts: &[T], vocab_size: usize) -> Self {
        Self::from_token_lists(&tokenize(texts), vocab_size)
    }

    /// Rank pre-tokenized documents. Ties break by first occurrence in the corpus, matching
    /// Python's `Counter.most_common` (stable sort on count over insertion order).
    #[must_use]
    pub fn from_token_lists(documents: &[Vec<u32>], vocab_size: usize) -> Self {
        assert!(
            u16::try_from(vocab_size + 2).is_ok(),
            "vocabulary no longer fits u16"
        );
        let mut counts: HashMap<u32, (u64, usize)> = HashMap::new();
        let mut seen = 0usize;
        for token in documents.iter().flatten() {
            counts
                .entry(*token)
                .and_modify(|(count, _)| *count += 1)
                .or_insert_with(|| {
                    seen += 1;
                    (1, seen)
                });
        }

        let mut ranked: Vec<(u32, (u64, usize))> = counts.into_iter().collect();
        ranked.sort_by_key(|&(_, (count, first_seen))| (std::cmp::Reverse(count), first_seen));

        let oov = vocab_size as u16;
        let mut lut = vec![oov; O200K_CEILING];
        for (rank, (token, _)) in ranked.iter().take(vocab_size).enumerate() {
            let token = *token as usize;
            assert!(
                token < O200K_CEILING,
                "token id {token} above the o200k ceiling"
            );
            lut[token] = rank as u16;
        }
        Self { lut, vocab_size }
    }

    /// Distinct ranked rows; also the OOV row index.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The row rare tokens collapse onto.
    #[must_use]
    pub const fn oov_id(&self) -> u16 {
        self.vocab_size as u16
    }

    /// The padding row the encoder fills grids with. Never produced by [`Self::row`].
    #[must_use]
    pub const fn pad_id(&self) -> u16 {
        self.vocab_size as u16 + 1
    }

    /// The compact row for a raw token id.
    #[must_use]
    pub fn row(&self, token: u32) -> u16 {
        let token = token as usize;
        assert!(
            token < O200K_CEILING,
            "token id {token} above the o200k ceiling"
        );
        self.lut[token]
    }

    /// Write the bundle's `token_vocabulary.safetensors`: one i32 tensor `token_to_row` of length
    /// [`O200K_CEILING`], exactly what the Python trainer ships.
    pub fn save(&self, path: &Path) -> Result<(), DataError> {
        let data: Vec<u8> = self
            .lut
            .iter()
            .flat_map(|&row| i32::from(row).to_le_bytes())
            .collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::I32,
            vec![O200K_CEILING],
            &data,
        )?;
        safetensors::serialize_to_file([("token_to_row", view)], None, path)?;
        Ok(())
    }

    /// Load a saved vocabulary. `vocab_size` is recovered as the OOV row value, the largest in
    /// the table.
    pub fn load(path: &Path) -> Result<Self, DataError> {
        let bytes = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor("token_to_row")?;
        if view.dtype() != safetensors::Dtype::I32 || view.shape() != [O200K_CEILING] {
            return Err(DataError::Contract(format!(
                "token_to_row must be i32 [{O200K_CEILING}], got {:?} {:?}",
                view.dtype(),
                view.shape()
            )));
        }
        let mut lut = Vec::with_capacity(O200K_CEILING);
        let mut max = 0i32;
        for chunk in view.data().chunks_exact(4) {
            let value = i32::from_le_bytes(chunk.try_into().expect("4-byte chunks"));
            if !(0..=i32::from(u16::MAX)).contains(&value) {
                return Err(DataError::Contract(format!(
                    "token_to_row value {value} outside the compact row range"
                )));
            }
            max = max.max(value);
            lut.push(value as u16);
        }
        let vocab_size = max as usize;
        // Same invariant `from_token_lists` asserts: the pad row (vocab_size + 1) must fit u16,
        // or `pad_id()` overflows.
        if u16::try_from(vocab_size + 2).is_err() {
            return Err(DataError::Contract(format!(
                "vocabulary of {vocab_size} rows no longer fits u16 with OOV and padding"
            )));
        }
        Ok(Self { lut, vocab_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_match_python_counter() {
        // Python oracle: Counter over [[5,7,5,9],[7,5,1,9],[2,2,9]].most_common() =
        // [(5,3), (9,3), (7,2), (2,2), (1,1)] — ties broken by first occurrence.
        let docs = vec![vec![5, 7, 5, 9], vec![7, 5, 1, 9], vec![2, 2, 9]];
        let vocab = Vocabulary::from_token_lists(&docs, 4);

        assert_eq!(vocab.row(5), 0);
        assert_eq!(vocab.row(9), 1);
        assert_eq!(vocab.row(7), 2);
        assert_eq!(vocab.row(2), 3);
        assert_eq!(
            vocab.row(1),
            vocab.oov_id(),
            "rank 5 falls off a 4-row vocabulary"
        );
        assert_eq!(vocab.row(123_456), vocab.oov_id(), "unseen tokens are OOV");
    }

    #[test]
    fn save_load_round_trip() {
        let docs = vec![vec![10, 20, 10], vec![30]];
        let vocab = Vocabulary::from_token_lists(&docs, 2);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token_vocabulary.safetensors");

        vocab.save(&path).unwrap();
        let loaded = Vocabulary::load(&path).unwrap();

        assert_eq!(loaded.vocab_size(), 2);
        for token in [10u32, 20, 30, 40] {
            assert_eq!(loaded.row(token), vocab.row(token));
        }
    }
}
