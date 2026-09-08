//! Text to the model's rectangular id grid — the one encoding shared by training and inference.
//!
//! Strip (optionally clean), tokenize, apply the vocabulary, and cut constant-size
//! non-overlapping chunks, padding the remainder. Training builds the encoder from a fresh
//! [`Vocabulary`]; inference reconstructs the exact same encoding from a shipped bundle via
//! [`DocumentEncoder::from_bundle`].

mod vocabulary;

pub use vocabulary::{O200K_CEILING, Vocabulary};

use crate::{
    as_strs,
    pre_processor::{text_cleaner_for_inference, tokenize},
};

/// Failures loading or validating encoder artifacts.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// Filesystem failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Vocabulary (de)serialization failure.
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// `encoder_config.json` (de)serialization failure.
    #[error(transparent)]
    Config(#[from] serde_json::Error),
    /// The artifact does not match the encoding contract.
    #[error("{0}")]
    Contract(String),
}

/// The bundle's `encoder_config.json`: everything inference needs to reproduce this encoding.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EncoderBundleConfig {
    /// Which build of this crate produced the bundle; inference must tokenize identically.
    pub preprocessing_version: String,
    /// The BPE scheme behind the token ids.
    pub tokenizer: String,
    /// Tokens per chunk.
    pub chunk_size: usize,
    /// The training-time chunk budget; the network accepts any count.
    pub max_chunks_seen_in_training: usize,
    /// Drop a trailing chunk this empty.
    pub min_fill: f64,
    /// Chunks are contiguous and non-overlapping.
    pub overlap: usize,
    /// Distinct token rows kept.
    pub vocab_size: usize,
    /// Row for tokens outside the vocabulary.
    pub oov_id: u16,
    /// Row for padding.
    pub pad_id: u16,
}

/// One document, encoded to the rectangular `[max_chunks * chunk_size]` grid of compact token
/// ids (padding included), plus its label.
///
/// Ids are `u16`: the vocabulary asserts `vocab_size + 2` fits, and the compact width keeps a
/// full encoded split ~4x smaller in memory than wide ids would.
#[derive(Debug, Clone)]
pub struct EncodedDocument {
    /// Compact token ids, row-major `[max_chunks, chunk_size]`.
    pub ids: Vec<u16>,
    /// 1.0 for AI, 0.0 for human.
    pub label: f32,
}

/// Whether the encoder cleans text before tokenizing.
///
/// The training parquets were cleaned during dataset curation, so the training path tokenizes
/// them as-is; raw user input at inference gets the universal cleanup first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextCleaning {
    /// Tokenize the input as-is (training on curated data).
    Off,
    /// Apply the universal inference cleaner first (raw user input).
    Inference,
}

/// Encodes documents into `[max_chunks, chunk_size]` grids of compact ids.
pub struct DocumentEncoder {
    vocabulary: Vocabulary,
    chunk_size: usize,
    max_chunks: usize,
    min_fill: f64,
    cleaning: TextCleaning,
}

impl DocumentEncoder {
    /// The frozen encoding: 192-token chunks, 8-chunk training budget, 10% minimum fill.
    /// Cleaning is off — the training path encodes curated data.
    #[must_use]
    pub const fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            chunk_size: 192,
            max_chunks: 8,
            min_fill: 0.10,
            cleaning: TextCleaning::Off,
        }
    }

    /// Reconstruct the encoder a bundle shipped: the exact grid shape and vocabulary the model
    /// was trained with, with inference cleaning on for raw user input.
    pub fn from_bundle(vocab_bytes: &[u8], config_json: &str) -> Result<Self, EncoderError> {
        let config: EncoderBundleConfig = serde_json::from_str(config_json)?;
        if config.tokenizer != "o200k_base" {
            return Err(EncoderError::Contract(format!(
                "bundle tokenizer {:?} is not o200k_base",
                config.tokenizer
            )));
        }
        if config.overlap != 0 {
            return Err(EncoderError::Contract(format!(
                "bundle overlap {} is not 0; this encoder cuts contiguous chunks",
                config.overlap
            )));
        }
        let vocabulary = Vocabulary::from_bytes(vocab_bytes)?;
        if vocabulary.vocab_size() != config.vocab_size
            || vocabulary.oov_id() != config.oov_id
            || vocabulary.pad_id() != config.pad_id
        {
            return Err(EncoderError::Contract(format!(
                "vocabulary ({} rows, oov {}, pad {}) disagrees with encoder_config.json \
                 ({}, {}, {})",
                vocabulary.vocab_size(),
                vocabulary.oov_id(),
                vocabulary.pad_id(),
                config.vocab_size,
                config.oov_id,
                config.pad_id
            )));
        }
        Ok(Self {
            vocabulary,
            chunk_size: config.chunk_size,
            max_chunks: config.max_chunks_seen_in_training,
            min_fill: config.min_fill,
            cleaning: TextCleaning::Inference,
        })
    }

    /// Override the grid shape (for tests and experiments; the defaults are the frozen recipe).
    #[must_use]
    pub const fn with_grid(mut self, chunk_size: usize, max_chunks: usize, min_fill: f64) -> Self {
        self.chunk_size = chunk_size;
        self.max_chunks = max_chunks;
        self.min_fill = min_fill;
        self
    }

    /// Override the cleaning mode.
    #[must_use]
    pub const fn with_cleaning(mut self, cleaning: TextCleaning) -> Self {
        self.cleaning = cleaning;
        self
    }

    /// Tokens per chunk.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Chunk slots per document grid.
    #[must_use]
    pub const fn max_chunks(&self) -> usize {
        self.max_chunks
    }

    /// The vocabulary this encoder applies.
    #[must_use]
    pub const fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Everything the bundle's `encoder_config.json` records.
    #[must_use]
    pub fn bundle_config(&self) -> EncoderBundleConfig {
        EncoderBundleConfig {
            preprocessing_version: env!("CARGO_PKG_VERSION").to_string(),
            tokenizer: "o200k_base".to_string(),
            chunk_size: self.chunk_size,
            max_chunks_seen_in_training: self.max_chunks,
            min_fill: self.min_fill,
            overlap: 0,
            vocab_size: self.vocabulary.vocab_size(),
            oov_id: self.vocabulary.oov_id(),
            pad_id: self.vocabulary.pad_id(),
        }
    }

    /// Clean (per the encoder's mode), trim, and tokenize; tokenization parallelizes internally
    /// for large batches.
    fn token_lists(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        match self.cleaning {
            TextCleaning::Off => {
                let trimmed: Vec<&str> = texts.iter().map(|t| t.trim()).collect();
                tokenize(&trimmed)
            }
            TextCleaning::Inference => {
                let cleaner = text_cleaner_for_inference();
                let cleaned: Vec<String> = cleaner.clean_batch(texts);
                let trimmed: Vec<&str> = cleaned.iter().map(|t| t.trim()).collect();
                tokenize(&trimmed)
            }
        }
    }

    /// Encode a batch of raw texts with their labels, one grid per document.
    ///
    /// A document longer than the grid is truncated — this is the evaluation-side encoding,
    /// where one document must yield exactly one prediction.
    pub fn encode_batch<T: AsRef<str>>(&self, texts: &[T], labels: &[f32]) -> Vec<EncodedDocument> {
        self.encode_strs(&as_strs(texts), labels)
    }

    fn encode_strs(&self, texts: &[&str], labels: &[f32]) -> Vec<EncodedDocument> {
        assert_eq!(texts.len(), labels.len());
        self.token_lists(texts)
            .iter()
            .zip(labels)
            .map(|(tokens, &label)| EncodedDocument {
                ids: self.grid(tokens),
                label,
            })
            .collect()
    }

    /// Encode a batch for training, splitting long documents instead of truncating them.
    ///
    /// A document longer than the `max_chunks * chunk_size` grid becomes several training
    /// examples — consecutive full grids plus the remainder — so no text is discarded (the
    /// campaign measured real headroom in the 2000+-token band that truncation was leaving on
    /// the table). Every fragment carries the document's label, which is exact for whole-document
    /// labels. A trailing fragment shorter than `min_fill * chunk_size` tokens is dropped, the
    /// window-level analogue of [`Self::grid`]'s trailing-chunk rule. Evaluation must use
    /// [`Self::encode_batch`] instead: fragments would count as extra documents in split-level
    /// metrics.
    pub fn encode_batch_split<T: AsRef<str>>(
        &self,
        texts: &[T],
        labels: &[f32],
    ) -> Vec<EncodedDocument> {
        self.encode_split_strs(&as_strs(texts), labels)
    }

    fn encode_split_strs(&self, texts: &[&str], labels: &[f32]) -> Vec<EncodedDocument> {
        assert_eq!(texts.len(), labels.len());
        self.token_lists(texts)
            .iter()
            .zip(labels)
            .flat_map(|(tokens, &label)| {
                self.split_grids(tokens)
                    .into_iter()
                    .map(move |ids| EncodedDocument { ids, label })
            })
            .collect()
    }

    /// Encode a batch for inference: each document's full token stream as its
    /// `max_chunks * chunk_size` window grids, nothing truncated.
    ///
    /// The model's chunk axis is dynamic — `max_chunks` is a training budget, not a limit — so
    /// inference concatenates a document's windows into one `[windows * max_chunks, chunk_size]`
    /// grid and the model masks out padding chunks wherever they fall. This is the same window
    /// tiling as [`Self::encode_batch_split`], kept label-free and grouped per document.
    pub fn encode_batch_windows<T: AsRef<str>>(&self, texts: &[T]) -> Vec<Vec<Vec<u16>>> {
        self.encode_windows_strs(&as_strs(texts))
    }

    fn encode_windows_strs(&self, texts: &[&str]) -> Vec<Vec<Vec<u16>>> {
        self.token_lists(texts)
            .iter()
            .map(|tokens| self.split_grids(tokens))
            .collect()
    }

    /// One grid per `max_chunks * chunk_size` window of the tokens; see
    /// [`Self::encode_batch_split`] for the tail rule.
    fn split_grids(&self, tokens: &[u32]) -> Vec<Vec<u16>> {
        let span = self.max_chunks * self.chunk_size;
        let full = tokens.len() / span;
        let tail = tokens.len() % span;
        let keep_tail = (tail as f64) >= self.min_fill * self.chunk_size as f64;
        let windows = (full + usize::from(keep_tail)).max(1);
        (0..windows)
            .map(|w| self.grid(&tokens[w * span..((w + 1) * span).min(tokens.len())]))
            .collect()
    }

    /// Fill a `[max_chunks * chunk_size]` grid with contiguous, non-overlapping chunks.
    ///
    /// A trailing chunk below `min_fill` is dropped unless it is the only one, since with no
    /// overlap it discards at most `min_fill * chunk_size` tokens. The final chunk is padded
    /// rather than back-filled from earlier text: duplication is measurably harmful and padding
    /// is masked out by the model.
    fn grid(&self, tokens: &[u32]) -> Vec<u16> {
        let cs = self.chunk_size;
        let pad = self.vocabulary.pad_id();
        let mut out = vec![pad; self.max_chunks * cs];
        if tokens.is_empty() {
            return out;
        }

        let n_full = tokens.len() / cs;
        let remainder = tokens.len() % cs;
        let keep_tail = (remainder as f64) >= self.min_fill * cs as f64;
        let n_chunks = (n_full + usize::from(keep_tail)).clamp(1, self.max_chunks);

        let take = tokens.len().min(n_chunks * cs);
        for (slot, token) in out.iter_mut().zip(&tokens[..take]) {
            *slot = self.vocabulary.row(*token);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder(chunk_size: usize, max_chunks: usize) -> DocumentEncoder {
        // Identity-ish vocabulary over tokens 0..6: rank = first-seen order of a fixed corpus.
        let vocab = Vocabulary::from_token_lists(&[vec![0, 1, 2, 3, 4, 5]], 6);
        DocumentEncoder::new(vocab).with_grid(chunk_size, max_chunks, 0.10)
    }

    fn grid(enc: &DocumentEncoder, tokens: &[u32]) -> Vec<u16> {
        enc.grid(tokens)
    }

    #[test]
    fn chunking_matches_python_semantics() {
        let enc = encoder(4, 3);
        let pad = enc.vocabulary().pad_id();

        // Empty: all pad (the model's slot-0 clamp keeps the forward finite).
        assert_eq!(grid(&enc, &[]), vec![pad; 12]);

        // Exact multiple: two full chunks, third slot padded.
        assert_eq!(
            grid(&enc, &[0, 1, 2, 3, 4, 5, 0, 1]),
            vec![0, 1, 2, 3, 4, 5, 0, 1, pad, pad, pad, pad]
        );

        // Remainder above min_fill (1 >= 0.4): trailing partial chunk kept.
        assert_eq!(
            grid(&enc, &[0, 1, 2, 3, 4]),
            vec![0, 1, 2, 3, 4, pad, pad, pad, pad, pad, pad, pad]
        );

        // A short single chunk is always kept even below min_fill of a *second* chunk.
        assert_eq!(grid(&enc, &[5]), {
            let mut want = vec![pad; 12];
            want[0] = 5;
            want
        });

        // Longer than the budget: truncated at max_chunks.
        let long: Vec<u32> = (0..20).map(|i| i % 6).collect();
        let got = grid(&enc, &long);
        assert_eq!(got.len(), 12);
        assert!(got.iter().all(|&r| r != pad), "a full grid has no padding");
    }

    #[test]
    fn min_fill_drops_a_nearly_empty_tail() {
        // chunk_size 10, min_fill 0.2: the threshold is 2 tokens, and Python's rule is
        // `remainder >= min_fill * chunk_size` — at the boundary the tail is kept.
        let enc = encoder(10, 2).with_grid(10, 2, 0.2);
        let pad = enc.vocabulary().pad_id();

        // remainder 1 < 2 (=0.2*10): tail dropped.
        let tokens: Vec<u32> = (0..11).map(|i| i % 6).collect();
        let got = enc.grid(&tokens);
        assert!(
            got[10..].iter().all(|&r| r == pad),
            "tail below min_fill is dropped"
        );

        // remainder 2 == 2: tail kept.
        let tokens: Vec<u32> = (0..12).map(|i| i % 6).collect();
        let got = enc.grid(&tokens);
        assert!(
            got[10..12].iter().all(|&r| r != pad),
            "tail at min_fill is kept"
        );
    }

    #[test]
    fn splitting_covers_every_window_and_drops_a_runt_tail() {
        // chunk_size 4, max_chunks 3: a window spans 12 tokens; min_fill 0.10 keeps any tail
        // window of >= 0.4 tokens, i.e. any non-empty one.
        let enc = encoder(4, 3);

        // 30 tokens: two full windows plus a 6-token tail -> three grids, recursively. Period 5
        // over the 6-token vocabulary so each window's first token differs.
        let tokens: Vec<u32> = (0..30).map(|i| i % 5).collect();
        let grids = enc.split_grids(&tokens);
        assert_eq!(grids.len(), 3);
        let pad = enc.vocabulary().pad_id();
        assert!(grids[0].iter().all(|&r| r != pad), "window 0 is full");
        assert!(grids[1].iter().all(|&r| r != pad), "window 1 is full");
        // The tail window holds 6 tokens: one full chunk plus a kept 2-token remainder.
        assert_eq!(grids[2].iter().filter(|&&r| r != pad).count(), 6);

        // Windows tile the document: window w starts at token 12*w.
        assert_eq!(grids[1][0], enc.vocabulary().row(tokens[12]));
        assert_eq!(grids[2][0], enc.vocabulary().row(tokens[24]));

        // A short document is one grid, identical to the truncating path.
        let short: Vec<u32> = (0..5).collect();
        assert_eq!(enc.split_grids(&short), vec![enc.grid(&short)]);
        assert_eq!(enc.split_grids(&[]), vec![enc.grid(&[])]);

        // A runt tail below min_fill of one chunk is dropped: threshold 2 tokens at min_fill 0.2.
        let enc = encoder(10, 2).with_grid(10, 2, 0.2);
        let tokens: Vec<u32> = (0..21).map(|i| i % 6).collect();
        assert_eq!(
            enc.split_grids(&tokens).len(),
            1,
            "1-token tail window dropped"
        );
        let tokens: Vec<u32> = (0..22).map(|i| i % 6).collect();
        assert_eq!(
            enc.split_grids(&tokens).len(),
            2,
            "tail at the threshold kept"
        );
    }

    #[test]
    fn oov_tokens_collapse_onto_the_oov_row() {
        let enc = encoder(4, 1);
        let got = enc.grid(&[0, 99_999, 3]);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], enc.vocabulary().oov_id());
        assert_eq!(got[2], 3);
    }

    #[test]
    fn from_bundle_round_trips_the_shipped_encoding() {
        let vocab = Vocabulary::from_token_lists(&[vec![0, 1, 2, 3, 4, 5]], 6);
        let training_encoder = DocumentEncoder::new(vocab).with_grid(4, 3, 0.10);
        let config_json = serde_json::to_string(&training_encoder.bundle_config()).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let vocab_path = dir.path().join("token_vocabulary.safetensors");
        training_encoder.vocabulary().save(&vocab_path).unwrap();
        let vocab_bytes = std::fs::read(&vocab_path).unwrap();

        let restored = DocumentEncoder::from_bundle(&vocab_bytes, &config_json).unwrap();
        assert_eq!(restored.chunk_size(), 4);
        assert_eq!(restored.max_chunks(), 3);
        assert_eq!(restored.cleaning, TextCleaning::Inference);

        // Grids agree between the training encoder and the reconstructed one.
        let tokens: Vec<u32> = (0..10).collect();
        assert_eq!(restored.grid(&tokens), training_encoder.grid(&tokens));

        // A config that disagrees with the vocabulary is refused.
        let mut bad: EncoderBundleConfig =
            serde_json::from_str(&config_json).expect("config parses");
        bad.vocab_size = 999;
        let bad_json = serde_json::to_string(&bad).unwrap();
        assert!(DocumentEncoder::from_bundle(&vocab_bytes, &bad_json).is_err());
    }
}
