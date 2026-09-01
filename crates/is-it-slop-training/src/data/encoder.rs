//! Text to the model's rectangular id grid.
//!
//! Port of `chunk_document`/`encode_corpus` in `notebooks/hierarchical_train.py`: strip, tokenize
//! through the shared preprocessing crate, apply the vocabulary, and cut constant-size
//! non-overlapping chunks, padding the remainder. No cleaner runs here — the released parquets
//! were cleaned during curation and the Python trainer applies none either (the known skew
//! recorded in `NEXT_RELEASE.md` Phase 2).
//!
//! This is the seam `NEXT_RELEASE` plans to move into `is-it-slop-preprocessing` as
//! `DocumentEncoder::from_bundle`, so training and inference encode through one implementation.

use is_it_slop_preprocessing::pre_processor::tokenize;

use super::{EncodedDocument, Vocabulary};

/// The bundle's `encoder_config.json`: everything inference needs to reproduce this encoding.
///
/// Key names match what `notebooks/hierarchical_train.py` wrote, so any consumer reads both.
#[derive(Debug, serde::Serialize)]
pub struct EncoderBundleConfig {
    /// Which build of the shared preprocessing crate produced this bundle; inference must
    /// tokenize identically.
    pub preprocessing_version: &'static str,
    /// The BPE scheme behind the token ids.
    pub tokenizer: &'static str,
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

/// Encodes documents into `[max_chunks, chunk_size]` grids of compact ids.
pub struct DocumentEncoder {
    vocabulary: Vocabulary,
    chunk_size: usize,
    max_chunks: usize,
    min_fill: f64,
}

impl DocumentEncoder {
    /// The frozen encoding: 192-token chunks, 8-chunk training budget, 10% minimum fill.
    #[must_use]
    pub const fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            chunk_size: 192,
            max_chunks: 8,
            min_fill: 0.10,
        }
    }

    /// Override the grid shape (for tests and experiments; the defaults are the frozen recipe).
    #[must_use]
    pub const fn with_grid(mut self, chunk_size: usize, max_chunks: usize, min_fill: f64) -> Self {
        self.chunk_size = chunk_size;
        self.max_chunks = max_chunks;
        self.min_fill = min_fill;
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

    /// Everything the bundle's `encoder_config.json` records, matching the Python trainer's keys.
    #[must_use]
    pub fn bundle_config(&self) -> EncoderBundleConfig {
        EncoderBundleConfig {
            // The preprocessing crate is version-locked to the workspace (`=` pin in the root
            // Cargo.toml), so this crate's own version identifies the tokenizer build.
            preprocessing_version: env!("CARGO_PKG_VERSION"),
            tokenizer: "o200k_base",
            chunk_size: self.chunk_size,
            max_chunks_seen_in_training: self.max_chunks,
            min_fill: self.min_fill,
            overlap: 0,
            vocab_size: self.vocabulary.vocab_size(),
            oov_id: self.vocabulary.oov_id(),
            pad_id: self.vocabulary.pad_id(),
        }
    }

    /// Encode a batch of raw texts with their labels.
    ///
    /// Texts are trimmed before tokenization, as the Python cache builder did. Tokenization
    /// parallelizes internally for large batches.
    pub fn encode_batch<T: AsRef<str> + Sync>(
        &self,
        texts: &[T],
        labels: &[f32],
    ) -> Vec<EncodedDocument> {
        assert_eq!(texts.len(), labels.len());
        let trimmed: Vec<&str> = texts.iter().map(|t| t.as_ref().trim()).collect();
        let token_lists = tokenize(&trimmed);
        token_lists
            .iter()
            .zip(labels)
            .map(|(tokens, &label)| EncodedDocument {
                ids: self.grid(tokens),
                label,
            })
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
    fn oov_tokens_collapse_onto_the_oov_row() {
        let enc = encoder(4, 1);
        let got = enc.grid(&[0, 99_999, 3]);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], enc.vocabulary().oov_id());
        assert_eq!(got[2], 3);
    }
}
