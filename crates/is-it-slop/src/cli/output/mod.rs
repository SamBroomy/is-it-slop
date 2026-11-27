//! Where the CLI's results go.
//!
//! A [`Sink`] receives one scored document at a time and writes it in one format; the format
//! is chosen once from `--output`. Human-readable sinks report failed documents on stderr so
//! stdout stays parseable; the JSON sinks carry them inline as `status: "error"` rows.

mod json;
mod jsonl;
mod label;
mod pretty;
mod score;
mod tsv;

use std::io::{self, Write};

use clap::ValueEnum;

use super::input::DocumentId;
use crate::{Classification, Threshold, UnifiedPrediction};

/// A prediction with the classification the run's threshold gives it.
#[derive(Debug)]
pub struct Scored {
    /// The model output.
    pub prediction: UnifiedPrediction,
    /// The prediction thresholded.
    pub class: Classification,
    /// The text behind each chunk prediction, when the sink asked for it
    /// ([`Sink::wants_chunk_texts`]).
    pub chunk_texts: Option<Vec<String>>,
}

impl Scored {
    /// Classify `prediction` at `threshold`.
    #[must_use]
    pub fn new(prediction: UnifiedPrediction, threshold: Threshold) -> Self {
        let class = prediction.classification(threshold);
        Self {
            prediction,
            class,
            chunk_texts: None,
        }
    }

    /// Attach the chunk texts, index for index with `prediction.chunk_predictions`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_texts` is not the same length as `prediction.chunk_predictions`.
    #[must_use]
    pub fn with_chunk_texts(mut self, chunk_texts: Vec<String>) -> Self {
        assert_eq!(
            chunk_texts.len(),
            self.prediction.chunk_predictions.len(),
            "chunk texts must match the prediction chunk count"
        );
        self.chunk_texts = Some(chunk_texts);
        self
    }
}

/// One document's outcome: scored, or failed before or during prediction.
pub type Item = Result<Scored, anyhow::Error>;

/// Writes results in one output format.
pub trait Sink {
    /// Called once before the first item.
    ///
    /// # Errors
    ///
    /// The default implementation is infallible; implementations may fail while writing.
    fn begin(&mut self, out: &mut dyn Write) -> io::Result<()> {
        let _ = out;
        Ok(())
    }

    /// Write one scored document.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()>;

    /// Report one document that could not be scored.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    fn emit_error(
        &mut self,
        out: &mut dyn Write,
        id: &DocumentId,
        error: &anyhow::Error,
    ) -> io::Result<()>;

    /// Called once after the last item.
    ///
    /// # Errors
    ///
    /// The default implementation is infallible; implementations may fail while flushing.
    fn finish(&mut self, out: &mut dyn Write, summary: &RunSummary) -> io::Result<()> {
        let _ = (out, summary);
        Ok(())
    }

    /// Whether emitted items should carry [`Scored::chunk_texts`]; the texts cost a second
    /// tokenization per document, so they are only produced on request.
    fn wants_chunk_texts(&self) -> bool {
        false
    }
}

/// Per-run settings a sink may need.
#[derive(Debug, Clone, Copy)]
pub struct SinkOptions {
    /// Whether the run can produce more than one document, so results need their id.
    pub labelled: bool,
    /// Whether to emit ANSI colour.
    pub color: bool,
    /// How much the pretty format shows per document.
    pub detail: Detail,
    /// Whether the pretty format lists every chunk with its score and text.
    pub chunks: bool,
}

/// How much the pretty format shows per document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Detail {
    /// The classification bar and nothing else.
    Quiet,
    /// Bar, confidence and warnings.
    #[default]
    Normal,
    /// Everything, plus the chunk analysis.
    Verbose,
}

/// The output formats `--output` offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum OutputFormat {
    /// Probability bar, confidence and warnings.
    #[default]
    Pretty,
    /// `Human` or `AI`, one per line.
    Label,
    /// The AI probability, one per line.
    Score,
    /// Tab-separated `id`, label and AI probability.
    Tsv,
    /// One JSON array of result objects.
    Json,
    /// One JSON result object per line.
    Jsonl,
}

impl OutputFormat {
    /// Build the sink for this format.
    #[must_use]
    pub fn sink(self, options: SinkOptions) -> Box<dyn Sink> {
        match self {
            Self::Pretty => Box::new(pretty::Pretty::new(options)),
            Self::Label => Box::new(label::Label::new(options)),
            Self::Score => Box::new(score::Score::new(options)),
            Self::Tsv => Box::new(tsv::Tsv),
            Self::Json => Box::new(json::Json::default()),
            Self::Jsonl => Box::new(jsonl::Jsonl),
        }
    }
}

/// When to colour output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum ColorChoice {
    /// Colour when stdout is a terminal and `NO_COLOR` is unset.
    #[default]
    Auto,
    /// Always colour.
    Always,
    /// Never colour.
    Never,
}

impl ColorChoice {
    /// Resolve against the real stdout.
    #[must_use]
    pub fn enabled(self) -> bool {
        match self {
            Self::Auto => supports_color::on(supports_color::Stream::Stdout).is_some(),
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// How the process exit code reports on the documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExitPolicy {
    /// 0 on success, 1 if any document failed.
    #[default]
    Standard,
    /// With `--exit-status`: 0 if any document is AI, 1 if none are, 2 if any failed.
    Classification,
}

impl ExitPolicy {
    const fn is_classification(self) -> bool {
        matches!(self, Self::Classification)
    }
}

/// Counts of how the run's documents came out; decides the exit code.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RunSummary {
    /// Documents classified as AI.
    pub ai: usize,
    /// Documents classified as human.
    pub human: usize,
    /// Documents that could not be scored.
    pub errors: usize,
}

impl RunSummary {
    /// Count one document's outcome.
    pub const fn record(&mut self, item: &Item) {
        match item {
            Ok(Scored {
                class: Classification::AI,
                ..
            }) => self.ai += 1,
            Ok(Scored {
                class: Classification::Human,
                ..
            }) => self.human += 1,
            Err(_) => self.errors += 1,
        }
    }

    /// Every document seen, scored or not.
    #[must_use]
    pub const fn documents(&self) -> usize {
        self.ai + self.human + self.errors
    }

    /// Any error is a failure (1). With [`ExitPolicy::Classification`], 0 means at least one
    /// document was AI, 1 means none were, and 2 means a document failed.
    #[must_use]
    pub const fn exit_code(&self, policy: ExitPolicy) -> u8 {
        let classification = policy.is_classification();
        if self.errors > 0 {
            if classification { 2 } else { 1 }
        } else if classification && self.ai == 0 {
            1
        } else {
            0
        }
    }
}

/// The JSON object for one scored document.
fn ok_row(id: &DocumentId, scored: &Scored) -> serde_json::Value {
    let prediction = &scored.prediction;
    serde_json::json!({
        "status": "ok",
        "id": id.to_json(),
        "class": scored.class,
        "class_label": scored.class.to_string(),
        "probabilities": {
            "human": prediction.prediction.human_probability(),
            "ai": prediction.prediction.ai_probability(),
        },
        "confidence": {
            "confidence": prediction.confidence(),
            "chunk_dispersion": prediction.chunk_dispersion(),
        },
        "chunk_info": {
            "num_chunks": prediction.num_chunks(),
            "chunk_dispersion": prediction.chunk_dispersion(),
        },
        "chunk_predictions": prediction.chunk_predictions.iter().map(|p| {
            (p.human_probability(), p.ai_probability())
        }).collect::<Vec<_>>(),
    })
}

/// The JSON object for one failed document.
fn error_row(id: &DocumentId, error: &anyhow::Error) -> serde_json::Value {
    serde_json::json!({
        "status": "error",
        "id": id.to_json(),
        "error": format!("{error:#}"),
    })
}

/// The stderr line for a failed document in the human-readable formats.
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "private formatting helper; the flag is named at every call site"
)]
fn report_error(labelled: bool, id: &DocumentId, error: &anyhow::Error) {
    if labelled {
        eprintln!("{id}: {error:#}");
    } else {
        eprintln!("Error: {error:#}");
    }
}

/// `"{id}: "` when results need their id, otherwise nothing.
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "private formatting helper; the flag is named at every call site"
)]
fn prefix(labelled: bool, id: &DocumentId) -> String {
    if labelled {
        format!("{id}: ")
    } else {
        String::new()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::cli::input::Source;

    pub(crate) fn scored(ai: f32, chunks: &[f32]) -> Scored {
        Scored::new(UnifiedPrediction::new(ai, chunks), Threshold::new(0.5))
    }

    pub(crate) fn file_id(index: usize) -> DocumentId {
        DocumentId::Record {
            source: Source::File(PathBuf::from("texts.txt")),
            index,
            id: None,
        }
    }

    pub(crate) fn failed() -> anyhow::Error {
        anyhow::anyhow!("boom")
    }

    /// Drive a sink over `items` into a buffer, returning the text it wrote.
    pub(crate) fn render(sink: &mut dyn Sink, items: &[(DocumentId, Item)]) -> String {
        let mut out = Vec::new();
        let mut summary = RunSummary::default();
        sink.begin(&mut out).unwrap();
        for (id, item) in items {
            match item {
                Ok(scored) => sink.emit(&mut out, id, scored).unwrap(),
                Err(error) => sink.emit_error(&mut out, id, error).unwrap(),
            }
            summary.record(item);
        }
        sink.finish(&mut out, &summary).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn summary_counts_and_exit_codes() {
        let mut summary = RunSummary::default();
        assert_eq!(summary.exit_code(ExitPolicy::Standard), 0);
        assert_eq!(
            summary.exit_code(ExitPolicy::Classification),
            1,
            "no documents means no AI found"
        );

        summary.record(&Ok(scored(0.1, &[0.1])));
        assert_eq!(summary.exit_code(ExitPolicy::Standard), 0);
        assert_eq!(summary.exit_code(ExitPolicy::Classification), 1);

        summary.record(&Ok(scored(0.9, &[0.9])));
        assert_eq!(summary.exit_code(ExitPolicy::Classification), 0);

        summary.record(&Err(failed()));
        assert_eq!(
            summary,
            RunSummary {
                ai: 1,
                human: 1,
                errors: 1
            }
        );
        assert_eq!(summary.documents(), 3);
        assert_eq!(summary.exit_code(ExitPolicy::Standard), 1);
        assert_eq!(summary.exit_code(ExitPolicy::Classification), 2);
    }

    #[test]
    fn error_row_shape() {
        let row = error_row(&file_id(2), &failed());
        assert_eq!(row["status"], "error");
        assert_eq!(row["id"], "texts.txt:2");
        assert_eq!(row["error"], "boom");
    }
}
