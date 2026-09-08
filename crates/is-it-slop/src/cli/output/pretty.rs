//! The human-readable sink: a probability bar, the calibrated confidence, and warnings.

use std::io::{self, Write};

use owo_colors::{OwoColorize, Style};

use super::{Detail, RunSummary, Scored, Sink, SinkOptions, report_error};
use crate::{Classification, Prediction, cli::input::DocumentId};

const STYLE_HEADER: Style = Style::new().bold();
const STYLE_DIM: Style = Style::new().dimmed();
const STYLE_WARN: Style = Style::new().yellow();
const STYLE_AI_LABEL: Style = Style::new().cyan().bold();
const STYLE_HUMAN_LABEL: Style = Style::new().truecolor(0xFF, 0x8C, 0x00).bold();
const STYLE_AI_BAR: Style = Style::new().cyan();
const STYLE_HUMAN_BAR: Style = Style::new().truecolor(0xFF, 0x8C, 0x00);

// "Human" (5 chars) is the widest label; BAR_WIDTH matches render_bar's visual width.
const LABEL_WIDTH: usize = 5;
const BAR_WIDTH: usize = 20;
/// Characters of each chunk's text shown in the chunk list.
const CHUNK_PREVIEW_CHARS: usize = 60;

pub struct Pretty {
    labelled: bool,
    color: bool,
    detail: Detail,
    chunks: bool,
}

impl Pretty {
    pub fn new(options: SinkOptions) -> Self {
        Self {
            labelled: options.labelled,
            color: options.color,
            detail: options.detail,
            chunks: options.chunks,
        }
    }

    fn paint(&self, text: &str, style: Style) -> String {
        if self.color {
            style.style(text).to_string()
        } else {
            text.to_owned()
        }
    }

    fn truecolor(&self, text: &str, (r, g, b): (u8, u8, u8)) -> String {
        if self.color {
            text.truecolor(r, g, b).to_string()
        } else {
            text.to_owned()
        }
    }

    fn write_block(&self, out: &mut dyn Write, scored: &Scored) -> io::Result<()> {
        let prediction = &scored.prediction;
        let human_prob = prediction.prediction.human_probability();
        let ai_prob = prediction.prediction.ai_probability();
        let metrics = prediction.confidence_metrics();

        let (bar_value, bar_style) = match scored.class {
            Classification::Human => (human_prob, STYLE_HUMAN_BAR),
            Classification::AI => (ai_prob, STYLE_AI_BAR),
        };
        let bar = self.paint(&render_bar(bar_value), bar_style);

        // Left = predicted class probability, right = other class probability.
        let left_pct = format!("{:.1}%", bar_value * 100.0);
        let right_pct = format!("{:.1}%", (1.0 - bar_value) * 100.0);
        let left_label = self.class_label(scored.class, &format!("{:>LABEL_WIDTH$}", scored.class));
        let right_label = self.class_label(
            scored.class.opposite(),
            &format!("{:<LABEL_WIDTH$}", scored.class.opposite()),
        );

        writeln!(out)?;
        writeln!(out, "  {left_label}  [{bar}]  {right_label}")?;

        // Indent to align percentages inside the [ ]: 2sp + LABEL_WIDTH + 2sp + "["
        let pct_indent = " ".repeat(LABEL_WIDTH + 5);
        let pct_right_width = BAR_WIDTH - left_pct.len();
        writeln!(out, "{pct_indent}{left_pct}{right_pct:>pct_right_width$}")?;

        if self.detail == Detail::Quiet {
            return Ok(());
        }

        if self.detail == Detail::Verbose && prediction.chunk_predictions.len() > 1 {
            let chunk_info = prediction.chunk_info();
            writeln!(out)?;
            writeln!(out, "{}", self.paint("Chunk Analysis", STYLE_HEADER))?;
            let chunks_t = ((chunk_info.num_chunks as f32 - 4.0) / 6.0).clamp(0.0, 1.0);
            writeln!(
                out,
                "  {:<12} {}",
                "Chunks",
                self.truecolor(&chunk_info.num_chunks.to_string(), rose_rgb(chunks_t))
            )?;
            if let Some(dispersion) = chunk_info.chunk_dispersion {
                writeln!(
                    out,
                    "  {:<12} {}",
                    "Dispersion",
                    self.truecolor(
                        &format!("{dispersion:.3}"),
                        rose_rgb(1.0 - (dispersion * 3.0).min(1.0))
                    )
                )?;
            }
        }

        writeln!(out)?;
        let confidence_label = self.paint(&format!("{:<12}", "Confidence"), STYLE_HEADER);
        let confidence = self.truecolor(
            &format!("{:>5.1}%", metrics.confidence * 100.0),
            confidence_rgb(metrics.confidence),
        );
        writeln!(out, "  {confidence_label} {confidence}")?;

        if metrics.chunk_dispersion.is_some_and(|d| d > 0.25) {
            writeln!(out)?;
            writeln!(
                out,
                "{}",
                self.paint(
                    "Chunks disagree — possible mixed human/AI content",
                    STYLE_WARN
                )
            )?;
        } else if metrics.confidence < 0.3 {
            writeln!(out)?;
            writeln!(
                out,
                "{}",
                self.paint("Low confidence — prediction uncertain", STYLE_WARN)
            )?;
        }

        // Last: the list can be long, and the summary above should not scroll away.
        if let Some(chunk_texts) = &scored.chunk_texts {
            self.write_chunks(out, &prediction.chunk_predictions, chunk_texts)?;
        }
        Ok(())
    }

    /// One line per chunk: its AI score as a bar and the start of its text. Chunk scores come
    /// from the auxiliary chunk head and are not calibrated, so the bar is always AI-oriented
    /// and coloured by which side of 0.5 it falls, not by the run's threshold.
    fn write_chunks(
        &self,
        out: &mut dyn Write,
        predictions: &[Prediction],
        texts: &[String],
    ) -> io::Result<()> {
        writeln!(out)?;
        writeln!(
            out,
            "{} {}",
            self.paint("Chunks", STYLE_HEADER),
            self.paint("(uncalibrated chunk scores)", STYLE_DIM)
        )?;
        let index_width = predictions.len().to_string().len();
        for (i, (prediction, text)) in predictions.iter().zip(texts).enumerate() {
            let ai = prediction.ai_probability();
            let style = if ai >= 0.5 {
                STYLE_AI_BAR
            } else {
                STYLE_HUMAN_BAR
            };
            let bar = self.paint(&render_bar(ai), style);
            let pct = format!("{:>5.1}% AI", ai * 100.0);
            let preview = self.paint(&preview(text), STYLE_DIM);
            writeln!(out, "  {:>index_width$}  [{bar}]  {pct}  {preview}", i + 1)?;
        }
        Ok(())
    }

    fn class_label(&self, class: Classification, text: &str) -> String {
        let style = match class {
            Classification::AI => STYLE_AI_LABEL,
            Classification::Human => STYLE_HUMAN_LABEL,
        };
        self.paint(text, style)
    }
}

/// The start of a chunk on one line: whitespace collapsed, cut at [`CHUNK_PREVIEW_CHARS`].
fn preview(text: &str) -> String {
    let mut out: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(CHUNK_PREVIEW_CHARS + 1)
        .collect();
    if out.chars().count() > CHUNK_PREVIEW_CHARS {
        out.truncate(
            out.char_indices()
                .nth(CHUNK_PREVIEW_CHARS)
                .map_or(out.len(), |(i, _)| i),
        );
        out.push('…');
    }
    out
}

impl Sink for Pretty {
    fn wants_chunk_texts(&self) -> bool {
        self.chunks
    }

    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        if self.labelled {
            writeln!(out, "--- {id} ---")?;
        }
        self.write_block(out, scored)?;
        if self.labelled {
            writeln!(out)?;
        }
        Ok(())
    }

    fn emit_error(
        &mut self,
        _out: &mut dyn Write,
        id: &DocumentId,
        error: &anyhow::Error,
    ) -> io::Result<()> {
        report_error(self.labelled, id, error);
        Ok(())
    }

    fn finish(&mut self, out: &mut dyn Write, summary: &RunSummary) -> io::Result<()> {
        if !self.labelled {
            return Ok(());
        }
        let errors = if summary.errors > 0 {
            format!(", {} failed", summary.errors)
        } else {
            String::new()
        };
        writeln!(
            out,
            "{} documents: {} AI, {} Human{errors}",
            summary.documents(),
            summary.ai,
            summary.human
        )
    }
}

fn render_bar(value: f32) -> String {
    let filled = ((value * BAR_WIDTH as f32).round() as usize).min(BAR_WIDTH);
    format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled))
}

fn confidence_rgb(value: f32) -> (u8, u8, u8) {
    let t = value.clamp(0.0, 1.0);
    let r = (130.0 + t * (204.0 - 130.0)) as u8;
    let g = (55.0 + t * (153.0 - 55.0)) as u8;
    let b = (185.0 + t * (255.0 - 185.0)) as u8;
    (r, g, b)
}

fn rose_rgb(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r = (185.0 + t * (255.0 - 185.0)) as u8;
    let g = (75.0 + t * (176.0 - 75.0)) as u8;
    let b = (115.0 + t * (204.0 - 115.0)) as u8;
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};
    use crate::cli::input::Source;

    fn options(labelled: bool) -> SinkOptions {
        SinkOptions {
            labelled,
            color: false,
            detail: Detail::Normal,
            chunks: false,
        }
    }

    #[test]
    fn chunk_list_shows_one_scored_line_per_chunk() {
        let sink = &mut Pretty::new(SinkOptions {
            chunks: true,
            ..options(false)
        });
        assert!(sink.wants_chunk_texts());
        let item = scored(0.9, &[0.95, 0.10]).with_chunk_texts(vec![
            "First   chunk\nof the document.".to_owned(),
            "x".repeat(200),
        ]);
        let text = render(sink, &[(DocumentId::Whole(Source::Argument), Ok(item))]);
        assert!(
            text.contains("Chunks (uncalibrated chunk scores)"),
            "{text}"
        );
        assert!(
            text.contains("  1  [███████████████████░]   95.0% AI  First chunk of the document."),
            "{text}"
        );
        let long_line = text
            .lines()
            .find(|l| l.starts_with("  2  "))
            .expect("second chunk");
        assert!(long_line.contains("10.0% AI"), "{long_line}");
        assert!(long_line.ends_with('…'), "{long_line}");
        assert_eq!(
            long_line.chars().filter(|&c| c == 'x').count(),
            CHUNK_PREVIEW_CHARS
        );

        let without = &mut Pretty::new(options(false));
        assert!(!without.wants_chunk_texts());
        let text = render(
            without,
            &[(DocumentId::Whole(Source::Argument), Ok(scored(0.9, &[0.9])))],
        );
        assert!(!text.contains("Chunks ("), "{text}");
    }

    #[test]
    fn single_document_has_bar_and_confidence_and_no_summary() {
        let text = render(
            &mut Pretty::new(options(false)),
            &[(DocumentId::Whole(Source::Argument), Ok(scored(0.9, &[0.9])))],
        );
        assert!(
            text.contains("   AI  [██████████████████░░]  Human"),
            "{text}"
        );
        assert!(text.contains("90.0%"), "{text}");
        assert!(text.contains("Confidence    80.0%"), "{text}");
        assert!(!text.contains("documents:"), "{text}");
        assert!(!text.contains("---"), "{text}");
    }

    #[test]
    fn labelled_run_has_headers_and_summary() {
        let text = render(
            &mut Pretty::new(options(true)),
            &[
                (file_id(1), Ok(scored(0.9, &[0.9]))),
                (file_id(2), Err(failed())),
                (file_id(3), Ok(scored(0.1, &[0.1]))),
            ],
        );
        assert!(text.contains("--- texts.txt:1 ---"), "{text}");
        assert!(text.contains("--- texts.txt:3 ---"), "{text}");
        assert!(!text.contains("texts.txt:2"), "errors go to stderr: {text}");
        assert!(
            text.ends_with("3 documents: 1 AI, 1 Human, 1 failed\n"),
            "{text}"
        );
    }

    #[test]
    fn quiet_and_verbose_toggle_sections() {
        let quiet = render(
            &mut Pretty::new(SinkOptions {
                detail: Detail::Quiet,
                ..options(false)
            }),
            &[(
                DocumentId::Whole(Source::Stdin),
                Ok(scored(0.9, &[0.9, 0.2])),
            )],
        );
        assert!(!quiet.contains("Confidence"), "{quiet}");

        let verbose = render(
            &mut Pretty::new(SinkOptions {
                detail: Detail::Verbose,
                ..options(false)
            }),
            &[(
                DocumentId::Whole(Source::Stdin),
                Ok(scored(0.9, &[0.9, 0.2])),
            )],
        );
        assert!(verbose.contains("Chunk Analysis"), "{verbose}");
        assert!(verbose.contains("Chunks       2"), "{verbose}");
        assert!(verbose.contains("Chunks disagree"), "{verbose}");
    }

    #[test]
    fn color_flag_controls_ansi() {
        let item = [(DocumentId::Whole(Source::Argument), Ok(scored(0.9, &[0.9])))];
        let plain = render(&mut Pretty::new(options(false)), &item);
        assert!(!plain.contains('\u{1b}'), "{plain}");
        let coloured = render(
            &mut Pretty::new(SinkOptions {
                color: true,
                ..options(false)
            }),
            &item,
        );
        assert!(coloured.contains('\u{1b}'), "{coloured}");
    }
}
