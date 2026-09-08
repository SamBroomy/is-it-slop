//! Tab-separated `id`, label and AI probability, always three columns so consumers can rely on
//! the shape whether or not the run had several documents.

use std::io::{self, Write};

use super::{Scored, Sink, report_error};
use crate::cli::input::DocumentId;

pub struct Tsv;

impl Sink for Tsv {
    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        writeln!(
            out,
            "{id}\t{}\t{:.4}",
            scored.class,
            scored.prediction.prediction.ai_probability()
        )
    }

    fn emit_error(
        &mut self,
        _out: &mut dyn Write,
        id: &DocumentId,
        error: &anyhow::Error,
    ) -> io::Result<()> {
        report_error(true, id, error);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};

    #[test]
    fn three_columns_and_errors_off_stdout() {
        let text = render(
            &mut Tsv,
            &[
                (file_id(1), Ok(scored(0.9, &[0.9]))),
                (file_id(2), Err(failed())),
            ],
        );
        assert_eq!(text, "texts.txt:1\tAI\t0.9000\n");
    }
}
