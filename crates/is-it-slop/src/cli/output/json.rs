//! One JSON array of result objects, written incrementally so it needs no buffering.

use std::io::{self, Write};

use super::{RunSummary, Scored, Sink, error_row, ok_row};
use crate::cli::input::DocumentId;

#[derive(Default)]
pub struct Json {
    written: usize,
}

impl Json {
    fn write_row(&mut self, out: &mut dyn Write, row: &serde_json::Value) -> io::Result<()> {
        if self.written > 0 {
            write!(out, ",")?;
        }
        self.written += 1;
        serde_json::to_writer(out, row)?;
        Ok(())
    }
}

impl Sink for Json {
    fn begin(&mut self, out: &mut dyn Write) -> io::Result<()> {
        write!(out, "[")
    }

    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        self.write_row(out, &ok_row(id, scored))
    }

    fn emit_error(
        &mut self,
        out: &mut dyn Write,
        id: &DocumentId,
        error: &anyhow::Error,
    ) -> io::Result<()> {
        self.write_row(out, &error_row(id, error))
    }

    fn finish(&mut self, out: &mut dyn Write, _summary: &RunSummary) -> io::Result<()> {
        writeln!(out, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};

    #[test]
    fn array_of_rows_including_errors() {
        let text = render(
            &mut Json::default(),
            &[
                (file_id(1), Ok(scored(0.9, &[0.9]))),
                (file_id(2), Err(failed())),
            ],
        );
        let rows: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["status"], "ok");
        assert_eq!(rows[0]["id"], "texts.txt:1");
        assert_eq!(rows[0]["class_label"], "AI");
        assert_eq!(rows[1]["status"], "error");
        assert_eq!(rows[1]["error"], "boom");
    }

    #[test]
    fn empty_run_is_an_empty_array() {
        assert_eq!(render(&mut Json::default(), &[]), "[]\n");
    }
}
