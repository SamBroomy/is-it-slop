//! One JSON result object per line.

use std::io::{self, Write};

use super::{Scored, Sink, error_row, ok_row};
use crate::cli::input::DocumentId;

pub(crate) struct Jsonl;

fn write_line(out: &mut dyn Write, row: &serde_json::Value) -> io::Result<()> {
    serde_json::to_writer(&mut *out, row)?;
    writeln!(out)
}

impl Sink for Jsonl {
    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        write_line(out, &ok_row(id, scored))
    }

    fn emit_error(
        &mut self,
        out: &mut dyn Write,
        id: &DocumentId,
        error: &anyhow::Error,
    ) -> io::Result<()> {
        write_line(out, &error_row(id, error))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};

    #[test]
    fn one_row_per_line_including_errors() {
        let text = render(
            &mut Jsonl,
            &[
                (file_id(1), Ok(scored(0.9, &[0.9]))),
                (file_id(2), Err(failed())),
            ],
        );
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["id"], "texts.txt:1");
        assert_eq!(rows[1]["status"], "error");
    }
}
