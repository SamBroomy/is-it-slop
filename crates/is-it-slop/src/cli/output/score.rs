//! The AI probability, one per line.

use std::io::{self, Write};

use super::{Scored, Sink, SinkOptions, prefix, report_error};
use crate::cli::input::DocumentId;

pub struct Score {
    labelled: bool,
}

impl Score {
    pub fn new(options: SinkOptions) -> Self {
        Self {
            labelled: options.labelled,
        }
    }
}

impl Sink for Score {
    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        writeln!(
            out,
            "{}{:.4}",
            prefix(self.labelled, id),
            scored.prediction.prediction.ai_probability()
        )
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
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};
    use crate::cli::input::Source;
    use crate::cli::output::Detail;

    #[test]
    fn four_decimal_scores() {
        let options = SinkOptions {
            labelled: false,
            color: false,
            detail: Detail::Normal,
            chunks: false,
        };
        let text = render(
            &mut Score::new(options),
            &[(
                DocumentId::Whole(Source::Argument),
                Ok(scored(0.12345, &[0.1])),
            )],
        );
        assert_eq!(text, "0.1235\n");

        let text = render(
            &mut Score::new(SinkOptions {
                labelled: true,
                ..options
            }),
            &[(file_id(4), Ok(scored(0.5, &[0.5])))],
        );
        assert_eq!(text, "texts.txt:4: 0.5000\n");
    }
}
