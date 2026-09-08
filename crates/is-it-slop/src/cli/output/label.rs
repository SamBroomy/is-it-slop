//! `Human` or `AI`, one per line.

use std::io::{self, Write};

use super::{Scored, Sink, SinkOptions, prefix, report_error};
use crate::cli::input::DocumentId;

pub struct Label {
    labelled: bool,
}

impl Label {
    pub fn new(options: SinkOptions) -> Self {
        Self {
            labelled: options.labelled,
        }
    }
}

impl Sink for Label {
    fn emit(&mut self, out: &mut dyn Write, id: &DocumentId, scored: &Scored) -> io::Result<()> {
        writeln!(out, "{}{}", prefix(self.labelled, id), scored.class)
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
    fn one_label_per_line_with_id_only_when_labelled() {
        let options = SinkOptions {
            labelled: true,
            color: false,
            detail: Detail::Normal,
            chunks: false,
        };
        let text = render(
            &mut Label::new(options),
            &[
                (file_id(1), Ok(scored(0.9, &[0.9]))),
                (file_id(2), Err(failed())),
                (file_id(3), Ok(scored(0.1, &[0.1]))),
            ],
        );
        assert_eq!(text, "texts.txt:1: AI\ntexts.txt:3: Human\n");

        let text = render(
            &mut Label::new(SinkOptions {
                labelled: false,
                ..options
            }),
            &[(DocumentId::Whole(Source::Argument), Ok(scored(0.1, &[0.1])))],
        );
        assert_eq!(text, "Human\n");
    }
}
