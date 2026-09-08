//! Where the CLI's documents come from.
//!
//! An [`Input`] is one or more locations (the positional argument, files, stdin) plus a
//! [`Framing`] that says how each location's bytes split into documents. It yields
//! [`Document`]s lazily and in order, so a line- or JSONL-framed file streams instead of being
//! read whole.

use std::{
    fmt,
    fs::File,
    io::{self, BufRead, BufReader, Cursor, IsTerminal, Read},
    iter,
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Deserialize;

use super::Cli;

/// How an input's bytes split into documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Framing {
    /// The whole input is one document.
    #[default]
    Text,
    /// One document per non-blank line.
    Lines,
    /// A JSON array of strings or `{"id": ..., "text": ...}` objects.
    Json,
    /// One JSON string or `{"id": ..., "text": ...}` object per line; stops at the first
    /// malformed record.
    Jsonl,
}

/// Where a document's bytes came from, as the user named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// The positional `TEXT` argument.
    Argument,
    /// Standard input.
    Stdin,
    /// A file given with `--file`.
    File(PathBuf),
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Argument => f.write_str("argument"),
            Self::Stdin => f.write_str("stdin"),
            Self::File(path) => write!(f, "{}", path.display()),
        }
    }
}

/// Identifies one document within a run's inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentId {
    /// A source read whole as one document.
    Whole(Source),
    /// The `index`-th (1-based) line or record of a framed source, with the record's own `id`
    /// when the JSON carried one.
    Record {
        /// The framed source.
        source: Source,
        /// 1-based position within the source: the line number for line framing, the record
        /// ordinal for JSON.
        index: usize,
        /// The `id` field of a JSON record, if present, kept as JSON so it passes through
        /// unchanged.
        id: Option<serde_json::Value>,
    },
}

/// Grep-style label: `path`, `path:3`, or `3` when the source has no name worth repeating.
impl fmt::Display for DocumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whole(source) => write!(f, "{source}"),
            Self::Record { source, index, id } => {
                if let Source::File(path) = source {
                    write!(f, "{}:", path.display())?;
                }
                match id {
                    Some(serde_json::Value::String(id)) => f.write_str(id),
                    Some(id) => write!(f, "{id}"),
                    None => write!(f, "{index}"),
                }
            }
        }
    }
}

impl DocumentId {
    /// The id for JSON output: a record's own id exactly as it was given, otherwise the
    /// display label as a string.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Record { id: Some(id), .. } => id.clone(),
            _ => serde_json::Value::String(self.to_string()),
        }
    }
}

/// One document to classify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Where the document came from.
    pub id: DocumentId,
    /// The raw text, exactly as read.
    pub text: String,
}

/// A document that could not be read, identified so it can be reported in its place.
#[derive(Debug)]
pub struct SourceError {
    /// What failed: the whole location, or the line or record within it.
    pub id: DocumentId,
    /// Why.
    pub error: anyhow::Error,
}

/// Something the CLI can open for bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Location {
    Argument(String),
    Stdin,
    File(PathBuf),
}

impl Location {
    fn source(&self) -> Source {
        match self {
            Self::Argument(_) => Source::Argument,
            Self::Stdin => Source::Stdin,
            Self::File(path) => Source::File(path.clone()),
        }
    }

    fn open(&self) -> Result<Box<dyn BufRead + '_>> {
        Ok(match self {
            Self::Argument(text) => Box::new(Cursor::new(text.as_bytes())),
            Self::Stdin => Box::new(io::stdin().lock()),
            Self::File(path) => {
                Box::new(BufReader::new(File::open(path).with_context(|| {
                    format!("Failed to open file: {}", path.display())
                })?))
            }
        })
    }
}

/// The run's documents: every location, framed the same way, in command-line order.
#[derive(Debug)]
pub struct Input {
    locations: Vec<Location>,
    framing: Framing,
}

impl Input {
    /// Build from the parsed arguments.
    ///
    /// `None` means nothing was given and stdin is a terminal, so there is nothing to read.
    #[must_use]
    pub fn from_cli(cli: &Cli) -> Option<Self> {
        let locations = if let Some(text) = &cli.text {
            vec![Location::Argument(text.clone())]
        } else if !cli.file.is_empty() {
            cli.file
                .iter()
                .map(|path| {
                    if path.as_os_str() == "-" {
                        Location::Stdin
                    } else {
                        Location::File(path.clone())
                    }
                })
                .collect()
        } else if io::stdin().is_terminal() {
            return None;
        } else {
            vec![Location::Stdin]
        };
        Some(Self {
            locations,
            framing: cli.input,
        })
    }

    /// Whether this input can yield more than one document.
    #[must_use]
    pub fn is_multi(&self) -> bool {
        self.framing != Framing::Text || self.locations.len() > 1
    }

    /// The documents, in order. A location that cannot be opened, read or parsed yields an
    /// error item in its place.
    pub fn documents(&self) -> impl Iterator<Item = Result<Document, SourceError>> + '_ {
        self.locations
            .iter()
            .flat_map(move |location| match location.open() {
                Ok(reader) => frame(reader, location.source(), self.framing),
                Err(error) => Box::new(iter::once(Err(SourceError {
                    id: DocumentId::Whole(location.source()),
                    error,
                }))),
            })
    }
}

/// Split one reader into documents according to `framing`.
fn frame<'a>(
    reader: Box<dyn BufRead + 'a>,
    source: Source,
    framing: Framing,
) -> Box<dyn Iterator<Item = Result<Document, SourceError>> + 'a> {
    match framing {
        Framing::Text => {
            let id = DocumentId::Whole(source.clone());
            let item = read_all(reader, &source)
                .map(|text| Document {
                    id: id.clone(),
                    text,
                })
                .map_err(|error| SourceError { id, error });
            Box::new(iter::once(item))
        }
        Framing::Lines => Box::new(reader.lines().enumerate().filter_map(move |(i, line)| {
            let index = i + 1;
            let id = DocumentId::Record {
                source: source.clone(),
                index,
                id: None,
            };
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    let error = anyhow::Error::new(e)
                        .context(format!("Failed to read line {index} of {source}"));
                    return Some(Err(SourceError { id, error }));
                }
            };
            let text = line.trim();
            (!text.is_empty()).then(|| {
                Ok(Document {
                    id,
                    text: text.to_owned(),
                })
            })
        })),
        Framing::Json => {
            let records: Result<Vec<JsonRecord>> = read_all(reader, &source).and_then(|text| {
                serde_json::from_str(&text)
                    .with_context(|| format!("Failed to parse JSON array from {source}"))
            });
            match records {
                Ok(records) => Box::new(
                    records
                        .into_iter()
                        .enumerate()
                        .map(move |(i, record)| Ok(record.into_document(source.clone(), i + 1))),
                ),
                Err(error) => Box::new(iter::once(Err(SourceError {
                    id: DocumentId::Whole(source),
                    error,
                }))),
            }
        }
        Framing::Jsonl => Box::new(
            serde_json::Deserializer::from_reader(reader)
                .into_iter::<JsonRecord>()
                .enumerate()
                .map(move |(i, record)| {
                    let index = i + 1;
                    record
                        .map(|record| record.into_document(source.clone(), index))
                        .map_err(|e| SourceError {
                            id: DocumentId::Record {
                                source: source.clone(),
                                index,
                                id: None,
                            },
                            error: anyhow::Error::new(e).context(format!(
                                "Failed to parse JSON record {index} from {source}"
                            )),
                        })
                }),
        ),
    }
}

fn read_all(mut reader: impl Read, source: &Source) -> Result<String> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .with_context(|| format!("Failed to read {source}"))?;
    Ok(text)
}

/// One element of a JSON array or one JSONL line: a bare string or an object with a `text`.
#[derive(Deserialize)]
#[serde(untagged)]
enum JsonRecord {
    Text(String),
    Object {
        #[serde(default)]
        id: Option<serde_json::Value>,
        text: String,
    },
}

impl JsonRecord {
    fn into_document(self, source: Source, index: usize) -> Document {
        let (id, text) = match self {
            Self::Text(text) => (None, text),
            Self::Object { id, text } => (id, text),
        };
        Document {
            id: DocumentId::Record { source, index, id },
            text,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn argument(text: &str, framing: Framing) -> Input {
        Input {
            locations: vec![Location::Argument(text.to_owned())],
            framing,
        }
    }

    fn collect(input: &Input) -> Vec<Result<Document, SourceError>> {
        input.documents().collect()
    }

    fn record(index: usize, id: Option<serde_json::Value>) -> DocumentId {
        DocumentId::Record {
            source: Source::Argument,
            index,
            id,
        }
    }

    #[test]
    fn text_framing_is_one_whole_document() {
        let docs = collect(&argument("  a\n\nb  ", Framing::Text));
        let [doc] = docs.as_slice() else {
            panic!("expected one document, got {docs:?}")
        };
        let doc = doc.as_ref().unwrap();
        assert_eq!(doc.id, DocumentId::Whole(Source::Argument));
        assert_eq!(doc.text, "  a\n\nb  ");
    }

    #[test]
    fn lines_framing_trims_and_skips_blank_lines_keeping_line_numbers() {
        let docs: Vec<Document> = collect(&argument("a\n\n  b \n   \nc", Framing::Lines))
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            docs,
            vec![
                Document {
                    id: record(1, None),
                    text: "a".into()
                },
                Document {
                    id: record(3, None),
                    text: "b".into()
                },
                Document {
                    id: record(5, None),
                    text: "c".into()
                },
            ]
        );
    }

    #[test]
    fn json_framing_accepts_strings_and_objects() {
        let json = r#"["plain", {"id": "x", "text": "with id"}, {"id": 7, "text": "numeric"},
                      {"id": null, "text": "null id"}, {"text": "no id"}]"#;
        let docs: Vec<Document> = collect(&argument(json, Framing::Json))
            .into_iter()
            .map(Result::unwrap)
            .collect();
        let ids: Vec<DocumentId> = docs.iter().map(|d| d.id.clone()).collect();
        assert_eq!(
            ids,
            vec![
                record(1, None),
                record(2, Some("x".into())),
                record(3, Some(7.into())),
                record(4, None),
                record(5, None),
            ]
        );
        assert_eq!(docs[1].text, "with id");
        assert_eq!(docs[2].id.to_string(), "7");
        assert_eq!(docs[2].id.to_json(), serde_json::json!(7));
        assert_eq!(docs[0].id.to_json(), serde_json::json!("1"));
    }

    #[test]
    fn json_framing_reports_parse_failure_once() {
        let docs = collect(&argument("[1, 2]", Framing::Json));
        assert_eq!(docs.len(), 1);
        let err = docs[0].as_ref().unwrap_err();
        assert_eq!(err.id, DocumentId::Whole(Source::Argument));
        assert!(
            err.error.to_string().contains("JSON array"),
            "{}",
            err.error
        );
    }

    #[test]
    fn jsonl_framing_streams_records_and_ignores_blank_lines() {
        let jsonl = "\"one\"\n\n{\"id\": \"b\", \"text\": \"two\"}\n";
        let docs: Vec<Document> = collect(&argument(jsonl, Framing::Jsonl))
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, record(1, None));
        assert_eq!(docs[1].id, record(2, Some("b".into())));
        assert_eq!(docs[1].text, "two");
    }

    #[test]
    fn jsonl_framing_yields_good_records_before_a_bad_one() {
        let docs = collect(&argument(
            "\"ok\"\n{not json}\n\"never reached\"\n",
            Framing::Jsonl,
        ));
        assert!(docs[0].is_ok());
        let err = docs[1].as_ref().unwrap_err();
        assert_eq!(err.id, record(2, None));
        assert!(err.error.to_string().contains("record 2"), "{}", err.error);
        assert_eq!(
            docs.len(),
            2,
            "the stream stops at the first malformed record"
        );
    }

    #[test]
    fn files_are_read_in_order_and_missing_files_are_error_items() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, "first").unwrap();
        std::fs::write(&b, "second").unwrap();
        let missing = dir.path().join("missing.txt");

        let input = Input {
            locations: vec![
                Location::File(a.clone()),
                Location::File(missing.clone()),
                Location::File(b.clone()),
            ],
            framing: Framing::Text,
        };
        let docs = collect(&input);
        assert_eq!(docs.len(), 3);
        let first = docs[0].as_ref().unwrap();
        assert_eq!(first.id, DocumentId::Whole(Source::File(a)));
        assert_eq!(first.text, "first");
        let err = docs[1].as_ref().unwrap_err();
        assert_eq!(err.id, DocumentId::Whole(Source::File(missing)));
        assert!(
            err.error.to_string().contains("missing.txt"),
            "{}",
            err.error
        );
        assert_eq!(docs[2].as_ref().unwrap().text, "second");
    }

    #[test]
    fn document_id_display_is_grep_style() {
        let file = Source::File(PathBuf::from("docs/essay.txt"));
        assert_eq!(
            DocumentId::Whole(file.clone()).to_string(),
            "docs/essay.txt"
        );
        assert_eq!(DocumentId::Whole(Source::Stdin).to_string(), "stdin");
        assert_eq!(
            DocumentId::Record {
                source: file.clone(),
                index: 3,
                id: None
            }
            .to_string(),
            "docs/essay.txt:3"
        );
        assert_eq!(
            DocumentId::Record {
                source: file,
                index: 3,
                id: Some("k".into())
            }
            .to_string(),
            "docs/essay.txt:k"
        );
        assert_eq!(record(3, None).to_string(), "3");
        assert_eq!(
            DocumentId::Record {
                source: Source::Stdin,
                index: 1,
                id: Some("k".into())
            }
            .to_string(),
            "k"
        );
    }

    #[test]
    fn from_cli_maps_dash_to_stdin_and_reports_multiplicity() {
        let cli = Cli::try_parse_from(["is-it-slop", "-f", "-", "-f", "a.txt"]).unwrap();
        let input = Input::from_cli(&cli).unwrap();
        assert_eq!(
            input.locations,
            vec![Location::Stdin, Location::File(PathBuf::from("a.txt"))]
        );
        assert!(input.is_multi());

        let cli = Cli::try_parse_from(["is-it-slop", "-f", "a.txt"]).unwrap();
        assert!(!Input::from_cli(&cli).unwrap().is_multi());

        let cli = Cli::try_parse_from(["is-it-slop", "-i", "lines", "hello"]).unwrap();
        let input = Input::from_cli(&cli).unwrap();
        assert_eq!(input.locations, vec![Location::Argument("hello".into())]);
        assert!(input.is_multi());
    }

    #[test]
    fn text_and_file_conflict() {
        assert!(Cli::try_parse_from(["is-it-slop", "-f", "a.txt", "hello"]).is_err());
    }
}
