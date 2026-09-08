//! End-to-end tests of the `is-it-slop` binary: input handling, output formats, exit status.

use std::{
    fmt::Write as _,
    io::Write,
    process::{Command, Output, Stdio},
};

fn run(args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_is-it-slop"))
        .args(args)
        .args(["--color", "never"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary spawns");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().expect("binary exits")
}

fn stdout_lines(output: &Output) -> Vec<String> {
    assert!(
        output.status.success(),
        "exit {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn write_files(dir: &std::path::Path, files: &[(&str, &str)]) -> Vec<String> {
    files
        .iter()
        .map(|(name, text)| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            path.to_str().unwrap().to_owned()
        })
        .collect()
}

#[test]
fn stdin_lines_are_labelled_by_line_number() {
    let output = run(
        &["--input", "lines", "-o", "label"],
        "First document here.\n\nThird line is the second document.\n",
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with("1: "), "{lines:?}");
    assert!(lines[1].starts_with("3: "), "{lines:?}");
}

#[test]
fn jsonl_ids_pass_through_to_jsonl_output() {
    let output = run(
        &["-i", "jsonl", "-o", "jsonl"],
        r#"{"id": "alpha", "text": "Some text to classify."}
{"text": "A record without an id."}
{"id": 7, "text": "A record with a numeric id."}
"#,
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 3, "{lines:?}");
    let rows: Vec<serde_json::Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["id"], "alpha");
    assert_eq!(rows[0]["status"], "ok");
    assert_eq!(rows[1]["id"], "2");
    assert_eq!(rows[2]["id"], 7, "numeric ids keep their type");
}

#[test]
fn json_is_always_an_array_with_ids() {
    let output = run(&["-o", "json", "Just one document."], "");
    let rows: Vec<serde_json::Value> =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert!(output.status.success());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["id"], "argument");
    assert_eq!(rows[0]["status"], "ok");
}

#[test]
fn multiple_files_are_prefixed_by_path() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_files(
        dir.path(),
        &[
            ("a.txt", "The first file's text."),
            ("b.txt", "The second file's text."),
        ],
    );

    let output = run(&["-f", &paths[0], "-f", &paths[1], "-o", "score"], "");
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].starts_with(&format!("{}: ", paths[0])),
        "{lines:?}"
    );
    assert!(
        lines[1].starts_with(&format!("{}: ", paths[1])),
        "{lines:?}"
    );
}

#[test]
fn single_file_has_no_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_files(dir.path(), &[("a.txt", "Just one file's text.")]);

    let output = run(&["-f", &paths[0], "-o", "label"], "");
    let lines = stdout_lines(&output);
    assert!(
        matches!(lines.as_slice(), [label] if label == "Human" || label == "AI"),
        "{lines:?}"
    );
}

#[test]
fn pretty_multi_document_run_ends_with_a_summary() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_files(
        dir.path(),
        &[("a.txt", "The first file's text."), ("b.txt", "  ")],
    );

    let output = run(&["-f", &paths[0], "-f", &paths[1]], "");
    assert_eq!(
        output.status.code(),
        Some(1),
        "a failed document fails the run"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("--- {} ---", paths[0])),
        "{stdout}"
    );
    assert!(
        stdout
            .trim_end()
            .ends_with("2 documents: 1 AI, 0 Human, 1 failed")
            || stdout
                .trim_end()
                .ends_with("2 documents: 0 AI, 1 Human, 1 failed"),
        "{stdout}"
    );
    assert!(stderr(&output).contains("b.txt"), "{}", stderr(&output));
}

#[test]
fn blank_document_in_a_multi_file_run_is_an_error_row_in_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_files(
        dir.path(),
        &[("empty.txt", "  \n"), ("full.txt", "Some real text.")],
    );

    let output = run(&["-f", &paths[0], "-f", &paths[1], "-o", "jsonl"], "");
    assert_eq!(output.status.code(), Some(1));
    let rows: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!(rows[0]["status"], "error");
    assert_eq!(rows[0]["id"], paths[0]);
    assert!(rows[0]["error"].as_str().unwrap().contains("non-empty"));
    assert_eq!(rows[1]["status"], "ok");
}

#[test]
fn missing_file_fails_naming_the_path() {
    let output = run(&["-f", "/nonexistent/for-sure.txt"], "");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("for-sure.txt"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn exit_status_reports_whether_any_document_is_ai() {
    // Uniform generated-sounding prose the model reliably calls AI.
    let ai = "Strong scientific evidence supports the assertion that the study of history \
              can provide a range of intellectual benefits. In particular, developing \
              critical thinking, contextual awareness, and a deeper understanding of \
              cause-and-effect relationships in human affairs.";
    let output = run(&["-e", "-o", "label", ai], "");
    let label = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let expected = i32::from(label != "AI");
    assert_eq!(output.status.code(), Some(expected), "label {label}");

    let output = run(&["-e", "-i", "lines", "-o", "label"], "\n\n");
    assert_eq!(output.status.code(), Some(1), "no documents means no AI");

    let output = run(&["-e", "-f", "/nonexistent/for-sure.txt"], "");
    assert_eq!(
        output.status.code(),
        Some(2),
        "errors are 2 under --exit-status"
    );
}

#[test]
fn verbose_and_quiet_only_apply_to_pretty() {
    let output = run(&["-o", "json", "-v", "some text"], "");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("--output pretty"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn results_keep_input_order_across_batches_with_errors_in_place() {
    // 70 records span three batches of 32; record 35 is blank and must fail in place.
    let mut stdin = String::new();
    for i in 1..=70 {
        let text = if i == 35 {
            String::new()
        } else {
            format!("Record number {i} has some ordinary text in it.")
        };
        writeln!(stdin, "{{\"id\": {i}, \"text\": \"{text}\"}}").unwrap();
    }
    let output = run(&["-i", "jsonl", "-o", "jsonl"], &stdin);
    assert_eq!(
        output.status.code(),
        Some(1),
        "one failed record fails the run"
    );
    let rows: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 70);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["id"], i as u64 + 1, "row {i} out of order: {row}");
        let expected = if i + 1 == 35 { "error" } else { "ok" };
        assert_eq!(row["status"], expected, "row {i}: {row}");
    }
}

#[test]
fn a_full_batch_is_written_before_stdin_closes() {
    use std::{
        io::{BufRead, BufReader},
        sync::mpsc,
        time::Duration,
    };

    let mut child = Command::new(env!("CARGO_BIN_EXE_is-it-slop"))
        .args(["-i", "lines", "-o", "score", "--color", "never"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    // Exactly one batch, and the pipe stays open so a collect-everything loop would never print.
    for i in 1..=32 {
        writeln!(stdin, "Line {i} of text that is still waiting for more.").unwrap();
    }
    stdin.flush().unwrap();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut first = String::new();
        BufReader::new(stdout).read_line(&mut first).unwrap();
        tx.send(first).unwrap();
    });
    let first = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("the first batch streams out while stdin is still open");
    assert!(first.starts_with("1: "), "{first:?}");

    drop(stdin);
    assert!(child.wait().unwrap().success());
}

#[test]
fn empty_lines_input_yields_an_empty_json_array() {
    let output = run(&["-i", "lines", "-o", "json"], "\n\n");
    assert_eq!(stdout_lines(&output), vec!["[]"]);
}
