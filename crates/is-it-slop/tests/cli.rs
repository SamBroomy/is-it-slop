//! End-to-end tests of the `is-it-slop` binary's input handling.

use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

fn run(args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_is-it-slop"))
        .args(args)
        .env("NO_COLOR", "1")
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

#[test]
fn stdin_lines_are_labelled_by_line_number() {
    let output = run(
        &["--input", "lines", "--label"],
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
        &["-i", "jsonl", "--jsonl"],
        r#"{"id": "alpha", "text": "Some text to classify."}
{"text": "A record without an id."}
"#,
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 2, "{lines:?}");
    let first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    let second: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(first["id"], "alpha");
    assert_eq!(first["status"], "ok");
    assert_eq!(second["id"], "2");
}

#[test]
fn multiple_files_are_prefixed_by_path() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    std::fs::write(&a, "The first file's text.").unwrap();
    std::fs::write(&b, "The second file's text.").unwrap();

    let output = run(
        &[
            "-f",
            a.to_str().unwrap(),
            "-f",
            b.to_str().unwrap(),
            "--score",
        ],
        "",
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].starts_with(&format!("{}: ", a.display())),
        "{lines:?}"
    );
    assert!(
        lines[1].starts_with(&format!("{}: ", b.display())),
        "{lines:?}"
    );
}

#[test]
fn single_file_has_no_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt");
    std::fs::write(&a, "Just one file's text.").unwrap();

    let output = run(&["-f", a.to_str().unwrap(), "--label"], "");
    let lines = stdout_lines(&output);
    assert!(
        matches!(lines.as_slice(), [label] if label == "Human" || label == "AI"),
        "{lines:?}"
    );
}

#[test]
fn blank_document_in_a_multi_file_run_fails_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.txt");
    let full = dir.path().join("full.txt");
    std::fs::write(&empty, "  \n").unwrap();
    std::fs::write(&full, "Some real text.").unwrap();

    let output = run(
        &[
            "-f",
            empty.to_str().unwrap(),
            "-f",
            full.to_str().unwrap(),
            "--score",
        ],
        "",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("empty.txt"), "{stderr}");
}

#[test]
fn missing_file_fails_naming_the_path() {
    let output = run(&["-f", "/nonexistent/for-sure.txt"], "");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("for-sure.txt"), "{stderr}");
}
