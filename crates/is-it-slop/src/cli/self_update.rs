//! Self-update functionality for standalone binaries.
//!
//! This module is only compiled when the `self-update` feature is enabled,
//! which should only be used for standalone binary distributions (CI-built releases).
//! Package manager installations (pip, homebrew, cargo install) should not enable this feature.

use std::process::{Command, ExitCode};

const INSTALL_SCRIPT_URL: &str =
    "https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.sh";
const GITHUB_API_LATEST: &str = "https://api.github.com/repos/SamBroomy/is-it-slop/releases/latest";

/// Run the self-update command.
///
/// Fetches the latest release tag from GitHub, compares to the current version,
/// and runs the install script to upgrade if a newer version is available.
#[must_use]
pub fn run_update() -> ExitCode {
    let current_version = env!("CARGO_PKG_VERSION");
    println!("Checking for updates...");
    println!("Current version: v{current_version}");

    let latest_tag = match fetch_latest_tag() {
        Ok(tag) => tag,
        Err(e) => {
            eprintln!("Failed to check for updates: {e}");
            eprintln!();
            eprintln!("You can manually upgrade by running:");
            eprintln!("  curl -fsSL {INSTALL_SCRIPT_URL} | sh");
            return ExitCode::FAILURE;
        }
    };

    let latest_version = latest_tag.strip_prefix('v').unwrap_or(&latest_tag);

    if latest_version == current_version {
        println!("Already up to date (v{current_version})");
        return ExitCode::SUCCESS;
    }

    println!();
    println!("Upgrading from v{current_version} to {latest_tag}...");
    println!();

    let status = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "curl -fsSL {INSTALL_SCRIPT_URL} | sh -s -- {latest_tag}"
        ))
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!();
            println!("Upgraded to {latest_tag}");
            ExitCode::SUCCESS
        }
        Ok(exit_status) => {
            eprintln!("Upgrade failed with exit code: {exit_status}");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Error running upgrade: {e}");
            eprintln!();
            eprintln!("You can manually upgrade by running:");
            eprintln!("  curl -fsSL {INSTALL_SCRIPT_URL} | sh -s -- {latest_tag}");
            ExitCode::FAILURE
        }
    }
}

fn fetch_latest_tag() -> Result<String, String> {
    let output = Command::new("curl")
        .args(["-fsSL", GITHUB_API_LATEST])
        .output()
        .map_err(|e| format!("Failed to run curl: {e}"))?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        if code == 403 {
            return Err("GitHub API rate limit exceeded (try again later)".to_string());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("GitHub API request failed ({code}): {stderr}"));
    }

    let response = String::from_utf8_lossy(&output.stdout);
    let tag = response
        .split("\"tag_name\":\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .ok_or_else(|| "Failed to parse GitHub API response".to_string())?;

    Ok(tag.to_string())
}
