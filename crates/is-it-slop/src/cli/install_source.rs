#![cfg(not(feature = "self-update"))]

//! Install source detection for showing appropriate upgrade instructions.
//!
//! This module is only compiled when the `self-update` feature is NOT enabled.
//! It detects how is-it-slop was installed and shows the appropriate upgrade command.

/// Detected installation source.
#[derive(Debug, Clone, Copy)]
pub enum InstallSource {
    /// Installed via pip
    Pip,
    /// Installed via uv
    Uv,
    /// Installed via Homebrew
    Homebrew,
    /// Installed via cargo
    Cargo,
}

impl InstallSource {
    /// Detect how is-it-slop was installed by examining the executable path.
    #[must_use]
    pub fn detect() -> Option<Self> {
        let exe_path = std::env::current_exe().ok()?;
        let canonical = exe_path.canonicalize().unwrap_or(exe_path);

        // Check for Homebrew Cellar
        if canonical.components().any(|c| c.as_os_str() == "Cellar") {
            return Some(Self::Homebrew);
        }

        // Check for pip/uv site-packages
        if canonical.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s.contains("site-packages") || s.contains("lib/python")
        }) {
            // Prefer uv if uv is in PATH
            if which::which("uv").is_ok() {
                return Some(Self::Uv);
            }
            return Some(Self::Pip);
        }

        // Check for cargo installation
        if let Some(home) = dirs::home_dir()
            && canonical.starts_with(home.join(".cargo"))
        {
            return Some(Self::Cargo);
        }

        None
    }

    /// Get a human-readable description of the install source.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Pip => "pip",
            Self::Uv => "uv",
            Self::Homebrew => "Homebrew",
            Self::Cargo => "Cargo",
        }
    }

    /// Get the upgrade command for this install source.
    #[must_use]
    pub const fn update_instructions(self) -> &'static str {
        match self {
            Self::Pip => "pip install --upgrade is-it-slop",
            Self::Uv => "uv tool upgrade is-it-slop",
            Self::Homebrew => "brew upgrade is-it-slop",
            Self::Cargo => "cargo install is-it-slop --locked --force",
        }
    }
}
