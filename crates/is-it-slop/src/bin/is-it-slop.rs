//! Command line interface for is-it-slop

use std::process::ExitCode;

use clap::Parser;
use is_it_slop::cli::{self, Cli, Commands, RunOutcome, SelfCommand};

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        return match command {
            Commands::Self_(namespace) => match namespace.command {
                SelfCommand::Update => handle_self_update(),
            },
        };
    }

    // Default behavior: text classification
    match cli::run(&cli) {
        Ok(RunOutcome::Normal | RunOutcome::ClassifyAi) => ExitCode::SUCCESS,
        Ok(RunOutcome::ClassifyHuman) => ExitCode::from(1),
        Err(e) => {
            eprintln!("Error: {e:#}");
            if cli.classify {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

/// Handle the self update command based on feature flags
#[cfg(feature = "self-update")]
fn handle_self_update() -> ExitCode {
    cli::self_update::run_update()
}

/// Handle the self update command for package manager installations
#[cfg(not(feature = "self-update"))]
fn handle_self_update() -> ExitCode {
    use cli::install_source::InstallSource;

    const BASE_MESSAGE: &str =
        "is-it-slop was installed through a package manager and cannot update itself.";

    if let Some(source) = InstallSource::detect() {
        eprintln!("{BASE_MESSAGE}\n");
        eprintln!("You installed is-it-slop using {}.", source.description());
        eprintln!("To update, run: {}", source.update_instructions());
    } else {
        eprintln!("{BASE_MESSAGE}");
        eprintln!("Please use your package manager to update is-it-slop.");
    }

    ExitCode::FAILURE
}
