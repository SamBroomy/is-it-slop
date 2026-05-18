//! Command line interface for is-it-slop

use std::process::ExitCode;

use clap::Parser;
use is_it_slop::cli::{self, Cli, RunOutcome};

fn main() -> ExitCode {
    let cli = Cli::parse();
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
