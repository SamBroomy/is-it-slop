//! Command line interface for is-it-slop

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::process::ExitCode;

use clap::Parser;
use is_it_slop::cli::{self, Cli};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli::run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
