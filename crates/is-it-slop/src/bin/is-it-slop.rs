//! Command line interface for is-it-slop

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::Parser;
use is_it_slop::cli::{self, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli::run(&cli)
}
