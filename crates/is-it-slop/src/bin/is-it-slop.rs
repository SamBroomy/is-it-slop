use anyhow::Result;
use clap::Parser;
use is_it_slop::cli::{self, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli::run(&cli)
}
