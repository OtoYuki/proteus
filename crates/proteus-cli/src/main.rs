//! `proteus` — the command-line client.
//!
//! This file is the entry point and nothing else: argument parsing lives in [`cli`], and each
//! subcommand owns a module under [`cmd`].

mod cli;
mod cmd;
mod esm_cmd;
mod report;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // RUST_LOG wins when set (e.g. RUST_LOG=off for clean demo output); default is info.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = cli::Cli::parse();
    let data_dir = cli::get_default_data_dir();
    let db_path = data_dir.join("proteus.db");
    let artifacts_dir = data_dir.join("artifacts");

    cmd::dispatch(cli.command, &db_path, &artifacts_dir, &data_dir).await
}
