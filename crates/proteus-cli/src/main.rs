//! `proteus` — the command-line client.
//!
//! This file is the entry point and nothing else: argument parsing lives in [`cli`], and each
//! subcommand owns a module under [`cmd`].

mod cli;
mod cmd;
mod esm_cmd;
mod job_ref;
mod report;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Rust ignores SIGPIPE, so a write to a closed pipe (`proteus analyze … | head -1`) panics
/// with "failed printing to stdout". Restore the default, as other Unix command-line tools
/// have it: the process ends quietly when its reader goes away.
#[cfg(unix)]
fn default_sigpipe() {
    extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    // SAFETY: installing the default disposition for SIGPIPE before any thread is spawned.
    unsafe {
        signal(SIGPIPE, SIG_DFL);
    }
}

#[cfg(not(unix))]
fn default_sigpipe() {}

fn main() -> Result<()> {
    default_sigpipe();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<()> {
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
