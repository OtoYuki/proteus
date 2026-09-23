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

/// A reader that goes away (`proteus analyze … | head -1`) makes the next `println!` panic with
/// "failed printing to stdout: Broken pipe". End quietly instead, with the status a shell gives
/// a process killed by SIGPIPE (128 + 13).
///
/// Deliberately not done by restoring the default SIGPIPE disposition: that applies to every
/// pipe and socket in the process, and `proteus serve` writes to executor stdin pipes and
/// container-engine sockets whose far end can close — which must be an error for one task,
/// not the end of the daemon.
fn quiet_broken_stdout() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if msg.starts_with("failed printing to stdout") && msg.contains("Broken pipe") {
            std::process::exit(141);
        }
        previous(info);
    }));
}

fn main() -> Result<()> {
    quiet_broken_stdout();
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
