//! `proteus esm` — ESM-2 mutation scoring and deep mutational scans.

use super::prelude::*;

/// Arguments of `proteus esm`.
#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: esm_cmd::EsmCommand,
    #[command(flatten)]
    esm: esm_cmd::EsmOptions,
}

pub async fn run(args: Args) -> Result<()> {
    let Args {
        command,
        esm,
    } = args;
    esm_cmd::run(command, esm).await?;
    Ok(())
}
