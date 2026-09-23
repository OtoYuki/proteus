//! One module per subcommand. Each owns its clap `Args` struct and a `run`, so the
//! parser surface and the work sit together and `main` stays a dispatch table.

pub mod analyze;
pub mod esm;
pub mod inspect;
pub mod mutate;
pub mod screen;
pub mod serve;
pub mod status;
pub mod submit;
pub mod view;

/// What every subcommand module imports.
pub(crate) mod prelude {
    pub use crate::cli::*;
    pub(crate) use crate::esm_cmd;
    pub(crate) use crate::job_ref;
    pub use crate::report::print_job_inspection;
    pub use anyhow::{bail, Context, Result};
    pub use comfy_table::presets::UTF8_FULL;
    pub use comfy_table::{Cell, Table};
    pub use indicatif::{ProgressBar, ProgressStyle};
    pub use proteus_core::confidence::ConfidenceSource;
    pub use proteus_core::models::PipelineTier;
    pub use proteus_core::sequence::validate_and_parse_fasta;
    pub use proteus_storage::create_sqlite_pool;
    pub use proteus_storage::repository::ProteusRepository;

    pub use proteus_engine::{
        ContainerExecutor, HostExecutor, PipelineScheduler, TesExecutionConfig, TesExecutor,
    };
    pub use proteus_server::{run_server_with_options, ServerOptions};
    pub use std::net::SocketAddr;
    pub use std::path::{Path, PathBuf};
    pub use std::sync::Arc;
    pub use uuid::Uuid;
}

use crate::cli::Commands;
use anyhow::Result;
use std::path::Path;

/// Run the parsed subcommand.
pub async fn dispatch(
    command: Commands,
    db_path: &Path,
    artifacts_dir: &Path,
    data_dir: &Path,
) -> Result<()> {
    let _ = (db_path, artifacts_dir, data_dir);
    match command {
        Commands::Submit(args) => submit::run(args, db_path, artifacts_dir).await,
        Commands::Status(args) => status::run(args, db_path).await,
        Commands::Inspect(args) => inspect::run(args, db_path).await,
        Commands::Analyze(args) => analyze::run(args).await,
        Commands::View(args) => view::run(args, db_path).await,
        Commands::Mutate(args) => mutate::run(args).await,
        Commands::Screen(args) => screen::run(args, db_path, artifacts_dir).await,
        Commands::Esm(args) => esm::run(args).await,
        Commands::Serve(args) => serve::run(args, db_path, artifacts_dir).await,
    }
}
