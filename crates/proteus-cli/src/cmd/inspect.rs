//! `proteus inspect` — Print the full biophysical report for a finished job.

use super::prelude::*;

/// Arguments of `proteus inspect`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Job UUID, or a unique prefix of one
    job_id: String,
}

pub async fn run(args: Args, db_path: &std::path::Path) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let Args { job_id } = args;
    let pool = create_sqlite_pool(&db_path).await?;
    let repo = ProteusRepository::new(pool);
    let job_id = job_ref::resolve(&repo, &job_id).await?;
    print_job_inspection(&repo, job_id).await?;
    Ok(())
}
