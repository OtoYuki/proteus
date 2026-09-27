//! `proteus delete` — Remove a job, its records and its files.

use super::prelude::*;

/// Arguments of `proteus delete`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Job UUID, or a unique prefix of one
    job_id: String,
    /// Keep the job's files (its model, PAE and logs) and remove only the records
    #[arg(long)]
    keep_files: bool,
}

pub async fn run(
    args: Args,
    db_path: &std::path::Path,
    artifacts_dir: &std::path::Path,
) -> Result<()> {
    let repo = ProteusRepository::new(create_sqlite_pool(db_path).await?);
    let id = job_ref::resolve(&repo, &args.job_id).await?;
    if let Some(job) = repo.get_job(id).await? {
        if job.status == proteus_core::models::JobStatus::Running {
            bail!(
                "job {} is still running; delete it when it has finished",
                job_ref::short(id)
            );
        }
    }
    let name = repo
        .list_jobs(i64::MAX)
        .await?
        .into_iter()
        .find(|j| j.job.id == id)
        .map(|j| j.header)
        .unwrap_or_default();
    if !repo.delete_job(id).await? {
        bail!("no job {}", job_ref::short(id));
    }
    // The job's own folder under the artifacts directory, and nothing outside it.
    let dir = artifacts_dir.join(id.to_string());
    let files = if !args.keep_files && dir.is_dir() && dir.starts_with(artifacts_dir) {
        std::fs::remove_dir_all(&dir).with_context(|| {
            format!(
                "the records are gone but {} could not be removed",
                dir.display()
            )
        })?;
        " and its files"
    } else {
        ""
    };
    println!("✓ deleted {} {name}{files}", job_ref::short(id));
    Ok(())
}
