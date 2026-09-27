//! `proteus rename` — Give a job the name the lists show.

use super::prelude::*;

/// Arguments of `proteus rename`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Job UUID, or a unique prefix of one
    job_id: String,
    /// The new name
    name: String,
}

pub async fn run(args: Args, db_path: &std::path::Path) -> Result<()> {
    let name = args.name.trim();
    if name.is_empty() {
        bail!("a job needs a name; give some text");
    }
    let repo = ProteusRepository::new(create_sqlite_pool(db_path).await?);
    let id = job_ref::resolve(&repo, &args.job_id).await?;
    if !repo.rename_job(id, name).await? {
        bail!("no job {}", job_ref::short(id));
    }
    println!("✓ renamed {} to {name}", job_ref::short(id));
    Ok(())
}
