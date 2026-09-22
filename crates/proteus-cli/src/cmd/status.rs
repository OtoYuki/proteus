//! `proteus status` — Print a job row.

use super::prelude::*;

/// Arguments of `proteus status`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Job UUID
    job_id: Uuid,
}

pub async fn run(args: Args, db_path: &std::path::Path) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let Args { job_id } = args;
    let pool = create_sqlite_pool(&db_path).await?;
    let repo = ProteusRepository::new(pool);

    let job = repo
        .get_job(job_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Job {} not found", job_id))?;

    let mut table = Table::new();
    table.load_style(UTF8_FULL);
    table.set_header(vec!["Field", "Value"]);

    table.add_row(vec!["Job ID", &job.id.to_string()]);
    table.add_row(vec!["Sequence ID", &job.sequence_id.to_string()]);
    table.add_row(vec!["Tier", &format!("{:?}", job.tier)]);
    table.add_row(vec![
        "Status",
        &match job.status {
            proteus_core::models::JobStatus::Completed => "Completed (OK)".to_string(),
            proteus_core::models::JobStatus::Failed => "Failed".to_string(),
            s => format!("{:?}", s),
        },
    ]);
    table.add_row(vec!["Created At", &job.created_at.to_rfc3339()]);

    if let Some(err) = job.error_log {
        table.add_row(vec!["Error", &err]);
    }

    println!("{table}");
    Ok(())
}
