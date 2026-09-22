//! `proteus submit` — Submit one sequence and wait for the prediction.

use super::prelude::*;

/// Arguments of `proteus submit`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Path to FASTA sequence file
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Raw FASTA string
    #[arg(long)]
    fasta: Option<String>,

    /// Computational tier
    #[arg(short, long, value_enum, default_value_t = CliTier::Fast)]
    tier: CliTier,

    /// Compute runner mode
    #[arg(long, value_enum, default_value_t = RunnerMode::Auto)]
    runner: RunnerMode,

    /// Run synchronously and wait for completion (`--wait=false` to enqueue and return)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    wait: bool,
}

pub async fn run(args: Args, db_path: &std::path::Path, artifacts_dir: &std::path::Path) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let artifacts_dir = artifacts_dir.to_path_buf();
    let Args {
        file,
        fasta,
        tier,
        runner,
        wait,
    } = args;
    let fasta_content = if let Some(path) = file {
        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read FASTA file at {:?}", path))?
    } else if let Some(content) = fasta {
        content
    } else {
        anyhow::bail!("Either --file or --fasta must be specified");
    };

    let sequence =
        validate_and_parse_fasta(&fasta_content).context("FASTA validation failed")?;

    println!(
        "Sequence validated: '{}' ({} residues)",
        sequence.header, sequence.length
    );

    let pool = create_sqlite_pool(&db_path).await?;
    let repo = ProteusRepository::new(pool);
    repo.insert_sequence(&sequence).await?;

    let job_id = Uuid::new_v4();
    let pipeline_tier: PipelineTier = tier.into();
    let job = proteus_core::models::PipelineJob {
        id: job_id,
        sequence_id: sequence.id,
        tier: pipeline_tier,
        status: proteus_core::models::JobStatus::Queued,
        priority: 1,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error_log: None,
    };
    repo.insert_job(&job).await?;

    println!("Job created: {}", job_id);

    if wait {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.green} {msg}")?,
        );
        pb.set_message("Executing bio-compute pipeline...");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));

        let compute_runner = resolve_runner(runner)?;
        let scheduler = PipelineScheduler::new(repo.clone(), compute_runner, artifacts_dir);
        scheduler.process_job(job_id).await?;

        pb.finish_with_message("Pipeline completed successfully!");

        print_job_inspection(&repo, job_id).await?;
    } else {
        println!(
            "Job enqueued in background. Use 'proteus status {}' to inspect.",
            job_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;
    use crate::cli::{Cli, Commands};
    use clap::Parser;

    /// `--wait` defaults on but must be switchable off, so a script can enqueue and return.
    #[test]
    fn submit_wait_can_be_switched_off() {
        let wait_of = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            Commands::Submit(Args { wait, .. }) => wait,
            _ => unreachable!(),
        };
        assert!(wait_of(&["proteus", "submit", "--fasta", ">x\nAC"]));
        assert!(wait_of(&["proteus", "submit", "--fasta", ">x\nAC", "--wait"]));
        assert!(!wait_of(&[
            "proteus",
            "submit",
            "--fasta",
            ">x\nAC",
            "--wait=false"
        ]));
    }
}
