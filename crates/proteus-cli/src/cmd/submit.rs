//! `proteus submit` — Submit one sequence, or a complex, and wait for the prediction.

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

    /// Alignment for every protein chain (sota tier): `none` (single sequence, the default),
    /// `server` (the public ColabFold MMseqs2 server — the sequences are SENT to it), or an
    /// .a3m file. Per-chain choices go in Boltz-style headers: `>A|protein|server`
    #[arg(long, value_name = "none|server|FILE")]
    msa: Option<String>,

    /// Structures to sample (sota tier), ranked by Boltz's confidence; the best is the job's
    /// model and the rest are kept beside it
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=25))]
    samples: Option<u16>,
}

pub async fn run(
    args: Args,
    db_path: &std::path::Path,
    artifacts_dir: &std::path::Path,
) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let artifacts_dir = artifacts_dir.to_path_buf();
    let Args {
        file,
        fasta,
        tier,
        runner,
        wait,
        msa,
        samples,
    } = args;
    let file_name = file
        .as_ref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .map(str::to_string);
    let fasta_content = if let Some(path) = file {
        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read FASTA file at {:?}", path))?
    } else if let Some(content) = fasta {
        content
    } else {
        anyhow::bail!("Either --file or --fasta must be specified");
    };

    let mut sequence = parse_input(&fasta_content, msa.as_deref(), samples)?;
    // Boltz-style headers name chains (`>A|protein`), not the job: take the file's name.
    if let (Some(stem), true) = (
        file_name.as_deref(),
        fasta_content
            .lines()
            .find(|l| l.trim_start().starts_with('>'))
            .is_some_and(|l| l.contains('|')),
    ) {
        let chains = proteus_core::complex::ComplexSpec::from_stored(&sequence.fasta)
            .map(|c| c.chains.len())
            .unwrap_or(1);
        sequence.header = if chains > 1 {
            format!("{stem} ({chains} chains)")
        } else {
            stem.to_string()
        };
    }
    if proteus_core::complex::ComplexSpec::is_spec(&sequence.fasta) {
        let spec = proteus_core::complex::ComplexSpec::from_stored(&sequence.fasta)?;
        if tier != CliTier::Sota {
            anyhow::bail!(
                "complexes, ligands, --msa and --samples need --tier sota (Boltz); the other \
                 tiers fold one chain"
            );
        }
        println!(
            "Input validated: '{}' — {} chain(s), {} protein residues{}{}",
            sequence.header,
            spec.chains.len(),
            spec.residues(),
            if spec.samples > 1 {
                format!(", {} samples", spec.samples)
            } else {
                String::new()
            },
            if spec.uses_msa_server() {
                ", MSA from the public ColabFold server (sequences leave this machine)"
            } else {
                ""
            }
        );
    } else {
        println!(
            "Sequence validated: '{}' ({} residues)",
            sequence.header, sequence.length
        );
    }

    let pool = create_sqlite_pool(&db_path).await?;
    let repo = ProteusRepository::new(pool);
    repo.insert_sequence(&sequence).await?;

    let job_id = Uuid::new_v4();
    let pipeline_tier: PipelineTier = tier.into();
    let job = proteus_core::models::PipelineJob {
        id: job_id,
        sequence_id: sequence.id,
        tier: pipeline_tier,
        // Pending hands the job to a running `proteus serve`; Queued keeps it for this
        // process, which runs it right away.
        status: if wait {
            proteus_core::models::JobStatus::Queued
        } else {
            proteus_core::models::JobStatus::Pending
        },
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
                .template(&format!(
                    "{{spinner:.{}}} {{msg}}",
                    tint(proteus_render::brand::Role::Accent)
                ))?,
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
            "Job queued. A `proteus serve` running on the same data directory picks it up \
             within a few seconds; `proteus status {}` shows its progress.",
            job_id
        );
    }
    Ok(())
}

/// A single-record FASTA stays the monomer it always was; several records, Boltz-style headers
/// (`>A|protein|…`, `>L|ccd`), `--msa` or `--samples` make it a
/// [`proteus_core::complex::ComplexSpec`], stored in its canonical form.
fn parse_input(
    text: &str,
    msa: Option<&str>,
    samples: Option<u16>,
) -> Result<proteus_core::models::Sequence> {
    use proteus_core::complex::{ComplexSpec, MsaSource};
    let normalized = if text.contains('\n') {
        text.to_string()
    } else {
        text.replace("\\n", "\n")
    };
    let records = normalized
        .lines()
        .filter(|l| l.trim_start().starts_with('>'))
        .count();
    let boltz_header = normalized.lines().any(|l| {
        let l = l.trim_start();
        l.starts_with('>')
            && l.split('|')
                .nth(1)
                .is_some_and(|k| matches!(k.trim(), "protein" | "ccd" | "smiles"))
    });
    let options = msa.is_some_and(|m| m != "none") || samples.is_some_and(|n| n > 1);
    if records <= 1 && !boltz_header && !options {
        return validate_and_parse_fasta(text).context("FASTA validation failed");
    }
    let mut spec = ComplexSpec::parse(&normalized).context("FASTA validation failed")?;
    match msa {
        None => {}
        Some("none") => spec.set_msa(MsaSource::Empty),
        Some("server") => spec.set_msa(MsaSource::Server),
        Some(path) => {
            let abs = std::fs::canonicalize(path)
                .with_context(|| format!("--msa: cannot open the alignment '{path}'"))?;
            spec.set_msa(MsaSource::File(abs.to_string_lossy().into_owned()));
        }
    }
    if let Some(n) = samples {
        spec.samples = n as usize;
    }
    let header = normalized
        .lines()
        .find_map(|l| l.trim().strip_prefix('>'))
        .map(|h| h.split('|').next().unwrap_or(h).trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "complex".into());
    let header = if spec.is_monomer() {
        header
    } else {
        format!("{header} +{} chain(s)", spec.chains.len() - 1)
    };
    Ok(proteus_core::models::Sequence {
        id: Uuid::new_v4(),
        header,
        length: spec.residues(),
        fasta: spec.to_stored(),
        created_at: chrono::Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn a_plain_fasta_stays_a_monomer_and_records_make_a_complex() {
        let m = super::parse_input(">ubq\nMQIFVK\n", None, None).unwrap();
        assert_eq!(m.fasta, "MQIFVK");
        let c = super::parse_input(">A|protein|empty\nMQIF\n>L|ccd\nATP\n", None, Some(3)).unwrap();
        assert!(c.fasta.starts_with("#proteus samples=3\n"));
        assert_eq!(c.length, 4);
        let s = super::parse_input(">ubq\nMQIFVK\n", Some("server"), None).unwrap();
        assert!(s.fasta.contains(">A|protein|server"));
        assert!(super::parse_input(">ubq\nMQIFVK\n", Some("/no/such.a3m"), None).is_err());
    }
    use crate::cli::{Cli, Commands};
    use clap::Parser;

    /// `--wait` defaults on but must be switchable off, so a script can enqueue and return.
    #[test]
    fn submit_wait_can_be_switched_off() {
        let wait_of = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command.unwrap() {
            Commands::Submit(Args { wait, .. }) => wait,
            _ => unreachable!(),
        };
        assert!(wait_of(&["proteus", "submit", "--fasta", ">x\nAC"]));
        assert!(wait_of(&[
            "proteus", "submit", "--fasta", ">x\nAC", "--wait"
        ]));
        assert!(!wait_of(&[
            "proteus",
            "submit",
            "--fasta",
            ">x\nAC",
            "--wait=false"
        ]));
    }
}
