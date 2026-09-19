use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};
use indicatif::{ProgressBar, ProgressStyle};
use proteus_core::metrics::analyze_pdb_file;
use proteus_core::models::PipelineTier;
use proteus_core::sequence::validate_and_parse_fasta;
use proteus_engine::oci::OciRunner;
use proteus_engine::simulated::SimulatedRunner;
use proteus_engine::{ComputeRunner, PipelineScheduler};
use proteus_server::run_server;
use proteus_storage::create_sqlite_pool;
use proteus_storage::repository::ProteusRepository;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "proteus")]
#[command(about = "High-throughput Bio-Compute Pipeline & Orchestration CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Submit a protein sequence to the bio-compute pipeline
    Submit {
        /// Path to FASTA sequence file
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Raw FASTA string
        #[arg(long)]
        fasta: Option<String>,

        /// Computational tier
        #[arg(short, long, value_enum, default_value_t = CliTier::Fast)]
        tier: CliTier,

        /// Run synchronously and wait for completion
        #[arg(long, default_value_t = true)]
        wait: bool,
    },

    /// Query the status of an existing computational job
    Status {
        /// Job UUID
        job_id: Uuid,
    },

    /// Inspect structural prediction and biophysical metrics for a job
    Inspect {
        /// Job UUID
        job_id: Uuid,
    },

    /// Direct offline biophysical analysis of a PDB file using native Rust engine
    Analyze {
        /// Path to PDB structure file
        #[arg(short, long)]
        pdb: PathBuf,

        /// Optional reference PDB for Kabsch RMSD alignment
        #[arg(short, long)]
        reference: Option<PathBuf>,
    },

    /// Run the headless background daemon (proteusd)
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Compute runner mode
        #[arg(long, value_enum, default_value_t = RunnerMode::Auto)]
        runner: RunnerMode,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum CliTier {
    Fast,
    Sota,
    Full,
}

impl From<CliTier> for PipelineTier {
    fn from(t: CliTier) -> Self {
        match t {
            CliTier::Fast => PipelineTier::FastScreening,
            CliTier::Sota => PipelineTier::HighFidelity,
            CliTier::Full => PipelineTier::FullValidation,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum RunnerMode {
    Auto,
    Oci,
    Simulated,
}

fn get_default_data_dir() -> PathBuf {
    dirs_next_or_home().join(".local/share/proteus")
}

fn dirs_next_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = Cli::parse();
    let data_dir = get_default_data_dir();
    let db_path = data_dir.join("proteus.db");
    let artifacts_dir = data_dir.join("artifacts");

    match cli.command {
        Commands::Submit {
            file,
            fasta,
            tier,
            wait,
        } => {
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

                let runner: Arc<dyn ComputeRunner> = match OciRunner::new() {
                    Ok(oci) => Arc::new(oci),
                    Err(_) => {
                        pb.println("Container runtime socket unavailable. Falling back to SimulatedRunner.");
                        Arc::new(SimulatedRunner::new())
                    }
                };

                let scheduler = PipelineScheduler::new(repo.clone(), runner, artifacts_dir);
                scheduler.process_job(job_id).await?;

                pb.finish_with_message("Pipeline completed successfully!");

                print_job_inspection(&repo, job_id).await?;
            } else {
                println!(
                    "Job enqueued in background. Use 'proteus status {}' to inspect.",
                    job_id
                );
            }
        }

        Commands::Status { job_id } => {
            let pool = create_sqlite_pool(&db_path).await?;
            let repo = ProteusRepository::new(pool);

            let job = repo
                .get_job(job_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Job {} not found", job_id))?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
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
        }

        Commands::Inspect { job_id } => {
            let pool = create_sqlite_pool(&db_path).await?;
            let repo = ProteusRepository::new(pool);
            print_job_inspection(&repo, job_id).await?;
        }

        Commands::Analyze { pdb, reference } => {
            println!("Analyzing structure file: {:?}", pdb);
            let metrics = analyze_pdb_file(&pdb, reference.as_deref())
                .context("Biophysical analysis failed")?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
            table.set_header(vec!["Biophysical Metric", "Value"]);

            table.add_row(vec![
                Cell::new("Radius of Gyration (Rg)"),
                Cell::new(format!("{:.3} Å", metrics.radius_of_gyration)),
            ]);
            table.add_row(vec![
                Cell::new("Contact Density (C-alpha <= 8Å)"),
                Cell::new(format!("{:.2}%", metrics.contact_density * 100.0)),
            ]);
            table.add_row(vec![
                Cell::new("Mean pLDDT"),
                Cell::new(format!("{:.2}", metrics.plddt_distribution.mean)),
            ]);
            table.add_row(vec![
                Cell::new("Median pLDDT"),
                Cell::new(format!("{:.2}", metrics.plddt_distribution.median)),
            ]);
            table.add_row(vec![
                Cell::new("Fraction High Conf (pLDDT >= 70)"),
                Cell::new(format!(
                    "{:.1}%",
                    metrics.plddt_distribution.high_confidence_fraction * 100.0
                )),
            ]);
            table.add_row(vec![
                Cell::new("Fraction Very High Conf (pLDDT >= 90)"),
                Cell::new(format!(
                    "{:.1}%",
                    metrics.plddt_distribution.very_high_confidence_fraction * 100.0
                )),
            ]);

            if let Some(rmsd) = metrics.rmsd_to_reference {
                table.add_row(vec![
                    Cell::new("Kabsch RMSD to Reference"),
                    Cell::new(format!("{:.3} Å", rmsd)),
                ]);
            }

            println!("{table}");
        }

        Commands::Serve { port, host, runner } => {
            let pool = create_sqlite_pool(&db_path).await?;
            let repo = ProteusRepository::new(pool);

            let compute_runner: Arc<dyn ComputeRunner> = match runner {
                RunnerMode::Simulated => {
                    println!("Runner mode: Simulated");
                    Arc::new(SimulatedRunner::new())
                }
                RunnerMode::Oci => {
                    println!("Runner mode: OCI Container Runner");
                    Arc::new(OciRunner::new()?)
                }
                RunnerMode::Auto => match OciRunner::new() {
                    Ok(oci) => {
                        println!("Auto-detected container engine at: {}", oci.socket_path());
                        Arc::new(oci)
                    }
                    Err(_) => {
                        println!("No container runtime found. Using SimulatedRunner.");
                        Arc::new(SimulatedRunner::new())
                    }
                },
            };

            let scheduler = PipelineScheduler::new(repo, compute_runner, artifacts_dir);
            let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
            run_server(addr, scheduler).await?;
        }
    }

    Ok(())
}

async fn print_job_inspection(repo: &ProteusRepository, job_id: Uuid) -> Result<()> {
    let pred = repo
        .get_prediction_by_job(job_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Prediction for job {} not found", job_id))?;

    let metrics = repo
        .get_metrics_by_prediction(pred.id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Metrics for prediction {} not found", pred.id))?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Metric", "Value", "Confidence Assessment"]);

    table.add_row(vec![
        Cell::new("Predicted PDB Path"),
        Cell::new(&pred.pdb_path),
        Cell::new("Artifact on disk"),
    ]);

    let plddt_cell = if let Some(p) = pred.plddt {
        format!("{:.2}", p)
    } else {
        "N/A".to_string()
    };
    let cat = pred
        .confidence_category
        .unwrap_or_else(|| "Unknown".to_string());
    table.add_row(vec![
        Cell::new("Global Confidence (pLDDT)"),
        Cell::new(plddt_cell),
        Cell::new(cat),
    ]);

    table.add_row(vec![
        Cell::new("Radius of Gyration (Rg)"),
        Cell::new(format!("{:.3} Å", metrics.radius_of_gyration)),
        Cell::new("Compactness metric"),
    ]);

    table.add_row(vec![
        Cell::new("Tertiary Contact Density"),
        Cell::new(format!("{:.2}%", metrics.contact_density * 100.0)),
        Cell::new("C-alpha <= 8.0Å pairs"),
    ]);

    table.add_row(vec![
        Cell::new("High Conf Residues (>=70)"),
        Cell::new(format!(
            "{:.1}%",
            metrics.plddt_distribution.high_confidence_fraction * 100.0
        )),
        Cell::new("Reliable backbone"),
    ]);

    println!("{table}");
    Ok(())
}
