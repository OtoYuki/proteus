use anyhow::{Context, Result};
use base64::Engine;
use clap::{Parser, Subcommand, ValueEnum};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};
use indicatif::{ProgressBar, ProgressStyle};
use proteus_core::confidence::ConfidenceSource;
use proteus_core::metrics::analyze_pdb_file;
use proteus_core::models::PipelineTier;
use proteus_core::ranking::evaluate_candidate_fitness;
use proteus_core::sequence::validate_and_parse_fasta;
use proteus_engine::oci::OciRunner;
use proteus_engine::simulated::SimulatedRunner;
use proteus_engine::{AutoRunner, ComputeRunner, EsmApiRunner, PipelineScheduler};
use proteus_server::run_server;
use proteus_storage::create_sqlite_pool;
use proteus_storage::repository::ProteusRepository;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
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

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ConfidenceSourceArg {
    Auto,
    Predicted,
    Experimental,
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

        /// Compute runner mode
        #[arg(long, value_enum, default_value_t = RunnerMode::Auto)]
        runner: RunnerMode,

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

        /// Treat the B-factor column as pLDDT (predicted) or as experimental B-factors;
        /// `auto` inspects the header and the value distribution.
        #[arg(long, value_enum, default_value_t = ConfidenceSourceArg::Auto)]
        confidence_source: ConfidenceSourceArg,
    },

    /// 3D structural ribbon visualization in the terminal (HalfBlock / Braille / Kitty)
    View {
        /// Target PDB file path or job UUID
        target: String,

        /// Optional reference PDB file path for 3D structural superposition and RMSD calculation
        #[arg(long)]
        compare: Option<PathBuf>,

        /// Run interactive 60 FPS TUI viewer with orbit camera controls
        #[arg(short, long)]
        interactive: bool,

        /// Enable side-by-side live biophysical telemetry dashboard (Ramachandran, pLDDT, SASA)
        #[arg(long)]
        dashboard: bool,

        /// Terminal rendering backend
        #[arg(short, long, value_enum, default_value_t = CliBackend::HalfBlock)]
        backend: CliBackend,

        /// Color scheme
        #[arg(short, long, value_enum, default_value_t = CliColorScheme::Plddt)]
        color: CliColorScheme,

        /// Terminal viewport width (defaults to terminal width or 80)
        #[arg(long)]
        width: Option<usize>,

        /// Terminal viewport height (defaults to terminal height or 30)
        #[arg(long)]
        height: Option<usize>,

        /// Open structure in browser via standalone Mol* WebGL 3D viewer
        #[arg(long)]
        web: bool,

        /// Export standalone Mol* WebGL 3D viewer HTML file
        #[arg(long)]
        html: Option<PathBuf>,
    },

    /// In-silico Deep Mutational Scanning (DMS) variant library generator
    Mutate {
        /// Path to scaffold FASTA file (or '-' for stdin)
        scaffold: String,

        /// Mutagenesis mode
        #[arg(short, long, value_enum, default_value_t = CliMutagenesisMode::Alanine)]
        mode: CliMutagenesisMode,

        /// 1-indexed window start position (inclusive)
        #[arg(long)]
        start: Option<usize>,

        /// 1-indexed window end position (inclusive)
        #[arg(long)]
        end: Option<usize>,

        /// Maximum number of mutant variants to generate
        #[arg(long)]
        max_variants: Option<usize>,

        /// Exclude the unmutated wildtype scaffold from output library
        #[arg(long)]
        no_wt: bool,

        /// Custom prefix for variant sequence headers
        #[arg(long)]
        prefix: Option<String>,

        /// Optional output file path for multi-FASTA library (writes to stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// High-throughput library screening funnel: batch folding, Pareto ranking, and leaderboard
    Screen {
        /// Path to multi-sequence FASTA library file (or '-' for stdin)
        library: String,

        /// Computational tier
        #[arg(short, long, value_enum, default_value_t = CliTier::Fast)]
        tier: CliTier,

        /// Compute runner mode
        #[arg(long, value_enum, default_value_t = RunnerMode::Auto)]
        runner: RunnerMode,

        /// Maximum concurrent worker threads
        #[arg(short, long, default_value_t = 4)]
        workers: usize,

        /// Minimum pLDDT cutoff threshold to pass screening
        #[arg(long, default_value_t = 70.0)]
        min_plddt: f64,

        /// Number of top ranked candidates to display in leaderboard
        #[arg(long, default_value_t = 10)]
        top: usize,

        /// Optional path to export structured screening dataset (.parquet, .csv, or .json)
        #[arg(short, long)]
        export: Option<PathBuf>,
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum CliMutagenesisMode {
    #[value(name = "alanine")]
    Alanine,
    #[value(name = "saturation")]
    Saturation,
}

impl From<CliMutagenesisMode> for proteus_core::mutagenesis::MutagenesisMode {
    fn from(m: CliMutagenesisMode) -> Self {
        match m {
            CliMutagenesisMode::Alanine => {
                proteus_core::mutagenesis::MutagenesisMode::AlanineScanning
            }
            CliMutagenesisMode::Saturation => {
                proteus_core::mutagenesis::MutagenesisMode::Saturation
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum CliBackend {
    #[value(name = "halfblock", alias = "half-block")]
    HalfBlock,
    #[value(name = "braille")]
    Braille,
    #[value(name = "kitty")]
    Kitty,
}

impl From<CliBackend> for proteus_render::terminal::TerminalBackend {
    fn from(b: CliBackend) -> Self {
        match b {
            CliBackend::HalfBlock => proteus_render::terminal::TerminalBackend::HalfBlock,
            CliBackend::Braille => proteus_render::terminal::TerminalBackend::Braille,
            CliBackend::Kitty => proteus_render::terminal::TerminalBackend::Kitty,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum CliColorScheme {
    #[value(name = "plddt")]
    Plddt,
    #[value(name = "ss", alias = "secondary-structure")]
    SecondaryStructure,
    #[value(name = "rainbow")]
    Rainbow,
}

impl From<CliColorScheme> for proteus_render::rasterizer::ColorScheme {
    fn from(c: CliColorScheme) -> Self {
        match c {
            CliColorScheme::Plddt => proteus_render::rasterizer::ColorScheme::Plddt,
            CliColorScheme::SecondaryStructure => {
                proteus_render::rasterizer::ColorScheme::SecondaryStructure
            }
            CliColorScheme::Rainbow => proteus_render::rasterizer::ColorScheme::Rainbow,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum RunnerMode {
    Auto,
    Oci,
    Simulated,
    EsmApi,
}

fn resolve_runner(mode: RunnerMode) -> Result<Arc<dyn ComputeRunner>> {
    match mode {
        RunnerMode::Auto => Ok(Arc::new(AutoRunner::new())),
        RunnerMode::Simulated => Ok(Arc::new(SimulatedRunner::new())),
        RunnerMode::EsmApi => Ok(Arc::new(EsmApiRunner::new())),
        RunnerMode::Oci => {
            let oci = OciRunner::new().context("Failed to initialize OCI container runner")?;
            Ok(Arc::new(oci))
        }
    }
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
            runner,
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

        Commands::Analyze {
            pdb,
            reference,
            confidence_source,
        } => {
            println!("Analyzing structure file: {:?}", pdb);
            let mut metrics = analyze_pdb_file(&pdb, reference.as_deref())
                .context("Biophysical analysis failed")?;
            let forced = match confidence_source {
                ConfidenceSourceArg::Auto => None,
                ConfidenceSourceArg::Predicted => Some(ConfidenceSource::Predicted),
                ConfidenceSourceArg::Experimental => Some(ConfidenceSource::ExperimentalBFactor),
            };
            if let Some(src) = forced {
                metrics.confidence_source = src;
                let residues = metrics
                    .secondary_structure_summary
                    .as_ref()
                    .map(|s| s.assignment.len())
                    .unwrap_or(1);
                metrics.candidate_fitness_score =
                    Some(evaluate_candidate_fitness(&metrics, residues).total_score);
            }

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
            match metrics.plddt() {
                Some(p) => {
                    table.add_row(vec![
                        Cell::new("Mean pLDDT"),
                        Cell::new(format!("{:.2}", p.mean)),
                    ]);
                    table.add_row(vec![
                        Cell::new("Median pLDDT"),
                        Cell::new(format!("{:.2}", p.median)),
                    ]);
                    table.add_row(vec![
                        Cell::new("Fraction High Conf (pLDDT >= 70)"),
                        Cell::new(format!("{:.1}%", p.high_confidence_fraction * 100.0)),
                    ]);
                    table.add_row(vec![
                        Cell::new("Fraction Very High Conf (pLDDT >= 90)"),
                        Cell::new(format!("{:.1}%", p.very_high_confidence_fraction * 100.0)),
                    ]);
                }
                None => {
                    table.add_row(vec![
                        Cell::new("pLDDT"),
                        Cell::new(
                            "n/a (experimental structure; B-factor column is not a confidence)",
                        ),
                    ]);
                }
            }

            if let Some(rmsd) = metrics.rmsd_to_reference {
                table.add_row(vec![
                    Cell::new("Kabsch RMSD to Reference"),
                    Cell::new(format!("{:.3} Å", rmsd)),
                ]);
            }

            if let Some(ref ss) = metrics.secondary_structure_summary {
                table.add_row(vec![
                    Cell::new("Secondary Structure Composition"),
                    Cell::new(format!(
                        "α-Helix: {:.1}% | β-Strand: {:.1}% | Coil: {:.1}%",
                        ss.helix_fraction * 100.0,
                        ss.strand_fraction * 100.0,
                        ss.coil_fraction * 100.0
                    )),
                ]);
            }

            if let Some(ref rama) = metrics.ramachandran_stats {
                table.add_row(vec![
                    Cell::new("Ramachandran Conformation"),
                    Cell::new(format!(
                        "Favored: {:.1}% | Allowed: {:.1}% | Outliers: {}",
                        rama.favored_fraction * 100.0,
                        rama.allowed_fraction * 100.0,
                        rama.outlier_count
                    )),
                ]);
            }

            if let Some(ref sasa) = metrics.sasa_metrics {
                table.add_row(vec![
                    Cell::new("Solvent Accessible Surface Area"),
                    Cell::new(format!(
                        "Total: {:.1} Å² (Hydrophobic Burial: {:.1}%)",
                        sasa.total_sasa,
                        sasa.hydrophobic_burial_ratio * 100.0
                    )),
                ]);
            }

            if let Some(ref clash) = metrics.clash_stats {
                table.add_row(vec![
                    Cell::new("MolProbity Clashscore (>0.4Å)"),
                    Cell::new(format!(
                        "{:.1} ({} severe steric overlaps)",
                        clash.clashscore, clash.clash_count
                    )),
                ]);
            }

            if let Some(ref net) = metrics.interaction_network {
                table.add_row(vec![
                    Cell::new("Hydrogen Bonds (H-Bonds)"),
                    Cell::new(format!(
                        "{} total ({} BB-BB, {} BB-SC, {} SC-SC)",
                        net.summary.total_hbonds,
                        net.summary.bb_bb_hbonds,
                        net.summary.bb_sc_hbonds,
                        net.summary.sc_sc_hbonds
                    )),
                ]);
                table.add_row(vec![
                    Cell::new("Ionic Salt Bridges (≤4.0Å)"),
                    Cell::new(format!(
                        "{} detected{}",
                        net.summary.total_salt_bridges,
                        if let Some(s) = net.salt_bridges.first() {
                            format!(
                                " (closest: {}{}:{}-{}{}:{} {:.2}Å)",
                                s.cation_res_name,
                                s.cation_res_seq,
                                s.cation_atom_name,
                                s.anion_res_name,
                                s.anion_res_seq,
                                s.anion_atom_name,
                                s.distance
                            )
                        } else {
                            "".to_string()
                        }
                    )),
                ]);
                table.add_row(vec![
                    Cell::new("Aromatic π-π Stacking"),
                    Cell::new(format!(
                        "{} conjugated pairs ({} parallel, {} T-shaped)",
                        net.summary.total_pi_pi_stacks,
                        net.pi_pi_stacks
                            .iter()
                            .filter(|p| p.category == proteus_core::PiStackingCategory::Parallel)
                            .count(),
                        net.pi_pi_stacks
                            .iter()
                            .filter(|p| p.category == proteus_core::PiStackingCategory::TShaped)
                            .count(),
                    )),
                ]);
                table.add_row(vec![
                    Cell::new("Cation-π Interactions"),
                    Cell::new(format!(
                        "{} active interactions",
                        net.summary.total_cation_pi
                    )),
                ]);
                table.add_row(vec![
                    Cell::new("Non-Covalent Network Density"),
                    Cell::new(format!(
                        "{:.1} contacts / 100 res",
                        net.summary.network_density
                    )),
                ]);
            }

            if let Some(fitness) = metrics.candidate_fitness_score {
                table.add_row(vec![
                    Cell::new("Candidate Fitness Score"),
                    Cell::new(format!("{:.1} / 100", fitness)),
                ]);
            }

            println!("{table}");
        }

        Commands::View {
            target,
            compare,
            interactive,
            dashboard,
            backend,
            color,
            width,
            height,
            web,
            html,
        } => {
            let target_path = PathBuf::from(&target);
            let (pdb_content, title) = if target_path.exists() {
                let content = tokio::fs::read_to_string(&target_path)
                    .await
                    .with_context(|| format!("Failed to read PDB file at {:?}", target_path))?;
                let name = target_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("PDB Structure")
                    .to_string();
                (content, name)
            } else if let Ok(job_id) = Uuid::parse_str(&target) {
                let pool = create_sqlite_pool(&db_path).await?;
                let repo = ProteusRepository::new(pool);
                let pred = repo
                    .get_prediction_by_job(job_id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("Prediction for job {} not found", job_id))?;
                let content = tokio::fs::read_to_string(&pred.pdb_path)
                    .await
                    .with_context(|| format!("Failed to read PDB at {:?}", pred.pdb_path))?;
                (content, format!("Job {job_id}"))
            } else {
                anyhow::bail!(
                    "Target '{}' is neither an existing file path nor a valid job UUID",
                    target
                );
            };

            if web || html.is_some() {
                let html_path = html.unwrap_or_else(|| {
                    let sanitized: String = title
                        .chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '_' })
                        .collect();
                    std::env::temp_dir().join(format!("proteus_view_{sanitized}.html"))
                });

                let b64_pdb =
                    base64::engine::general_purpose::STANDARD.encode(pdb_content.as_bytes());
                let html_content = format!(
                    r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Proteus 3D Structure Viewer - {title}</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/molstar@3.30.0/build/viewer/molstar.css" />
    <script type="text/javascript" src="https://unpkg.com/molstar@3.30.0/build/viewer/molstar.js"></script>
    <style>
        body, html {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #fff; }}
        #app {{ width: 100%; height: 100%; position: absolute; }}
        #header {{ position: absolute; top: 16px; left: 20px; z-index: 1000; background: rgba(15, 23, 42, 0.85); padding: 12px 20px; border-radius: 12px; backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); }}
        #header h1 {{ margin: 0; font-size: 16px; font-weight: 700; color: #38bdf8; letter-spacing: -0.025em; }}
        #header p {{ margin: 4px 0 0 0; font-size: 12px; color: #94a3b8; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Proteus Bio-Compute 3D Viewer</h1>
        <p>{title}</p>
    </div>
    <div id="app"></div>
    <script>
        const pdbB64 = `{b64_pdb}`;
        document.addEventListener('DOMContentLoaded', async () => {{
            const viewer = await molstar.Viewer.create('app', {{
                layoutIsExpanded: false,
                layoutShowControls: true,
                layoutShowRemoteState: false,
                layoutShowSequence: true,
                layoutShowLog: false,
                viewportShowExpand: false,
            }});
            const rawPdb = atob(pdbB64);
            const blob = new Blob([rawPdb], {{ type: 'text/plain' }});
            const url = URL.createObjectURL(blob);
            await viewer.loadStructureFromUrl(url, 'pdb', false, {{
                representationStyle: {{
                    type: 'cartoon',
                    color: 'secondary-structure',
                }}
            }});
        }});
    </script>
</body>
</html>"#
                );

                tokio::fs::write(&html_path, html_content)
                    .await
                    .with_context(|| format!("Failed to write HTML file to {:?}", html_path))?;

                println!(
                    "Generated standalone 3D WebGL viewer HTML -> {:?}",
                    html_path
                );

                if web {
                    println!("Launching default browser via xdg-open...");
                    let _ = std::process::Command::new("xdg-open")
                        .arg(&html_path)
                        .spawn();
                }
                return Ok(());
            }

            let render_backend: proteus_render::terminal::TerminalBackend = backend.into();
            let render_color: proteus_render::rasterizer::ColorScheme = color.into();

            let (term_cols, term_rows): (u16, u16) =
                crossterm::terminal::size().unwrap_or((80, 24));
            let w = width.unwrap_or(term_cols as usize);
            let h = height.unwrap_or(term_rows.saturating_sub(4).max(16) as usize);

            if let Some(ref_path) = compare {
                let ref_content = tokio::fs::read_to_string(&ref_path)
                    .await
                    .with_context(|| format!("Failed to read reference PDB at {:?}", ref_path))?;

                if interactive {
                    let sup_data = proteus_render::prepare_superposition_for_rendering(
                        &pdb_content,
                        &ref_content,
                    )
                    .context("Failed to superimpose structures for 3D rendering")?;

                    let ref_name = ref_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Reference");
                    let dual_title = format!("{title} (Cyan) vs {ref_name} (Ruby)");
                    let config = proteus_render::tui::ViewerConfig {
                        title: dual_title,
                        initial_color_scheme: proteus_render::rasterizer::ColorScheme::Solid(
                            proteus_render::rasterizer::ColorRGB::new(6, 182, 212),
                        ),
                        auto_rotate: true,
                        secondary_mesh: Some((
                            sup_data.ref_mesh,
                            proteus_render::rasterizer::ColorRGB::new(244, 63, 94),
                        )),
                        rmsd: Some(sup_data.rmsd),
                        disulfide_mesh: None,
                        dashboard_enabled: false,
                        dashboard_data: None,
                    };
                    proteus_render::tui::run_interactive_viewer(
                        &sup_data.target_mesh,
                        sup_data.camera,
                        config,
                    )
                    .context("Interactive dual-structure 3D viewer error")?;
                } else {
                    let (snapshot, rmsd) = proteus_render::render_superposition_snapshot(
                        &pdb_content,
                        &ref_content,
                        w,
                        h,
                        render_backend,
                    )
                    .context("Failed to render superposition snapshot")?;

                    println!("{snapshot}");
                    println!(
                        "\x1b[1mSuperposition:\x1b[0m Target (Cyan) vs Reference (Ruby) | \x1b[32mRMSD: {:.3} Å\x1b[0m",
                        rmsd
                    );
                }
            } else if interactive || dashboard {
                let structure_data = proteus_render::parse_pdb_structure(&pdb_content)
                    .context("Failed to parse structure for 3D rendering")?;

                let dashboard_data = Some(proteus_render::tui::DashboardData {
                    title: title.clone(),
                    num_residues: structure_data.num_residues,
                    num_disulfides: structure_data.num_disulfides,
                    metrics: structure_data.metrics,
                    plddts: structure_data.plddts,
                    ramachandran_points: structure_data.ramachandran_points,
                });

                let config = proteus_render::tui::ViewerConfig {
                    title,
                    initial_color_scheme: render_color,
                    auto_rotate: true,
                    secondary_mesh: None,
                    rmsd: None,
                    disulfide_mesh: structure_data.disulfide_mesh,
                    dashboard_enabled: dashboard || term_cols >= 100,
                    dashboard_data,
                };
                proteus_render::tui::run_interactive_viewer(
                    &structure_data.ribbon_mesh,
                    structure_data.camera,
                    config,
                )
                .context("Interactive 3D viewer error")?;
            } else {
                let snapshot = proteus_render::render_pdb_snapshot(
                    &pdb_content,
                    w,
                    h,
                    render_backend,
                    render_color,
                )
                .context("Failed to render 3D snapshot")?;

                println!("{snapshot}");
            }
        }

        Commands::Mutate {
            scaffold,
            mode,
            start,
            end,
            max_variants,
            no_wt,
            prefix,
            output,
        } => {
            let content: String = if scaffold == "-" {
                use tokio::io::AsyncReadExt;
                let mut buf = String::new();
                tokio::io::stdin()
                    .read_to_string(&mut buf)
                    .await
                    .context("Failed to read scaffold FASTA from stdin")?;
                buf
            } else {
                let p = Path::new(&scaffold);
                tokio::fs::read_to_string(p)
                    .await
                    .with_context(|| format!("Failed to read scaffold file at {:?}", p))?
            };

            let seq = proteus_core::sequence::validate_and_parse_fasta(&content)
                .context("Scaffold sequence validation failed")?;

            let config = proteus_core::mutagenesis::MutagenesisConfig {
                mode: mode.into(),
                window_start: start,
                window_end: end,
                max_variants,
                include_wildtype: !no_wt,
                prefix,
            };

            let library = proteus_core::mutagenesis::generate_mutant_library(&seq, &config)
                .context("Mutant variant generation failed")?;

            let formatted = proteus_core::sequence::format_multi_fasta(&library);

            if let Some(out_path) = output {
                tokio::fs::write(&out_path, &formatted)
                    .await
                    .with_context(|| format!("Failed to write mutant library to {:?}", out_path))?;
                eprintln!(
                    "Generated {} variant sequences (scaffold len: {}) -> {:?}",
                    library.len(),
                    seq.length,
                    out_path
                );
            } else {
                print!("{formatted}");
            }
        }

        Commands::Screen {
            library,
            tier,
            runner,
            workers,
            min_plddt,
            top,
            export,
        } => {
            let content: String = if library == "-" {
                eprintln!("Reading sequence library from standard input (stdin)...");
                use tokio::io::AsyncReadExt;
                let mut buf = String::new();
                tokio::io::stdin()
                    .read_to_string(&mut buf)
                    .await
                    .context("Failed to read multi-FASTA library from stdin")?;
                buf
            } else {
                let p = Path::new(&library);
                eprintln!("Reading sequence library from: {:?}", p);
                tokio::fs::read_to_string(p)
                    .await
                    .with_context(|| format!("Failed to read library file at {:?}", p))?
            };

            let sequences = proteus_core::sequence::validate_and_parse_multi_fasta(&content)
                .context("Multi-FASTA library parsing failed")?;

            let total_seqs = sequences.len();
            eprintln!("Loaded {total_seqs} candidate sequences for screening funnel");

            let pool = create_sqlite_pool(&db_path).await?;
            let repo = ProteusRepository::new(pool);

            let mut job_ids = Vec::with_capacity(total_seqs);
            let pipeline_tier: PipelineTier = tier.into();

            for seq in &sequences {
                repo.insert_sequence(seq).await?;
                let job_id = Uuid::new_v4();
                let job = proteus_core::models::PipelineJob {
                    id: job_id,
                    sequence_id: seq.id,
                    tier: pipeline_tier.clone(),
                    status: proteus_core::models::JobStatus::Queued,
                    priority: 1,
                    created_at: chrono::Utc::now(),
                    started_at: None,
                    completed_at: None,
                    error_log: None,
                };
                repo.insert_job(&job).await?;
                job_ids.push(job_id);
            }

            let pb = ProgressBar::new(total_seqs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) {msg}",
                    )?
                    .progress_chars("█▓▒░"),
            );
            pb.set_message("Screening candidate library in parallel...");

            let compute_runner = resolve_runner(runner)?;
            let scheduler = PipelineScheduler::new(repo.clone(), compute_runner, artifacts_dir);

            let results = scheduler.process_batch(&job_ids, workers).await;
            pb.finish_with_message("Screening batch execution complete!");

            let successful_count = results.iter().filter(|r| r.is_ok()).count();
            eprintln!(
                "\nCompleted: {}/{} successful ({} parallel workers)",
                successful_count, total_seqs, workers
            );

            // Fetch predictions and metrics for ranking
            struct CandidateRank {
                job_id: Uuid,
                header: String,
                length: usize,
                plddt: f64,
                rg: f64,
                hydrophobic_burial: f64,
                helix_pct: f64,
                strand_pct: f64,
                coil_pct: f64,
                favored_rama: f64,
                rama_outliers: usize,
                clashscore: f64,
                hbond_count: usize,
                salt_bridge_count: usize,
                pi_stacking_count: usize,
                cation_pi_count: usize,
                fitness: f64,
            }

            let mut candidates: Vec<CandidateRank> = Vec::new();

            for (seq, &job_id) in sequences.iter().zip(job_ids.iter()) {
                if let Ok(Some(pred)) = repo.get_prediction_by_job(job_id).await {
                    if let Ok(Some(metrics)) = repo.get_metrics_by_prediction(pred.id).await {
                        let plddt = pred.plddt.unwrap_or(metrics.plddt_distribution.mean);
                        if plddt >= min_plddt {
                            let burial = metrics
                                .sasa_metrics
                                .as_ref()
                                .map_or(0.0, |s| s.hydrophobic_burial_ratio * 100.0);
                            let (helix, strand, coil) = metrics
                                .secondary_structure_summary
                                .as_ref()
                                .map_or((0.0, 0.0, 0.0), |s| {
                                    (
                                        s.helix_fraction * 100.0,
                                        s.strand_fraction * 100.0,
                                        s.coil_fraction * 100.0,
                                    )
                                });
                            let (favored_rama, rama_outliers) =
                                metrics.ramachandran_stats.as_ref().map_or((0.0, 0), |r| {
                                    (r.favored_fraction * 100.0, r.outlier_count)
                                });
                            let clashscore =
                                metrics.clash_stats.as_ref().map_or(0.0, |c| c.clashscore);
                            let (
                                hbond_count,
                                salt_bridge_count,
                                pi_stacking_count,
                                cation_pi_count,
                            ) = metrics
                                .interaction_network
                                .as_ref()
                                .map_or((0, 0, 0, 0), |net| {
                                    (
                                        net.summary.total_hbonds,
                                        net.summary.total_salt_bridges,
                                        net.summary.total_pi_pi_stacks,
                                        net.summary.total_cation_pi,
                                    )
                                });
                            let fitness = metrics.candidate_fitness_score.unwrap_or(0.0);

                            candidates.push(CandidateRank {
                                job_id,
                                header: seq.header.clone(),
                                length: seq.length,
                                plddt,
                                rg: metrics.radius_of_gyration,
                                hydrophobic_burial: burial,
                                helix_pct: helix,
                                strand_pct: strand,
                                coil_pct: coil,
                                favored_rama,
                                rama_outliers,
                                clashscore,
                                hbond_count,
                                salt_bridge_count,
                                pi_stacking_count,
                                cation_pi_count,
                                fitness,
                            });
                        }
                    }
                }
            }

            // Rank by composite fitness score descending
            candidates.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            println!(
                "\n=== Screening Funnel Leaderboard (Cutoff: pLDDT >= {:.1}) ===",
                min_plddt
            );
            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
            table.set_header(vec![
                "Rank",
                "Candidate Header",
                "Len",
                "pLDDT",
                "Rg (Å)",
                "Core Burial",
                "Clash",
                "H-Bonds",
                "Salt/π",
                "Fitness / 100",
                "Job ID",
            ]);

            for (idx, c) in candidates.iter().take(top).enumerate() {
                table.add_row(vec![
                    Cell::new(format!("#{}", idx + 1)),
                    Cell::new(&c.header),
                    Cell::new(c.length),
                    Cell::new(format!("{:.2}", c.plddt)),
                    Cell::new(format!("{:.2}", c.rg)),
                    Cell::new(format!("{:.1}%", c.hydrophobic_burial)),
                    Cell::new(format!("{:.1}", c.clashscore)),
                    Cell::new(c.hbond_count),
                    Cell::new(format!(
                        "{}/{}",
                        c.salt_bridge_count,
                        c.pi_stacking_count + c.cation_pi_count
                    )),
                    Cell::new(format!("{:.1}", c.fitness)),
                    Cell::new(c.job_id.to_string()),
                ]);
            }

            println!("{table}");

            if let Some(winner) = candidates.first() {
                println!(
                    "\nTop Candidate: '{}' (Fitness: {:.1})\nView structure in terminal: proteus view {}",
                    winner.header, winner.fitness, winner.job_id
                );
            }

            if let Some(export_path) = export {
                let mut records = Vec::with_capacity(candidates.len());
                for (idx, c) in candidates.iter().enumerate() {
                    records.push(proteus_storage::ScreeningRecord {
                        rank: idx + 1,
                        job_id: c.job_id,
                        header: c.header.clone(),
                        length: c.length,
                        plddt: c.plddt,
                        rg: c.rg,
                        hydrophobic_burial_pct: c.hydrophobic_burial,
                        helix_pct: c.helix_pct,
                        strand_pct: c.strand_pct,
                        coil_pct: c.coil_pct,
                        favored_ramachandran_pct: c.favored_rama,
                        rama_outliers: c.rama_outliers,
                        clashscore: c.clashscore,
                        hbond_count: c.hbond_count,
                        salt_bridge_count: c.salt_bridge_count,
                        pi_stacking_count: c.pi_stacking_count,
                        cation_pi_count: c.cation_pi_count,
                        fitness: c.fitness,
                    });
                }
                proteus_storage::save_screening_dataset(&records, &export_path)
                    .await
                    .with_context(|| format!("Failed to export dataset to {:?}", export_path))?;
                eprintln!(
                    "Successfully exported {} ranked candidates -> {:?}",
                    records.len(),
                    export_path
                );
            }
        }

        Commands::Serve { port, host, runner } => {
            let pool = create_sqlite_pool(&db_path).await?;
            let repo = ProteusRepository::new(pool);
            let compute_runner = resolve_runner(runner)?;
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

    if let Some(ref ss) = metrics.secondary_structure_summary {
        table.add_row(vec![
            Cell::new("Secondary Structure"),
            Cell::new(format!(
                "H: {:.1}% | E: {:.1}% | C: {:.1}%",
                ss.helix_fraction * 100.0,
                ss.strand_fraction * 100.0,
                ss.coil_fraction * 100.0
            )),
            Cell::new("P-SEA assignment"),
        ]);
    }

    if let Some(ref rama) = metrics.ramachandran_stats {
        table.add_row(vec![
            Cell::new("Ramachandran Regions"),
            Cell::new(format!(
                "Favored: {:.1}% | Outliers: {}",
                rama.favored_fraction * 100.0,
                rama.outlier_count
            )),
            Cell::new("Backbone stereochemistry"),
        ]);
    }

    if let Some(ref sasa) = metrics.sasa_metrics {
        table.add_row(vec![
            Cell::new("Hydrophobic Core Burial"),
            Cell::new(format!("{:.1}%", sasa.hydrophobic_burial_ratio * 100.0)),
            Cell::new("Shrake-Rupley SASA"),
        ]);
    }

    if let Some(fitness) = metrics.candidate_fitness_score {
        table.add_row(vec![
            Cell::new("Candidate Fitness Score"),
            Cell::new(format!("{:.1} / 100", fitness)),
            Cell::new("Multi-objective ranking"),
        ]);
    }

    println!("{table}");
    Ok(())
}
