//! `proteus screen` — Fold, score and rank a variant library; export the table.

use super::prelude::*;
use proteus_render::brand::Role;

/// Arguments of `proteus screen`.
#[derive(clap::Args, Debug)]
pub struct Args {
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

    /// Ranking signal: structure-only fitness, ESM-2 zero-shot score, or a hybrid
    #[arg(long, value_enum, default_value_t = esm_cmd::Scorer::Structure)]
    scorer: esm_cmd::Scorer,

    #[command(flatten)]
    esm: esm_cmd::EsmOptions,
}

pub async fn run(
    args: Args,
    db_path: &std::path::Path,
    artifacts_dir: &std::path::Path,
) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let artifacts_dir = artifacts_dir.to_path_buf();
    let Args {
        library,
        tier,
        runner,
        workers,
        min_plddt,
        top,
        export,
        scorer,
        esm,
    } = args;
    let content: String = if library == "-" {
        eprintln!("reading the library from stdin");
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        tokio::io::stdin()
            .read_to_string(&mut buf)
            .await
            .context("Failed to read multi-FASTA library from stdin")?;
        buf
    } else {
        let p = Path::new(&library);
        eprintln!("reading the library from {}", p.display());
        tokio::fs::read_to_string(p)
            .await
            .with_context(|| format!("Failed to read library file at {:?}", p))?
    };

    let sequences = proteus_core::sequence::validate_and_parse_multi_fasta(&content)
        .context("Multi-FASTA library parsing failed")?;
    if let Some(p) = &export {
        proteus_storage::check_export_path(p)
            .with_context(|| format!("cannot export to {:?}", p))?;
    }

    let total_seqs = sequences.len();
    eprintln!("{total_seqs} candidates to fold and measure");

    // ESM-2 scores are sequence-only: compute them before (and independently of) folding.
    let esm_scores = if scorer == esm_cmd::Scorer::Structure {
        None
    } else {
        Some(esm_cmd::score_library(&sequences, &esm)?)
    };

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
            .template(&format!(
                "[{{elapsed_precise}}] {{bar:40.{}/{}}} {{pos}}/{{len}} ({{percent}}%) {{msg}}",
                tint(proteus_render::brand::Role::Accent),
                tint(proteus_render::brand::Role::Line)
            ))?
            .progress_chars("█▓▒░"),
    );
    pb.set_message("folding and measuring");

    let compute_runner = resolve_runner(runner)?;
    let scheduler = PipelineScheduler::new(repo.clone(), compute_runner, artifacts_dir);

    // Advance the bar as each job ends, not once at the end of the batch.
    let mut events = scheduler.subscribe();
    let ticker = {
        let pb = pb.clone();
        let ours: std::collections::HashSet<Uuid> = job_ids.iter().copied().collect();
        tokio::spawn(async move {
            use proteus_engine::EngineEvent;
            loop {
                match events.recv().await {
                    Ok(EngineEvent::JobCompleted { job_id, .. })
                    | Ok(EngineEvent::JobFailed { job_id, .. })
                        if ours.contains(&job_id) =>
                    {
                        pb.inc(1)
                    }
                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }
        })
    };
    let results = scheduler.process_batch(&job_ids, workers).await;
    ticker.abort();
    pb.set_position(total_seqs as u64);
    pb.finish_with_message("done");

    let successful_count = results.iter().filter(|r| r.is_ok()).count();
    eprintln!(
        "{}/{} folded ({} workers in parallel)",
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
        heavy_atom_overlap_score: f64,
        hbond_count: usize,
        salt_bridge_count: usize,
        pi_stacking_count: usize,
        cation_pi_count: usize,
        fitness: f64,
        esm2_score: Option<f32>,
        rank_key: f64,
        engine: String,
    }

    let mut candidates: Vec<CandidateRank> = Vec::new();
    let mut simulated_dropped = 0usize;
    // Ranked structures that did not run at the requested tier, by fallback reason.
    let mut downgraded: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    for (seq, &job_id) in sequences.iter().zip(job_ids.iter()) {
        if let Ok(Some(pred)) = repo.get_prediction_by_job(job_id).await {
            let engine = proteus_engine::engine_name(pred.metadata.as_ref()).to_string();
            if !rankable(&engine, runner) {
                simulated_dropped += 1;
                continue;
            }
            if let Some(d) = proteus_engine::tier_downgrade(pred.metadata.as_ref()) {
                *downgraded
                    .entry(format!("requested '{}': {}", d.requested, d.reason))
                    .or_default() += 1;
            }
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
                    let (favored_rama, rama_outliers) = metrics
                        .ramachandran_stats
                        .as_ref()
                        .map_or((0.0, 0), |r| (r.favored_fraction * 100.0, r.outlier_count));
                    let heavy_atom_overlap_score = metrics
                        .steric_overlap
                        .as_ref()
                        .map_or(0.0, |c| c.heavy_atom_overlap_score);
                    let (hbond_count, salt_bridge_count, pi_stacking_count, cation_pi_count) =
                        metrics
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
                    let esm2_score = esm_scores
                        .as_ref()
                        .and_then(|m| m.get(&seq.id).copied())
                        .flatten();
                    // An entry the ESM-2 scorer could not score (no mutation tag, or a tag
                    // that does not apply) ranks after every scored one; falling back to the
                    // 0-100 fitness would put it above every ESM-2 log-ratio.
                    let rank_key = match (scorer, esm2_score) {
                        (esm_cmd::Scorer::Structure, _) => fitness,
                        (_, None) => f64::NEG_INFINITY,
                        (esm_cmd::Scorer::Esm2, Some(e)) => e as f64,
                        (esm_cmd::Scorer::Hybrid, Some(e)) => esm_cmd::hybrid(fitness, e),
                    };

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
                        heavy_atom_overlap_score,
                        hbond_count,
                        salt_bridge_count,
                        pi_stacking_count,
                        cation_pi_count,
                        fitness,
                        esm2_score,
                        rank_key,
                        engine,
                    });
                }
            }
        }
    }

    if candidates.is_empty() {
        let failed = total_seqs - successful_count;
        anyhow::bail!(
            "no candidate reached the leaderboard: {failed} of {total_seqs} jobs failed \
             (run with RUST_LOG=info to see why), {simulated_dropped} produced simulated \
             placeholders, and the remaining {} fell below --min-plddt {min_plddt:.1}",
            successful_count.saturating_sub(simulated_dropped)
        );
    }

    // Rank by the selected signal, descending
    candidates.sort_by(|a, b| {
        b.rank_key
            .partial_cmp(&a.rank_key)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if simulated_dropped > 0 {
        eprintln!(
            "\nWARNING: {simulated_dropped} of {total_seqs} candidates were folded by the offline \
             simulator (no container image and the ESMFold API was unavailable). Their \
             structures are synthetic helices, not predictions, and are excluded from the \
             ranking. Re-run online, provide a runner image, or pass --runner simulated to \
             rank them anyway."
        );
    }
    if !downgraded.is_empty() {
        let n: usize = downgraded.values().sum();
        eprintln!(
            "\nWARNING: {n} of {} ranked structures did not run at the requested tier \
             (--tier {}); `proteus inspect <job>` shows the engine and reason:",
            candidates.len(),
            format!("{tier:?}").to_lowercase()
        );
        for (reason, count) in &downgraded {
            eprintln!("  {count}× {reason}");
        }
    }
    let a = proteus_render::brand::ansi::Ansi::for_stdout();
    println!(
        "\n{} {}{}",
        a.paint(Role::Accent, "(leaderboard)"),
        a.paint(Role::Dim, &format!("pLDDT ≥ {min_plddt:.1}")),
        if runner == RunnerMode::Simulated {
            format!(
                " {}",
                a.paint(
                    Role::Warm,
                    "! SIMULATED: synthetic structures, not predictions"
                )
            )
        } else {
            String::new()
        }
    );
    let mut table = Table::new();
    table.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    // A short job id split over two lines breaks grep and copy-paste.
    fit_table(&mut table);
    // A column of dashes says nothing, so ESM-2 only appears when something scored.
    let show_esm = candidates.iter().take(top).any(|c| c.esm2_score.is_some());
    let mut header = vec![
        "Rank",
        "Candidate",
        "Len",
        "pLDDT",
        "Rg (Å)",
        "Burial",
        "Clash/1k",
        "H-Bonds",
        "Salt/π",
        "Fitness",
    ];
    if show_esm {
        header.push("ESM-2");
    }
    header.push("Job");
    table.set_header(header);

    for (idx, c) in candidates.iter().take(top).enumerate() {
        let mut row = vec![
            Cell::new(format!("#{}", idx + 1)),
            Cell::new(&c.header),
            Cell::new(c.length),
            Cell::new(format!("{:.2}", c.plddt)),
            Cell::new(format!("{:.2}", c.rg)),
            Cell::new(format!("{:.1}%", c.hydrophobic_burial)),
            Cell::new(format!("{:.1}", c.heavy_atom_overlap_score)),
            Cell::new(c.hbond_count),
            Cell::new(format!(
                "{}/{}",
                c.salt_bridge_count,
                c.pi_stacking_count + c.cation_pi_count
            )),
            Cell::new(format!("{:.1}", c.fitness)),
        ];
        if show_esm {
            row.push(Cell::new(
                c.esm2_score
                    .map(|e| format!("{e:+.2}"))
                    .unwrap_or_else(|| "–".into()),
            ));
        }
        row.push(Cell::new(job_ref::short(c.job_id)));
        table.add_row(row);
    }

    if top > 0 {
        println!("{table}");
    }
    if scorer != esm_cmd::Scorer::Structure {
        println!(
            "Ranked by {:?} (ESM-2 {} marginals, {})",
            scorer,
            if esm.masked { "masked" } else { "wild-type" },
            esm.model
        );
    }

    if let Some(winner) = candidates.first() {
        println!(
            "\n{} best: {} {} fitness {:.1}\n  {} proteus view {}",
            a.paint(Role::Accent, "✓"),
            winner.header,
            a.paint(Role::Dim, "·"),
            winner.fitness,
            a.paint(Role::Dim, "~ $"),
            job_ref::short(winner.job_id)
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
                heavy_atom_overlap_score: c.heavy_atom_overlap_score,
                hbond_count: c.hbond_count,
                salt_bridge_count: c.salt_bridge_count,
                pi_stacking_count: c.pi_stacking_count,
                cation_pi_count: c.cation_pi_count,
                fitness: c.fitness,
                esm2_score: c.esm2_score.map(f64::from),
                engine: c.engine.clone(),
            });
        }
        proteus_storage::save_screening_dataset(&records, &export_path)
            .await
            .with_context(|| format!("Failed to export dataset to {:?}", export_path))?;
        eprintln!(
            "wrote {} ranked candidates to {}",
            records.len(),
            export_path.display()
        );
    }
    Ok(())
}
