//! Terminal report for a finished job: the table `proteus inspect` prints.

use anyhow::Result;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};
use proteus_storage::repository::ProteusRepository;
use uuid::Uuid;

pub async fn print_job_inspection(repo: &ProteusRepository, job_id: Uuid) -> Result<()> {
    let Some(pred) = repo.get_prediction_by_job(job_id).await? else {
        // No prediction yet: say where the job is, and why it stopped if it failed.
        return match repo.get_job(job_id).await? {
            Some(job) => Err(anyhow::anyhow!(
                "job {job_id} has no prediction to inspect: it is {:?}{}",
                job.status,
                job.error_log.map(|e| format!(" ({e})")).unwrap_or_default()
            )),
            None => Err(anyhow::anyhow!("no job {job_id}")),
        };
    };

    let metrics = repo
        .get_metrics_by_prediction(pred.id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Metrics for prediction {} not found", pred.id))?;

    let mut table = Table::new();
    table.load_style(UTF8_FULL);
    table.set_header(vec!["Metric", "Value", "Confidence Assessment"]);

    table.add_row(vec![
        Cell::new("Predicted PDB Path"),
        Cell::new(&pred.pdb_path),
        Cell::new("Artifact on disk"),
    ]);

    let engine = proteus_engine::engine_name(pred.metadata.as_ref());
    table.add_row(vec![
        Cell::new("Engine"),
        Cell::new(engine),
        Cell::new(if engine == proteus_engine::ENGINE_SIMULATED {
            "SIMULATED: synthetic helix, not a prediction"
        } else {
            "Structure source"
        }),
    ]);

    let tier_requested = match repo.get_job(job_id).await? {
        Some(job) => job.tier_slug().to_string(),
        None => "unknown".to_string(),
    };
    let tier_tracked = pred
        .metadata
        .as_ref()
        .and_then(|m| m.get("tier_honoured"))
        .is_some();
    let (tier_cell, tier_note) = match proteus_engine::tier_downgrade(pred.metadata.as_ref()) {
        Some(d) => (
            format!("{tier_requested} (NOT honoured)"),
            format!("DOWNGRADED: {}", d.reason),
        ),
        None if tier_tracked => (tier_requested, "Ran at the requested tier".to_string()),
        // Explicit --runner: the user picked the engine, so no downgrade is recorded.
        None => (tier_requested, "Runner chosen explicitly".to_string()),
    };
    table.add_row(vec![
        Cell::new("Tier"),
        Cell::new(tier_cell),
        Cell::new(tier_note),
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
            Cell::new("DSSP (Kabsch–Sander) assignment"),
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
