//! `proteus analyze` — All-atom biophysics of a structure file, with no database involved.

use super::prelude::*;

/// Arguments of `proteus analyze`.
#[derive(clap::Args, Debug)]
pub struct Args {
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
}

pub async fn run(args: Args) -> Result<()> {
    let Args {
        pdb,
        reference,
        confidence_source,
    } = args;
    println!("Analyzing structure file: {:?}", pdb);
    let mut metrics =
        analyze_pdb_file(&pdb, reference.as_deref()).context("Biophysical analysis failed")?;
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
    table.load_style(UTF8_FULL);
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
                Cell::new("n/a (experimental structure; B-factor column is not a confidence)"),
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

    if let Some(ref clash) = metrics.steric_overlap {
        table.add_row(vec![
            Cell::new("Heavy-atom steric overlap (>0.4 Å, no H)"),
            Cell::new(format!(
                "{:.1} per 1k atoms ({} overlaps)",
                clash.heavy_atom_overlap_score, clash.clash_count
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
    Ok(())
}
