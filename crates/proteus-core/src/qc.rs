//! One flat row of quality metrics per structure file, for triaging many models at once.
//!
//! [`crate::metrics::analyze_pdb_file`] returns the full nested report for one structure;
//! [`structure_qc`] flattens the same numbers into a single record with the identifying
//! columns (file, sequence, chain and residue counts) a table of thousands of models needs.

use crate::confidence::ConfidenceSource;
use crate::error::CoreError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Empirical radius of gyration of a folded, globular protein of `n` residues, in Å
/// (Rg ≈ 2.2·N^0.38; the same law the fitness score's compactness term uses).
pub fn expected_folded_rg(n: usize) -> f64 {
    2.2 * (n as f64).powf(0.38)
}

/// Flat quality metrics of one structure file. Fractions are reported as percentages.
/// pLDDT columns are `None` unless the B-factor column was detected (or declared) to be a
/// predicted confidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructureQc {
    /// Path as given on the command line (or as found while walking a directory).
    pub file: String,
    /// File name without its structure and compression extensions.
    pub model: String,
    pub n_chains: usize,
    /// Protein residues with a C-alpha atom, over all chains.
    pub n_residues: usize,
    /// One-letter sequence, chains separated by `/`. Non-standard residues are `X`.
    pub sequence: String,
    /// `predicted`, `experimental` or `unknown`.
    pub confidence_source: String,
    pub plddt_mean: Option<f64>,
    pub plddt_median: Option<f64>,
    pub plddt_ge70_pct: Option<f64>,
    pub plddt_ge90_pct: Option<f64>,
    /// C-alpha radius of gyration, Å, over every chain in the file.
    pub rg: f64,
    /// [`expected_folded_rg`] for `n_residues`.
    pub rg_expected: f64,
    /// `rg / rg_expected`: ≈ 1 for a compact single domain, well above 1 for extended or
    /// unfolded models (or multi-domain / multi-chain assemblies).
    pub rg_ratio: f64,
    pub helix_pct: f64,
    pub strand_pct: f64,
    pub coil_pct: f64,
    /// Eight-state DSSP string, one character per residue.
    pub dssp: String,
    pub rama_favored_pct: f64,
    pub rama_allowed_pct: f64,
    pub rama_outliers: usize,
    pub sasa_total: f64,
    pub hydrophobic_burial_pct: f64,
    /// Severe heavy-atom overlaps (> 0.40 Å) per 1000 atoms. Not a MolProbity clashscore.
    pub heavy_atom_overlap_score: f64,
    pub overlap_count: usize,
    pub hbond_count: usize,
    pub salt_bridge_count: usize,
    pub pi_stacking_count: usize,
    pub cation_pi_count: usize,
    /// Kabsch C-alpha RMSD to the reference structure, when one was given.
    pub rmsd_to_reference: Option<f64>,
    /// Composite triage score, 0–100 (see the README for its definition and its limits).
    pub fitness: f64,
}

/// `name.pdb.gz` → `name`, `x.cif` → `x`, `model_0.ent` → `model_0`.
pub fn model_name(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut stem = name.as_str();
    if let Some(s) = stem.strip_suffix(".gz") {
        stem = s;
    }
    for ext in [".pdb", ".ent", ".cif", ".mmcif"] {
        if let Some(s) = stem.strip_suffix(ext) {
            stem = s;
            break;
        }
    }
    stem.to_string()
}

/// Whether `path` names a structure file by extension (`.pdb`, `.ent`, `.cif`, `.mmcif`,
/// each optionally gzipped). Used when walking a directory; explicit file arguments are
/// accepted whatever their name and sniffed by content.
pub fn is_structure_file_name(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let name = name.strip_suffix(".gz").unwrap_or(&name);
    [".pdb", ".ent", ".cif", ".mmcif"]
        .iter()
        .any(|ext| name.ends_with(ext))
}

fn one_letter(three: &str) -> char {
    match three {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        "SEC" => 'U',
        "PYL" => 'O',
        "MSE" => 'M',
        _ => 'X',
    }
}

/// Analyse one structure file into a [`StructureQc`] row.
///
/// `confidence` overrides the automatic pLDDT-vs-B-factor detection when `Some`.
pub fn structure_qc(
    path: &Path,
    reference: Option<&pdbtbx::PDB>,
    confidence: Option<ConfidenceSource>,
) -> Result<StructureQc, CoreError> {
    let loaded = crate::io::load_structure(path)?;
    let detailed = crate::metrics::analyze_pdb_detailed_with_source(
        &loaded.pdb,
        reference,
        Some(&loaded.header_preview),
        confidence,
    )?;
    let m = detailed.metrics;

    let protein = crate::io::protein_heavy_atoms(&loaded.pdb);
    let backbone = crate::backbone::extract_backbone(&protein);
    let mut sequence = String::with_capacity(backbone.len() + 4);
    let mut chains: Vec<&str> = Vec::new();
    let mut n_residues = 0;
    for r in backbone.iter().filter(|r| r.ca.is_some()) {
        if chains.last() != Some(&r.chain_id.as_str()) {
            if !chains.is_empty() {
                sequence.push('/');
            }
            chains.push(&r.chain_id);
        }
        sequence.push(one_letter(&r.name));
        n_residues += 1;
    }
    chains.sort_unstable();
    chains.dedup();

    let plddt = m.plddt().cloned();
    let ss = m.secondary_structure_summary.as_ref();
    let rama = m.ramachandran_stats.as_ref();
    let sasa = m.sasa_metrics.as_ref();
    let overlap = m.steric_overlap.as_ref();
    let net = m.interaction_network.as_ref().map(|n| &n.summary);
    let rg_expected = expected_folded_rg(n_residues);

    Ok(StructureQc {
        file: path.display().to_string(),
        model: model_name(path),
        n_chains: chains.len(),
        n_residues,
        sequence,
        confidence_source: match m.confidence_source {
            ConfidenceSource::Predicted => "predicted",
            ConfidenceSource::ExperimentalBFactor => "experimental",
            ConfidenceSource::Unknown => "unknown",
        }
        .to_string(),
        plddt_mean: plddt.as_ref().map(|p| p.mean),
        plddt_median: plddt.as_ref().map(|p| p.median),
        plddt_ge70_pct: plddt.as_ref().map(|p| p.high_confidence_fraction * 100.0),
        plddt_ge90_pct: plddt
            .as_ref()
            .map(|p| p.very_high_confidence_fraction * 100.0),
        rg: m.radius_of_gyration,
        rg_expected,
        rg_ratio: if rg_expected > 0.0 {
            m.radius_of_gyration / rg_expected
        } else {
            0.0
        },
        helix_pct: ss.map_or(0.0, |s| s.helix_fraction * 100.0),
        strand_pct: ss.map_or(0.0, |s| s.strand_fraction * 100.0),
        coil_pct: ss.map_or(0.0, |s| s.coil_fraction * 100.0),
        dssp: ss.map(|s| s.dssp.clone()).unwrap_or_default(),
        rama_favored_pct: rama.map_or(0.0, |r| r.favored_fraction * 100.0),
        rama_allowed_pct: rama.map_or(0.0, |r| r.allowed_fraction * 100.0),
        rama_outliers: rama.map_or(0, |r| r.outlier_count),
        sasa_total: sasa.map_or(0.0, |s| s.total_sasa),
        hydrophobic_burial_pct: sasa.map_or(0.0, |s| s.hydrophobic_burial_ratio * 100.0),
        heavy_atom_overlap_score: overlap.map_or(0.0, |o| o.heavy_atom_overlap_score),
        overlap_count: overlap.map_or(0, |o| o.clash_count),
        hbond_count: net.map_or(0, |n| n.total_hbonds),
        salt_bridge_count: net.map_or(0, |n| n.total_salt_bridges),
        pi_stacking_count: net.map_or(0, |n| n.total_pi_pi_stacks),
        cation_pi_count: net.map_or(0, |n| n.total_cation_pi),
        rmsd_to_reference: m.rmsd_to_reference,
        fitness: m.candidate_fitness_score.unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name)
    }

    #[test]
    fn model_name_strips_structure_and_compression_extensions() {
        assert_eq!(model_name(Path::new("a/b/design_07.pdb.gz")), "design_07");
        assert_eq!(model_name(Path::new("x.cif")), "x");
        assert_eq!(model_name(Path::new("x.mmcif.gz")), "x");
        assert_eq!(model_name(Path::new("pdb1crn.ent")), "pdb1crn");
        assert_eq!(model_name(Path::new("notes.txt")), "notes.txt");
    }

    #[test]
    fn directory_walk_accepts_structure_extensions_only() {
        for ok in ["a.pdb", "a.PDB", "a.cif.gz", "a.mmcif", "a.ent.gz"] {
            assert!(is_structure_file_name(Path::new(ok)), "{ok}");
        }
        for no in ["a.fasta", "a.json", "a.gz", "pdb", "a.pdbqt"] {
            assert!(!is_structure_file_name(Path::new(no)), "{no}");
        }
    }

    #[test]
    fn crambin_row_matches_the_full_report() {
        let path = data("1crn.pdb");
        let qc = structure_qc(&path, None, None).unwrap();
        let full = crate::metrics::analyze_pdb_file(&path, None).unwrap();

        assert_eq!(qc.model, "1crn");
        assert_eq!(qc.n_chains, 1);
        assert_eq!(qc.n_residues, 46);
        assert_eq!(
            qc.sequence,
            "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
        );
        assert_eq!(qc.confidence_source, "experimental");
        assert_eq!(qc.plddt_mean, None, "an X-ray B-factor is not a pLDDT");
        assert_eq!(qc.dssp.chars().count(), 46);
        assert!((qc.rg - full.radius_of_gyration).abs() < 1e-12);
        assert_eq!(qc.fitness, full.candidate_fitness_score.unwrap());
        assert_eq!(
            qc.hbond_count,
            full.interaction_network.unwrap().summary.total_hbonds
        );
        // A compact globular protein sits near the folded-protein law.
        assert!(
            (0.8..1.2).contains(&qc.rg_ratio),
            "rg_ratio {}",
            qc.rg_ratio
        );
    }

    #[test]
    fn declaring_the_confidence_source_exposes_or_hides_plddt() {
        let path = data("1crn.pdb");
        let forced = structure_qc(&path, None, Some(ConfidenceSource::Predicted)).unwrap();
        assert_eq!(forced.confidence_source, "predicted");
        assert!(forced.plddt_mean.is_some());
        let auto = structure_qc(&path, None, None).unwrap();
        assert_ne!(
            forced.fitness, auto.fitness,
            "pLDDT weight enters the score"
        );
    }

    #[test]
    fn a_declared_prediction_gets_the_same_rescaling_as_a_detected_one() {
        // Reported: forcing `predicted` on an ESMFold-style 0-1 file skipped the 0-1 -> 0-100
        // rescale, reporting pLDDT 0.25 and scoring the model as if it were near zero.
        let path = data("edge/esm01_low.pdb");
        let forced = structure_qc(&path, None, Some(ConfidenceSource::Predicted)).unwrap();
        let detected = structure_qc(&path, None, None).unwrap();
        assert_eq!(detected.confidence_source, "predicted");
        assert!(
            (forced.plddt_mean.unwrap() - 25.0).abs() < 1e-9,
            "{forced:?}"
        );
        assert_eq!(forced.plddt_mean, detected.plddt_mean);
        assert_eq!(forced.fitness, detected.fitness);
    }

    #[test]
    fn a_method_declared_late_in_an_mmcif_still_counts() {
        // Reported: `_exptl.method 'X-RAY DIFFRACTION'` beyond the first 16 KiB was never
        // read, so an X-ray entry with B-factors of 40-80 was reported as pLDDT.
        let qc = structure_qc(&data("edge/xray_late_exptl.cif"), None, None).unwrap();
        assert_eq!(qc.confidence_source, "experimental");
        assert_eq!(qc.plddt_mean, None);
    }

    #[test]
    fn median_plddt_of_an_even_count_is_the_mean_of_the_middle_two() {
        // Reported: B-factors 50/60/90/95 gave a median of 90, not 75.
        let mut text = String::from("TITLE     ALPHAFOLD PREDICTION\n");
        for (i, b) in [50.0, 60.0, 90.0, 95.0].iter().enumerate() {
            text.push_str(&format!(
                "ATOM  {:>5}  CA  GLY A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}           C\n",
                i + 1,
                i + 1,
                3.8 * i as f64,
                0.0,
                0.0,
                b
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.pdb");
        std::fs::write(&path, text).unwrap();
        let qc = structure_qc(&path, None, None).unwrap();
        assert_eq!(qc.plddt_median, Some(75.0));
    }

    #[test]
    fn chains_are_counted_and_separated_in_the_sequence() {
        let mut text = String::new();
        let mut serial = 1;
        for (chain, x0) in [("A", 0.0), ("B", 30.0)] {
            for i in 0..3 {
                let x = x0 + 3.8 * i as f64;
                text.push_str(&format!(
                    "ATOM  {serial:>5}  CA  GLY {chain}{:>4}    {x:>8.3}{:>8.3}{:>8.3}  1.00 90.00           C\n",
                    i + 1,
                    0.0,
                    0.0
                ));
                serial += 1;
            }
        }
        text.push_str("END\n");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dimer.pdb");
        std::fs::write(&path, text).unwrap();
        let qc = structure_qc(&path, None, None).unwrap();
        assert_eq!(qc.n_chains, 2);
        assert_eq!(qc.n_residues, 6);
        assert_eq!(qc.sequence, "GGG/GGG");
    }
}
