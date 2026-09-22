//! Decide whether a structure's B-factor column carries a predictor's pLDDT or
//! crystallographic/NMR displacement parameters.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSource {
    /// B-factor column holds pLDDT (AlphaFold, ESMFold, Boltz, OpenFold, …).
    Predicted,
    /// Experimental structure: the B-factor column is a displacement parameter, not a confidence.
    ExperimentalBFactor,
    #[default]
    Unknown,
}

impl ConfidenceSource {
    pub fn is_predicted(self) -> bool {
        matches!(self, ConfidenceSource::Predicted)
    }
}

/// Header substrings (upper-cased) that identify an experimental structure. `RESOLUTION.` and
/// `REFINEMENT` appear in REMARK 2/3 of every deposited X-ray entry, which pdbtbx retains
/// even though it drops the EXPDTA record.
const EXPERIMENTAL_MARKERS: &[&str] = &[
    "X-RAY",
    "DIFFRACTION",
    "NMR",
    "ELECTRON MICROSCOPY",
    "ELECTRON CRYSTALLOGRAPHY",
    "NEUTRON",
    "FIBER",
    "RESOLUTION.",
    "REFINEMENT",
    "R VALUE",
    "_REFINE.",
    "_EXPTL.METHOD",
];

/// Header substrings (upper-cased) that identify a predicted model.
const PREDICTED_MARKERS: &[&str] = &[
    "ALPHAFOLD",
    "ESMFOLD",
    "ESM-FOLD",
    "BOLTZ",
    "OPENFOLD",
    "COLABFOLD",
    "CHAI-1",
    "PROTENIX",
    "ROSETTAFOLD",
    "PLDDT",
    "_MA_QA_METRIC",
    "PREDICTED",
];

/// Header text pdbtbx retains: the identifier and REMARK lines.
pub fn pdb_header_text(pdb: &pdbtbx::PDB) -> String {
    let mut s = String::new();
    if let Some(id) = pdb.identifier.as_deref() {
        s.push_str(id);
        s.push('\n');
    }
    for (number, line) in pdb.remarks() {
        s.push_str(&format!("REMARK {number} {line}\n"));
    }
    s
}

/// Classify the confidence source from any available header text plus the per-residue
/// B-factor values. Predicted markers win over experimental markers only when no
/// experimental marker is present, because AlphaFold-DB headers never mention refinement
/// while deposited entries never mention pLDDT.
pub fn detect_confidence_source(header: &str, b_factors: &[f64]) -> ConfidenceSource {
    let header = header.to_ascii_uppercase();
    if EXPERIMENTAL_MARKERS.iter().any(|m| header.contains(m)) {
        return ConfidenceSource::ExperimentalBFactor;
    }
    if PREDICTED_MARKERS.iter().any(|m| header.contains(m)) {
        return ConfidenceSource::Predicted;
    }
    if b_factors.is_empty() {
        return ConfidenceSource::Unknown;
    }
    // Value heuristic: pLDDT lives in [0, 100] (or [0, 1] for raw ESMFold), is mostly high and
    // has a modest spread; crystallographic B-factors are usually < 30 on average.
    let n = b_factors.len() as f64;
    let max = b_factors.iter().copied().fold(f64::MIN, f64::max);
    let scale = if max <= 1.0 && max > 0.0 { 100.0 } else { 1.0 };
    let vals: Vec<f64> = b_factors.iter().map(|v| v * scale).collect();
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let in_range = vals.iter().all(|v| (0.0..=100.0).contains(v));
    if in_range && mean > 30.0 && var.sqrt() < 40.0 {
        ConfidenceSource::Predicted
    } else {
        ConfidenceSource::ExperimentalBFactor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crambin_is_experimental() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
            .unwrap();
        let b: Vec<f64> = pdb
            .atoms()
            .filter(|a| a.name() == "CA")
            .map(|a| a.b_factor())
            .collect();
        assert_eq!(
            detect_confidence_source(&pdb_header_text(&pdb), &b),
            ConfidenceSource::ExperimentalBFactor
        );
        // Even with no header, crambin's B-factors (mean ≈ 5.8) are not pLDDT-like.
        assert_eq!(
            detect_confidence_source("", &b),
            ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn expdta_xray_wins_over_values() {
        assert_eq!(
            detect_confidence_source("EXPDTA    X-RAY DIFFRACTION", &[85.0, 90.0, 92.0]),
            ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn alphafold_title_is_predicted() {
        assert_eq!(
            detect_confidence_source(
                "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR P69905",
                &[85.0, 90.0, 92.0]
            ),
            ConfidenceSource::Predicted
        );
    }

    #[test]
    fn headerless_plddt_like_values_are_predicted() {
        assert_eq!(
            detect_confidence_source("REMARK 1 NOTHING USEFUL", &[71.0, 88.5, 93.2, 60.4]),
            ConfidenceSource::Predicted
        );
        assert_eq!(
            detect_confidence_source("", &[0.71, 0.885, 0.932, 0.604]),
            ConfidenceSource::Predicted
        );
    }

    #[test]
    fn headerless_bfactor_like_values_are_experimental() {
        assert_eq!(
            detect_confidence_source("REMARK 1 NOTHING USEFUL", &[5.0, 8.1, 12.4, 6.6]),
            ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn no_information_is_unknown() {
        assert_eq!(detect_confidence_source("", &[]), ConfidenceSource::Unknown);
    }
}
