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

/// Whether `word` occurs in `text` as a whole token: not preceded or followed by a letter or
/// digit. "NMR" must match `SOLUTION NMR` but not `NMRAL1` or `NmrA-like` (protein names in
/// AlphaFold DB titles).
fn contains_word(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    text.match_indices(word).any(|(i, m)| {
        // A boundary is only required where the marker itself ends in a letter or digit:
        // `_REFINE.` must still match `_REFINE.LS_D_RES_HIGH`.
        let edge = |c: Option<u8>| c.is_some_and(|b| b.is_ascii_alphanumeric());
        let first = word.as_bytes().first().copied();
        let last = word.as_bytes().last().copied();
        let before = i.checked_sub(1).map(|j| bytes[j]);
        let after = bytes.get(i + m.len()).copied();
        !(edge(first) && edge(before)) && !(edge(last) && edge(after))
    })
}

/// The experimental method a file declares in a machine-readable field (`EXPDTA` or
/// `_exptl.method`), upper-cased, if any.
fn declared_method(header: &str) -> Option<String> {
    let mut lines = header.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("EXPDTA") {
            return Some(rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("_EXPTL.METHOD") {
            let inline = rest.trim().trim_matches(['\'', '"']).trim();
            if !inline.is_empty() {
                return Some(inline.to_string());
            }
            // Loop form: the first data row after the tags carries the value.
            for row in lines.by_ref() {
                if row.starts_with('_') || row.trim().is_empty() {
                    continue;
                }
                return Some(row.trim().to_string());
            }
        }
    }
    None
}

/// A `CRYST1` record with a real unit cell. Predicted models and NMR entries write the
/// placeholder `1.000 1.000 1.000`.
fn has_real_unit_cell(header: &str) -> bool {
    header.lines().filter(|l| l.starts_with("CRYST1")).any(|l| {
        l.get(6..33)
            .map(|cell| {
                cell.split_whitespace()
                    .filter_map(|v| v.parse::<f64>().ok())
                    .any(|v| v > 1.5)
            })
            .unwrap_or(false)
    })
}

/// Classify the confidence source from any available header text plus the per-residue
/// B-factor values, strongest evidence first:
///
/// 1. A declared experimental method (`EXPDTA`, `_exptl.method`) other than a theoretical
///    model means experimental; ModelCIF quality metrics (`_ma_qa_metric`) mean predicted.
/// 2. Method words and predictor names in the free text, matched as whole words.
/// 3. A real crystallographic unit cell means experimental.
/// 4. The values: pLDDT lies in [0, 100], or [0, 1] as ESMFold writes it — a scale no
///    B-factor column uses — and is mostly high; crystallographic B-factors average lower.
///
/// A header-less low-confidence prediction on the 0–100 scale (mean ≤ 30) is
/// indistinguishable from B-factors by value alone and is classed experimental; pass the
/// source explicitly (`--confidence-source predicted`) for such files.
pub fn detect_confidence_source(header: &str, b_factors: &[f64]) -> ConfidenceSource {
    let header = header.to_ascii_uppercase();
    if let Some(method) = declared_method(&header) {
        if !method.contains("THEORETICAL") && !method.contains("PREDICT") {
            return ConfidenceSource::ExperimentalBFactor;
        }
    }
    if header.contains("_MA_QA_METRIC") {
        return ConfidenceSource::Predicted;
    }
    if EXPERIMENTAL_MARKERS
        .iter()
        .any(|m| contains_word(&header, m))
    {
        return ConfidenceSource::ExperimentalBFactor;
    }
    if PREDICTED_MARKERS.iter().any(|m| contains_word(&header, m)) {
        return ConfidenceSource::Predicted;
    }
    if has_real_unit_cell(&header) {
        return ConfidenceSource::ExperimentalBFactor;
    }
    if b_factors.is_empty() {
        return ConfidenceSource::Unknown;
    }
    let n = b_factors.len() as f64;
    let max = b_factors.iter().copied().fold(f64::MIN, f64::max);
    let min = b_factors.iter().copied().fold(f64::MAX, f64::min);
    if max <= 1.0 && min >= 0.0 && max > 0.0 {
        // No crystallographic B-factor column is confined to [0, 1].
        return ConfidenceSource::Predicted;
    }
    let mean = b_factors.iter().sum::<f64>() / n;
    let var = b_factors.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let in_range = b_factors.iter().all(|v| (0.0..=100.0).contains(v));
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
    #[test]
    fn protein_names_that_contain_a_method_word_do_not_decide_it() {
        // Reported: AF-Q9HBL8 (NmrA-like redox sensor, NMRAL1) was labelled experimental
        // because "NMR" matched inside the protein name, hiding a mean pLDDT of 96.6.
        let header = "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR NMRA-LIKE FAMILY DOMAIN-CONTAINING PROTEIN 1 (Q9HBL8)\nNMRK1";
        assert_eq!(
            detect_confidence_source(header, &[96.0, 97.0, 95.5]),
            ConfidenceSource::Predicted
        );
        assert_eq!(
            detect_confidence_source("EXPDTA    SOLUTION NMR", &[80.0]),
            ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn machine_readable_method_beats_everything_else() {
        // Reported: `_exptl.method` sits 70-210 KB into deposited mmCIF, past the header
        // preview, so an X-ray entry with high B-factors was read as a prediction.
        let loop_form = "LOOP_\n_EXPTL.ENTRY_ID\n_EXPTL.METHOD\n1ABC 'X-RAY DIFFRACTION'\n";
        assert_eq!(
            detect_confidence_source(loop_form, &[60.0, 70.0, 80.0]),
            ConfidenceSource::ExperimentalBFactor
        );
        assert_eq!(
            detect_confidence_source("_exptl.method   'ELECTRON MICROSCOPY'", &[60.0, 70.0]),
            ConfidenceSource::ExperimentalBFactor
        );
        assert_eq!(
            detect_confidence_source("_ma_qa_metric.id", &[20.0, 25.0]),
            ConfidenceSource::Predicted
        );
        assert_eq!(
            detect_confidence_source(
                "CRYST1   59.062   68.451   30.517  90.00  90.00  90.00 P 21 21 21",
                &[60.0, 70.0]
            ),
            ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn a_unit_interval_column_is_plddt_whatever_its_mean() {
        // Reported: a header-less ESMFold model with mean pLDDT 0.25 was classed experimental,
        // which dropped the pLDDT term and raised its score.
        assert_eq!(
            detect_confidence_source("", &[0.2, 0.25, 0.3]),
            ConfidenceSource::Predicted
        );
    }
}
