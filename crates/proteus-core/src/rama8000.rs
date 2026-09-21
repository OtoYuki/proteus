//! MolProbity Top8000 Ramachandran evaluation.
//!
//! Data and semantics come from cctbx (`mmtbx/validation/ramachandran/rama_eval.h`,
//! `mmtbx/validation/ramalyze.py`); see `data/rama8000/NOTICE` for provenance and licence.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::structure::RamachandranRegion;

const GRID: usize = 180;
const FAVORED_THRESHOLD: f64 = 0.02;

/// Residue class selecting one of the six Top8000 distributions (ramalyze precedence:
/// Gly, then Pro by ω, then pre-Pro, then Ile/Val, else general).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RamaClass {
    General,
    Glycine,
    CisPro,
    TransPro,
    PrePro,
    IleVal,
}

impl RamaClass {
    /// Select the class from the residue name, the following residue's name (when
    /// connected by a peptide bond) and ω (cis when |ω| < 90°, as in ramalyze).
    pub fn classify(name: &str, next_name: Option<&str>, omega: Option<f64>) -> RamaClass {
        match name {
            "GLY" => RamaClass::Glycine,
            "PRO" => match omega {
                Some(w) if w.abs() < 90.0 => RamaClass::CisPro,
                _ => RamaClass::TransPro,
            },
            _ if next_name == Some("PRO") => RamaClass::PrePro,
            "ILE" | "VAL" => RamaClass::IleVal,
            _ => RamaClass::General,
        }
    }

    fn table(self) -> &'static [f32] {
        static TABLES: OnceLock<[Vec<f32>; 6]> = OnceLock::new();
        let t = TABLES.get_or_init(|| {
            [
                decode(include_bytes!("../data/rama8000/general.f32")),
                decode(include_bytes!("../data/rama8000/glycine.f32")),
                decode(include_bytes!("../data/rama8000/cispro.f32")),
                decode(include_bytes!("../data/rama8000/transpro.f32")),
                decode(include_bytes!("../data/rama8000/prepro.f32")),
                decode(include_bytes!("../data/rama8000/ileval.f32")),
            ]
        });
        &t[self as usize]
    }

    fn allowed_threshold(self) -> f64 {
        match self {
            RamaClass::General => 0.0005,
            RamaClass::CisPro => 0.0020,
            _ => 0.0010,
        }
    }
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len(),
        GRID * GRID * 4,
        "rama8000 grid must be 180x180 f32"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn wrap(mut v: f64) -> f64 {
    while v > 180.0 {
        v -= 360.0;
    }
    while v < -180.0 {
        v += 360.0;
    }
    v
}

/// Neighbouring odd-degree bin centres around `v` and their bin indices
/// (cctbx `get_bins_and_values`).
fn bins(v: f64) -> (usize, usize, f64, f64) {
    let mut lower = v.floor();
    if (lower as i64) % 2 == 0 {
        lower -= 1.0;
    }
    let mut higher = v.ceil();
    if (higher as i64) % 2 == 0 {
        higher += 1.0;
    }
    if lower == higher {
        higher += 2.0;
    }
    (bin_index(lower), bin_index(higher), lower, higher)
}

fn bin_index(v: f64) -> usize {
    let mut b = ((v + 179.0) / 2.0) as i64;
    if b > 179 {
        b -= 180;
    }
    if b < 0 {
        b += 180;
    }
    b as usize
}

/// Interpolated Top8000 density at (φ, ψ) for the given class, in [0, 1].
pub fn score(class: RamaClass, phi: f64, psi: f64) -> f64 {
    let t = class.table();
    let (p, s) = (wrap(phi), wrap(psi));
    let (p0, p1, x1, x2) = bins(p);
    let (s0, s1, y1, y2) = bins(s);
    let v = |a: usize, b: usize| t[a * GRID + b] as f64;
    // cctbx linear_interpolation_2d(x1, y1, x2, y2, v1=(x1,y1), v2=(x2,y2), v3=(x1,y2), v4=(x2,y1), x, y)
    let (v11, v22, v12, v21) = (v(p0, s0), v(p1, s1), v(p0, s1), v(p1, s0));
    let fx = (p - x1) / (x2 - x1);
    let fy = (s - y1) / (y2 - y1);
    (1.0 - fx) * (1.0 - fy) * v11 + fx * fy * v22 + (1.0 - fx) * fy * v12 + fx * (1.0 - fy) * v21
}

/// MolProbity classification: favored ≥ 2 %, allowed ≥ 0.05 % (general), ≥ 0.2 % (cis-Pro),
/// ≥ 0.1 % (all other classes), otherwise outlier.
pub fn evaluate(class: RamaClass, phi: f64, psi: f64) -> RamachandranRegion {
    let s = score(class, phi, psi);
    if s >= FAVORED_THRESHOLD {
        RamachandranRegion::Favored
    } else if s >= class.allowed_threshold() {
        RamachandranRegion::Allowed
    } else {
        RamachandranRegion::Outlier
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::RamachandranRegion as R;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    /// Oracle values printed by cctbx `RamachandranEval().evaluate(cls, [phi, psi])`.
    #[test]
    fn scores_match_cctbx_rama_eval() {
        assert!(close(score(RamaClass::General, -63.0, -42.0), 0.9973580));
        assert!(close(score(RamaClass::General, -120.0, 130.0), 0.5453480));
        assert!(close(score(RamaClass::General, 60.0, 60.0), 0.0225505));
        assert!(close(score(RamaClass::Glycine, 80.0, -170.0), 0.5347310));
        assert!(close(score(RamaClass::TransPro, -65.0, 145.0), 0.8440870));
        assert!(close(score(RamaClass::PrePro, -80.0, 80.0), 0.0000244));
        assert!(close(score(RamaClass::IleVal, -110.0, 125.0), 0.6720705));
        assert!(close(score(RamaClass::General, -179.5, 179.5), 0.0060123));
        assert!(close(score(RamaClass::General, 0.0, 0.0), 0.0));
        assert!(close(score(RamaClass::TransPro, 60.0, 60.0), 0.0));
    }

    #[test]
    fn regions_follow_cctbx_thresholds() {
        assert_eq!(evaluate(RamaClass::General, -63.0, -42.0), R::Favored);
        assert_eq!(evaluate(RamaClass::General, -120.0, 130.0), R::Favored);
        assert_eq!(evaluate(RamaClass::General, 60.0, 60.0), R::Favored);
        assert_eq!(evaluate(RamaClass::General, -179.5, 179.5), R::Allowed);
        assert_eq!(evaluate(RamaClass::General, 0.0, 0.0), R::Outlier);
        assert_eq!(evaluate(RamaClass::Glycine, 80.0, -170.0), R::Favored);
        assert_eq!(evaluate(RamaClass::TransPro, 60.0, 60.0), R::Outlier);
        assert_eq!(evaluate(RamaClass::PrePro, -80.0, 80.0), R::Outlier);
    }

    #[test]
    fn class_selection() {
        assert_eq!(
            RamaClass::classify("ALA", Some("PRO"), Some(180.0)),
            RamaClass::PrePro
        );
        assert_eq!(
            RamaClass::classify("GLY", Some("PRO"), Some(180.0)),
            RamaClass::Glycine
        );
        assert_eq!(
            RamaClass::classify("PRO", Some("ALA"), Some(178.0)),
            RamaClass::TransPro
        );
        assert_eq!(
            RamaClass::classify("PRO", Some("ALA"), Some(-5.0)),
            RamaClass::CisPro
        );
        assert_eq!(
            RamaClass::classify("PRO", Some("PRO"), None),
            RamaClass::TransPro
        );
        assert_eq!(RamaClass::classify("ILE", None, None), RamaClass::IleVal);
        assert_eq!(
            RamaClass::classify("VAL", Some("PRO"), None),
            RamaClass::PrePro
        );
        assert_eq!(RamaClass::classify("XYZ", None, None), RamaClass::General);
    }

    #[test]
    fn crambin_is_almost_all_favored() {
        let (pdb, _) = pdbtbx::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"),
            pdbtbx::StrictnessLevel::Loose,
        )
        .unwrap();
        let d = crate::metrics::analyze_pdb_detailed(&pdb, None).unwrap();
        let s = d.metrics.ramachandran_stats.unwrap();
        // cctbx ramalyze on 1CRN: 44 evaluated, 43 favored, 1 allowed, 0 outliers.
        assert_eq!(s.total_evaluated, 44);
        assert_eq!(s.outlier_count, 0, "{s:?}");
        assert!((s.favored_fraction - 43.0 / 44.0).abs() < 1e-9, "{s:?}");
        assert!((s.allowed_fraction - 1.0 / 44.0).abs() < 1e-9, "{s:?}");
    }
}
