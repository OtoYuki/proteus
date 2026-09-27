//! MolProbity Top8000 side-chain rotamer evaluation (a port of cctbx `rotalyze`).
//!
//! Semantics follow cctbx exactly, file by file:
//!
//! * which residues are evaluated and how their χ angles are measured:
//!   `mmtbx/validation/rotalyze.py` and `mmtbx/rotamer/sidechain_angles.py` with the atom
//!   definitions of `mmtbx/rotamer/sidechain_angles.props`;
//! * which contour grid each residue uses and how it is interpolated:
//!   `mmtbx/rotamer/rotamer_eval.py` (`aminoAcids`, `RotamerEval.evaluate`) and
//!   `mmtbx/rotamer/n_dim_table.py` (`NDimTable.valueAt`);
//! * rotamer names: `mmtbx/rotamer/rotamer_eval.py` (`RotamerID`, `NamedRot`) with the boxes
//!   of `mmtbx/rotamer/rotamer_names.props`.
//!
//! See `data/rota8000/NOTICE` for data provenance and licences.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// `rotalyze.py` `ALLOWED_THRESHOLD`: a rotamer is favored when its grid value is ≥ 2 %.
const ALLOWED_THRESHOLD: f64 = 0.02;
/// `rotalyze.py` `OUTLIER_THRESHOLD`: below 0.3 % it is an outlier, in between allowed.
const OUTLIER_THRESHOLD: f64 = 0.003;

/// MolProbity rotamer class (`rotalyze.evaluateScore`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RotamerEvaluation {
    Favored,
    Allowed,
    Outlier,
}

impl RotamerEvaluation {
    /// Classify a Top8000 grid value (a fraction, not a percentage): favored ≥ 0.02,
    /// allowed ≥ 0.003, otherwise outlier. Both bounds are inclusive, as in cctbx.
    pub fn from_value(value: f64) -> Self {
        if value >= ALLOWED_THRESHOLD {
            RotamerEvaluation::Favored
        } else if value >= OUTLIER_THRESHOLD {
            RotamerEvaluation::Allowed
        } else {
            RotamerEvaluation::Outlier
        }
    }
}

/// One evaluated side chain, as a row of cctbx `rotalyze(outliers_only=False).results`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RotamerResult {
    pub chain_id: String,
    pub seq_num: isize,
    pub insertion_code: Option<String>,
    pub resname: String,
    /// χ1, χ2, … in degrees on [0, 360), one per χ the residue defines (Pro has three). This is
    /// cctbx `chi_angles` (`RotamerID.wrap_chis(symmetry=False)`), so the last χ of
    /// Asp/Glu/Phe/Tyr is *not* folded onto [0, 180) here.
    pub chi: Vec<f64>,
    /// Interpolated Top8000 grid value in percent (cctbx `score`).
    pub score: f64,
    pub evaluation: RotamerEvaluation,
    /// cctbx `rotamer_name`: `"OUTLIER"` for outliers, otherwise the first box of
    /// `rotamer_names.props` containing the χ angles (e.g. `"mt"`, `"tpp-160"`, `"Cg_endo"`),
    /// or `"UNCLASSIFIED"` when none does. Always `Some`.
    pub name: Option<String>,
}

/// χ definitions from `sidechain_angles.props`: for each χ the four atom names. MET/MSE list
/// two spellings of SD/SE (` SD ;SD  `), which are identical once the names are trimmed.
/// GLY and ALA are in the props file with no χ; `measureChiAngles` returns `[]` for them and
/// `RotamerEval.evaluate` has no table, so rotalyze skips them.
fn chi_atoms(resname: &str) -> Option<&'static [[&'static str; 4]]> {
    const N_CA_CB_CG: [&str; 4] = ["N", "CA", "CB", "CG"];
    const CA_CB_CG_CD: [&str; 4] = ["CA", "CB", "CG", "CD"];
    Some(match resname {
        "VAL" => &[["N", "CA", "CB", "CG1"]],
        "LEU" => &[N_CA_CB_CG, ["CA", "CB", "CG", "CD1"]],
        "ILE" => &[["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
        "PRO" => &[N_CA_CB_CG, CA_CB_CG_CD, ["CB", "CG", "CD", "N"]],
        "PHE" | "TYR" | "TRP" => &[N_CA_CB_CG, ["CA", "CB", "CG", "CD1"]],
        "SER" => &[["N", "CA", "CB", "OG"]],
        "THR" => &[["N", "CA", "CB", "OG1"]],
        "CYS" => &[["N", "CA", "CB", "SG"]],
        "MET" => &[
            N_CA_CB_CG,
            ["CA", "CB", "CG", "SD"],
            ["CB", "CG", "SD", "CE"],
        ],
        "MSE" => &[
            N_CA_CB_CG,
            ["CA", "CB", "CG", "SE"],
            ["CB", "CG", "SE", "CE"],
        ],
        "LYS" => &[
            N_CA_CB_CG,
            CA_CB_CG_CD,
            ["CB", "CG", "CD", "CE"],
            ["CG", "CD", "CE", "NZ"],
        ],
        "ARG" => &[
            N_CA_CB_CG,
            CA_CB_CG_CD,
            ["CB", "CG", "CD", "NE"],
            ["CG", "CD", "NE", "CZ"],
        ],
        "HIS" => &[N_CA_CB_CG, ["CA", "CB", "CG", "ND1"]],
        "ASP" | "ASN" => &[N_CA_CB_CG, ["CA", "CB", "CG", "OD1"]],
        "GLN" | "GLU" => &[N_CA_CB_CG, CA_CB_CG_CD, ["CB", "CG", "CD", "OE1"]],
        _ => return None,
    })
}

/// `rotamer_eval.aminoAcids`: the grid file for each residue (`rotalyze` maps MSE to MET).
/// Phe and Tyr share one table; Leu uses the pruned `leu` table (not `leu-raw`) and Pro the
/// 1-D χ1 table (not `pro3d`), so only χ1 of Pro is scored although all three are measured.
fn table_index(resname: &str) -> Option<usize> {
    Some(match resname {
        "ARG" => 0,
        "ASN" => 1,
        "ASP" => 2,
        "CYS" => 3,
        "GLN" => 4,
        "GLU" => 5,
        "HIS" => 6,
        "ILE" => 7,
        "LEU" => 8,
        "LYS" => 9,
        "MET" | "MSE" => 10,
        "PHE" | "TYR" => 11,
        "PRO" => 12,
        "SER" => 13,
        "THR" => 14,
        "TRP" => 15,
        "VAL" => 16,
        _ => return None,
    })
}

const TABLE_FILES: [&[u8]; 17] = [
    include_bytes!("../data/rota8000/arg.bin"),
    include_bytes!("../data/rota8000/asn.bin"),
    include_bytes!("../data/rota8000/asp.bin"),
    include_bytes!("../data/rota8000/cys.bin"),
    include_bytes!("../data/rota8000/gln.bin"),
    include_bytes!("../data/rota8000/glu.bin"),
    include_bytes!("../data/rota8000/his.bin"),
    include_bytes!("../data/rota8000/ile.bin"),
    include_bytes!("../data/rota8000/leu.bin"),
    include_bytes!("../data/rota8000/lys.bin"),
    include_bytes!("../data/rota8000/met.bin"),
    include_bytes!("../data/rota8000/phetyr.bin"),
    include_bytes!("../data/rota8000/pro.bin"),
    include_bytes!("../data/rota8000/ser.bin"),
    include_bytes!("../data/rota8000/thr.bin"),
    include_bytes!("../data/rota8000/trp.bin"),
    include_bytes!("../data/rota8000/val.bin"),
];

struct Dim {
    min: f64,
    n_bins: i64,
    wrap: bool,
    /// `wBin = (maxVal - minVal) / nBins`.
    width: f64,
}

/// A dense n-dimensional grid (`NDimTable`), stored as indices into its distinct values.
struct Table {
    dims: Vec<Dim>,
    values: Vec<f32>,
    index: Vec<u16>,
}

impl Table {
    /// Decode the format written by `scripts/convert_rota8000.py`.
    fn decode(blob: &[u8]) -> Table {
        let mut raw = Vec::new();
        flate2::read::ZlibDecoder::new(blob)
            .read_to_end(&mut raw)
            .expect("rota8000 table is valid zlib");
        let mut r = Reader(&raw);
        assert_eq!(r.take(4), b"ROT8", "rota8000 magic");
        assert_eq!(r.u8(), 1, "rota8000 format version");
        let n_dim = r.u8() as usize;
        let dims: Vec<Dim> = (0..n_dim)
            .map(|_| {
                let (min, max, n_bins, wrap) = (r.f64(), r.f64(), r.u32() as i64, r.u8() != 0);
                Dim {
                    min,
                    n_bins,
                    wrap,
                    width: (max - min) / n_bins as f64,
                }
            })
            .collect();
        let n_values = r.u32() as usize;
        let values = (0..n_values)
            .map(|_| f32::from_le_bytes(r.take(4).try_into().unwrap()))
            .collect();
        let size: usize = dims.iter().map(|d| d.n_bins as usize).product();
        let (lo, hi) = (r.take(size), r.take(size));
        assert!(r.0.is_empty(), "rota8000 trailing bytes");
        let index = lo
            .iter()
            .zip(hi)
            .map(|(&l, &h)| u16::from_le_bytes([l, h]))
            .collect();
        Table {
            dims,
            values,
            index,
        }
    }

    fn get(i: usize) -> &'static Table {
        static TABLES: [OnceLock<Table>; 17] = [const { OnceLock::new() }; 17];
        TABLES[i].get_or_init(|| Table::decode(TABLE_FILES[i]))
    }

    /// `NDimTable.bin2index_limit`: wrap each bin index where the dimension wraps, clamp it
    /// into range otherwise, and flatten row-major.
    fn index_limit(&self, bins: &[i64]) -> usize {
        let mut idx = 0i64;
        for (d, &b) in self.dims.iter().zip(bins) {
            let b = if d.wrap { b.rem_euclid(d.n_bins) } else { b };
            idx = idx * d.n_bins + b.clamp(0, d.n_bins - 1);
        }
        idx as usize
    }

    /// `NDimTable.valueAt`: multilinear interpolation between the bin containing `pt` and,
    /// per dimension, the neighbouring bin on the side of `pt` relative to the bin centre.
    /// `pt` holds the raw χ angles (on (−180°, 180°], not wrapped); the negative bin indices
    /// this gives on a 0–360° grid are wrapped by `index_limit`, and on the 0–180° grids of
    /// Asp χ2, Glu χ3 and Phe/Tyr χ2 the same wrap folds the angle modulo 180°. Only the
    /// first `n_dim` coordinates are read. The summation order matches cctbx term for term.
    fn value_at(&self, pt: &[f64]) -> f64 {
        let n = self.dims.len();
        let mut home = [0i64; 4];
        let mut neighbor = [0i64; 4];
        let mut contrib = [0f64; 4];
        for (i, d) in self.dims.iter().enumerate() {
            // whereIs: floor((x - min) / w), capped (above only) at nBins - 1.
            let b = ((pt[i] - d.min) / d.width)
                .floor()
                .min((d.n_bins - 1) as f64) as i64;
            let centre = d.min + d.width * (b as f64 + 0.5);
            home[i] = b;
            neighbor[i] = if pt[i] < centre { b - 1 } else { b + 1 };
            contrib[i] = ((pt[i] - centre) / d.width).abs();
        }
        let mut value = 0.0;
        let mut current = [0i64; 4];
        for mask in 0..(1u32 << n) {
            let mut coeff = 1.0;
            for dim in 0..n {
                if mask & (1 << dim) == 0 {
                    current[dim] = home[dim];
                    coeff *= 1.0 - contrib[dim];
                } else {
                    current[dim] = neighbor[dim];
                    coeff *= contrib[dim];
                }
            }
            let cell = self.index[self.index_limit(&current[..n])];
            value += coeff * self.values[cell as usize] as f64;
        }
        value
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        head
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
}

/// Top8000 grid value (a fraction in [0, 1]) for a residue with the given raw χ angles in
/// degrees, as cctbx `RotamerEval.evaluate`. `None` for residues without a table (GLY, ALA,
/// non-standard) or with fewer χ than the table has dimensions.
pub fn rotamer_value(resname: &str, chi: &[f64]) -> Option<f64> {
    let t = Table::get(table_index(resname)?);
    (chi.len() >= t.dims.len()).then(|| t.value_at(chi))
}

/// One `rotamer_names.props` line: `NamedRot`.
struct NamedRot {
    name: String,
    bounds: Vec<i32>,
}

impl NamedRot {
    /// `NamedRot.contains`: every χ within its inclusive [low, high] box.
    fn contains(&self, angles: &[f64]) -> bool {
        self.bounds.chunks(2).enumerate().all(|(i, b)| {
            angles
                .get(i)
                .is_some_and(|&a| !(a < b[0] as f64 || a > b[1] as f64))
        })
    }
}

/// `RotamerID.names`: per lower-case residue name, the boxes in file order (names repeat, e.g.
/// Phe `m-10` has two boxes, so this is a list, not a map).
fn rotamer_names() -> &'static HashMap<String, Vec<NamedRot>> {
    static NAMES: OnceLock<HashMap<String, Vec<NamedRot>>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names: HashMap<String, Vec<NamedRot>> = HashMap::new();
        for line in include_str!("../data/rota8000/rotamer_names.props").lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let mut parts = line.split('=');
            let key = parts.next().unwrap().trim();
            let ranges = parts.next().unwrap().trim().trim_matches('"');
            let mut key = key.split(' ');
            let (aa, rot) = (key.next().unwrap(), key.next().unwrap());
            let bounds = ranges.split(", ").map(|b| b.parse().unwrap()).collect();
            names.entry(aa.to_string()).or_default().push(NamedRot {
                name: rot.to_string(),
                bounds,
            });
        }
        names
    })
}

/// `RotamerID.wrap_chis(symmetry=False)`: each χ modulo 360 onto [0, 360) (Python float `%`).
fn wrap_chis(chi: &[f64]) -> Vec<f64> {
    chi.iter().map(|c| c.rem_euclid(360.0)).collect()
}

/// `RotamerID.identify`: re-wrap with symmetry (last χ of ASP/GLU/PHE/TYR modulo 180, the
/// `wrap_sym` rule) and return the first matching box's name, or `""`.
fn identify(resname: &str, wrapped: &[f64]) -> String {
    let aa = resname.to_ascii_lowercase();
    let mut chis = wrap_chis(wrapped);
    if matches!(aa.as_str(), "asp" | "glu" | "phe" | "tyr") {
        if let Some(last) = chis.last_mut() {
            *last = last.rem_euclid(180.0);
        }
    }
    rotamer_names()
        .get(&aa)
        .and_then(|rots| rots.iter().find(|r| r.contains(&chis)))
        .map(|r| r.name.clone())
        .unwrap_or_default()
}

/// `scitbx::math::dihedral::angle(deg=true)`: acos form, sign from d21·(n1×n2), `None` when
/// three consecutive atoms are collinear (cctbx then skips the residue).
fn dihedral(p: [[f64; 3]; 4]) -> Option<f64> {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let d01 = sub(p[0], p[1]);
    let d21 = sub(p[2], p[1]);
    let d23 = sub(p[2], p[3]);
    let n0121 = cross(d01, d21);
    let n2123 = cross(d21, d23);
    let (l1, l2) = (dot(n0121, n0121), dot(n2123, n2123));
    if l1 == 0.0 || l2 == 0.0 {
        return None;
    }
    let cos = (dot(n0121, n2123) / (l1 * l2).sqrt()).clamp(-1.0, 1.0);
    let mut angle = cos.acos();
    if dot(d21, cross(n0121, n2123)) < 0.0 {
        angle = -angle;
    }
    Some(angle / (std::f64::consts::PI / 180.0))
}

/// Evaluate every side chain of the first model the way cctbx `rotalyze` does.
///
/// `pdb` is expected to be reduced by [`crate::io::protein_heavy_atoms`] (first altloc only),
/// matching the hierarchy cctbx sees after `remove_alt_confs`. A residue is evaluated when its
/// name is one of the 18 amino acids with a Top8000 table or MSE (scored with the Met table)
/// and every atom of every χ it defines is present (`measureChiAngles` returns `None` for a
/// missing atom and rotalyze skips the residue); GLY, ALA and other residue names are skipped.
/// Results are in file order.
pub fn evaluate_rotamers(pdb: &pdbtbx::PDB) -> Vec<RotamerResult> {
    let mut out = Vec::new();
    let Some(model) = pdb.models().next() else {
        return out;
    };
    for chain in model.chains() {
        for residue in chain.residues() {
            let resname = residue.name().unwrap_or("").trim();
            let Some(defs) = chi_atoms(resname) else {
                continue;
            };
            // construct_complete_sidechain: later atoms with the same name win.
            let mut atoms: HashMap<&str, [f64; 3]> = HashMap::new();
            for a in residue.atoms() {
                let (x, y, z) = a.pos();
                atoms.insert(a.name().trim(), [x, y, z]);
            }
            let chi: Option<Vec<f64>> = defs
                .iter()
                .map(|names| {
                    let mut p = [[0.0; 3]; 4];
                    for (slot, name) in p.iter_mut().zip(names) {
                        *slot = *atoms.get(name)?;
                    }
                    dihedral(p)
                })
                .collect();
            let Some(chi) = chi else {
                continue;
            };
            let Some(value) = rotamer_value(resname, &chi) else {
                continue;
            };
            let evaluation = RotamerEvaluation::from_value(value);
            let wrapped = wrap_chis(&chi);
            let name = match evaluation {
                RotamerEvaluation::Outlier => "OUTLIER".to_string(),
                _ => match identify(resname, &wrapped) {
                    n if n.is_empty() => "UNCLASSIFIED".to_string(),
                    n => n,
                },
            };
            out.push(RotamerResult {
                chain_id: chain.id().to_string(),
                seq_num: residue.serial_number(),
                insertion_code: residue.insertion_code().map(str::to_string),
                resname: resname.to_string(),
                chi: wrapped,
                score: value * 100.0,
                evaluation,
                name: Some(name),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle values printed by cctbx 2025.11 `RotamerEval().evaluate(aa, chis)` and
    /// `RotamerID().identify(aa, wrap_chis(aa, chis, symmetry=False))`, raw χ in degrees.
    #[test]
    fn values_and_names_match_cctbx_rotamer_eval() {
        let cases: &[(&str, &[f64], f64, &str)] = &[
            ("SER", &[60.0], 0.7596812546253204, "p"),
            ("VAL", &[-179.8], 0.5814588427543649, "t"),
            ("PHE", &[-60.7, 97.9], 0.9443350990295409, "m-80"),
            // χ2 of Phe/Tyr is folded modulo 180 by the 0–180° grid and by wrap_sym.
            ("TYR", &[-60.7, -82.1], 0.9443350990295409, "m-80"),
            ("ASP", &[-80.6, -19.8], 0.5496158515214923, "m-30"),
            ("ASP", &[180.0, 180.0], 0.6090320199728012, "t0"),
            ("GLU", &[-73.9, -54.5, -18.4], 0.29951864592628796, "mm-30"),
            (
                "LYS",
                &[-175.6, 176.2, -172.0, -174.2],
                0.8182872854598999,
                "tttt",
            ),
            (
                "ARG",
                &[-55.9, -71.7, 114.2, -128.0],
                0.0035136376792309788,
                "mmp-170",
            ),
            // Pro: only χ1 is scored (1-D table); the name uses all three χ.
            ("PRO", &[-29.0, 40.0, -30.0], 0.909697026014328, "Cg_exo"),
            ("MET", &[80.4, -172.2, 177.5], 0.09581611536588747, "ptt"),
            ("LEU", &[-117.8, 30.2], 0.0001722736348005126, "mp"),
            ("ILE", &[125.7, -175.4], 0.0, "tt"),
        ];
        for &(aa, chi, value, name) in cases {
            let v = rotamer_value(aa, chi).unwrap();
            assert!((v - value).abs() < 1e-12, "{aa} {chi:?}: {v} != {value}");
            assert_eq!(identify(aa, &wrap_chis(chi)), name, "{aa} {chi:?}");
        }
    }

    fn crambin() -> Vec<RotamerResult> {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
            .unwrap();
        evaluate_rotamers(&crate::io::protein_heavy_atoms(&pdb))
    }

    fn find(rs: &[RotamerResult], seq: isize) -> &RotamerResult {
        rs.iter().find(|r| r.seq_num == seq).unwrap()
    }

    /// Reference: `validate/reference/geometry/1crn_pdb.json.gz` (cctbx rotalyze).
    #[test]
    fn crambin_matches_cctbx_rotalyze() {
        let rs = crambin();
        let close = |a: f64, b: f64, tol: f64| (a - b).abs() < tol;
        let r = find(&rs, 5);
        assert_eq!(r.resname, "PRO");
        assert_eq!(r.chi.len(), 3);
        assert!(close(r.chi[0], 32.042, 1e-3) && close(r.chi[1], 318.618, 1e-3));
        assert!(close(r.score, 58.5194, 1e-4), "{r:?}");
        assert_eq!(r.name.as_deref(), Some("Cg_endo"));
        let r = find(&rs, 10);
        assert_eq!(r.resname, "ARG");
        assert!(close(r.score, 23.2741, 1e-4), "{r:?}");
        assert_eq!(r.evaluation, RotamerEvaluation::Favored);
        assert_eq!(r.name.as_deref(), Some("tpp-160"));
        let r = find(&rs, 13);
        assert_eq!(r.resname, "PHE");
        assert!(close(r.chi[1], 270.018, 1e-3), "{r:?}");
        assert!(close(r.score, 63.6559, 1e-4), "{r:?}");
        assert_eq!(r.name.as_deref(), Some("t80"));
        let r = find(&rs, 8);
        assert_eq!(r.resname, "VAL");
        assert!(close(r.score, 3.083, 1e-4), "{r:?}");
    }

    #[test]
    fn thresholds_are_inclusive_fractions() {
        assert_eq!(
            RotamerEvaluation::from_value(0.02),
            RotamerEvaluation::Favored
        );
        assert_eq!(
            RotamerEvaluation::from_value(0.019_999),
            RotamerEvaluation::Allowed
        );
        assert_eq!(
            RotamerEvaluation::from_value(0.003),
            RotamerEvaluation::Allowed
        );
        assert_eq!(
            RotamerEvaluation::from_value(0.002_999),
            RotamerEvaluation::Outlier
        );
    }

    #[test]
    fn gly_ala_and_unknown_residues_are_not_evaluated() {
        let rs = crambin();
        assert!(rs
            .iter()
            .all(|r| !matches!(r.resname.as_str(), "GLY" | "ALA")));
        assert_eq!(rotamer_value("GLY", &[]), None);
        assert_eq!(rotamer_value("UNK", &[60.0]), None);
    }
}
