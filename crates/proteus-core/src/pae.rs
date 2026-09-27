//! Predicted aligned error (PAE) and the global confidences (pTM, ipTM) a structure predictor
//! writes next to its model.
//!
//! pLDDT says how sure the model is of each residue's local environment; PAE says how sure it
//! is of where residue *j* sits when the prediction is superposed on residue *i*. Two domains
//! can each be pLDDT > 90 and still be placed relative to each other by guesswork, which only
//! PAE shows. `values[i * n + j]` is that expected error in Å: row `i` is the aligned residue,
//! column `j` the scored one, as AlphaFold DB, ColabFold and Boltz all lay it out.
//!
//! Formats read:
//! - Boltz: `pae_<model>.npz` (array `pae`, float32 N×N) and `confidence_<model>.json` (`ptm`,
//!   `iptm`).
//! - AlphaFold DB: `…-predicted_aligned_error_v*.json`, `[{"predicted_aligned_error": [[…]],
//!   "max_predicted_aligned_error": 31.75}]`, and the v1/v2 layout with `residue1`/`residue2`/
//!   `distance` lists.
//! - ColabFold: `…_scores_rank_…json` (`pae`, `max_pae`, `ptm`, `iptm`).
//! - AlphaFold 3: `…_full_data_<k>.json` (`pae`) and `…_summary_confidences_<k>.json` (AlphaFold
//!   Server); `<name>_confidences.json` and `<name>_summary_confidences.json` beside
//!   `<name>_model.cif`, or `confidences.json` beside a sample's `model.cif` (a local run).
//! - OpenFold3: `<q>_seed_<s>_sample_<n>_confidences.json` or `.npz` (`pae`, float16 in the
//!   `.npz`) and `…_confidences_aggregated.json` beside `…_model.cif`.
//! - Protenix: `<job>_full_data_sample_<k>.json` (`token_pair_pae`, written with
//!   `--need_atom_confidence`) and `<job>_summary_confidence_sample_<k>.json` beside
//!   `<job>_sample_<k>.cif`.
//! - Chai-1: `scores.model_idx_<k>.npz` (`ptm`, `iptm`) beside `pred.model_idx_<k>.cif`. Chai-1
//!   writes no PAE file; `pae.model_idx_<k>.npy` saved from its Python API is read if present.
//! - A bare `.npy` holding the N×N matrix (float16, float32 or float64).
//!
//! AlphaFold 3-style predictors write one row per token, not per residue; see
//! [`protein_token_rows`] for how a complex with ligands is matched to its residues.

use crate::error::CoreError;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};

/// AlphaFold 2's PAE histogram tops out at 31.75 Å; its colour scale runs from 0 to there.
pub const DEFAULT_MAX_PAE: f32 = 31.75;

/// An N×N predicted-aligned-error matrix, row-major, in Å.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictedAlignedError {
    pub n: usize,
    pub values: Vec<f32>,
    /// Upper end of the scale: the file's own maximum when it states one, else
    /// [`DEFAULT_MAX_PAE`] or the largest value, whichever is larger.
    pub max: f32,
}

impl PredictedAlignedError {
    fn new(n: usize, values: Vec<f32>, stated_max: Option<f32>) -> Result<Self, CoreError> {
        if n == 0 || values.len() != n * n {
            return Err(parse_err(format!(
                "a PAE matrix must be square and non-empty; got {} values for {n} rows",
                values.len()
            )));
        }
        if let Some(bad) = values.iter().find(|v| !v.is_finite() || **v < 0.0) {
            return Err(parse_err(format!(
                "PAE values are distances in Å and cannot be {bad}"
            )));
        }
        let observed = values.iter().copied().fold(0.0f32, f32::max);
        let max = stated_max
            .filter(|m| m.is_finite() && *m > 0.0)
            .unwrap_or(DEFAULT_MAX_PAE)
            .max(observed);
        Ok(Self { n, values, max })
    }

    /// Expected error at residue `j` when aligned on residue `i`, in Å.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.values[i * self.n + j]
    }

    /// Mean over the whole matrix, in Å.
    pub fn mean(&self) -> f64 {
        self.values.iter().map(|v| *v as f64).sum::<f64>() / self.values.len() as f64
    }

    /// Mean PAE between two residue ranges, symmetrised: the average of the `a`→`b` and
    /// `b`→`a` blocks. What "are these two domains placed confidently?" reads off the plot.
    pub fn block_mean(&self, a: std::ops::Range<usize>, b: std::ops::Range<usize>) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for i in a.clone() {
            for j in b.clone() {
                sum += (self.get(i, j) + self.get(j, i)) as f64 / 2.0;
                count += 1;
            }
        }
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }
}

/// What a predictor said about its model as a whole, and where it was read from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PredictionConfidence {
    pub pae: Option<PredictedAlignedError>,
    /// Predicted TM-score of the whole model, 0–1.
    pub ptm: Option<f64>,
    /// Interface pTM, 0–1, for complexes; absent (or 0 in Boltz's file) for one chain.
    pub iptm: Option<f64>,
    /// pTM of each chain, in chain order (Boltz `chains_ptm`).
    pub chain_ptm: Vec<f64>,
    /// ipTM of each pair of chains, in chain order (Boltz `pair_chains_iptm`); the diagonal is
    /// the chain's own pTM.
    pub pair_iptm: Vec<Vec<f64>>,
    /// ipTM over protein–ligand interfaces (Boltz `ligand_iptm`), when there are any.
    pub ligand_iptm: Option<f64>,
    /// Boltz's ranking score (0.8·complex pLDDT + 0.2·ipTM, or pTM for one chain).
    pub confidence_score: Option<f64>,
    /// The files these came from, for provenance in the viewer.
    pub sources: Vec<PathBuf>,
}

impl PredictionConfidence {
    pub fn is_empty(&self) -> bool {
        self.pae.is_none() && self.ptm.is_none() && self.iptm.is_none()
    }
}

impl PredictedAlignedError {
    /// Collapse token groups: `groups[k]` lists the matrix indices that become output index `k`
    /// (one residue each for a protein, every atom of a ligand for a ligand), averaging over
    /// each block. How a Boltz/AF3 matrix with one token per ligand atom is shown per ligand.
    pub fn collapse(&self, groups: &[Vec<usize>]) -> Option<Self> {
        if groups.iter().flatten().any(|&i| i >= self.n) || groups.iter().any(Vec::is_empty) {
            return None;
        }
        let m = groups.len();
        let mut values = vec![0.0f32; m * m];
        for (a, ga) in groups.iter().enumerate() {
            for (b, gb) in groups.iter().enumerate() {
                let mut sum = 0.0f32;
                for &i in ga {
                    for &j in gb {
                        sum += self.get(i, j);
                    }
                }
                values[a * m + b] = sum / (ga.len() * gb.len()) as f32;
            }
        }
        Some(Self {
            n: m,
            values,
            max: self.max,
        })
    }
}

/// Two chains with a modified residue and a ligand between them: 11 tokens, protein rows
/// 0, 2, 6 and 10 (see `token_rows_follow_the_file_order`).
#[cfg(test)]
pub(crate) const TOKEN_TEST_PDB: &[u8] = b"\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00 90.00           O
HETATM    5  N   SEP A   2       3.332   1.536   0.000  1.00 90.00           N
HETATM    6  H   SEP A   2       3.800   0.700   0.000  1.00 90.00           H
HETATM    7  CA  SEP A   2       3.970   2.845   0.000  1.00 90.00           C
HETATM    8  C   SEP A   2       5.486   2.715   0.000  1.00 90.00           C
HETATM    9  O   SEP A   2       6.022   1.604   0.000  1.00 90.00           O
HETATM   10  P   SEP A   2       3.500   4.500   1.000  1.00 90.00           P
ATOM     11  N   ALA A   3       6.180   3.850   0.000  1.00 90.00           N
ATOM     12  CA  ALA A   3       7.633   3.859   0.000  1.00 90.00           C
ATOM     13  C   ALA A   3       8.176   5.281   0.000  1.00 90.00           C
ATOM     14  O   ALA A   3       7.418   6.251   0.000  1.00 90.00           O
HETATM   15  C1  LIG C   1      10.000  10.000  10.000  1.00 80.00           C
HETATM   16  C2  LIG C   1      11.000  10.000  10.000  1.00 80.00           C
HETATM   17  O1  LIG C   1      12.000  10.000  10.000  1.00 80.00           O
ATOM     18  N   ALA B   1      20.000   0.000   0.000  1.00 90.00           N
ATOM     19  CA  ALA B   1      21.458   0.000   0.000  1.00 90.00           C
ATOM     20  C   ALA B   1      22.009   1.420   0.000  1.00 90.00           C
ATOM     21  O   ALA B   1      21.251   2.390   0.000  1.00 90.00           O
HETATM   22  O   HOH W   1      30.000  30.000  30.000  1.00 10.00           O
END
";

/// Where each protein residue sits in an AlphaFold 3-style token layout, and how many tokens
/// the model has: `(n_tokens, rows)`, `rows[k]` being the PAE row of the `k`-th protein residue
/// in the order [`crate::io::protein_heavy_atoms`] keeps them.
///
/// AlphaFold 3, Boltz, Chai-1, Protenix and OpenFold3 give a standard residue or nucleotide one
/// token and anything else one token per heavy atom, in input order, which is file order. A
/// modified residue in a protein chain is one of those; its row is its CA atom's, as `ipsae.py`
/// takes it. Waters are not tokenised. `ipsae.py`'s `index N is out of bounds` failures on
/// Boltz-2 complexes (DunbrackLab/IPSAE #20, #28) are a miscount of exactly this.
pub fn protein_token_rows(pdb: &pdbtbx::PDB) -> (usize, Vec<usize>) {
    use crate::io::{is_hydrogen, is_protein_residue, is_standard_amino_acid};
    const NUCLEOTIDES: &[&str] = &["A", "C", "G", "U", "DA", "DC", "DG", "DT", "N", "DN"];
    let mut pdb = pdb.clone();
    crate::io::split_merged_residues(&mut pdb);
    let mut next = 0usize;
    let mut rows = Vec::new();
    for residue in pdb.chains().flat_map(|c| c.residues()) {
        let name = residue.name().unwrap_or("").trim();
        if matches!(name, "HOH" | "WAT" | "DOD") {
            continue;
        }
        let heavy: Vec<&pdbtbx::Atom> = residue
            .conformers()
            .next()
            .into_iter()
            .flat_map(|c| c.atoms())
            .filter(|a| !is_hydrogen(a))
            .collect();
        if heavy.is_empty() {
            continue;
        }
        let protein = is_protein_residue(residue);
        if is_standard_amino_acid(name) || NUCLEOTIDES.contains(&name) {
            if protein {
                rows.push(next);
            }
            next += 1;
        } else {
            if protein {
                let ca = heavy
                    .iter()
                    .position(|a| a.name().trim() == "CA")
                    .unwrap_or(0);
                rows.push(next + ca);
            }
            next += heavy.len();
        }
    }
    (next, rows)
}

fn parse_err(msg: impl Into<String>) -> CoreError {
    CoreError::ParseError(msg.into())
}

/// Read a PAE matrix from `.npz`, `.npy` or `.json`, by extension.
pub fn read_pae(path: &Path) -> Result<PredictedAlignedError, CoreError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let bytes = std::fs::read(path)
        .map_err(|e| parse_err(format!("cannot read {}: {e}", path.display())))?;
    let named = |e: CoreError| parse_err(format!("{}: {e}", path.display()));
    match ext.as_str() {
        "npz" => pae_from_npz(&bytes).map_err(named),
        "npy" => {
            let (shape, values) = parse_npy(&bytes).map_err(named)?;
            square(shape, values, None).map_err(named)
        }
        "json" => {
            let v: Value = serde_json::from_slice(&bytes)
                .map_err(|e| parse_err(format!("{}: not JSON: {e}", path.display())))?;
            pae_from_json(&v)
                .ok_or_else(|| {
                    parse_err(format!(
                        "{}: no PAE matrix found (looked for predicted_aligned_error, pae, \
                         token_pair_pae, residue1/residue2/distance)",
                        path.display()
                    ))
                })?
                .map_err(named)
        }
        _ => Err(parse_err(format!(
            "{}: a PAE file is .npz, .npy or .json",
            path.display()
        ))),
    }
}

fn square(
    shape: Vec<usize>,
    values: Vec<f32>,
    max: Option<f32>,
) -> Result<PredictedAlignedError, CoreError> {
    match shape.as_slice() {
        [a, b] if a == b => PredictedAlignedError::new(*a, values, max),
        // Boltz writes one matrix per model; a leading 1 is a batch of one.
        [1, a, b] if a == b => PredictedAlignedError::new(*a, values, max),
        _ => Err(parse_err(format!("PAE array has shape {shape:?}, not N×N"))),
    }
}

fn pae_from_npz(bytes: &[u8]) -> Result<PredictedAlignedError, CoreError> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| parse_err(format!("not an .npz archive: {e}")))?;
    let names: Vec<String> = zip.file_names().map(str::to_string).collect();
    let name = ["pae.npy", "predicted_aligned_error.npy"]
        .iter()
        .find(|n| names.iter().any(|m| m == *n))
        .map(|s| s.to_string())
        .or_else(|| (names.len() == 1).then(|| names[0].clone()))
        .ok_or_else(|| {
            parse_err(format!(
                "no 'pae' array in the archive (it holds {})",
                names.join(", ")
            ))
        })?;
    let mut entry = zip
        .by_name(&name)
        .map_err(|e| parse_err(format!("{name}: {e}")))?;
    let mut raw = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut raw)
        .map_err(|e| parse_err(format!("{name}: {e}")))?;
    let (shape, values) = parse_npy(&raw)?;
    square(shape, values, None)
}

/// IEEE 754 half precision to single, subnormals included.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A little-endian float `.npy` array: shape and values in C order.
pub fn parse_npy(b: &[u8]) -> Result<(Vec<usize>, Vec<f32>), CoreError> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(parse_err("not a .npy array (bad magic)"));
    }
    let (header_len, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 if b.len() >= 12 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => {
            return Err(parse_err(format!(
                ".npy format version {v} is not supported"
            )))
        }
    };
    let header = b
        .get(start..start + header_len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| parse_err(".npy header is truncated or not text"))?;
    let field = |key: &str| -> Option<&str> {
        let at = header.find(&format!("'{key}'"))? + key.len() + 2;
        let rest = header[at..].trim_start().strip_prefix(':')?.trim_start();
        Some(rest)
    };
    let descr = field("descr")
        .and_then(|r| r.strip_prefix('\''))
        .and_then(|r| r.split('\'').next())
        .ok_or_else(|| parse_err(".npy header has no descr"))?;
    let fortran = field("fortran_order").is_some_and(|r| r.starts_with("True"));
    let shape: Vec<usize> = field("shape")
        .and_then(|r| r.strip_prefix('('))
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| parse_err(".npy header has no shape"))?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| parse_err(".npy shape is not a list of integers"))?;
    let count: usize = shape.iter().product();
    let data = &b[start + header_len..];
    let values: Vec<f32> = match descr {
        "<f4" | "|f4" if data.len() >= count * 4 => data[..count * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        "<f8" if data.len() >= count * 8 => data[..count * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        // OpenFold3 writes its `.npz` confidences as float16 by default.
        "<f2" if data.len() >= count * 2 => data[..count * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "<f2" | "<f4" | "|f4" | "<f8" => {
            return Err(parse_err(".npy data is shorter than its shape"))
        }
        other => {
            return Err(parse_err(format!(
                ".npy dtype {other} is not supported (little-endian float16, float32 or float64)"
            )))
        }
    };
    let values = if fortran && shape.len() == 2 {
        let (r, c) = (shape[0], shape[1]);
        let mut t = vec![0.0f32; count];
        for i in 0..r {
            for j in 0..c {
                t[i * c + j] = values[j * r + i];
            }
        }
        t
    } else {
        values
    };
    Ok((shape, values))
}

fn number(v: &Value) -> Option<f32> {
    v.as_f64().map(|x| x as f32)
}

/// A PAE matrix from any of the JSON layouts in the module docs, or `None` when the document
/// holds none.
fn pae_from_json(v: &Value) -> Option<Result<PredictedAlignedError, CoreError>> {
    // AlphaFold DB wraps the object in a one-element list.
    let obj = match v {
        Value::Array(a) if a.len() == 1 && a[0].is_object() => &a[0],
        other => other,
    };
    let stated_max = ["max_predicted_aligned_error", "max_pae"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(number));
    if let Some(rows) = ["predicted_aligned_error", "pae", "token_pair_pae"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(Value::as_array))
    {
        let n = rows.len();
        let mut values = Vec::with_capacity(n * n);
        for row in rows {
            let Some(row) = row.as_array() else {
                return Some(Err(parse_err("PAE rows must be lists of numbers")));
            };
            if row.len() != n {
                return Some(Err(parse_err(format!(
                    "PAE matrix is not square: {n} rows, a row of {}",
                    row.len()
                ))));
            }
            for x in row {
                match number(x) {
                    Some(x) => values.push(x),
                    None => return Some(Err(parse_err("PAE entries must be numbers"))),
                }
            }
        }
        return Some(PredictedAlignedError::new(n, values, stated_max));
    }
    // AlphaFold DB v1/v2: three flat lists, residues numbered from 1.
    let list = |k: &str| obj.get(k).and_then(Value::as_array);
    if let (Some(r1), Some(r2), Some(d)) = (list("residue1"), list("residue2"), list("distance")) {
        if r1.len() != d.len() || r2.len() != d.len() {
            return Some(Err(parse_err(
                "residue1, residue2 and distance have different lengths",
            )));
        }
        let n = r1.iter().filter_map(Value::as_u64).max().unwrap_or(0) as usize;
        if n * n != d.len() {
            return Some(Err(parse_err(format!(
                "{} PAE entries for {n} residues",
                d.len()
            ))));
        }
        let mut values = vec![0.0f32; n * n];
        for k in 0..d.len() {
            let (Some(i), Some(j), Some(x)) = (r1[k].as_u64(), r2[k].as_u64(), number(&d[k]))
            else {
                return Some(Err(parse_err("PAE lists must hold numbers")));
            };
            if i == 0 || j == 0 {
                return Some(Err(parse_err("PAE residue numbers start at 1")));
            }
            values[(i as usize - 1) * n + (j as usize - 1)] = x;
        }
        return Some(PredictedAlignedError::new(n, values, stated_max));
    }
    None
}

/// pTM and ipTM from Chai-1's `scores.model_idx_N.npz` (one-element arrays `ptm`, `iptm`).
/// A single chain's ipTM is 0 there, as in Boltz's file, and is dropped the same way.
fn scores_from_npz(bytes: &[u8]) -> Result<(Option<f64>, Option<f64>), CoreError> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| parse_err(format!("not an .npz archive: {e}")))?;
    let mut first = |name: &str| -> Result<Option<f64>, CoreError> {
        let Ok(mut entry) = zip.by_name(name) else {
            return Ok(None);
        };
        let mut raw = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut raw)
            .map_err(|e| parse_err(format!("{name}: {e}")))?;
        let (_, v) = parse_npy(&raw)?;
        Ok(v.first().map(|x| f64::from(*x)).filter(|x| x.is_finite()))
    };
    let ptm = first("ptm.npy")?;
    let iptm = first("iptm.npy")?.filter(|x| *x > 0.0);
    Ok((ptm, iptm))
}

/// pTM and ipTM from a scores document (Boltz `confidence_*.json`, ColabFold `*_scores_*.json`,
/// AlphaFold 3 `*_summary_confidences_*.json`).
fn scores_from_json(v: &Value) -> (Option<f64>, Option<f64>) {
    let get = |k: &str| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
    let ptm = get("ptm");
    // Boltz writes iptm 0.0 for a single chain, which is "not applicable", not "no confidence".
    let iptm = get("iptm").filter(|x| *x > 0.0);
    (ptm, iptm)
}

/// A JSON object keyed by chain index ("0", "1", …), in index order.
fn indexed(o: &serde_json::Map<String, Value>) -> Vec<(usize, &Value)> {
    let mut e: Vec<(usize, &Value)> = o
        .iter()
        .filter_map(|(k, x)| k.parse::<usize>().ok().map(|i| (i, x)))
        .collect();
    e.sort_by_key(|(i, _)| *i);
    e
}

/// Boltz's per-chain scores: `chains_ptm` {"0": x, …} and `pair_chains_iptm` {"0": {"1": x}}.
fn chain_scores_from_json(v: &Value) -> (Vec<f64>, Vec<Vec<f64>>) {
    let chain_ptm = v
        .get("chains_ptm")
        .and_then(Value::as_object)
        .map(|o| {
            indexed(o)
                .into_iter()
                .filter_map(|(_, x)| x.as_f64())
                .collect()
        })
        .unwrap_or_default();
    let pair_iptm = v
        .get("pair_chains_iptm")
        .and_then(Value::as_object)
        .map(|o| {
            indexed(o)
                .into_iter()
                .filter_map(|(_, row)| row.as_object())
                .map(|row| {
                    indexed(row)
                        .into_iter()
                        .filter_map(|(_, x)| x.as_f64())
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();
    (chain_ptm, pair_iptm)
}

/// The files a predictor writes next to a model, by naming convention. Only files that exist
/// are returned. `(pae_file, scores_file)`.
pub fn sidecar_files(structure: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let Some(dir) = structure.parent() else {
        return (None, None);
    };
    let name = structure.file_name().and_then(|s| s.to_str()).unwrap_or("");
    // Strip `.gz` and then the structure extension.
    let name = name.strip_suffix(".gz").unwrap_or(name);
    let stem = name.rsplit_once('.').map_or(name, |(s, _)| s);
    let exists = |p: PathBuf| p.is_file().then_some(p);

    let mut pae = None;
    let mut scores = None;
    // Boltz: input_model_0.pdb → pae_input_model_0.npz, confidence_input_model_0.json.
    pae = pae.or_else(|| exists(dir.join(format!("pae_{stem}.npz"))));
    scores = scores.or_else(|| exists(dir.join(format!("confidence_{stem}.json"))));
    // AlphaFold DB: AF-P69905-F1-model_v6 → AF-P69905-F1-predicted_aligned_error_v6.json.
    if let Some((head, tail)) = stem.rsplit_once("-model_v") {
        pae = pae
            .or_else(|| exists(dir.join(format!("{head}-predicted_aligned_error_v{tail}.json"))));
    }
    // ColabFold: X_unrelaxed_rank_001_… → X_scores_rank_001_….json (pae and ptm in one file).
    for tag in ["_unrelaxed_", "_relaxed_"] {
        if let Some((head, tail)) = stem.split_once(tag) {
            let f = exists(dir.join(format!("{head}_scores_{tail}.json")));
            pae = pae.or_else(|| f.clone());
            scores = scores.or(f);
        }
    }
    // AlphaFold 3: fold_x_model_0 → fold_x_full_data_0.json, fold_x_summary_confidences_0.json.
    if let Some((head, k)) = stem.rsplit_once("_model_") {
        if k.chars().all(|c| c.is_ascii_digit()) && !k.is_empty() {
            pae = pae.or_else(|| exists(dir.join(format!("{head}_full_data_{k}.json"))));
            scores =
                scores.or_else(|| exists(dir.join(format!("{head}_summary_confidences_{k}.json"))));
        }
    }
    // AlphaFold 3 run locally: name/name_model.cif → name_confidences.json (pae) and
    // name_summary_confidences.json; each sample's seed-1_sample-0/model.cif → confidences.json.
    // OpenFold3 names a sample the same way (q_seed_42_sample_1_model.cif), with its PAE in
    // `_confidences.json` or `.npz` and its pTM/ipTM in `_confidences_aggregated.json`.
    if let Some(head) = stem.strip_suffix("_model") {
        pae = pae
            .or_else(|| exists(dir.join(format!("{head}_confidences.json"))))
            .or_else(|| exists(dir.join(format!("{head}_confidences.npz"))));
        scores = scores
            .or_else(|| exists(dir.join(format!("{head}_summary_confidences.json"))))
            .or_else(|| exists(dir.join(format!("{head}_confidences_aggregated.json"))));
    }
    // Protenix: job_sample_0.cif → job_full_data_sample_0.json (`token_pair_pae`, written only
    // with --need_atom_confidence) and job_summary_confidence_sample_0.json.
    if let Some((head, k)) = stem.rsplit_once("_sample_") {
        if k.chars().all(|c| c.is_ascii_digit()) && !k.is_empty() {
            pae = pae.or_else(|| exists(dir.join(format!("{head}_full_data_sample_{k}.json"))));
            scores = scores
                .or_else(|| exists(dir.join(format!("{head}_summary_confidence_sample_{k}.json"))));
        }
    }
    // Chai-1: pred.model_idx_0.cif → scores.model_idx_0.npz. Chai-1 writes no PAE file; one
    // saved from its API as pae.model_idx_0.npy (or .npz) is picked up.
    if let Some(k) = stem.strip_prefix("pred.model_idx_") {
        pae = pae
            .or_else(|| exists(dir.join(format!("pae.model_idx_{k}.npy"))))
            .or_else(|| exists(dir.join(format!("pae.model_idx_{k}.npz"))));
        scores = scores.or_else(|| exists(dir.join(format!("scores.model_idx_{k}.npz"))));
    }
    if stem == "model" {
        pae = pae.or_else(|| exists(dir.join("confidences.json")));
        scores = scores.or_else(|| exists(dir.join("summary_confidences.json")));
    }
    (pae, scores)
}

/// Read the confidences for a model: from `pae_file`/`scores_file` when given, otherwise from
/// the files its predictor writes next to it ([`sidecar_files`]). A file that is found but
/// cannot be read is an error; no file at all is an empty result.
pub fn read_confidence(
    structure: &Path,
    pae_file: Option<&Path>,
) -> Result<PredictionConfidence, CoreError> {
    let (found_pae, found_scores) = sidecar_files(structure);
    let pae_path = pae_file.map(Path::to_path_buf).or(found_pae);
    let mut out = PredictionConfidence::default();
    if let Some(p) = &pae_path {
        out.pae = Some(read_pae(p)?);
        out.sources.push(p.clone());
    }
    // A scores file can be the PAE file itself (ColabFold, a user's --pae JSON).
    let mut score_paths: Vec<PathBuf> = found_scores.into_iter().collect();
    if let Some(p) = pae_path.filter(|p| p.extension().is_some_and(|e| e == "json")) {
        if !score_paths.contains(&p) {
            score_paths.push(p);
        }
    }
    for p in score_paths {
        let bytes = std::fs::read(&p)
            .map_err(|e| parse_err(format!("cannot read {}: {e}", p.display())))?;
        if p.extension().is_some_and(|e| e == "npz") {
            let (ptm, iptm) =
                scores_from_npz(&bytes).map_err(|e| parse_err(format!("{}: {e}", p.display())))?;
            if ptm.is_some() || iptm.is_some() {
                out.ptm = out.ptm.or(ptm);
                out.iptm = out.iptm.or(iptm);
                out.sources.push(p);
            }
            continue;
        }
        let v: Value = serde_json::from_slice(&bytes)
            .map_err(|e| parse_err(format!("{}: not JSON: {e}", p.display())))?;
        let (ptm, iptm) = scores_from_json(&v);
        if out.chain_ptm.is_empty() {
            let (chain_ptm, pair_iptm) = chain_scores_from_json(&v);
            out.chain_ptm = chain_ptm;
            out.pair_iptm = pair_iptm;
        }
        let get = |k: &str| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
        out.ligand_iptm = out.ligand_iptm.or(get("ligand_iptm").filter(|x| *x > 0.0));
        out.confidence_score = out.confidence_score.or(get("confidence_score"));
        if ptm.is_some() || iptm.is_some() {
            out.ptm = out.ptm.or(ptm);
            out.iptm = out.iptm.or(iptm);
            if !out.sources.contains(&p) {
                out.sources.push(p);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_f32(shape: &[usize], values: &[f32]) -> Vec<u8> {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape_txt = if dims.len() == 1 {
            format!("({},)", dims[0])
        } else {
            format!("({})", dims.join(", "))
        };
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_txt}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((header.len() as u16).to_le_bytes());
        b.extend(header.as_bytes());
        for v in values {
            b.extend(v.to_le_bytes());
        }
        b
    }

    fn npz(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut out);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, data) in entries {
                w.start_file(*name, opts).unwrap();
                std::io::Write::write_all(&mut w, data).unwrap();
            }
            w.finish().unwrap();
        }
        out.into_inner()
    }

    #[test]
    fn npy_round_trips_shape_and_values() {
        let (shape, v) = parse_npy(&npy_f32(&[2, 2], &[0.5, 1.0, 2.0, 30.0])).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(v, vec![0.5, 1.0, 2.0, 30.0]);
    }

    #[test]
    fn a_boltz_npz_is_read_from_its_pae_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("pae_x_model_0.npz");
        std::fs::write(
            &p,
            npz(&[(
                "pae.npy",
                npy_f32(&[3, 3], &[0., 1., 2., 3., 0., 4., 5., 6., 0.]),
            )]),
        )
        .unwrap();
        let pae = read_pae(&p).unwrap();
        assert_eq!(pae.n, 3);
        assert_eq!(pae.get(0, 2), 2.0);
        assert_eq!(pae.get(2, 0), 5.0);
        assert_eq!(pae.max, DEFAULT_MAX_PAE);
    }

    #[test]
    fn alphafold_db_json_in_both_layouts() {
        let current = serde_json::json!([{
            "predicted_aligned_error": [[0, 3], [4, 0]],
            "max_predicted_aligned_error": 31.75
        }]);
        let p = pae_from_json(&current).unwrap().unwrap();
        assert_eq!((p.n, p.get(0, 1), p.get(1, 0), p.max), (2, 3.0, 4.0, 31.75));

        let legacy = serde_json::json!([{
            "residue1": [1, 1, 2, 2], "residue2": [1, 2, 1, 2],
            "distance": [0.0, 3.0, 4.0, 0.0], "max_predicted_aligned_error": 31.75
        }]);
        assert_eq!(pae_from_json(&legacy).unwrap().unwrap(), p);
    }

    #[test]
    fn malformed_matrices_are_refused_not_guessed() {
        let ragged = serde_json::json!({"pae": [[0, 1], [2]]});
        assert!(pae_from_json(&ragged).unwrap().is_err());
        let negative = serde_json::json!({"pae": [[0, -1], [2, 0]]});
        assert!(pae_from_json(&negative).unwrap().is_err());
        assert!(pae_from_json(&serde_json::json!({"plddt": [90]})).is_none());
    }

    /// Names as each predictor's own writer makes them (checked against the sources on
    /// 2026-09-27): Protenix `runner/dumper.py`, OpenFold3 `core/runners/writer.py`, Chai-1
    /// `chai1.py` (which writes no PAE; `pae.model_idx_N.npy` is what a user saves from its API).
    #[test]
    fn protenix_openfold3_and_chai1_names_map_to_their_files() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        for f in [
            "job_sample_0.cif",
            "job_full_data_sample_0.json",
            "job_summary_confidence_sample_0.json",
            "q_seed_42_sample_1_model.cif",
            "q_seed_42_sample_1_confidences.npz",
            "q_seed_42_sample_1_confidences_aggregated.json",
            "pred.model_idx_2.cif",
            "scores.model_idx_2.npz",
            "pae.model_idx_2.npy",
        ] {
            std::fs::write(d.join(f), "").unwrap();
        }
        let (pae, scores) = sidecar_files(&d.join("job_sample_0.cif"));
        assert_eq!(pae.unwrap(), d.join("job_full_data_sample_0.json"));
        assert_eq!(
            scores.unwrap(),
            d.join("job_summary_confidence_sample_0.json")
        );
        let (pae, scores) = sidecar_files(&d.join("q_seed_42_sample_1_model.cif"));
        assert_eq!(pae.unwrap(), d.join("q_seed_42_sample_1_confidences.npz"));
        assert_eq!(
            scores.unwrap(),
            d.join("q_seed_42_sample_1_confidences_aggregated.json")
        );
        let (pae, scores) = sidecar_files(&d.join("pred.model_idx_2.cif"));
        assert_eq!(pae.unwrap(), d.join("pae.model_idx_2.npy"));
        assert_eq!(scores.unwrap(), d.join("scores.model_idx_2.npz"));
    }

    #[test]
    fn protenix_token_pair_pae_and_float16_npy_are_read() {
        let v = serde_json::json!({"token_pair_pae": [[0.5, 3.0], [4.0, 0.5]]});
        let p = pae_from_json(&v).unwrap().unwrap();
        assert_eq!((p.n, p.get(0, 1), p.get(1, 0)), (2, 3.0, 4.0));
        // float16 bit patterns of 0.5, 1.0, 2.0 and 30.0.
        let halves: [u16; 4] = [0x3800, 0x3c00, 0x4000, 0x4f80];
        let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 2), }";
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(header.len() as u16).to_le_bytes());
        b.extend_from_slice(header.as_bytes());
        for h in halves {
            b.extend_from_slice(&h.to_le_bytes());
        }
        let (shape, values) = parse_npy(&b).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(values, vec![0.5, 1.0, 2.0, 30.0]);
    }

    /// Chai-1's `scores.model_idx_N.npz` holds pTM and ipTM as one-element arrays.
    #[test]
    fn chai1_scores_npz_gives_ptm_and_iptm() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("pred.model_idx_0.cif"), "").unwrap();
        std::fs::write(
            d.join("scores.model_idx_0.npz"),
            npz(&[
                ("ptm.npy", npy_f32(&[1], &[0.81])),
                ("iptm.npy", npy_f32(&[1], &[0.62])),
                ("aggregate_score.npy", npy_f32(&[1], &[0.7])),
            ]),
        )
        .unwrap();
        let c = read_confidence(&d.join("pred.model_idx_0.cif"), None).unwrap();
        assert!(c.pae.is_none());
        assert!((c.ptm.unwrap() - 0.81).abs() < 1e-6);
        assert!((c.iptm.unwrap() - 0.62).abs() < 1e-6);
    }

    #[test]
    fn boltz_sidecars_are_found_and_single_chain_iptm_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("input_model_0.pdb");
        std::fs::write(&model, "").unwrap();
        std::fs::write(
            dir.path().join("pae_input_model_0.npz"),
            npz(&[("pae.npy", npy_f32(&[2, 2], &[0., 1., 1., 0.]))]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("confidence_input_model_0.json"),
            r#"{"ptm": 0.914, "iptm": 0.0, "complex_plddt": 0.93}"#,
        )
        .unwrap();
        let c = read_confidence(&model, None).unwrap();
        assert_eq!(c.pae.unwrap().n, 2);
        assert_eq!(c.ptm, Some(0.914));
        assert_eq!(c.iptm, None);
        assert_eq!(c.sources.len(), 2);
    }

    #[test]
    fn alphafold_db_and_colabfold_names_map_to_their_pae_files() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        for f in [
            "AF-P69905-F1-model_v6.pdb",
            "AF-P69905-F1-predicted_aligned_error_v6.json",
            "x_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb",
            "x_scores_rank_001_alphafold2_ptm_model_1_seed_000.json",
            "fold_y_model_0.cif",
            "fold_y_full_data_0.json",
            "fold_y_summary_confidences_0.json",
        ] {
            std::fs::write(d.join(f), "").unwrap();
        }
        let (pae, _) = sidecar_files(&d.join("AF-P69905-F1-model_v6.pdb"));
        assert_eq!(
            pae.unwrap(),
            d.join("AF-P69905-F1-predicted_aligned_error_v6.json")
        );
        let (pae, scores) =
            sidecar_files(&d.join("x_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"));
        let want = d.join("x_scores_rank_001_alphafold2_ptm_model_1_seed_000.json");
        assert_eq!((pae.as_ref(), scores.as_ref()), (Some(&want), Some(&want)));
        let (pae, scores) = sidecar_files(&d.join("fold_y_model_0.cif"));
        assert_eq!(pae.unwrap(), d.join("fold_y_full_data_0.json"));
        assert_eq!(scores.unwrap(), d.join("fold_y_summary_confidences_0.json"));
        assert_eq!(sidecar_files(&d.join("plain.pdb")), (None, None));

        // AlphaFold 3 run locally: the top model and one sample's directory.
        let run = d.join("design_7");
        std::fs::create_dir_all(run.join("seed-1_sample-0")).unwrap();
        for f in [
            "design_7_model.cif",
            "design_7_confidences.json",
            "design_7_summary_confidences.json",
            "seed-1_sample-0/model.cif",
            "seed-1_sample-0/confidences.json",
            "seed-1_sample-0/summary_confidences.json",
        ] {
            std::fs::write(run.join(f), "").unwrap();
        }
        let (pae, scores) = sidecar_files(&run.join("design_7_model.cif"));
        assert_eq!(pae.unwrap(), run.join("design_7_confidences.json"));
        assert_eq!(
            scores.unwrap(),
            run.join("design_7_summary_confidences.json")
        );
        let (pae, scores) = sidecar_files(&run.join("seed-1_sample-0/model.cif"));
        assert_eq!(pae.unwrap(), run.join("seed-1_sample-0/confidences.json"));
        assert_eq!(
            scores.unwrap(),
            run.join("seed-1_sample-0/summary_confidences.json")
        );
    }

    #[test]
    fn ligand_tokens_collapse_to_one_row_and_column() {
        // Two residues, then a three-atom ligand.
        let mut v = vec![1.0f32; 25];
        v[0] = 0.0;
        v[6] = 0.0;
        let p = PredictedAlignedError::new(5, v, None).unwrap();
        let c = p.collapse(&[vec![0], vec![1], vec![2, 3, 4]]).unwrap();
        assert_eq!(c.n, 3);
        assert_eq!(c.get(0, 0), 0.0);
        assert_eq!(c.get(2, 2), 1.0);
        assert!(p.collapse(&[vec![0], vec![9]]).is_none());
    }

    /// AlphaFold 3 / Boltz tokens in file order: GLY (1 token), SEP, a modified residue, one per
    /// heavy atom (N, CA, C, O, P: tokens 1–5, its row the CA's, 2), ALA (6), a three-atom
    /// ligand between chains (7–9), then chain B's ALA (10) and a water, which predictors do
    /// not tokenise.
    #[test]
    fn token_rows_follow_the_file_order() {
        let pdb = crate::io::open_structure_bytes(TOKEN_TEST_PDB, Some("x.pdb")).unwrap();
        assert_eq!(protein_token_rows(&pdb), (11, vec![0, 2, 6, 10]));
    }

    #[test]
    fn boltz_chain_scores_are_read_in_chain_order() {
        let v = serde_json::json!({
            "chains_ptm": {"1": 0.8, "0": 0.9},
            "pair_chains_iptm": {"0": {"0": 0.9, "1": 0.7}, "1": {"0": 0.7, "1": 0.8}}
        });
        let (c, p) = chain_scores_from_json(&v);
        assert_eq!(c, vec![0.9, 0.8]);
        assert_eq!(p, vec![vec![0.9, 0.7], vec![0.7, 0.8]]);
    }

    #[test]
    fn block_mean_is_symmetrised() {
        let p = PredictedAlignedError::new(2, vec![0.0, 2.0, 6.0, 0.0], None).unwrap();
        assert_eq!(p.block_mean(0..1, 1..2), 4.0);
    }
}
