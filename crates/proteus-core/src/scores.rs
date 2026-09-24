//! Per-residue and per-substitution scores read from a table, for colouring a structure by
//! them: a deep mutational scan, a variant-effect predictor, or any per-position value.
//!
//! Three layouts are recognised, from CSV/TSV or JSON (an array of objects):
//! - **Matrix**, as `proteus esm scan --export` writes: a position column, a `wt` column and one
//!   column per amino acid (`position,wt,A,C,…,Y`).
//! - **Variants**, one row per substitution: a column of `L43A`-style names (AlphaMissense's
//!   `protein_variant`, a `mutation` column, or a `header` holding `mutation=L43A` as
//!   `proteus screen` exports write) and a numeric column.
//! - **Per residue**: a position column (`position`, `residue`, `resi`, …) and a numeric column.
//!
//! A position's value is its own for the per-residue layout and the mean over its substitutions
//! otherwise. Positions are matched to the structure by residue number or by sequence index,
//! whichever agrees with the file's wild-type letters; a table that fits neither is refused.

use crate::error::CoreError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// The twenty amino acids in the order the matrix is stored.
pub const AMINO_ACIDS: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

fn aa_index(c: char) -> Option<usize> {
    AMINO_ACIDS
        .iter()
        .position(|a| *a == c.to_ascii_uppercase())
}

/// Scores read from a table, before they are placed on a structure.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScoreTable {
    /// The column the values came from.
    pub column: String,
    /// Position (as written in the file) → wild-type letter, when the file says.
    pub wild_type: HashMap<isize, char>,
    /// Position → value of the position itself (per-residue layout).
    pub per_position: HashMap<isize, f64>,
    /// Position → substitution → value (matrix and variant layouts).
    pub substitutions: HashMap<isize, HashMap<char, f64>>,
}

/// Scores placed on a structure's residues, in ribbon order.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueScores {
    pub column: String,
    /// One value per residue: the position's own, or the mean of its substitutions.
    pub values: Vec<Option<f64>>,
    /// Per residue, one value per [`AMINO_ACIDS`] entry, when the file has substitutions.
    pub matrix: Option<Vec<[Option<f32>; 20]>>,
    /// How positions were matched: `"residue number"` or `"sequence index"`.
    pub matched_by: &'static str,
    /// Residues that received a value.
    pub covered: usize,
    /// Wild-type letters in the file that disagree with the structure.
    pub mismatches: usize,
}

fn err(msg: impl Into<String>) -> CoreError {
    CoreError::ParseError(msg.into())
}

/// `L43A` → (L, 43, A). Also accepts `p.Leu43Ala`-free short forms with a negative or an
/// insertion-free number; anything else is `None`.
pub fn parse_variant(s: &str) -> Option<(char, isize, char)> {
    let s = s.trim();
    let s = s.strip_prefix("p.").unwrap_or(s);
    // Letter, at least one digit, letter: anything shorter cannot be a variant (and would
    // make the slice below cross itself).
    if s.chars().count() < 3 {
        return None;
    }
    let mut chars = s.chars();
    let wt = chars.next()?;
    let mt = s.chars().last()?;
    let num = &s[wt.len_utf8()..s.len() - mt.len_utf8()];
    if !wt.is_ascii_alphabetic() || !(mt.is_ascii_alphabetic() || mt == '*') || num.is_empty() {
        return None;
    }
    Some((
        wt.to_ascii_uppercase(),
        num.parse().ok()?,
        mt.to_ascii_uppercase(),
    ))
}

/// A `mutation=L43A` tag inside a longer string (a `proteus screen` header).
fn variant_in(s: &str) -> Option<(char, isize, char)> {
    let at = s.find("mutation=")? + "mutation=".len();
    let rest = &s[at..];
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '-')
        .unwrap_or(rest.len());
    parse_variant(&rest[..end])
}

const POSITION_COLUMNS: [&str; 7] = [
    "position",
    "pos",
    "residue",
    "resi",
    "resid",
    "residue_number",
    "resnum",
];

/// Rows as (header → cell text), from CSV/TSV or a JSON array of objects.
/// A table: its column names, then one map of column → cell text per row.
type Rows = (Vec<String>, Vec<HashMap<String, String>>);

fn read_rows(path: &Path) -> Result<Rows, CoreError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let is_json = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        || text.trim_start().starts_with('[');
    if is_json {
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| err(format!("{}: not JSON: {e}", path.display())))?;
        let arr = v
            .as_array()
            .ok_or_else(|| err(format!("{}: expected an array of rows", path.display())))?;
        let mut headers: Vec<String> = Vec::new();
        let mut rows = Vec::with_capacity(arr.len());
        for row in arr {
            let obj = row
                .as_object()
                .ok_or_else(|| err(format!("{}: rows must be objects", path.display())))?;
            let mut m = HashMap::new();
            for (k, v) in obj {
                if !headers.contains(k) {
                    headers.push(k.clone());
                }
                let cell = match v {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                m.insert(k.clone(), cell);
            }
            rows.push(m);
        }
        return Ok((headers, rows));
    }
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let head = lines
        .next()
        .ok_or_else(|| err(format!("{}: empty table", path.display())))?;
    let sep = if head.contains('\t') { '\t' } else { ',' };
    let split = |l: &str| -> Vec<String> {
        l.split(sep)
            .map(|c| c.trim().trim_matches('"').to_string())
            .collect()
    };
    let headers = split(head);
    let rows = lines
        .map(|l| {
            headers
                .iter()
                .cloned()
                .zip(split(l))
                .collect::<HashMap<_, _>>()
        })
        .collect();
    Ok((headers, rows))
}

fn number(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("nan") || s.eq_ignore_ascii_case("na") {
        return None;
    }
    s.parse::<f64>().ok().filter(|x| x.is_finite())
}

/// Read a score table. `column` picks the value column; without it the first numeric column
/// that is not a position, rank or length is used.
pub fn read_score_table(path: &Path, column: Option<&str>) -> Result<ScoreTable, CoreError> {
    let (headers, rows) = read_rows(path)?;
    if rows.is_empty() {
        return Err(err(format!("{}: no rows", path.display())));
    }
    let lower = |h: &str| h.trim().to_ascii_lowercase();
    let find = |names: &[&str]| {
        headers
            .iter()
            .find(|h| names.contains(&lower(h).as_str()))
            .cloned()
    };
    let position_col = find(&POSITION_COLUMNS);
    let wt_col = find(&["wt", "wildtype", "wild_type", "ref"]);
    let aa_cols: Vec<(String, char)> = headers
        .iter()
        .filter_map(|h| {
            let t = h.trim();
            (t.len() == 1)
                .then(|| {
                    t.chars()
                        .next()
                        .and_then(|c| aa_index(c).map(|_| (h.clone(), c)))
                })
                .flatten()
        })
        .collect();
    let mut t = ScoreTable::default();

    // Matrix: position, wt and at least ten amino-acid columns.
    if let (Some(pc), true) = (&position_col, aa_cols.len() >= 10) {
        t.column = column.unwrap_or("mean over substitutions").to_string();
        for r in &rows {
            let Some(pos) = r.get(pc).and_then(|s| s.trim().parse::<isize>().ok()) else {
                continue;
            };
            if let Some(w) = wt_col
                .as_ref()
                .and_then(|c| r.get(c))
                .and_then(|s| s.trim().chars().next())
            {
                t.wild_type.insert(pos, w.to_ascii_uppercase());
            }
            let subs = t.substitutions.entry(pos).or_default();
            for (h, aa) in &aa_cols {
                if let Some(v) = r.get(h).and_then(|s| number(s)) {
                    subs.insert(*aa, v);
                }
            }
        }
        return Ok(t);
    }

    // Variant names: a column (or a header tag) where most cells parse as L43A.
    let variant_col = headers.iter().find(|h| {
        let hits = rows
            .iter()
            .take(50)
            .filter(|r| {
                r.get(*h)
                    .is_some_and(|s| parse_variant(s).is_some() || variant_in(s).is_some())
            })
            .count();
        hits * 2 > rows.len().min(50)
    });
    let skip = |h: &str| {
        let l = lower(h);
        POSITION_COLUMNS.contains(&l.as_str())
            || matches!(l.as_str(), "rank" | "length" | "len" | "index" | "")
            || variant_col.is_some_and(|v| v == h)
    };
    let value_col = match column {
        Some(c) => headers
            .iter()
            .find(|h| h.as_str() == c)
            .cloned()
            .ok_or_else(|| {
                err(format!(
                    "{}: no column '{c}' (columns: {})",
                    path.display(),
                    headers.join(", ")
                ))
            })?,
        None => headers
            .iter()
            .find(|h| {
                !skip(h)
                    && rows
                        .iter()
                        .take(20)
                        .filter(|r| r.get(*h).and_then(|s| number(s)).is_some())
                        .count()
                        * 2
                        > rows.len().min(20)
            })
            .cloned()
            .ok_or_else(|| {
                err(format!(
                    "{}: no numeric column to colour by (columns: {})",
                    path.display(),
                    headers.join(", ")
                ))
            })?,
    };
    t.column = value_col.clone();
    if let Some(vc) = variant_col {
        for r in &rows {
            let cell = r.get(vc).map(String::as_str).unwrap_or("");
            let Some((wt, pos, mt)) = parse_variant(cell).or_else(|| variant_in(cell)) else {
                continue;
            };
            let Some(v) = r.get(&value_col).and_then(|s| number(s)) else {
                continue;
            };
            t.wild_type.insert(pos, wt);
            if mt != '*' && aa_index(mt).is_some() {
                t.substitutions.entry(pos).or_default().insert(mt, v);
            }
        }
        return Ok(t);
    }
    if let Some(pc) = position_col {
        for r in &rows {
            let (Some(pos), Some(v)) = (
                r.get(&pc).and_then(|s| s.trim().parse::<isize>().ok()),
                r.get(&value_col).and_then(|s| number(s)),
            ) else {
                continue;
            };
            t.per_position.insert(pos, v);
            if let Some(w) = wt_col
                .as_ref()
                .and_then(|c| r.get(c))
                .and_then(|s| s.trim().chars().next())
            {
                t.wild_type.insert(pos, w.to_ascii_uppercase());
            }
        }
        return Ok(t);
    }
    Err(err(format!(
        "{}: found neither a position column ({}) nor a column of variants like L43A",
        path.display(),
        POSITION_COLUMNS.join(", ")
    )))
}

impl ScoreTable {
    fn positions(&self) -> Vec<isize> {
        let mut p: Vec<isize> = self
            .per_position
            .keys()
            .chain(self.substitutions.keys())
            .copied()
            .collect();
        p.sort_unstable();
        p.dedup();
        p
    }

    /// Place the table on residues given by their numbers and one-letter codes (ribbon order).
    /// Residue numbers are tried first, then 1-based sequence indices; the mapping whose
    /// wild-type letters agree best wins, and one with under 90 % agreement is refused.
    pub fn place(&self, numbers: &[isize], letters: &[char]) -> Result<ResidueScores, CoreError> {
        let positions = self.positions();
        if positions.is_empty() {
            return Err(err("the table has no values"));
        }
        let by_number: HashMap<isize, usize> =
            numbers.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        let by_index = |p: isize| (p >= 1 && (p as usize) <= numbers.len()).then(|| p as usize - 1);
        let score = |map: &dyn Fn(isize) -> Option<usize>| {
            let (mut placed, mut agree, mut checked) = (0usize, 0usize, 0usize);
            for p in &positions {
                if let Some(i) = map(*p) {
                    placed += 1;
                    if let Some(w) = self.wild_type.get(p) {
                        checked += 1;
                        agree += (letters[i] == *w) as usize;
                    }
                }
            }
            (placed, agree, checked)
        };
        let num_map = |p: isize| by_number.get(&p).copied();
        let a = score(&num_map);
        let b = score(&by_index);
        // With wild-type letters, agreement decides; without, coverage does.
        let key = |(placed, agree, checked): (usize, usize, usize)| {
            ((agree * 1000).checked_div(checked).unwrap_or(0), placed)
        };
        let (use_number, (placed, agree, checked)) = if key(a) >= key(b) {
            (true, a)
        } else {
            (false, b)
        };
        if placed == 0 {
            return Err(err(format!(
                "none of the table's positions ({}–{}) is a residue of the structure",
                positions[0],
                positions[positions.len() - 1]
            )));
        }
        if checked > 0 && agree * 10 < checked * 9 {
            return Err(err(format!(
                "the table's wild-type letters match the structure at {agree} of {checked} \
                 positions: it is for another sequence or another numbering"
            )));
        }
        let map = |p: isize| {
            if use_number {
                num_map(p)
            } else {
                by_index(p)
            }
        };
        let mut values = vec![None; numbers.len()];
        let mut matrix = (!self.substitutions.is_empty()).then(|| vec![[None; 20]; numbers.len()]);
        for p in &positions {
            let Some(i) = map(*p) else { continue };
            if let Some(v) = self.per_position.get(p) {
                values[i] = Some(*v);
            }
            if let Some(subs) = self.substitutions.get(p) {
                // The wild type scores 0 in a scan by construction; it is not a substitution.
                let wt = self.wild_type.get(p).copied().unwrap_or(letters[i]);
                let vals: Vec<f64> = subs
                    .iter()
                    .filter(|(aa, _)| **aa != wt)
                    .map(|(_, v)| *v)
                    .collect();
                if values[i].is_none() && !vals.is_empty() {
                    values[i] = Some(vals.iter().sum::<f64>() / vals.len() as f64);
                }
                if let Some(m) = matrix.as_mut() {
                    for (aa, v) in subs {
                        if let Some(k) = aa_index(*aa) {
                            m[i][k] = Some(*v as f32);
                        }
                    }
                }
            }
        }
        Ok(ResidueScores {
            column: self.column.clone(),
            covered: values.iter().filter(|v| v.is_some()).count(),
            values,
            matrix,
            matched_by: if use_number {
                "residue number"
            } else {
                "sequence index"
            },
            mismatches: checked - agree,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &tempfile::TempDir, name: &str, text: &str) -> std::path::PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn variant_names_parse() {
        assert_eq!(parse_variant("L43A"), Some(('L', 43, 'A')));
        assert_eq!(parse_variant("p.M1C"), Some(('M', 1, 'C')));
        assert_eq!(parse_variant("K-2E"), Some(('K', -2, 'E')));
        assert_eq!(parse_variant("W12*"), Some(('W', 12, '*')));
        assert_eq!(parse_variant("43"), None);
        assert_eq!(parse_variant("1"), None);
        assert_eq!(parse_variant("AA"), None);
        assert_eq!(parse_variant("rank"), None);
        assert_eq!(
            variant_in("1crn_crambin_P5A [mutation=P5A]"),
            Some(('P', 5, 'A'))
        );
    }

    #[test]
    fn an_esm_scan_matrix_averages_substitutions_and_skips_the_wild_type() {
        let dir = tempfile::tempdir().unwrap();
        let mut csv = String::from("position,wt");
        for aa in AMINO_ACIDS {
            csv.push_str(&format!(",{aa}"));
        }
        csv.push('\n');
        // Position 1 (M): every substitution -2, the wild type 0.
        csv.push_str("1,M");
        for aa in AMINO_ACIDS {
            csv.push_str(if aa == 'M' { ",0" } else { ",-2" });
        }
        csv.push('\n');
        let t = read_score_table(&write(&dir, "scan.csv", &csv), None).unwrap();
        let s = t.place(&[1, 2], &['M', 'Q']).unwrap();
        assert_eq!(s.values, vec![Some(-2.0), None]);
        let m = s.matrix.unwrap();
        assert_eq!(m[0][aa_index('M').unwrap()], Some(0.0));
        assert_eq!(m[0][aa_index('A').unwrap()], Some(-2.0));
        assert_eq!(s.matched_by, "residue number");
    }

    #[test]
    fn alphamissense_rows_and_a_named_column() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(
            &dir,
            "am.csv",
            "protein_variant,am_pathogenicity,am_class\nM1A,0.4,Amb\nM1C,0.6,Amb\nV2D,0.9,LPath\n",
        );
        let t = read_score_table(&p, None).unwrap();
        assert_eq!(t.column, "am_pathogenicity");
        let s = t.place(&[1, 2], &['M', 'V']).unwrap();
        assert_eq!(s.values[0], Some(0.5));
        assert_eq!(s.values[1], Some(0.9));
        assert!(read_score_table(&p, Some("nope")).is_err());
    }

    #[test]
    fn numbering_falls_back_to_sequence_index_when_letters_say_so() {
        let dir = tempfile::tempdir().unwrap();
        // A crystal structure numbered from 101; the table numbers the sequence from 1.
        let p = write(&dir, "v.csv", "mutation,score\nA1G,1.0\nC2G,2.0\nD3G,3.0\n");
        let t = read_score_table(&p, None).unwrap();
        let s = t.place(&[101, 102, 103], &['A', 'C', 'D']).unwrap();
        assert_eq!(s.matched_by, "sequence index");
        assert_eq!(s.values, vec![Some(1.0), Some(2.0), Some(3.0)]);
    }

    #[test]
    fn a_table_for_another_sequence_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(&dir, "v.csv", "mutation,score\nW1G,1\nW2G,1\nW3G,1\n");
        let t = read_score_table(&p, None).unwrap();
        let e = t
            .place(&[1, 2, 3], &['A', 'C', 'D'])
            .unwrap_err()
            .to_string();
        assert!(e.contains("another sequence"), "{e}");
    }

    #[test]
    fn per_residue_json_rows() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(
            &dir,
            "r.json",
            r#"[{"residue": 1, "conservation": 0.2}, {"residue": 2, "conservation": 0.8}]"#,
        );
        let t = read_score_table(&p, None).unwrap();
        let s = t.place(&[1, 2], &['A', 'C']).unwrap();
        assert_eq!(s.values, vec![Some(0.2), Some(0.8)]);
        assert!(s.matrix.is_none());
    }
}
