//! Side-chain rotamers against cctbx `rotalyze` (MolProbity Top8000).
//!
//! Reference: `validate/reference/geometry/*.json.gz`, key `rotalyze`, written by
//! `validate/geometry_ref.py` (the structure reduced as `protein_heavy_atoms` reduces it, then
//! `rotalyze(outliers_only=False)`). Ignored by default because it needs the downloaded corpus:
//!
//! ```text
//! cargo test -p proteus-core --release --test rotamer_validation -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use proteus_core::rotamer::{evaluate_rotamers, RotamerEvaluation, RotamerResult};
use serde::Deserialize;

/// χ within 0.01°, score within 0.01 (percent); the reference rounds χ to 3 and the score to 4
/// decimals.
const CHI_TOL: f64 = 0.01;
const SCORE_TOL: f64 = 0.01;

#[derive(Deserialize)]
struct Reference {
    rotalyze: BTreeMap<String, RefRotamer>,
}

#[derive(Deserialize)]
struct RefRotamer {
    resname: String,
    score: f64,
    evaluation: String,
    rotamer: String,
    chi: Vec<Option<f64>>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate")
}

/// `<id>_pdb` → `corpus/<id>.pdb`, `<id>_cif` → `corpus/<id>.cif`,
/// `esmfold_<id>` → `predicted/esmfold/<id>.pdb`.
fn structure_path(stem: &str) -> PathBuf {
    if let Some(id) = stem.strip_prefix("esmfold_") {
        return root().join("predicted/esmfold").join(format!("{id}.pdb"));
    }
    let (id, fmt) = stem.rsplit_once('_').unwrap();
    root().join("corpus").join(format!("{id}.{fmt}"))
}

fn load(path: &Path) -> pdbtbx::PDB {
    let loaded = proteus_core::io::load_structure(path).unwrap();
    proteus_core::io::protein_heavy_atoms(&loaded.pdb)
}

fn key(r: &RotamerResult) -> String {
    format!(
        "{}:{}:{}",
        r.chain_id.trim(),
        r.seq_num,
        r.insertion_code.as_deref().unwrap_or("").trim()
    )
}

fn evaluation_label(e: RotamerEvaluation) -> &'static str {
    match e {
        RotamerEvaluation::Favored => "Favored",
        RotamerEvaluation::Allowed => "Allowed",
        RotamerEvaluation::Outlier => "OUTLIER",
    }
}

/// Angular difference, so 359.9996 (rounded by the reference to 360.0) matches 0.0004.
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

#[derive(Default)]
struct Row {
    id: String,
    n_ref: usize,
    n_ours: usize,
    matched: usize,
    max_chi: f64,
    max_score: f64,
    eval_mismatch: usize,
    name_mismatch: usize,
    ms: f64,
}

#[test]
#[ignore]
fn rotamers_match_cctbx_rotalyze() {
    let mut refs: Vec<PathBuf> = std::fs::read_dir(root().join("reference/geometry"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".json.gz"))
        .collect();
    refs.sort();
    assert!(!refs.is_empty(), "no reference files");

    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut counts = BTreeMap::<&str, (usize, usize)>::new();

    for path in refs {
        let stem = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".json.gz")
            .to_string();
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let reference: Reference = serde_json::from_str(&text).unwrap();
        let structure = structure_path(&stem);
        assert!(structure.exists(), "missing {}", structure.display());

        let pdb = load(&structure);
        let t = Instant::now();
        let ours = evaluate_rotamers(&pdb);
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let mut by_key: BTreeMap<String, &RotamerResult> = BTreeMap::new();
        for r in &ours {
            if by_key.insert(key(r), r).is_some() {
                failures.push(format!("{stem}: duplicate residue key {}", key(r)));
            }
        }
        let mut row = Row {
            id: stem.clone(),
            n_ref: reference.rotalyze.len(),
            n_ours: ours.len(),
            ..Row::default()
        };
        for k in by_key.keys() {
            if !reference.rotalyze.contains_key(k) {
                failures.push(format!("{stem} {k}: evaluated by Proteus, not by cctbx"));
            }
        }
        for (k, want) in &reference.rotalyze {
            let Some(got) = by_key.get(k) else {
                failures.push(format!(
                    "{stem} {k} {}: evaluated by cctbx, not by Proteus",
                    want.resname
                ));
                continue;
            };
            row.matched += 1;
            let c = counts.entry(evaluation_label(got.evaluation)).or_default();
            c.0 += 1;
            if got.resname != want.resname {
                failures.push(format!(
                    "{stem} {k}: resname {} vs {}",
                    got.resname, want.resname
                ));
            }
            let want_chi: Vec<f64> = want.chi.iter().flatten().copied().collect();
            if got.chi.len() != want_chi.len() {
                failures.push(format!("{stem} {k}: chi {:?} vs {:?}", got.chi, want_chi));
            } else {
                for (a, b) in got.chi.iter().zip(&want_chi) {
                    row.max_chi = row.max_chi.max(angle_diff(*a, *b));
                    if angle_diff(*a, *b) > CHI_TOL {
                        failures.push(format!("{stem} {k}: chi {:?} vs {:?}", got.chi, want_chi));
                        break;
                    }
                }
            }
            let ds = (got.score - want.score).abs();
            row.max_score = row.max_score.max(ds);
            if ds > SCORE_TOL {
                failures.push(format!("{stem} {k}: score {} vs {}", got.score, want.score));
            }
            if evaluation_label(got.evaluation) != want.evaluation {
                row.eval_mismatch += 1;
                failures.push(format!(
                    "{stem} {k} {}: evaluation {:?} (score {:.4}) vs {} (score {})",
                    want.resname, got.evaluation, got.score, want.evaluation, want.score
                ));
            } else {
                c.1 += 1;
            }
            if got.name.as_deref() != Some(want.rotamer.as_str()) {
                row.name_mismatch += 1;
                failures.push(format!(
                    "{stem} {k} {}: rotamer {:?} vs {}",
                    want.resname, got.name, want.rotamer
                ));
            }
        }
        row.ms = ms;
        rows.push(row);
    }

    println!(
        "{:<18} {:>6} {:>6} {:>7} {:>10} {:>10} {:>5} {:>5} {:>8}",
        "structure", "cctbx", "ours", "matched", "max|dchi|", "max|dscore|", "eval", "name", "ms"
    );
    for r in &rows {
        println!(
            "{:<18} {:>6} {:>6} {:>7} {:>10.2e} {:>10.2e} {:>5} {:>5} {:>8.2}",
            r.id,
            r.n_ref,
            r.n_ours,
            r.matched,
            r.max_chi,
            r.max_score,
            r.eval_mismatch,
            r.name_mismatch,
            r.ms
        );
    }
    let total_ref: usize = rows.iter().map(|r| r.n_ref).sum();
    let total_matched: usize = rows.iter().map(|r| r.matched).sum();
    let max_chi = rows.iter().map(|r| r.max_chi).fold(0.0, f64::max);
    let max_score = rows.iter().map(|r| r.max_score).fold(0.0, f64::max);
    println!(
        "\n{} structures, {total_matched}/{total_ref} residues compared to the reference, max |dchi| {max_chi:.2e} deg, \
         max |dscore| {max_score:.2e} %",
        rows.len()
    );
    for (label, (n, agree)) in &counts {
        println!("  {label:<8} {agree}/{n} evaluations agree");
    }
    if !failures.is_empty() {
        for f in failures.iter().take(60) {
            eprintln!("{f}");
        }
        panic!("{} rotamer mismatches against cctbx", failures.len());
    }
}

/// Evaluation time on 1AON (GroEL–GroES, ~58k atoms), cold (table decoding) and warm.
#[test]
#[ignore]
fn rotamer_timing_1aon() {
    let pdb = load(&root().join("corpus/1aon.pdb"));
    let atoms = pdb.atom_count();
    let t = Instant::now();
    let n = evaluate_rotamers(&pdb).len();
    let cold = t.elapsed();
    let t = Instant::now();
    let runs = 10;
    for _ in 0..runs {
        assert_eq!(evaluate_rotamers(&pdb).len(), n);
    }
    let warm = t.elapsed() / runs;
    println!("1aon: {atoms} atoms, {n} rotamers, cold {cold:?}, warm {warm:?}");
}
