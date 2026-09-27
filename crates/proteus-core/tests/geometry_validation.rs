//! Covalent-geometry checks against cctbx, structure by structure.
//!
//! `validate/geometry_reference.py` runs cctbx (Phenix's restraint library: geostd, CDL v1.2,
//! EH99 cis-Pro; `mmtbx.validation` cbetadev and omegalyze) on every corpus entry and on the
//! committed ESMFold models, after reducing each structure the way Proteus does. This test
//! reproduces the numbers with `proteus_core::geometry` and requires:
//!
//! * the same number of bond, angle, chirality and planarity restraints;
//! * the same > 4σ outliers, except restraints within `z_boundary` of the cutoff, where the
//!   fifth decimal of a coordinate decides;
//! * the same RMSZ, Cβ deviations and outlier flags, and ω and ω types,
//!
//! to the tolerances in `[geometry]` of `validate/tolerances.toml`.
//!
//! Run with `make validate`. With `PROTEUS_GEOMETRY_FULL=<dir>` pointing at `--full` dumps
//! (`geometry_reference.py --full`), every individual restraint's target, σ and model value is
//! compared as well.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use proteus_core::geometry::{self, RestraintKind};
use serde::Deserialize;

/// `[geometry]` of validate/tolerances.toml.
#[derive(Deserialize)]
struct Tolerances {
    z_boundary: f64,
    rmsz_rel: f64,
    cbeta_abs: f64,
    omega_abs_deg: f64,
    ideal_abs: f64,
    model_abs: f64,
}

fn tolerances() -> Tolerances {
    #[derive(Deserialize)]
    struct File {
        geometry: Tolerances,
    }
    let text = std::fs::read_to_string(root().join("tolerances.toml")).unwrap();
    toml::from_str::<File>(&text).unwrap().geometry
}

#[derive(Deserialize)]
struct Reference {
    #[serde(default)]
    kind: String,
    restraints: HashMap<String, RefKind>,
    cbetadev: HashMap<String, RefCbeta>,
    omegalyze: HashMap<String, RefOmega>,
}

#[derive(Deserialize)]
struct RefKind {
    n: usize,
    rmsz: Option<f64>,
    outliers: Vec<RefRestraint>,
    #[serde(default)]
    all: Vec<RefRestraint>,
}

#[derive(Deserialize, Clone)]
struct RefRestraint {
    atoms: Vec<String>,
    ideal: f64,
    sigma: f64,
    model: f64,
    z: f64,
}

#[derive(Deserialize)]
struct RefCbeta {
    deviation: f64,
    outlier: bool,
}

#[derive(Deserialize)]
struct RefOmega {
    omega: f64,
    #[serde(rename = "type")]
    omega_type: String,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate")
}

/// cctbx label "A:52:A:ASN:CA" → Proteus label "A 52A ASN CA".
fn label(cctbx: &str) -> String {
    let p: Vec<&str> = cctbx.split(':').collect();
    format!("{} {}{} {} {}", p[0], p[1], p[2], p[3], p[4])
}

/// Order-free identity of a restraint: kind plus the sorted atom labels.
fn key(kind: &str, atoms: &[String]) -> (String, Vec<String>) {
    let mut a = atoms.to_vec();
    a.sort();
    (kind.to_string(), a)
}

fn kind_name(k: RestraintKind) -> &'static str {
    match k {
        RestraintKind::Bond => "bond",
        RestraintKind::Angle => "angle",
        RestraintKind::Chirality => "chirality",
        RestraintKind::Planarity => "planarity",
    }
}

fn read_gz(path: &Path) -> String {
    let mut s = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

fn structure_path(stem: &str) -> PathBuf {
    if let Some(id) = stem.strip_prefix("esmfold_") {
        return root().join("predicted/esmfold").join(format!("{id}.pdb"));
    }
    let (id, fmt) = stem.rsplit_once('_').unwrap();
    root().join("corpus").join(format!("{id}.{fmt}"))
}

fn residue_key(id: &geometry::ResidueId) -> String {
    format!("{}:{}:{}", id.chain, id.seq_num, id.insertion_code)
}

#[test]
#[ignore]
fn covalent_geometry_matches_cctbx() {
    let tol = tolerances();
    let full_dir = std::env::var("PROTEUS_GEOMETRY_FULL")
        .ok()
        .map(PathBuf::from);
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root().join("reference/geometry"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".json.gz"))
        .collect();
    entries.sort();
    assert!(
        !entries.is_empty(),
        "no references; run geometry_reference.py"
    );
    // `PROTEUS_GEOMETRY_ONLY=1crn_pdb,esmfold_1crn_A` restricts the run while debugging.
    if let Ok(only) = std::env::var("PROTEUS_GEOMETRY_ONLY") {
        let only: HashSet<&str> = only.split(',').collect();
        entries.retain(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            only.contains(name.trim_end_matches(".json.gz"))
        });
    }

    let mut failures: Vec<String> = Vec::new();
    // Outliers that differ only within tol.z_boundary of the cutoff, and the largest RMSZ gap.
    let mut boundary = 0usize;
    let mut worst_rmsz = 0.0f64;
    let mut table = String::from(
        "| structure | kind | bonds | bond out (ours/cctbx) | bond RMSZ | angles | angle out | chir | planes | Cβ out | cis/twisted |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for path in &entries {
        let stem = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches(".json.gz")
            .to_string();
        let r: Reference = serde_json::from_str(&read_gz(path)).unwrap();
        let file = structure_path(&stem);
        if !file.exists() {
            if std::env::var("PROTEUS_VALIDATE_OFFLINE").is_ok() {
                continue;
            }
            panic!("missing {} — run `make fetch`", file.display());
        }
        let loaded = proteus_core::io::load_structure(&file).unwrap();
        let protein = proteus_core::io::protein_heavy_atoms(&loaded.pdb);
        let g = geometry::analyze(&protein);
        let details = geometry::restraint_details(&protein);
        let mut fail = |msg: String| failures.push(format!("{stem}: {msg}"));

        // Restraint counts, RMSZ, outliers.
        let mut by_kind: BTreeMap<&str, Vec<&geometry::RestraintDetail>> = BTreeMap::new();
        for d in &details {
            by_kind.entry(kind_name(d.kind)).or_default().push(d);
        }
        let stats = [
            ("bond", &g.bonds),
            ("angle", &g.angles),
            ("chirality", &g.chiralities),
            ("planarity", &g.planes),
        ];
        for (name, ours) in stats {
            let Some(theirs) = r.restraints.get(name) else {
                if ours.n != 0 {
                    fail(format!("{name}: {} restraints, cctbx none", ours.n));
                }
                continue;
            };
            if ours.n != theirs.n {
                fail(format!("{name}: {} restraints, cctbx {}", ours.n, theirs.n));
            }
            if let (Some(a), Some(b)) = (ours.rmsz, theirs.rmsz) {
                worst_rmsz = worst_rmsz.max((a - b).abs() / b.max(1.0));
                if (a - b).abs() > tol.rmsz_rel * b.max(1.0) {
                    fail(format!("{name}: RMSZ {a:.5}, cctbx {b:.5}"));
                }
            }
            let ours_out: HashMap<_, f64> = by_kind
                .get(name)
                .into_iter()
                .flatten()
                .filter(|d| d.z.abs() > geometry::SIGMA_CUTOFF)
                .map(|d| (key(name, &d.atoms), d.z))
                .collect();
            let all_ours: HashMap<_, f64> = by_kind
                .get(name)
                .into_iter()
                .flatten()
                .map(|d| (key(name, &d.atoms), d.z))
                .collect();
            let theirs_out: HashMap<_, f64> = theirs
                .outliers
                .iter()
                .map(|o| {
                    let atoms: Vec<String> = o.atoms.iter().map(|a| label(a)).collect();
                    (key(name, &atoms), o.z)
                })
                .collect();
            for (k, z) in &theirs_out {
                if !ours_out.contains_key(k) {
                    let near = (z.abs() - geometry::SIGMA_CUTOFF).abs() < tol.z_boundary;
                    if near {
                        boundary += 1;
                    } else {
                        fail(format!(
                            "{name} outlier missed: {:?} cctbx z {z:.2}, ours {:?}",
                            k.1,
                            all_ours.get(k)
                        ));
                    }
                }
            }
            for (k, z) in &ours_out {
                if !theirs_out.contains_key(k) {
                    if (z.abs() - geometry::SIGMA_CUTOFF).abs() < tol.z_boundary {
                        boundary += 1;
                    } else {
                        fail(format!("{name} extra outlier: {:?} z {z:.2}", k.1));
                    }
                }
            }
            // Per-restraint comparison when a full dump is available.
            if let Some(dir) = &full_dir {
                let full = dir.join(format!("{stem}.json"));
                if full.exists() {
                    let fr: Reference =
                        serde_json::from_str(&std::fs::read_to_string(&full).unwrap()).unwrap();
                    let ours_all: HashMap<_, &&geometry::RestraintDetail> = by_kind
                        .get(name)
                        .into_iter()
                        .flatten()
                        .map(|d| (key(name, &d.atoms), d))
                        .collect();
                    let mut seen = HashSet::new();
                    for t in &fr.restraints[name].all {
                        let atoms: Vec<String> = t.atoms.iter().map(|a| label(a)).collect();
                        let k = key(name, &atoms);
                        seen.insert(k.clone());
                        let Some(o) = ours_all.get(&k) else {
                            fail(format!(
                                "{name} missing: {atoms:?} ideal {} σ {}",
                                t.ideal, t.sigma
                            ));
                            continue;
                        };
                        let chiral = name == "chirality";
                        let ideal_ok = if chiral {
                            (o.ideal.abs() - t.ideal.abs()).abs() < tol.ideal_abs
                        } else {
                            (o.ideal - t.ideal).abs() < tol.ideal_abs
                        };
                        let model_ok = if chiral {
                            (o.model.abs() - t.model.abs()).abs() < tol.model_abs
                        } else {
                            (o.model - t.model).abs() < tol.model_abs
                        };
                        if !ideal_ok || (o.esd - t.sigma).abs() > tol.ideal_abs || !model_ok {
                            fail(format!(
                                "{name} {atoms:?}: ours ideal {:.4} σ {:.4} model {:.4}; cctbx ideal {} σ {} model {}",
                                o.ideal, o.esd, o.model, t.ideal, t.sigma, t.model
                            ));
                        }
                    }
                    for (k, d) in &ours_all {
                        if !seen.contains(k) {
                            fail(format!(
                                "{name} extra: {:?} ideal {:.4} σ {:.4}",
                                d.atoms, d.ideal, d.esd
                            ));
                        }
                    }
                }
            }
        }

        // Cβ deviation.
        let ours_cb: HashMap<String, f64> = g
            .cbeta
            .iter()
            .map(|c| (residue_key(&c.residue), c.deviation))
            .collect();
        if ours_cb.len() != r.cbetadev.len() {
            fail(format!(
                "Cβ: {} residues, cctbx {}",
                ours_cb.len(),
                r.cbetadev.len()
            ));
        }
        for (k, t) in &r.cbetadev {
            match ours_cb.get(k) {
                None => fail(format!("Cβ missing {k}")),
                Some(d) => {
                    if (d - t.deviation).abs() > tol.cbeta_abs {
                        fail(format!("Cβ {k}: {d:.4} vs {:.4}", t.deviation));
                    } else if (*d >= geometry::CBETA_OUTLIER) != t.outlier
                        && (d - geometry::CBETA_OUTLIER).abs() > tol.cbeta_abs
                    {
                        fail(format!("Cβ {k}: outlier flag differs at {d:.4}"));
                    }
                }
            }
        }

        // ω.
        let ours_w: HashMap<String, &geometry::OmegaResult> = g
            .omegas
            .iter()
            .map(|w| (residue_key(&w.residue), w))
            .collect();
        if ours_w.len() != r.omegalyze.len() {
            fail(format!(
                "ω: {} peptides, cctbx {}",
                ours_w.len(),
                r.omegalyze.len()
            ));
        }
        for (k, t) in &r.omegalyze {
            match ours_w.get(k) {
                None => fail(format!("ω missing {k}")),
                Some(w) => {
                    let ty = format!("{:?}", w.omega_type).to_lowercase();
                    if (w.omega - t.omega).abs() > tol.omega_abs_deg || ty != t.omega_type {
                        fail(format!(
                            "ω {k}: {:.3} {ty} vs {:.3} {}",
                            w.omega, t.omega, t.omega_type
                        ));
                    }
                }
            }
        }

        let theirs = |n: &str| r.restraints.get(n).map_or(0, |k| k.outliers.len());
        table.push_str(&format!(
            "| {stem} | {} | {} | {}/{} | {:.3} | {} | {}/{} | {} | {} | {} | {}/{} |\n",
            r.kind,
            g.bonds.n,
            g.bonds.outliers,
            theirs("bond"),
            g.bonds.rmsz.unwrap_or(0.0),
            g.angles.n,
            g.angles.outliers,
            theirs("angle"),
            g.chiralities.outliers,
            g.planes.outliers,
            g.cbeta_outliers,
            g.cis_nonproline + g.cis_proline,
            g.twisted,
        ));
    }
    println!("{table}");
    println!(
        "{} structures; outliers differing only within {} of 4σ: {boundary}; largest relative RMSZ difference: {worst_rmsz:.2e}",
        entries.len(),
        tol.z_boundary
    );
    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(80).collect();
        panic!(
            "{} disagreements with cctbx (first {}):\n{}",
            failures.len(),
            shown.len(),
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Wall time of the full covalent-geometry analysis (restraints, Cβ, ω, rotamers) on 1AON
/// (GroEL–GroES, 58 674 atoms after reduction), cold and warm.
#[test]
#[ignore]
fn geometry_timing_1aon() {
    let file = root().join("corpus/1aon.pdb");
    let loaded = proteus_core::io::load_structure(&file).unwrap();
    let protein = proteus_core::io::protein_heavy_atoms(&loaded.pdb);
    let t = std::time::Instant::now();
    let g = geometry::analyze(&protein);
    let cold = t.elapsed();
    let runs = 5;
    let t = std::time::Instant::now();
    for _ in 0..runs {
        assert_eq!(geometry::analyze(&protein).bonds.n, g.bonds.n);
    }
    let warm = t.elapsed() / runs;
    println!(
        "1aon: {} atoms, {} bonds, {} angles, {} rotamers; cold {cold:?}, warm {warm:?}",
        protein.atom_count(),
        g.bonds.n,
        g.angles.n,
        g.rotamer_residues
    );
}

/// The claim the geometry checks make about predicted models: an unrelaxed model (ESMFold,
/// as served) is told apart from the same model after an AlphaFold2-style restrained Amber
/// minimisation (`validate/predicted/relax.py`), although the two have the same fold.
#[test]
#[ignore]
fn relaxed_and_unrelaxed_models_separate() {
    let dir = root().join("predicted");
    let mut names: Vec<String> = std::fs::read_dir(dir.join("esmfold"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".pdb"))
        .collect();
    names.sort();
    assert_eq!(names.len(), 13);
    let peptide_cn = |details: &[geometry::RestraintDetail]| {
        let v: Vec<f64> = details
            .iter()
            .filter(|d| {
                d.kind == RestraintKind::Bond && d.origin == geometry::RestraintOrigin::Peptide
            })
            .map(|d| d.model)
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    println!(
        "| model | Cα RMSD | bond RMSZ (unrelaxed → relaxed) | bond > 4σ | angle RMSZ | mean peptide C–N (Å) | rotamer out % |\n|---|---|---|---|---|---|---|"
    );
    let mut failures = Vec::new();
    for name in &names {
        let load = |sub: &str| {
            let l = proteus_core::io::load_structure(&dir.join(sub).join(name)).unwrap();
            (proteus_core::io::protein_heavy_atoms(&l.pdb), l.pdb)
        };
        let (raw, raw_full) = load("esmfold");
        let (rel, rel_full) = load("esmfold_relaxed");
        let rmsd = proteus_core::qc::ca_rmsd(&rel_full, &raw_full).unwrap();
        let (a, b) = (geometry::analyze(&raw), geometry::analyze(&rel));
        let (ca, cb) = (
            peptide_cn(&geometry::restraint_details(&raw)),
            peptide_cn(&geometry::restraint_details(&rel)),
        );
        let z = |g: &geometry::CovalentGeometry| g.bonds.rmsz.unwrap();
        println!(
            "| {name} | {rmsd:.2} | {:.2} → {:.2} | {} → {} | {:.2} → {:.2} | {ca:.3} → {cb:.3} | {:.1} → {:.1} |",
            z(&a),
            z(&b),
            a.bonds.outliers,
            b.bonds.outliers,
            a.angles.rmsz.unwrap(),
            b.angles.rmsz.unwrap(),
            a.rotamer_outlier_pct().unwrap_or(0.0),
            b.rotamer_outlier_pct().unwrap_or(0.0),
        );
        // Every unrelaxed model has bond outliers and no relaxed one does, with the fold
        // unchanged: the separation is the bond geometry, not the structure.
        if a.bonds.outliers == 0 || b.bonds.outliers != 0 {
            failures.push(format!(
                "{name}: bond outliers {} unrelaxed, {} relaxed",
                a.bonds.outliers, b.bonds.outliers
            ));
        }
        if z(&b) >= z(&a) {
            failures.push(format!("{name}: relaxing did not lower bond RMSZ"));
        }
        if rmsd > 0.2 {
            failures.push(format!("{name}: relaxation moved Cα by {rmsd:.2} Å"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
