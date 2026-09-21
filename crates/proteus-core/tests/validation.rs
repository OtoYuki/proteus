//! Corpus validation against mdtraj / freesasa / cctbx reference values.
//!
//! Run with `make validate` (needs `validate/corpus/` fetched). Ignored by default because it
//! depends on the downloaded corpus. Tolerances live in `validate/tolerances.toml`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    id: String,
    format: String,
    kind: String,
    n_residues: usize,
    residue_keys: Vec<String>,
    residue_keys_ordinal: Vec<String>,
    rg_ca: f64,
    sasa_freesasa_lr: Option<f64>,
    sasa_mdtraj_sr: f64,
    dssp8: Vec<String>,
    dssp3: Vec<String>,
    phi: Vec<Option<f64>>,
    psi: Vec<Option<f64>>,
    rama: Rama,
}

#[derive(Deserialize)]
struct Rama {
    favored: usize,
    allowed: usize,
    outliers: usize,
    labels: HashMap<String, String>,
    labels_ordinal: HashMap<String, String>,
}

#[derive(Deserialize)]
struct Tolerances {
    rg_ca_abs: f64,
    sasa_vs_freesasa_lr_rel: f64,
    sasa_vs_mdtraj_sr_rel: f64,
    phi_psi_abs_deg: f64,
    dssp3_min_agreement: f64,
    dssp8_min_agreement: f64,
    rama_label_min_agreement: f64,
    min_residue_key_match: f64,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate")
}

struct Row {
    id: String,
    format: String,
    kind: String,
    n: usize,
    rg_err: f64,
    sasa_sr_rel: f64,
    sasa_lr_rel: Option<f64>,
    angle_ok: usize,
    angle_n: usize,
    dssp8: f64,
    dssp3: f64,
    rama_agree: f64,
    ours: (usize, usize, usize),
    theirs: (usize, usize, usize),
}

#[test]
#[ignore]
fn corpus_matches_reference_implementations() {
    let tol: Tolerances =
        toml::from_str(&std::fs::read_to_string(root().join("tolerances.toml")).unwrap()).unwrap();
    let mut rows: Vec<Row> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root().join("reference"))
        .unwrap()
        .map(|e| e.unwrap().path())
        // Structure corpus only; validate/reference/esm/ holds ESM-2 logits for proteus-esm.
        .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    entries.sort();

    for path in entries {
        let r: Reference = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let structure = root().join("corpus").join(format!("{}.{}", r.id, r.format));
        if !structure.exists() {
            if std::env::var("PROTEUS_VALIDATE_OFFLINE").is_ok() {
                eprintln!("skip {} (offline)", r.id);
                continue;
            }
            panic!("missing {} — run `make fetch`", structure.display());
        }

        let loaded = proteus_core::io::load_structure(&structure).unwrap();
        let d = proteus_core::metrics::analyze_pdb_detailed_with_header(
            &loaded.pdb,
            None,
            Some(&loaded.header_preview),
        )
        .unwrap();
        let protein = proteus_core::io::protein_heavy_atoms(&loaded.pdb);
        let bb = proteus_core::backbone::extract_backbone(&protein);
        let m = &d.metrics;
        let ss = m.secondary_structure_summary.as_ref().unwrap();
        let rs = m.ramachandran_stats.as_ref().unwrap();
        let mut bad: Vec<String> = Vec::new();

        // Residue identity alignment: "chain:resseq:icode" by chain id, falling back to
        // chain ordinal when the two readers disagree on mmCIF asym ids.
        let id_keys: Vec<String> = bb
            .iter()
            .map(|b| {
                format!(
                    "{}:{}:{}",
                    b.chain_id.trim(),
                    b.seq_num,
                    b.insertion_code.as_deref().unwrap_or("")
                )
            })
            .collect();
        let mut chain_ordinal: HashMap<&str, usize> = HashMap::new();
        for b in &bb {
            let next = chain_ordinal.len();
            chain_ordinal.entry(b.chain_id.as_str()).or_insert(next);
        }
        let ord_keys: Vec<String> = bb
            .iter()
            .map(|b| {
                format!(
                    "{}:{}:{}",
                    chain_ordinal[b.chain_id.as_str()],
                    b.seq_num,
                    b.insertion_code.as_deref().unwrap_or("")
                )
            })
            .collect();
        let count_matches = |keys: &[String], reference: &[String]| {
            let set: std::collections::HashSet<&str> =
                reference.iter().map(|s| s.as_str()).collect();
            keys.iter().filter(|k| set.contains(k.as_str())).count()
        };
        let id_matched = count_matches(&id_keys, &r.residue_keys);
        let ord_matched = count_matches(&ord_keys, &r.residue_keys_ordinal);
        let use_ordinal = ord_matched > id_matched;
        let (keys, matched, rama_labels) = if use_ordinal {
            (ord_keys, ord_matched, &r.rama.labels_ordinal)
        } else {
            (id_keys, id_matched, &r.rama.labels)
        };
        let match_frac = matched as f64 / keys.len().max(1) as f64;
        if match_frac < tol.min_residue_key_match {
            bad.push(format!(
                "residue keys: only {matched}/{} match mdtraj (ordinal={use_ordinal}; ours[0..3]={:?}, theirs[0..3]={:?})",
                keys.len(),
                &keys[..keys.len().min(3)],
                &r.residue_keys[..r.residue_keys.len().min(3)]
            ));
        }
        if bb.len() != r.n_residues {
            bad.push(format!("residue count {} vs {}", bb.len(), r.n_residues));
        }

        // Rg and SASA
        let rg_err = (m.radius_of_gyration - r.rg_ca).abs();
        if rg_err > tol.rg_ca_abs {
            bad.push(format!("rg {:.3} vs {:.3}", m.radius_of_gyration, r.rg_ca));
        }
        let sasa = m.sasa_metrics.as_ref().unwrap().total_sasa;
        let sasa_sr_rel = (sasa - r.sasa_mdtraj_sr).abs() / r.sasa_mdtraj_sr;
        if sasa_sr_rel > tol.sasa_vs_mdtraj_sr_rel {
            bad.push(format!(
                "sasa vs mdtraj S&R {:.1} vs {:.1} ({:.2}%)",
                sasa,
                r.sasa_mdtraj_sr,
                sasa_sr_rel * 100.0
            ));
        }
        let sasa_lr_rel = r.sasa_freesasa_lr.map(|lr| (sasa - lr).abs() / lr);
        if let Some(rel) = sasa_lr_rel {
            if rel > tol.sasa_vs_freesasa_lr_rel {
                bad.push(format!(
                    "sasa vs freesasa L&R {:.1} vs {:.1} ({:.2}%)",
                    sasa,
                    r.sasa_freesasa_lr.unwrap(),
                    rel * 100.0
                ));
            }
        }

        // phi/psi, DSSP, Ramachandran labels — all keyed by residue identity.
        let (mut angle_n, mut angle_ok) = (0usize, 0usize);
        let (mut d8_n, mut d8_ok, mut d3_n, mut d3_ok) = (0usize, 0usize, 0usize, 0usize);
        let (mut lab_n, mut lab_ok) = (0usize, 0usize);
        let mut angle_examples: Vec<String> = Vec::new();
        let ref_index: HashMap<&str, usize> = if use_ordinal {
            r.residue_keys_ordinal
                .iter()
                .enumerate()
                .map(|(i, k)| (k.as_str(), i))
                .collect()
        } else {
            r.residue_keys
                .iter()
                .enumerate()
                .map(|(i, k)| (k.as_str(), i))
                .collect()
        };
        for (i, key) in keys.iter().enumerate() {
            let (phi, psi, region) = d.ramachandran_points[i];
            let Some(&j) = ref_index.get(key.as_str()) else {
                continue;
            };
            for (got, want) in [(phi, r.phi[j]), (psi, r.psi[j])] {
                if let (Some(g), Some(w)) = (got, want) {
                    angle_n += 1;
                    let mut diff = (g - w).abs();
                    if diff > 180.0 {
                        diff = 360.0 - diff;
                    }
                    if diff <= tol.phi_psi_abs_deg {
                        angle_ok += 1;
                    } else if angle_examples.len() < 3 {
                        angle_examples.push(format!("{key} {g:.1} vs {w:.1}"));
                    }
                }
            }
            d8_n += 1;
            if ss.dssp.chars().nth(i).map(|c| c.to_string()).as_deref() == Some(r.dssp8[j].as_str())
            {
                d8_ok += 1;
            }
            d3_n += 1;
            if ss.assignment[i].as_char().to_string() == r.dssp3[j] {
                d3_ok += 1;
            }
            if let Some(want) = rama_labels.get(key) {
                if phi.is_some() && psi.is_some() {
                    lab_n += 1;
                    let g = match region {
                        proteus_core::structure::RamachandranRegion::Favored => "F",
                        proteus_core::structure::RamachandranRegion::Allowed => "A",
                        proteus_core::structure::RamachandranRegion::Outlier => "O",
                    };
                    if g == want {
                        lab_ok += 1;
                    }
                }
            }
        }
        if angle_n > 0 && angle_ok < angle_n {
            bad.push(format!(
                "phi/psi {}/{} beyond {}° e.g. {}",
                angle_n - angle_ok,
                angle_n,
                tol.phi_psi_abs_deg,
                angle_examples.join("; ")
            ));
        }
        let d8 = d8_ok as f64 / d8_n.max(1) as f64;
        let d3 = d3_ok as f64 / d3_n.max(1) as f64;
        if d8 < tol.dssp8_min_agreement {
            bad.push(format!("dssp8 agreement {:.3} ({d8_ok}/{d8_n})", d8));
        }
        if d3 < tol.dssp3_min_agreement {
            bad.push(format!("dssp3 agreement {:.3} ({d3_ok}/{d3_n})", d3));
        }
        let rama_agree = lab_ok as f64 / lab_n.max(1) as f64;
        if rama_agree < tol.rama_label_min_agreement {
            bad.push(format!(
                "rama labels agreement {:.4} ({lab_ok}/{lab_n})",
                rama_agree
            ));
        }

        let allowed = (rs.allowed_fraction * rs.total_evaluated as f64).round() as usize;
        rows.push(Row {
            id: r.id.clone(),
            format: r.format.clone(),
            kind: r.kind.clone(),
            n: bb.len(),
            rg_err,
            sasa_sr_rel,
            sasa_lr_rel,
            angle_ok,
            angle_n,
            dssp8: d8,
            dssp3: d3,
            rama_agree,
            ours: (
                rs.total_evaluated - rs.outlier_count - allowed,
                allowed,
                rs.outlier_count,
            ),
            theirs: (r.rama.favored, r.rama.allowed, r.rama.outliers),
        });
        if !bad.is_empty() {
            failures.push(format!("{} ({}): {}", r.id, r.format, bad.join("; ")));
        }
    }

    let mut table = String::new();
    table.push_str("| id | fmt | kind | res | Δrg Å | SASA vs mdtraj | SASA vs freesasa | φ/ψ ≤tol | DSSP-8 | DSSP-3 | Rama labels | F/A/O proteus | F/A/O cctbx |\n");
    table.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for w in &rows {
        table.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {:.2}% | {} | {}/{} | {:.1}% | {:.1}% | {:.2}% | {}/{}/{} | {}/{}/{} |\n",
            w.id,
            w.format,
            w.kind,
            w.n,
            w.rg_err,
            w.sasa_sr_rel * 100.0,
            w.sasa_lr_rel
                .map(|v| format!("{:.2}%", v * 100.0))
                .unwrap_or_else(|| "–".into()),
            w.angle_ok,
            w.angle_n,
            w.dssp8 * 100.0,
            w.dssp3 * 100.0,
            w.rama_agree * 100.0,
            w.ours.0,
            w.ours.1,
            w.ours.2,
            w.theirs.0,
            w.theirs.1,
            w.theirs.2
        ));
    }
    println!("{table}");
    std::fs::write(root().join("last_run.md"), &table).ok();
    assert!(
        failures.is_empty(),
        "validation failures:\n{}",
        failures.join("\n")
    );
}
