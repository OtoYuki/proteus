//! Hydrogen-bond network vs mdtraj `baker_hubbard` on structures with explicit hydrogens.
//!
//! Proteus detects H-bonds from **heavy atoms only** (donor/acceptor distance plus antecedent
//! angles), because predicted models never ship hydrogens. mdtraj uses the explicit H (D–H···A
//! distance and angle). The criteria are related but not identical, so this test measures
//! **recall** — the fraction of mdtraj's bonds that Proteus also finds — and reports
//! **precision** — the fraction of Proteus' bonds that mdtraj confirms — rather than asserting
//! set equality. Proteus finds 1.3–1.7× as many bonds as Baker–Hubbard with explicit H; that
//! over-detection is the cost of working without hydrogens and is gated, not hidden.
//!
//! Reference: `validate/reference/hbonds/*.json` from `validate/hbond_reference.py`.
//! Run: `make validate` (needs `validate/corpus/`), or `cargo test -p proteus-core --test hbond_validation -- --ignored`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Minimum fraction of mdtraj's non-local H-bonds that Proteus must recover. Observed on all
/// six hydrogen-bearing corpus entries: 1d3z 100 %, 1gb1 100 %, 2kod 98.1 %, 2l3b 97.1 %,
/// 1g6j 94.9 %, 1l2y 85.7 % (a 20-residue mini-protein with 14 reference bonds).
const MIN_RECALL: f64 = 0.85;

/// Minimum fraction of Proteus' non-local bonds that mdtraj confirms. Observed: 59–76 %.
/// The floor exists so that a change cannot silently trade precision for recall.
const MIN_PRECISION: f64 = 0.50;

/// Proteus only reports bonds at least this far apart in sequence; mdtraj reports i→i+1 too,
/// so those are excluded from the comparison rather than counted as misses.
const MIN_SEQ_SEPARATION: isize = 2;

#[derive(Deserialize)]
struct Reference {
    id: String,
    n_hydrogens: usize,
    hbonds: Vec<RefBond>,
}

#[derive(Deserialize)]
struct RefBond {
    donor_resseq: isize,
    acceptor_resseq: isize,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate")
}

#[test]
#[ignore]
fn hydrogen_bonds_recover_the_mdtraj_network() {
    let dir = root().join("reference/hbonds");
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e} — run validate/hbond_reference.py", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    entries.sort();

    for path in entries {
        let r: Reference = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(r.n_hydrogens > 0, "{}: reference has no hydrogens", r.id);
        let structure = root().join("corpus").join(format!("{}.pdb", r.id));
        if !structure.exists() {
            if std::env::var("PROTEUS_VALIDATE_OFFLINE").is_ok() {
                eprintln!("skip {} (offline)", r.id);
                continue;
            }
            panic!("missing {} — run `make fetch`", structure.display());
        }

        let pdb = proteus_core::io::open_structure(&structure).unwrap();
        let protein = proteus_core::io::protein_heavy_atoms(&pdb);
        let net = proteus_core::interactions::compute_interaction_network(&protein);
        let ours: HashSet<(isize, isize)> = net
            .hbonds
            .iter()
            .map(|b| (b.donor_res_seq, b.acceptor_res_seq))
            .collect();

        let theirs: HashSet<(isize, isize)> = r
            .hbonds
            .iter()
            .map(|b| (b.donor_resseq, b.acceptor_resseq))
            .filter(|(d, a)| (d - a).abs() >= MIN_SEQ_SEPARATION)
            .collect();

        let ours_nonlocal: HashSet<(isize, isize)> = ours
            .iter()
            .copied()
            .filter(|(d, a)| (d - a).abs() >= MIN_SEQ_SEPARATION)
            .collect();
        let hit = theirs.intersection(&ours).count();
        let recall = hit as f64 / theirs.len().max(1) as f64;
        let precision = hit as f64 / ours_nonlocal.len().max(1) as f64;
        eprintln!(
            "{}: recall {hit}/{} = {:.1}%, precision {hit}/{} = {:.1}%",
            r.id,
            theirs.len(),
            recall * 100.0,
            ours_nonlocal.len(),
            precision * 100.0
        );
        if precision < MIN_PRECISION {
            failures.push(format!(
                "{}: precision {:.1}% below {:.0}%",
                r.id,
                precision * 100.0,
                MIN_PRECISION * 100.0
            ));
        }
        if recall < MIN_RECALL {
            let missed: Vec<String> = theirs
                .difference(&ours)
                .take(5)
                .map(|(d, a)| format!("{d}->{a}"))
                .collect();
            failures.push(format!(
                "{}: recall {:.1}% below {:.0}%; missed e.g. {}",
                r.id,
                recall * 100.0,
                MIN_RECALL * 100.0,
                missed.join(", ")
            ));
        }
        checked += 1;
    }

    assert!(
        checked > 0,
        "no hydrogen-bearing reference structures were checked"
    );
    assert_eq!(
        checked, 6,
        "expected all six hydrogen-bearing corpus entries"
    );
    assert!(
        failures.is_empty(),
        "H-bond validation failures:\n{}",
        failures.join("\n")
    );
}
