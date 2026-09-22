//! Salt bridges, π–π stacking and cation–π against PLIP.
//!
//! These three were the rows in the README that said "no widely used reference implementation
//! with the same definitions". [PLIP](https://github.com/pharmai/plip) (Salentin et al. 2015)
//! is one, run in intra-chain mode by `validate/plip_reference.py`.
//!
//! The criteria are genuinely different, so this measures **recall** (how many of PLIP's
//! interactions Proteus also finds) and reports **precision** (how many of Proteus's PLIP
//! confirms), exactly as the mdtraj H-bond harness does — not set equality:
//!
//! | interaction | PLIP | Proteus |
//! |---|---|---|
//! | salt bridge | ≤ 5.5 Å between charge centres | ≤ 4.0 Å between closest atoms |
//! | π–π | ≤ 5.5 Å centroids, offset ≤ 2.0 Å | ≤ 6.5 Å centroids, no offset test |
//! | cation–π | ≤ 6.0 Å, offset ≤ 2.0 Å | ≤ 6.0 Å, angle to normal ≤ 45° |
//!
//! Proteus's salt-bridge cutoff is the stricter one (4.0 Å atom-to-atom against PLIP's 5.5 Å
//! centre-to-centre), so *missing* some of PLIP's is expected and is a threshold choice, not a
//! defect. The floors below are set from the observed values with headroom; they exist to catch
//! a regression, not to claim agreement Proteus does not have.
//!
//! Comparison is by **residue pair** (`chain:resseq`), orientation-free: multiple atom-level
//! contacts between the same two residues collapse to one, as they do in the reference.
//!
//! Scope: intra-chain only — PLIP's INTRA mode profiles one chain against itself. Inter-chain
//! contacts are covered by `inter_chain_contacts_between_equally_numbered_residues_are_kept`
//! in `interactions.rs`.
//!
//! Run with `make validate`. Ignored by default: needs `validate/corpus/` and the generated
//! `validate/reference/plip/`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct PlipReference {
    id: String,
    salt_bridges: Vec<(String, i64, String, i64)>,
    pi_stacking: Vec<(String, i64, String, i64)>,
    cation_pi: Vec<(String, i64, String, i64)>,
}

/// Orientation-free residue-pair key, matching the reference's normalisation.
type Pair = ((String, i64), (String, i64));

fn pair(a_chain: &str, a_res: isize, b_chain: &str, b_res: isize) -> Pair {
    let a = (a_chain.trim().to_string(), a_res as i64);
    let b = (b_chain.trim().to_string(), b_res as i64);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn reference_pairs(raw: &[(String, i64, String, i64)]) -> BTreeSet<Pair> {
    raw.iter()
        .map(|(c1, r1, c2, r2)| pair(c1, *r1 as isize, c2, *r2 as isize))
        .collect()
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validate")
        .canonicalize()
        .expect("validate/ directory")
}

/// Recall and precision of `ours` against `theirs`, plus the raw counts.
struct Overlap {
    shared: usize,
    ours: usize,
    theirs: usize,
}

impl Overlap {
    fn new(ours: &BTreeSet<Pair>, theirs: &BTreeSet<Pair>) -> Self {
        Self {
            shared: ours.intersection(theirs).count(),
            ours: ours.len(),
            theirs: theirs.len(),
        }
    }
    fn recall(&self) -> f64 {
        if self.theirs == 0 {
            1.0
        } else {
            self.shared as f64 / self.theirs as f64
        }
    }
    fn precision(&self) -> f64 {
        if self.ours == 0 {
            1.0
        } else {
            self.shared as f64 / self.ours as f64
        }
    }
}

#[test]
#[ignore]
fn interactions_overlap_plip() {
    let ref_dir = root().join("reference/plip");
    let corpus = root().join("corpus");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&ref_dir)
        .unwrap_or_else(|e| panic!("{}: {e} — run `make reference`", ref_dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no PLIP references in {}",
        ref_dir.display()
    );

    // Running totals, because per-structure counts are small and a per-structure floor would
    // be noise. The corpus totals are what the README quotes.
    let (mut sb, mut pp, mut cp) = (
        Overlap {
            shared: 0,
            ours: 0,
            theirs: 0,
        },
        Overlap {
            shared: 0,
            ours: 0,
            theirs: 0,
        },
        Overlap {
            shared: 0,
            ours: 0,
            theirs: 0,
        },
    );
    let mut rows = Vec::new();

    for file in &files {
        let r: PlipReference = serde_json::from_str(&std::fs::read_to_string(file).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        let structure = corpus.join(format!("{}.pdb", r.id));
        let Ok(text) = std::fs::read_to_string(&structure) else {
            eprintln!("skipping {}: {} not fetched", r.id, structure.display());
            continue;
        };
        let pdb = proteus_core::io::open_structure_bytes(text.as_bytes(), Some("x.pdb")).unwrap();
        let net = proteus_core::interactions::compute_interaction_network(&pdb);

        // Proteus reports inter-chain contacts too; PLIP's INTRA mode does not see them, so
        // restrict the comparison to same-chain pairs on both sides.
        let ours_sb: BTreeSet<Pair> = net
            .salt_bridges
            .iter()
            .filter(|s| s.cation_chain_id == s.anion_chain_id)
            .map(|s| {
                pair(
                    &s.cation_chain_id,
                    s.cation_res_seq,
                    &s.anion_chain_id,
                    s.anion_res_seq,
                )
            })
            .collect();
        let ours_pp: BTreeSet<Pair> = net
            .pi_pi_stacks
            .iter()
            .filter(|s| s.ring1_chain_id == s.ring2_chain_id)
            .map(|s| {
                pair(
                    &s.ring1_chain_id,
                    s.ring1_res_seq,
                    &s.ring2_chain_id,
                    s.ring2_res_seq,
                )
            })
            .collect();
        let ours_cp: BTreeSet<Pair> = net
            .cation_pi_interactions
            .iter()
            .filter(|s| s.cation_chain_id == s.ring_chain_id)
            .map(|s| {
                pair(
                    &s.cation_chain_id,
                    s.cation_res_seq,
                    &s.ring_chain_id,
                    s.ring_res_seq,
                )
            })
            .collect();

        for (ours, theirs, acc) in [
            (&ours_sb, reference_pairs(&r.salt_bridges), &mut sb),
            (&ours_pp, reference_pairs(&r.pi_stacking), &mut pp),
            (&ours_cp, reference_pairs(&r.cation_pi), &mut cp),
        ] {
            let o = Overlap::new(ours, &theirs);
            acc.shared += o.shared;
            acc.ours += o.ours;
            acc.theirs += o.theirs;
        }
        rows.push((
            r.id.clone(),
            Overlap::new(&ours_sb, &reference_pairs(&r.salt_bridges)),
            Overlap::new(&ours_cp, &reference_pairs(&r.cation_pi)),
        ));
    }

    println!("\n| structure | salt bridges (ours/PLIP/shared) | cation-π (ours/PLIP/shared) |");
    println!("|---|---|---|");
    for (id, s, c) in &rows {
        println!(
            "| {id} | {}/{}/{} | {}/{}/{} |",
            s.ours, s.theirs, s.shared, c.ours, c.theirs, c.shared
        );
    }
    println!(
        "\ncorpus totals\n  salt bridges: recall {:.1}% precision {:.1}% ({} ours vs {} PLIP)\n\
  pi-pi:        recall {:.1}% precision {:.1}% ({} ours vs {} PLIP)\n\
  cation-pi:    recall {:.1}% precision {:.1}% ({} ours vs {} PLIP)",
        100.0 * sb.recall(),
        100.0 * sb.precision(),
        sb.ours,
        sb.theirs,
        100.0 * pp.recall(),
        100.0 * pp.precision(),
        pp.ours,
        pp.theirs,
        100.0 * cp.recall(),
        100.0 * cp.precision(),
        cp.ours,
        cp.theirs,
    );

    // Floors set below the observed corpus values with headroom. They catch a regression; they
    // are not a claim of agreement. Observed 2026-09-22 over 15 structures:
    //   salt bridges  recall 72.0 %  precision 97.7 %   (87 ours, 118 PLIP)
    //   π–π           recall 81.8 %  precision 81.8 %   (11 ours,  11 PLIP)
    //   cation–π      recall 65.4 %  precision 73.9 %   (23 ours,  26 PLIP)
    // Salt-bridge recall is deliberately well under 100 %: Proteus's 4.0 Å atom-to-atom cutoff
    // is stricter than PLIP's 5.5 Å centre-to-centre, so it reports a subset on purpose — and
    // the 97.7 % precision is the evidence that the subset is the right one.
    for (name, o, min_recall, min_precision) in [
        ("salt bridge", &sb, 0.65, 0.90),
        ("π–π stacking", &pp, 0.70, 0.70),
        ("cation–π", &cp, 0.55, 0.60),
    ] {
        assert!(
            o.recall() >= min_recall,
            "{name} recall against PLIP fell to {:.1}% ({}/{} of PLIP's), floor {:.0}%",
            100.0 * o.recall(),
            o.shared,
            o.theirs,
            100.0 * min_recall
        );
        assert!(
            o.precision() >= min_precision,
            "{name} precision against PLIP fell to {:.1}% ({}/{} of ours confirmed), floor \
             {:.0}% — Proteus is reporting contacts PLIP does not see",
            100.0 * o.precision(),
            o.shared,
            o.ours,
            100.0 * min_precision
        );
    }
}
