//! What the composite fitness score is actually good for, measured.
//!
//! The score is a Proteus-defined weighted sum, not a predictor of experimental fitness, and
//! no reference implementation exists to compare it against. What it *claims* is narrower and
//! testable: **a structure that looks like a folded protein should score above one that does
//! not.** That is the claim `proteus screen` relies on when it ranks candidates, so that is
//! what this harness measures — on real deposited structures, against decoys made from them.
//!
//! Three decoy families, each breaking something different:
//!
//! | decoy | what it breaks | why the score should notice |
//! |---|---|---|
//! | coordinate noise, σ = 0.5–3.0 Å | local geometry | Ramachandran outliers, steric overlap, broken H-bonds |
//! | expanded ×1.5 | compactness | radius of gyration, burial, contact network |
//! | ideal poly-helix of the same length | everything but secondary structure | no tertiary network, wrong Rg |
//!
//! The last one is not hypothetical: it is exactly what Proteus's own offline simulator emits,
//! so this doubles as the check that a placeholder can never outrank a real model.
//!
//! Run with `make validate` (needs `validate/corpus/`). Ignored by default like the other
//! corpus tests. Deliberately *not* a claim that the score predicts stability or activity —
//! for that, see the ESM-2 ProteinGym numbers in `bench/README.md`.

use std::path::{Path, PathBuf};

use proteus_core::metrics::analyze_pdb_detailed;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validate")
        .canonicalize()
        .expect("validate/ directory")
}

/// A deterministic LCG: decoys must be identical on every run and every machine, so a
/// regression is a real change in the score and never a reroll.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box–Muller, one normal per call.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Rewrite ATOM/HETATM coordinates through `f`. Column-exact, so the result re-parses.
fn map_coordinates(pdb: &str, mut f: impl FnMut(f64, f64, f64) -> (f64, f64, f64)) -> String {
    let mut out = String::with_capacity(pdb.len());
    for line in pdb.lines() {
        if (line.starts_with("ATOM") || line.starts_with("HETATM")) && line.len() >= 54 {
            let (x, y, z) = (
                line[30..38].trim().parse::<f64>().unwrap_or(0.0),
                line[38..46].trim().parse::<f64>().unwrap_or(0.0),
                line[46..54].trim().parse::<f64>().unwrap_or(0.0),
            );
            let (x, y, z) = f(x, y, z);
            out.push_str(&format!(
                "{}{x:8.3}{y:8.3}{z:8.3}{}\n",
                &line[..30],
                &line[54..]
            ));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn centroid(pdb: &str) -> (f64, f64, f64) {
    let (mut sx, mut sy, mut sz, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    map_coordinates(pdb, |x, y, z| {
        sx += x;
        sy += y;
        sz += z;
        n += 1.0;
        (x, y, z)
    });
    let n = n.max(1.0);
    (sx / n, sy / n, sz / n)
}

fn jittered(pdb: &str, sigma: f64, seed: u64) -> String {
    let mut rng = Lcg(seed);
    map_coordinates(pdb, |x, y, z| {
        (
            x + sigma * rng.normal(),
            y + sigma * rng.normal(),
            z + sigma * rng.normal(),
        )
    })
}

fn expanded(pdb: &str, factor: f64) -> String {
    let (cx, cy, cz) = centroid(pdb);
    map_coordinates(pdb, |x, y, z| {
        (
            cx + (x - cx) * factor,
            cy + (y - cy) * factor,
            cz + (z - cz) * factor,
        )
    })
}

/// An ideal α-helix poly-alanine C-alpha trace: the shape Proteus's offline simulator emits.
fn ideal_helix(n_residues: usize) -> String {
    let mut out = String::from("HEADER    IDEAL HELIX DECOY\n");
    for i in 0..n_residues {
        let theta = i as f64 * 100.0_f64.to_radians();
        let (x, y, z) = (2.3 * theta.cos(), 2.3 * theta.sin(), 1.5 * i as f64);
        out.push_str(&format!(
            "ATOM  {:5}  CA  ALA A{:4}    {x:8.3}{y:8.3}{z:8.3}  1.00 50.00           C\n",
            i + 1,
            i + 1
        ));
    }
    out.push_str("END\n");
    out
}

fn score(pdb_text: &str) -> Option<f64> {
    let pdb = proteus_core::io::open_structure_bytes(pdb_text.as_bytes(), Some("x.pdb")).ok()?;
    analyze_pdb_detailed(&pdb, None)
        .ok()?
        .metrics
        .candidate_fitness_score
}

/// Deposited X-ray structures, spanning sizes and folds. PDB format only: the decoys are made
/// by rewriting fixed-width coordinate columns.
const TARGETS: &[&str] = &[
    "1crn", "1ubq", "2ci2", "1pgb", "1mbn", "3ptb", "2lzm", "1stn",
];

#[test]
#[ignore]
fn native_structures_outrank_their_decoys() {
    let dir = root().join("corpus");
    let mut rows = Vec::new();
    let mut checked = 0;

    for id in TARGETS {
        let path = dir.join(format!("{id}.pdb"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipping {id}: {} not fetched", path.display());
            continue;
        };
        let Some(native) = score(&text) else {
            panic!("{id}: native structure did not score")
        };
        let n_res = text
            .lines()
            .filter(|l| l.starts_with("ATOM") && l.len() > 15 && &l[12..16] == " CA ")
            .count();

        let decoys = [
            ("noise 0.5Å", score(&jittered(&text, 0.5, 0x5eed))),
            ("noise 1.0Å", score(&jittered(&text, 1.0, 0x5eed))),
            ("noise 3.0Å", score(&jittered(&text, 3.0, 0x5eed))),
            ("expanded 1.5×", score(&expanded(&text, 1.5))),
            ("ideal helix", score(&ideal_helix(n_res))),
        ];

        for (name, decoy) in decoys {
            let decoy = decoy.unwrap_or_else(|| panic!("{id}: decoy '{name}' did not score"));
            rows.push((id.to_string(), name, native, decoy));
            assert!(
                native > decoy,
                "{id}: decoy '{name}' scored {decoy:.1}, at or above the native {native:.1} — \
                 the ranking `proteus screen` performs would put a broken model first"
            );
            checked += 1;
        }

        // Heavier corruption must not score better than lighter corruption.
        let (n05, n10, n30) = (
            score(&jittered(&text, 0.5, 0x5eed)).unwrap(),
            score(&jittered(&text, 1.0, 0x5eed)).unwrap(),
            score(&jittered(&text, 3.0, 0x5eed)).unwrap(),
        );
        assert!(
            n05 >= n10 && n10 >= n30,
            "{id}: score is not monotone in coordinate noise: 0.5Å {n05:.1}, 1.0Å {n10:.1}, \
             3.0Å {n30:.1}"
        );
    }

    assert!(
        checked >= 20,
        "only {checked} native/decoy pairs were scored; is validate/corpus/ fetched?"
    );

    println!("\n| structure | native | {:<14} | Δ |", "decoy");
    println!("|---|---|---|---|");
    for (id, name, native, decoy) in &rows {
        println!(
            "| {id} | {native:.1} | {name:<14} {decoy:.1} | {:+.1} |",
            decoy - native
        );
    }
    let worst = rows
        .iter()
        .map(|(_, _, n, d)| n - d)
        .fold(f64::INFINITY, f64::min);
    println!("\nsmallest native−decoy margin: {worst:.1} points over {checked} pairs");
}
