# SP1 — Scientific Correctness & Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every biophysical number Proteus emits either match a reference implementation within a committed tolerance in CI, or carry an explicit approximation label.

**Architecture:** Fix the dihedral sign; introduce a chain-aware backbone extractor; add a standalone Kabsch–Sander DSSP crate; replace box-Ramachandran with embedded Top8000 (cctbx `rama8000`) contour grids; add pLDDT provenance; relabel the heavy-atom clash metric; route all structure opening through one mmCIF-aware function; add a `validate/` harness (mdtraj, freesasa, cctbx) whose committed reference JSON drives an integration test and a CI job.

**Tech Stack:** Rust 2021 (workspace, `nalgebra`, `pdbtbx 0.11`, `serde`), Python 3.12 via `uv venv` (`mdtraj`, `freesasa`, `cctbx-base`, `numpy`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-21-sp1-scientific-correctness-design.md`

## Global Constraints

- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` must pass before every commit.
- No `unsafe`. No new runtime dependencies in `proteus-core` beyond what exists; `proteus-dssp` depends only on `nalgebra` (+ `serde` optional).
- Angles in degrees, IUPAC sign convention (right-handed α-helix φ ≈ −57°, ψ ≈ −47°).
- Ramachandran thresholds exactly as cctbx `rama_eval.h`: favored ≥ 0.02; allowed ≥ 0.0005 (general), ≥ 0.002 (cis-Pro), ≥ 0.001 (all others).
- DSSP semantics follow DSSP 2.x as ported by mdtraj (H overrides E; G/I fill loop only; no PPII).
- Validation tolerances live only in `validate/tolerances.toml`; tests read them, never hard-code them.
- Python reference tooling runs from `validate/.venv` created by `uv venv --python 3.12` (ephemeral `uv run --with cctbx-base` is broken: cctbx pickles its install path).
- Commit after every task; commit messages `type(scope): summary` + the Co-Authored-By trailer from the session reminder.
- The harness is what proves correctness; a task is not done until its numbers are shown (test output pasted in the commit body is fine).

---

## File map

| path | responsibility |
|---|---|
| `crates/proteus-core/src/structure.rs` | dihedral, Ramachandran types + classification (rewritten), SS summary types; delegates SS to `proteus-dssp` |
| `crates/proteus-core/src/backbone.rs` (new) | `BackboneResidue` extraction from `pdbtbx::PDB`, chain breaks, ω |
| `crates/proteus-core/src/rama8000.rs` (new) | embedded Top8000 grids, bilinear lookup, `RamaClass` |
| `crates/proteus-core/data/rama8000/*.f32`, `NOTICE` (new) | six 180×180 LE-f32 grids + cctbx licence |
| `scripts/convert_rama8000.py` (new) | regenerates the `.f32` files from cctbx `rama8000_tables.h` |
| `crates/proteus-core/src/confidence.rs` (new) | `ConfidenceSource` detection |
| `crates/proteus-core/src/io.rs` (new) | `open_structure(path)` — PDB / mmCIF / `.gz` |
| `crates/proteus-core/src/clash.rs` | rename to `StericOverlapStats` / `heavy_atom_overlap_score` |
| `crates/proteus-core/src/metrics.rs` | uses backbone/dssp/rama8000/confidence |
| `crates/proteus-core/src/ranking.rs` | weight renormalisation without pLDDT |
| `crates/proteus-dssp/` (new crate) | Kabsch–Sander DSSP |
| `crates/proteus-cli/src/main.rs` | labels, `open_structure`, `--confidence-source` |
| `crates/proteus-render/src/tui/dashboard.rs`, `lib.rs` | `RamachandranRegion` variants, labels |
| `crates/proteus-storage/src/export.rs` | column rename, `schema_version` |
| `crates/proteus-server/src/api.rs` | OpenAPI component list |
| `validate/` (new) | corpus manifest, fetch, reference generator, tolerances, README |
| `crates/proteus-core/tests/validation.rs` (new) | integration test over the corpus |
| `.github/workflows/validate.yml` (new) | CI job |
| `Makefile` (new) | `make validate`, `make reference` |

---

### Task 1: Fix `compute_dihedral` sign and pin φ/ψ to mdtraj

**Files:**
- Modify: `crates/proteus-core/src/structure.rs:53-80` (`compute_dihedral`), tests at `:376-430`
- Test: `crates/proteus-core/src/structure.rs` (unit tests), `crates/proteus-core/tests/data/1crn_phipsi_mdtraj.csv` (new)

**Interfaces:**
- Produces: `pub fn compute_dihedral(p1,p2,p3,p4: &Vector3<f64>) -> Result<f64, CoreError>` — unchanged signature, IUPAC sign.

- [ ] **Step 1: Generate the mdtraj reference CSV**

```bash
mkdir -p validate && uv venv --python 3.12 validate/.venv -q && uv pip install -q --python validate/.venv/bin/python mdtraj numpy
validate/.venv/bin/python - <<'EOF'
import mdtraj as md, numpy as np
t = md.load("crates/proteus-core/tests/data/1crn.pdb")
_, phi = md.compute_phi(t); _, psi = md.compute_psi(t)
phi = np.degrees(phi[0]); psi = np.degrees(psi[0])
res = list(t.topology.residues)
with open("crates/proteus-core/tests/data/1crn_phipsi_mdtraj.csv", "w") as f:
    f.write("resseq,resname,phi,psi\n")
    for i, r in enumerate(res):
        p = "" if i == 0 else f"{phi[i-1]:.3f}"
        s = "" if i == len(res)-1 else f"{psi[i]:.3f}"
        f.write(f"{r.resSeq},{r.name},{p},{s}\n")
EOF
head -4 crates/proteus-core/tests/data/1crn_phipsi_mdtraj.csv
```
Expected first data row: `1,THR,,-147.7xx` (psi only), second: `2,THR,-107.8xx,144.3xx`.

- [ ] **Step 2: Write the failing tests**

Append to the `mod tests` block in `structure.rs`:
```rust
    #[test]
    fn dihedral_sign_matches_iupac_on_crambin() {
        let (pdb, _) = pdbtbx::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"),
            pdbtbx::StrictnessLevel::Loose,
        )
        .unwrap();
        let csv = include_str!("../tests/data/1crn_phipsi_mdtraj.csv");
        let mut expect: Vec<(Option<f64>, Option<f64>)> = Vec::new();
        for line in csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let p = f[2].parse::<f64>().ok();
            let s = f[3].parse::<f64>().ok();
            expect.push((p, s));
        }
        let bb: Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> = pdb
            .residues()
            .map(|r| {
                let get = |n: &str| {
                    r.atoms()
                        .find(|a| a.name() == n)
                        .map(|a| Vector3::new(a.x(), a.y(), a.z()))
                        .unwrap()
                };
                (get("N"), get("CA"), get("C"))
            })
            .collect();
        for i in 0..bb.len() {
            if i > 0 {
                let phi = compute_dihedral(&bb[i - 1].2, &bb[i].0, &bb[i].1, &bb[i].2).unwrap();
                assert!((phi - expect[i].0.unwrap()).abs() < 0.05, "phi res {} got {phi}", i + 1);
            }
            if i + 1 < bb.len() {
                let psi = compute_dihedral(&bb[i].0, &bb[i].1, &bb[i].2, &bb[i + 1].0).unwrap();
                assert!((psi - expect[i].1.unwrap()).abs() < 0.05, "psi res {} got {psi}", i + 1);
            }
        }
    }

    #[test]
    fn dihedral_mirror_negates() {
        let p = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.5, 0.0),
            Vector3::new(0.7, 1.5, 0.9),
        ];
        let a = compute_dihedral(&p[0], &p[1], &p[2], &p[3]).unwrap();
        let m: Vec<Vector3<f64>> = p.iter().map(|v| Vector3::new(v.x, v.y, -v.z)).collect();
        let b = compute_dihedral(&m[0], &m[1], &m[2], &m[3]).unwrap();
        assert!((a + b).abs() < 1e-9);
        assert!(a > 0.0, "right-handed twist must be positive, got {a}");
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p proteus-core dihedral -- --nocapture`
Expected: `dihedral_sign_matches_iupac_on_crambin` FAILS with `phi res 2 got 107.8…`; `dihedral_mirror_negates` FAILS on the `a > 0.0` assertion.

- [ ] **Step 4: Fix the implementation**

Replace the body of `compute_dihedral` after the collinearity check:
```rust
    let n1 = b1.cross(&b2);
    let n2 = b2.cross(&b3);
    // IUPAC/Blondel–Karplus signed torsion: atan2(|b2| b1·n2, n1·n2)
    let y = b2_norm * b1.dot(&n2);
    let x = n1.dot(&n2);
    Ok(y.atan2(x).to_degrees())
```
Delete the `m1` line.

- [ ] **Step 5: Run the whole core suite; fix tests that pinned the wrong sign**

Run: `cargo test -p proteus-core`
Any existing test asserting `CoreHelix` for e.g. `(60.0, 45.0)` must be flipped to `(-60.0, -45.0)`. `test_ramachandran_classification` may now fail on its own hard-coded values — update those values to IUPAC ones (helix `(-63.0, -42.0)`, strand `(-120.0, 130.0)`, αL `(57.0, 47.0)`).
Expected: all pass.

- [ ] **Step 6: Clippy/fmt and commit**

```bash
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add crates/proteus-core/src/structure.rs crates/proteus-core/tests/data/1crn_phipsi_mdtraj.csv
git commit -m "fix(core): correct backbone dihedral sign convention (all phi/psi were negated)

Pins 44 phi/psi pairs of 1CRN to mdtraj within 0.05 deg."
```

---

### Task 2: Chain-aware backbone extraction (`backbone.rs`)

**Files:**
- Create: `crates/proteus-core/src/backbone.rs`
- Modify: `crates/proteus-core/src/lib.rs` (add `pub mod backbone;`), `crates/proteus-core/src/metrics.rs:191-255` (use it)

**Interfaces:**
- Produces:
```rust
pub struct BackboneResidue {
    pub chain_id: String,
    pub seq_num: isize,
    pub name: String,          // 3-letter, trimmed, uppercase
    pub n: Option<Vector3<f64>>,
    pub ca: Option<Vector3<f64>>,
    pub c: Option<Vector3<f64>>,
    pub o: Option<Vector3<f64>>,
    pub b_factor: f64,         // CA B-factor
    pub chain_break_before: bool,
}
pub fn extract_backbone(pdb: &pdbtbx::PDB) -> Vec<BackboneResidue>;
pub fn omega(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64>; // CA(i-1),C(i-1),N(i),CA(i)
pub fn phi(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64>;
pub fn psi(cur: &BackboneResidue, next: &BackboneResidue) -> Option<f64>;
```
`chain_break_before` is true when the chain id differs from the previous residue or `|C(i−1) − N(i)| > 2.5 Å` or either atom is missing. `phi`/`psi`/`omega` return `None` across a break.

- [ ] **Step 1: Write the failing tests** (`backbone.rs` bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn crambin() -> pdbtbx::PDB {
        pdbtbx::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"),
            pdbtbx::StrictnessLevel::Loose,
        )
        .unwrap()
        .0
    }

    #[test]
    fn extracts_46_residues_with_all_backbone_atoms() {
        let bb = extract_backbone(&crambin());
        assert_eq!(bb.len(), 46);
        assert!(bb.iter().all(|r| r.n.is_some() && r.ca.is_some() && r.c.is_some() && r.o.is_some()));
        assert_eq!(bb[0].name, "THR");
        assert_eq!(bb[0].seq_num, 1);
        assert!(bb[0].chain_break_before);
        assert!(bb[1..].iter().all(|r| !r.chain_break_before));
    }

    #[test]
    fn phi_psi_omega_on_crambin() {
        let bb = extract_backbone(&crambin());
        let phi7 = phi(&bb[5], &bb[6]).unwrap();
        let psi7 = psi(&bb[6], &bb[7]).unwrap();
        assert!((phi7 + 63.6).abs() < 0.1, "{phi7}");
        assert!((psi7 + 42.1).abs() < 0.1, "{psi7}");
        let w = omega(&bb[5], &bb[6]).unwrap();
        assert!(w.abs() > 150.0, "trans peptide expected, got {w}");
        assert!(phi(&bb[45], &bb[0]).is_none() || bb[0].chain_break_before);
    }

    #[test]
    fn two_chains_break() {
        let text = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n\
TER\n\
ATOM      5  N   GLY B   1      20.000   0.000   0.000  1.00  0.00           N\n\
ATOM      6  CA  GLY B   1      21.458   0.000   0.000  1.00  0.00           C\n\
ATOM      7  C   GLY B   1      22.009   1.420   0.000  1.00  0.00           C\n\
ATOM      8  O   GLY B   1      21.251   2.390   0.000  1.00  0.00           O\n\
END\n";
        let (pdb, _) = pdbtbx::open_raw(std::io::BufReader::new(text.as_bytes()), pdbtbx::StrictnessLevel::Loose).unwrap();
        let bb = extract_backbone(&pdb);
        assert_eq!(bb.len(), 2);
        assert!(bb[1].chain_break_before);
        assert!(phi(&bb[0], &bb[1]).is_none());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p proteus-core backbone`
Expected: compile error — module missing.

- [ ] **Step 3: Implement `backbone.rs`**

```rust
//! Chain-aware backbone extraction shared by Ramachandran, DSSP and confidence analysis.

use nalgebra::Vector3;

use crate::structure::compute_dihedral;

/// Maximum C(i-1)–N(i) distance for a continuous peptide bond (DSSP convention).
pub const MAX_PEPTIDE_BOND: f64 = 2.5;

#[derive(Debug, Clone)]
pub struct BackboneResidue {
    pub chain_id: String,
    pub seq_num: isize,
    pub name: String,
    pub n: Option<Vector3<f64>>,
    pub ca: Option<Vector3<f64>>,
    pub c: Option<Vector3<f64>>,
    pub o: Option<Vector3<f64>>,
    pub b_factor: f64,
    pub chain_break_before: bool,
}

impl BackboneResidue {
    pub fn is_proline(&self) -> bool {
        self.name == "PRO"
    }
    pub fn is_glycine(&self) -> bool {
        self.name == "GLY"
    }
}

/// Extract one entry per residue that has a C-alpha, in file order, with chain breaks marked.
pub fn extract_backbone(pdb: &pdbtbx::PDB) -> Vec<BackboneResidue> {
    let mut out: Vec<BackboneResidue> = Vec::new();
    for chain in pdb.chains() {
        for residue in chain.residues() {
            let mut r = BackboneResidue {
                chain_id: chain.id().to_string(),
                seq_num: residue.serial_number(),
                name: residue.name().map(|n| n.trim().to_uppercase()).unwrap_or_default(),
                n: None,
                ca: None,
                c: None,
                o: None,
                b_factor: 0.0,
                chain_break_before: false,
            };
            for atom in residue.atoms() {
                let v = Vector3::new(atom.x(), atom.y(), atom.z());
                match atom.name().trim() {
                    "N" => r.n = Some(v),
                    "CA" => {
                        r.ca = Some(v);
                        r.b_factor = atom.b_factor();
                    }
                    "C" => r.c = Some(v),
                    "O" | "O1" | "OXT" if r.o.is_none() => r.o = Some(v),
                    _ => {}
                }
            }
            if r.ca.is_none() {
                continue;
            }
            r.chain_break_before = match out.last() {
                None => true,
                Some(prev) => {
                    prev.chain_id != r.chain_id
                        || match (prev.c, r.n) {
                            (Some(c), Some(n)) => (c - n).norm() > MAX_PEPTIDE_BOND,
                            _ => true,
                        }
                }
            };
            out.push(r);
        }
    }
    out
}

/// φ(i) = C(i−1)–N(i)–CA(i)–C(i). `None` across a chain break or with missing atoms.
pub fn phi(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64> {
    if cur.chain_break_before {
        return None;
    }
    compute_dihedral(&prev.c?, &cur.n?, &cur.ca?, &cur.c?).ok()
}

/// ψ(i) = N(i)–CA(i)–C(i)–N(i+1).
pub fn psi(cur: &BackboneResidue, next: &BackboneResidue) -> Option<f64> {
    if next.chain_break_before {
        return None;
    }
    compute_dihedral(&cur.n?, &cur.ca?, &cur.c?, &next.n?).ok()
}

/// ω(i) = CA(i−1)–C(i−1)–N(i)–CA(i); ≈180° trans, ≈0° cis.
pub fn omega(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64> {
    if cur.chain_break_before {
        return None;
    }
    compute_dihedral(&prev.ca?, &prev.c?, &cur.n?, &cur.ca?).ok()
}
```
Add `pub mod backbone;` to `lib.rs` (do **not** glob re-export it — `phi`/`psi` are too generic).

- [ ] **Step 4: Run tests**

Run: `cargo test -p proteus-core backbone`
Expected: 3 pass. (If `residue.serial_number()` type differs in pdbtbx 0.11, cast: `residue.serial_number() as isize`.)

- [ ] **Step 5: Use it in `metrics.rs`**

Replace the local `struct ResidueBackbone` and the residue loop (`metrics.rs:199-254`) with:
```rust
    let backbones = crate::backbone::extract_backbone(pdb);
    for r in &backbones {
        if let Some(ca) = r.ca {
            ca_coords.push(ca);
            plddts.push(r.b_factor);
        }
    }
    for atom in pdb.atoms() {
        let coord = Vector3::new(atom.x(), atom.y(), atom.z());
        let elem_symbol = atom
            .element()
            .map(|e| e.symbol().to_string())
            .unwrap_or_else(|| atom.name().trim().chars().next().unwrap_or('C').to_string());
        all_atoms.push(crate::sasa::AtomDescriptor::new(coord, elem_symbol));
    }
```
and replace the φ/ψ loop (`metrics.rs:289-338`) body with:
```rust
    for i in 0..n_res {
        let phi = if i > 0 { crate::backbone::phi(&backbones[i - 1], &backbones[i]) } else { None };
        let psi = if i + 1 < n_res { crate::backbone::psi(&backbones[i], &backbones[i + 1]) } else { None };
        let next_name = if i + 1 < n_res && !backbones[i + 1].chain_break_before {
            Some(backbones[i + 1].name.as_str())
        } else {
            None
        };
        let context = crate::structure::ResidueContext::from_names(&backbones[i].name, next_name);
        let region = match (phi, psi) {
            (Some(p), Some(s)) => crate::structure::classify_ramachandran_context(p, s, context),
            _ => crate::structure::RamachandranRegion::Outlier,
        };
        ramachandran_points.push((phi, psi, region));
        phi_psi_context.push((phi, psi, context));
    }
```
(`ResidueContext`/`classify_ramachandran_context` are replaced in Task 4; this keeps the tree compiling now.)

- [ ] **Step 6: Run full workspace tests, clippy, fmt, commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add crates/proteus-core && git commit -m "feat(core): chain-aware backbone extraction with peptide-bond chain breaks"
```

---

### Task 3: `proteus-dssp` crate — Kabsch–Sander secondary structure

**Files:**
- Create: `crates/proteus-dssp/Cargo.toml`, `crates/proteus-dssp/src/lib.rs`, `crates/proteus-dssp/src/hbond.rs`, `crates/proteus-dssp/src/assign.rs`, `crates/proteus-dssp/README.md`
- Modify: `Cargo.toml` (workspace members + `proteus-dssp = { path = ... }`), `crates/proteus-core/Cargo.toml`, `crates/proteus-core/src/structure.rs:82-186` (delete P-SEA, adapt), `crates/proteus-core/src/metrics.rs` (call site)
- Test: `crates/proteus-dssp/src/assign.rs` unit tests; `crates/proteus-core/tests/data/1crn_dssp_mdtraj.txt` (new)

**Interfaces:**
- Produces (`proteus_dssp`):
```rust
pub struct Residue { pub n: [f64; 3], pub ca: [f64; 3], pub c: [f64; 3], pub o: [f64; 3], pub is_proline: bool, pub chain_break_before: bool }
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Ss { H, B, E, G, I, T, S, Loop }
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Simple { Helix, Strand, Coil }
impl Ss { pub fn as_char(self) -> char; pub fn simplify(self) -> Simple }
pub fn assign(residues: &[Residue]) -> Vec<Ss>;
```
- Consumes: nothing from the workspace (standalone).

- [ ] **Step 1: Generate the mdtraj 8-state reference for crambin**

```bash
validate/.venv/bin/python - <<'EOF'
import mdtraj as md
t = md.load("crates/proteus-core/tests/data/1crn.pdb")
ss = md.compute_dssp(t, simplified=False)[0]
open("crates/proteus-core/tests/data/1crn_dssp_mdtraj.txt","w").write("".join(ss).replace(" ", "-") + "\n")
print("".join(ss))
EOF
```
Expected string (mdtraj, 46 chars): ` EE   HHHHHHHHHHH  TT HHHHHHHH  EE      HHH  ` modulo exact spacing — copy whatever mdtraj prints; that file is the oracle.

- [ ] **Step 2: Scaffold the crate**

`crates/proteus-dssp/Cargo.toml`:
```toml
[package]
name = "proteus-dssp"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
description = "Pure-Rust Kabsch–Sander DSSP secondary-structure assignment (DSSP 2.x semantics)."
repository.workspace = true
keywords = ["bioinformatics", "dssp", "protein", "secondary-structure"]
categories = ["science::bioinformatics"]

[dependencies]
serde = { workspace = true, optional = true }

[features]
default = []
serde = ["dep:serde"]
```
Root `Cargo.toml`: add `"crates/proteus-dssp"` to `members` and `proteus-dssp = { path = "crates/proteus-dssp" }` under `[workspace.dependencies]`.

`src/lib.rs`:
```rust
//! Kabsch–Sander DSSP secondary-structure assignment in pure Rust.
//!
//! Semantics follow DSSP 2.x (as ported by mdtraj): α-helix (H) overrides sheet
//! assignments, 3₁₀ (G) and π (I) helices only fill unassigned residues, no PPII.
#![forbid(unsafe_code)]

mod assign;
mod hbond;

pub use assign::{assign, Residue, Simple, Ss};
```

- [ ] **Step 3: Write failing tests** (`src/assign.rs` bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Ideal α-helix backbone from standard internal geometry (φ=−57, ψ=−47, ω=180).
    fn ideal_helix(n: usize) -> Vec<Residue> {
        build_chain(n, -57.0, -47.0)
    }

    /// Build N residues from φ/ψ with NeRF; ω = 180°, standard bond lengths/angles.
    fn build_chain(n: usize, phi: f64, psi: f64) -> Vec<Residue> {
        use std::f64::consts::PI;
        fn place(a: [f64; 3], b: [f64; 3], c: [f64; 3], bond: f64, angle: f64, torsion: f64) -> [f64; 3] {
            let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let cross = |p: [f64; 3], q: [f64; 3]| [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
            let norm = |p: [f64; 3]| { let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(); [p[0] / l, p[1] / l, p[2] / l] };
            let bc = norm(sub(c, b));
            let nrm = norm(cross(sub(b, a), bc));
            let m = [bc, cross(nrm, bc), nrm];
            let (ang, tor) = (angle * PI / 180.0, torsion * PI / 180.0);
            let d2 = [-bond * ang.cos(), bond * ang.sin() * tor.cos(), bond * ang.sin() * tor.sin()];
            [
                c[0] + m[0][0] * d2[0] + m[1][0] * d2[1] + m[2][0] * d2[2],
                c[1] + m[0][1] * d2[0] + m[1][1] * d2[1] + m[2][1] * d2[2],
                c[2] + m[0][2] * d2[0] + m[1][2] * d2[1] + m[2][2] * d2[2],
            ]
        }
        let mut n_ = [0.0, 0.0, 0.0];
        let mut ca = [1.458, 0.0, 0.0];
        let mut c = place([0.0, 1.0, 0.0], n_, ca, 1.525, 111.2, -120.0);
        let mut out = Vec::new();
        for i in 0..n {
            // O is placed in the peptide plane: N(i+1) is built first, O opposite.
            let n_next = place(n_, ca, c, 1.329, 116.2, psi);
            let o = place(n_next, ca, c, 1.231, 120.5, 180.0);
            out.push(Residue { n: n_, ca, c, o, is_proline: false, chain_break_before: i == 0 });
            let ca_next = place(ca, c, n_next, 1.458, 121.7, 180.0);
            let c_next = place(c, n_next, ca_next, 1.525, 111.2, phi);
            n_ = n_next;
            ca = ca_next;
            c = c_next;
        }
        out
    }

    #[test]
    fn ideal_helix_is_mostly_h() {
        let ss = assign(&ideal_helix(20));
        let s: String = ss.iter().map(|x| x.as_char()).collect();
        let h = ss.iter().filter(|x| **x == Ss::H).count();
        assert!(h >= 14, "expected a long H run, got {s}");
        assert!(!s.contains('E'), "{s}");
    }

    #[test]
    fn extended_chain_alone_is_coil() {
        let ss = assign(&build_chain(12, -120.0, 130.0));
        assert!(ss.iter().all(|x| matches!(x, Ss::Loop | Ss::S | Ss::T)), "{:?}", ss);
    }

    #[test]
    fn crambin_matches_mdtraj_8_state() {
        let (pdb, _) = pdbtbx::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../proteus-core/tests/data/1crn.pdb"),
            pdbtbx::StrictnessLevel::Loose,
        )
        .unwrap();
        let expect = include_str!("../../proteus-core/tests/data/1crn_dssp_mdtraj.txt").trim().to_string();
        let residues: Vec<Residue> = pdb
            .residues()
            .map(|r| {
                let get = |n: &str| {
                    let a = r.atoms().find(|a| a.name() == n).unwrap();
                    [a.x(), a.y(), a.z()]
                };
                Residue { n: get("N"), ca: get("CA"), c: get("C"), o: get("O"), is_proline: r.name() == Some("PRO"), chain_break_before: false }
            })
            .collect();
        let mut residues = residues;
        residues[0].chain_break_before = true;
        let got: String = assign(&residues).iter().map(|s| s.as_char()).collect();
        let agree = got.chars().zip(expect.chars()).filter(|(a, b)| a == b).count();
        eprintln!("mdtraj  {expect}\nproteus {got}\nagree {agree}/46");
        assert!(agree >= 44, "8-state agreement {agree}/46 below 44");
    }
}
```
Add to `crates/proteus-dssp/Cargo.toml`: `[dev-dependencies] pdbtbx = { workspace = true }`.

- [ ] **Step 4: Run to verify failure**

Run: `cargo test -p proteus-dssp`
Expected: compile errors (`assign`, `Residue`, `Ss` undefined).

- [ ] **Step 5: Implement `hbond.rs`**

```rust
//! Backbone hydrogen placement and Kabsch–Sander H-bond energies.

use crate::assign::Residue;

pub const COUPLING: f64 = -27.888; // -0.42 * 0.20 * 332 kcal/mol·Å
pub const MIN_ENERGY: f64 = -9.9;
pub const MAX_BOND_ENERGY: f64 = -0.5;
pub const MIN_CA_DISTANCE: f64 = 9.0;
const MIN_DISTANCE: f64 = 0.5;

pub type V = [f64; 3];

pub fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn norm(a: V) -> f64 { dot(a, a).sqrt() }
pub fn dist(a: V, b: V) -> f64 { norm(sub(a, b)) }

/// Amide hydrogen: H = N + unit(C(i−1) − O(i−1)). Proline and chain starts get H = N,
/// which makes the energy identically 0 (never a bond), matching DSSP.
pub fn place_hydrogens(res: &[Residue]) -> Vec<V> {
    let mut h = Vec::with_capacity(res.len());
    for i in 0..res.len() {
        let r = &res[i];
        if i == 0 || r.is_proline || r.chain_break_before {
            h.push(r.n);
            continue;
        }
        let p = &res[i - 1];
        let d = sub(p.c, p.o);
        let l = norm(d);
        if l < 1e-6 {
            h.push(r.n);
        } else {
            h.push([r.n[0] + d[0] / l, r.n[1] + d[1] / l, r.n[2] + d[2] / l]);
        }
    }
    h
}

/// Energy of the N–H(donor) ··· O=C(acceptor) bond, kcal/mol.
pub fn energy(donor: &Residue, donor_h: V, acceptor: &Residue) -> f64 {
    let r_ho = dist(donor_h, acceptor.o);
    let r_hc = dist(donor_h, acceptor.c);
    let r_nc = dist(donor.n, acceptor.c);
    let r_no = dist(donor.n, acceptor.o);
    if r_ho < MIN_DISTANCE || r_hc < MIN_DISTANCE || r_nc < MIN_DISTANCE || r_no < MIN_DISTANCE {
        return MIN_ENERGY;
    }
    let e = COUPLING / r_ho - COUPLING / r_hc + COUPLING / r_nc - COUPLING / r_no;
    e.max(MIN_ENERGY)
}

#[derive(Clone, Copy, Debug)]
pub struct Partner {
    pub index: Option<usize>,
    pub energy: f64,
}

impl Default for Partner {
    fn default() -> Self {
        Partner { index: None, energy: 0.0 }
    }
}

/// Per residue: the two strongest acceptors of its N–H and the two strongest donors to its C=O.
#[derive(Clone, Debug, Default)]
pub struct Bonds {
    pub acceptor: [Partner; 2], // residues whose C=O this residue's N–H bonds to
    pub donor: [Partner; 2],    // residues whose N–H bonds to this residue's C=O
}

fn insert(slot: &mut [Partner; 2], index: usize, energy: f64) {
    if energy < slot[0].energy {
        slot[1] = slot[0];
        slot[0] = Partner { index: Some(index), energy };
    } else if energy < slot[1].energy {
        slot[1] = Partner { index: Some(index), energy };
    }
}

pub fn compute_bonds(res: &[Residue]) -> Vec<Bonds> {
    let h = place_hydrogens(res);
    let mut bonds = vec![Bonds::default(); res.len()];
    for i in 0..res.len() {
        for j in (i + 1)..res.len() {
            if dist(res[i].ca, res[j].ca) >= MIN_CA_DISTANCE {
                continue;
            }
            let e = energy(&res[i], h[i], &res[j]);
            insert(&mut bonds[i].acceptor, j, e);
            insert(&mut bonds[j].donor, i, e);
            if j != i + 1 {
                let e = energy(&res[j], h[j], &res[i]);
                insert(&mut bonds[j].acceptor, i, e);
                insert(&mut bonds[i].donor, j, e);
            }
        }
    }
    bonds
}

/// True if residue `a`'s N–H donates to residue `b`'s C=O with E < −0.5.
pub fn test_bond(bonds: &[Bonds], a: usize, b: usize) -> bool {
    bonds[a]
        .acceptor
        .iter()
        .any(|p| p.index == Some(b) && p.energy < MAX_BOND_ENERGY)
}
```

- [ ] **Step 6: Implement `assign.rs`**

```rust
//! DSSP 2.x secondary-structure assignment (helices, bridges/ladders, bends, turns).

use crate::hbond::{compute_bonds, dist, dot, norm, sub, test_bond, Bonds, V};

#[derive(Clone, Debug)]
pub struct Residue {
    pub n: V,
    pub ca: V,
    pub c: V,
    pub o: V,
    pub is_proline: bool,
    pub chain_break_before: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Ss {
    H,
    B,
    E,
    G,
    I,
    T,
    S,
    Loop,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Simple {
    Helix,
    Strand,
    Coil,
}

impl Ss {
    pub fn as_char(self) -> char {
        match self {
            Ss::H => 'H',
            Ss::B => 'B',
            Ss::E => 'E',
            Ss::G => 'G',
            Ss::I => 'I',
            Ss::T => 'T',
            Ss::S => 'S',
            Ss::Loop => '-',
        }
    }
    pub fn simplify(self) -> Simple {
        match self {
            Ss::H | Ss::G | Ss::I => Simple::Helix,
            Ss::E | Ss::B => Simple::Strand,
            _ => Simple::Coil,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HelixFlag {
    None,
    Start,
    End,
    StartAndEnd,
    Middle,
}

impl HelixFlag {
    fn is_start(self) -> bool {
        matches!(self, HelixFlag::Start | HelixFlag::StartAndEnd)
    }
}

/// No chain break strictly inside (a, b] — i.e. residues a..=b are one continuous segment.
fn no_chain_break(res: &[Residue], a: usize, b: usize) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    !res[lo + 1..=hi].iter().any(|r| r.chain_break_before)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BridgeType {
    Parallel,
    Antiparallel,
}

#[derive(Debug)]
struct Bridge {
    kind: BridgeType,
    i: Vec<usize>,
    j: std::collections::VecDeque<usize>,
}

fn test_bridge(res: &[Residue], bonds: &[Bonds], a: usize, b: usize) -> Option<BridgeType> {
    if a == 0 || b == 0 || a + 1 >= res.len() || b + 1 >= res.len() {
        return None;
    }
    let (a1, a3, b1, b3) = (a - 1, a + 1, b - 1, b + 1);
    if !(no_chain_break(res, a1, a3) && no_chain_break(res, b1, b3)) {
        return None;
    }
    let tb = |x, y| test_bond(bonds, x, y);
    if (tb(a3, b) && tb(b, a1)) || (tb(b3, a) && tb(a, b1)) {
        Some(BridgeType::Parallel)
    } else if (tb(a3, b1) && tb(b3, a1)) || (tb(b, a) && tb(a, b)) {
        Some(BridgeType::Antiparallel)
    } else {
        None
    }
}

fn assign_sheets(res: &[Residue], bonds: &[Bonds], ss: &mut [Ss]) {
    let n = res.len();
    let mut bridges: Vec<Bridge> = Vec::new();
    for i in 1..n.saturating_sub(1) {
        for j in (i + 1)..n.saturating_sub(1) {
            let Some(kind) = test_bridge(res, bonds, i, j) else { continue };
            let mut found = false;
            for br in bridges.iter_mut() {
                if br.kind != kind || i != *br.i.last().unwrap() + 1 {
                    continue;
                }
                match kind {
                    BridgeType::Parallel if *br.j.back().unwrap() + 1 == j => {
                        br.i.push(i);
                        br.j.push_back(j);
                        found = true;
                    }
                    BridgeType::Antiparallel if *br.j.front().unwrap() == j + 1 => {
                        br.i.push(i);
                        br.j.push_front(j);
                        found = true;
                    }
                    _ => {}
                }
                if found {
                    break;
                }
            }
            if !found {
                bridges.push(Bridge { kind, i: vec![i], j: std::collections::VecDeque::from(vec![j]) });
            }
        }
    }

    // Merge ladders across β-bulges.
    let mut i = 0;
    while i < bridges.len() {
        let mut j = i + 1;
        while j < bridges.len() {
            let (ibi, iei) = (bridges[i].i[0], *bridges[i].i.last().unwrap());
            let (jbi, jei) = (bridges[j].i[0], *bridges[j].i.last().unwrap());
            let (ibj, iej) = (*bridges[i].j.front().unwrap(), *bridges[i].j.back().unwrap());
            let (jbj, jej) = (*bridges[j].j.front().unwrap(), *bridges[j].j.back().unwrap());
            let same = bridges[i].kind == bridges[j].kind;
            let cont_i = no_chain_break(res, ibi.min(jbi), iei.max(jei));
            let cont_j = no_chain_break(res, ibj.min(jbj), iej.max(jej));
            if !same || !cont_i || !cont_j || jbi < iei || jbi - iei >= 6 || (iei >= jbi && ibi <= jei) {
                j += 1;
                continue;
            }
            let bulge = match bridges[i].kind {
                BridgeType::Parallel => (jbj > iej && jbj - iej < 6 && ibj < jbj) || (jbj > iej && jbj - iej < 3),
                BridgeType::Antiparallel => (ibj > jej && ibj - jej < 6 && jej < ibj) || (ibj > jej && ibj - jej < 3),
            };
            if !bulge {
                j += 1;
                continue;
            }
            let other = bridges.remove(j);
            bridges[i].i.extend(other.i);
            match bridges[i].kind {
                BridgeType::Parallel => bridges[i].j.extend(other.j),
                BridgeType::Antiparallel => {
                    for x in other.j.into_iter().rev() {
                        bridges[i].j.push_front(x);
                    }
                }
            }
            // do not advance j: re-examine the element that shifted into position j
        }
        i += 1;
    }

    for br in &bridges {
        let kind = if br.i.len() > 1 { Ss::E } else { Ss::B };
        for k in br.i[0]..=*br.i.last().unwrap() {
            if ss[k] != Ss::E {
                ss[k] = kind;
            }
        }
        let (lo, hi) = (*br.j.front().unwrap(), *br.j.back().unwrap());
        for k in lo.min(hi)..=lo.max(hi) {
            if ss[k] != Ss::E {
                ss[k] = kind;
            }
        }
    }
}

fn assign_helices(res: &[Residue], bonds: &[Bonds], ss: &mut [Ss]) {
    let n = res.len();
    // flags[stride-3][i]
    let mut flags = vec![vec![HelixFlag::None; n]; 3];
    for stride in 3..=5usize {
        let f = &mut flags[stride - 3];
        for i in 0..n.saturating_sub(stride) {
            if test_bond(bonds, i + stride, i) && no_chain_break(res, i, i + stride) {
                f[i + stride] = HelixFlag::End;
                for k in (i + 1)..(i + stride) {
                    if f[k] == HelixFlag::None {
                        f[k] = HelixFlag::Middle;
                    }
                }
                f[i] = if f[i] == HelixFlag::End { HelixFlag::StartAndEnd } else { HelixFlag::Start };
            }
        }
    }
    // α (4): unconditional
    for i in 1..n.saturating_sub(4) {
        if flags[1][i].is_start() && flags[1][i - 1].is_start() {
            for k in i..=(i + 3) {
                ss[k] = Ss::H;
            }
        }
    }
    // 3₁₀ (3): only into loop/G
    for i in 1..n.saturating_sub(3) {
        if flags[0][i].is_start() && flags[0][i - 1].is_start() {
            let empty = (i..=(i + 2)).all(|k| matches!(ss[k], Ss::Loop | Ss::G));
            if empty {
                for k in i..=(i + 2) {
                    ss[k] = Ss::G;
                }
            }
        }
    }
    // π (5): only into loop/I
    for i in 1..n.saturating_sub(5) {
        if flags[2][i].is_start() && flags[2][i - 1].is_start() {
            let empty = (i..=(i + 4)).all(|k| matches!(ss[k], Ss::Loop | Ss::I));
            if empty {
                for k in i..=(i + 4) {
                    ss[k] = Ss::I;
                }
            }
        }
    }
    // turns and bends
    for i in 1..n.saturating_sub(1) {
        if ss[i] != Ss::Loop {
            continue;
        }
        let mut is_turn = false;
        'outer: for stride in 3..=5usize {
            for k in 1..stride {
                if i >= k && flags[stride - 3][i - k].is_start() {
                    is_turn = true;
                    break 'outer;
                }
            }
        }
        if is_turn {
            ss[i] = Ss::T;
        } else if is_bend(res, i) {
            ss[i] = Ss::S;
        }
    }
}

fn is_bend(res: &[Residue], i: usize) -> bool {
    if i < 2 || i + 2 >= res.len() || !no_chain_break(res, i - 2, i + 2) {
        return false;
    }
    let u = sub(res[i].ca, res[i - 2].ca);
    let v = sub(res[i + 2].ca, res[i].ca);
    let c = dot(u, v) / (norm(u) * norm(v));
    let kappa = c.clamp(-1.0, 1.0).acos().to_degrees();
    kappa > 70.0
}

/// Assign 8-state DSSP secondary structure.
pub fn assign(residues: &[Residue]) -> Vec<Ss> {
    let n = residues.len();
    let mut ss = vec![Ss::Loop; n];
    if n < 3 {
        return ss;
    }
    let bonds = compute_bonds(residues);
    assign_sheets(residues, &bonds, &mut ss);
    assign_helices(residues, &bonds, &mut ss);
    let _ = dist; // keep helper linked for future use
    ss
}
```
Remove the `let _ = dist;` line and the unused import if clippy complains — keep only what is used.

- [ ] **Step 7: Run tests; iterate until crambin agreement ≥ 44/46**

Run: `cargo test -p proteus-dssp -- --nocapture`
Expected: `ideal_helix_is_mostly_h` PASS, `extended_chain_alone_is_coil` PASS, `crambin_matches_mdtraj_8_state` prints both strings and PASSES. If agreement < 44, diff the strings: a mismatch on the *first/last residue of a helix* usually means the `flags[i]`/`flags[i-1]` start test is off by one; an `E`↔`-` mismatch means `test_bridge` argument order (donor is the first argument of `test_bond`).

- [ ] **Step 8: Wire into `proteus-core`, delete P-SEA**

`crates/proteus-core/Cargo.toml`: add `proteus-dssp = { workspace = true, features = ["serde"] }`.
In `structure.rs` replace `assign_secondary_structure` (`:82-186`) with:
```rust
/// Assign secondary structure with Kabsch–Sander DSSP (via `proteus-dssp`).
pub fn assign_secondary_structure(
    backbone: &[crate::backbone::BackboneResidue],
) -> SecondaryStructureSummary {
    let residues: Vec<proteus_dssp::Residue> = backbone
        .iter()
        .filter_map(|r| {
            Some(proteus_dssp::Residue {
                n: r.n?.into(),
                ca: r.ca?.into(),
                c: r.c?.into(),
                o: r.o?.into(),
                is_proline: r.is_proline(),
                chain_break_before: r.chain_break_before,
            })
        })
        .collect();
    let dssp = proteus_dssp::assign(&residues);
    let assignment: Vec<SecondaryStructure> = dssp
        .iter()
        .map(|s| match s.simplify() {
            proteus_dssp::Simple::Helix => SecondaryStructure::Helix,
            proteus_dssp::Simple::Strand => SecondaryStructure::Strand,
            proteus_dssp::Simple::Coil => SecondaryStructure::Coil,
        })
        .collect();
    let n = assignment.len().max(1) as f64;
    let count = |k: SecondaryStructure| assignment.iter().filter(|s| **s == k).count() as f64 / n;
    SecondaryStructureSummary {
        helix_fraction: count(SecondaryStructure::Helix),
        strand_fraction: count(SecondaryStructure::Strand),
        coil_fraction: count(SecondaryStructure::Coil),
        dssp: dssp.iter().map(|s| s.as_char()).collect(),
        assignment,
    }
}
```
Add `pub dssp: String` to `SecondaryStructureSummary` (8-state string, `#[serde(default)]`). `Vector3<f64>` → `[f64;3]`: use `[v.x, v.y, v.z]` if `.into()` does not resolve. `SecondaryStructure` needs `PartialEq` (add to its derive if missing). Update the call in `metrics.rs`: `let ss_summary = crate::structure::assign_secondary_structure(&backbones);`. Delete the old P-SEA tests; add:
```rust
    #[test]
    fn crambin_three_state_fractions_match_dssp() {
        let (pdb, _) = pdbtbx::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"), pdbtbx::StrictnessLevel::Loose).unwrap();
        let bb = crate::backbone::extract_backbone(&pdb);
        let s = assign_secondary_structure(&bb);
        assert!((s.helix_fraction - 0.478).abs() < 0.05, "{}", s.helix_fraction);
        assert!((s.strand_fraction - 0.087).abs() < 0.05, "{}", s.strand_fraction);
        assert!((s.coil_fraction - 0.435).abs() < 0.05, "{}", s.coil_fraction);
    }
```
Fix every other place that constructs `SecondaryStructureSummary` (grep `SecondaryStructureSummary {`) to add `dssp: String::new()`.

- [ ] **Step 9: Workspace tests, clippy, fmt, commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add Cargo.toml Cargo.lock crates/proteus-dssp crates/proteus-core
git commit -m "feat(dssp): add pure-Rust Kabsch-Sander DSSP crate and replace P-SEA

Crambin 8-state agreement with mdtraj: <N>/46 (paste from test output)."
```

---

### Task 4: Top8000 Ramachandran contours (`rama8000.rs`)

**Files:**
- Create: `scripts/convert_rama8000.py`, `crates/proteus-core/data/rama8000/{general,glycine,cispro,transpro,prepro,ileval}.f32`, `crates/proteus-core/data/rama8000/NOTICE`, `crates/proteus-core/src/rama8000.rs`
- Modify: `crates/proteus-core/src/structure.rs` (types + delete box classifier, `:188-340`), `crates/proteus-core/src/metrics.rs` (call site), `crates/proteus-render/src/tui/dashboard.rs:160-162,498-499`, `crates/proteus-core/src/lib.rs`

**Interfaces:**
- Produces:
```rust
// rama8000.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum RamaClass { General, Glycine, CisPro, TransPro, PrePro, IleVal }
impl RamaClass { pub fn classify(name: &str, next_name: Option<&str>, omega: Option<f64>) -> RamaClass; }
pub fn score(class: RamaClass, phi: f64, psi: f64) -> f64;         // interpolated density
pub fn evaluate(class: RamaClass, phi: f64, psi: f64) -> RamachandranRegion;
// structure.rs
#[derive(...)] pub enum RamachandranRegion { Favored, Allowed, Outlier }
pub struct RamachandranStats { favored_fraction, allowed_fraction, outlier_fraction, outlier_count, total_evaluated }  // unchanged
pub fn evaluate_ramachandran(points: &[(Option<f64>, Option<f64>, RamaClass)]) -> RamachandranStats;
```
- Removed: `ResidueContext`, `classify_ramachandran`, `classify_ramachandran_context`, `evaluate_ramachandran_angles`, `evaluate_ramachandran_with_context`, variants `CoreHelix/CoreStrand/LeftHandedHelix`.

- [ ] **Step 1: Conversion script + data + NOTICE**

`scripts/convert_rama8000.py`:
```python
#!/usr/bin/env python3
"""Convert cctbx mmtbx/validation/ramachandran/rama8000_tables.h into six 180x180 LE-f32 grids.

Usage: python scripts/convert_rama8000.py path/to/rama8000_tables.h
Source: https://github.com/cctbx/cctbx_project (BSD-3, LBNL). Data: Top8000 (Richardson lab, Duke).
"""
import re, struct, sys, pathlib

NAMES = {"general": "general", "glycine": "glycine", "cis_pro": "cispro",
         "trans_pro": "transpro", "pre_pro": "prepro", "ile_val": "ileval"}
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(__file__).resolve().parents[1] / "crates/proteus-core/data/rama8000"
out.mkdir(parents=True, exist_ok=True)
for key, fname in NAMES.items():
    m = re.search(r"linear_table_%s\[\] = \{([^}]*)\}" % key, src, re.S)
    vals = [float(v) for v in m.group(1).replace("\n", "").split(",") if v.strip()]
    assert len(vals) == 180 * 180, (key, len(vals))
    (out / f"{fname}.f32").write_bytes(struct.pack("<%df" % len(vals), *vals))
    print(fname, len(vals), "min", min(vals), "max", max(vals))
```
Run: `python3 scripts/convert_rama8000.py /tmp/rama8000_tables.h` (file already downloaded; else `curl -sL -o /tmp/rama8000_tables.h https://raw.githubusercontent.com/cctbx/cctbx_project/master/mmtbx/validation/ramachandran/rama8000_tables.h`).
Expected: six lines, each `32400`, general max ≈ 0.6.

`crates/proteus-core/data/rama8000/NOTICE`:
```
Ramachandran percentile contour grids (Top8000) — six 180x180 grids, 2-degree bins
centred at odd degrees (-179 ... 179), row index = phi bin, column index = psi bin,
little-endian f32, generated by scripts/convert_rama8000.py from

  cctbx_project/mmtbx/validation/ramachandran/rama8000_tables.h
  https://github.com/cctbx/cctbx_project

cctbx Copyright (c) 2006 - 2026, The Regents of the University of California, through
Lawrence Berkeley National Laboratory. Redistributed under the cctbx BSD-3-style licence
(see https://github.com/cctbx/cctbx_project/blob/master/LICENSE.txt). The underlying
distributions are from the Top8000 dataset (Richardson laboratory, Duke University;
Lovell et al. 2003, Proteins 50:437; Chen et al. 2010, Acta Cryst D66:12).
Evaluation thresholds replicate mmtbx/validation/ramachandran/rama_eval.h.
```

- [ ] **Step 2: Write failing tests** (`rama8000.rs` bottom)

Get four oracle values from cctbx first:
```bash
validate/.venv/bin/python - <<'EOF'
from mmtbx.validation.ramachandran import rama_eval
r = rama_eval()
for cls, phi, psi in [("general",-63.0,-42.0),("general",-120.0,130.0),("general",60.0,60.0),("glycine",80.0,-170.0),("trans-proline",-65.0,145.0),("pre-proline",-80.0,80.0),("isoleucine or valine",-110.0,125.0),("general",-179.5,179.5)]:
    print(cls, phi, psi, "%.6f" % r.get_score(cls, phi, psi), r.evaluate_angles(cls, phi, psi))
EOF
```
(`uv pip install -q --python validate/.venv/bin/python cctbx-base` first if not present.) Paste the printed scores into the test below in place of the `EXPECT_*` placeholders — those are the only numbers you are allowed to type.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::RamachandranRegion as R;

    fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-5 }

    #[test]
    fn scores_match_cctbx_rama_eval() {
        assert!(close(score(RamaClass::General, -63.0, -42.0), EXPECT_GEN_HELIX));
        assert!(close(score(RamaClass::General, -120.0, 130.0), EXPECT_GEN_STRAND));
        assert!(close(score(RamaClass::General, 60.0, 60.0), EXPECT_GEN_AL));
        assert!(close(score(RamaClass::Glycine, 80.0, -170.0), EXPECT_GLY));
        assert!(close(score(RamaClass::TransPro, -65.0, 145.0), EXPECT_TPRO));
        assert!(close(score(RamaClass::PrePro, -80.0, 80.0), EXPECT_PREPRO));
        assert!(close(score(RamaClass::IleVal, -110.0, 125.0), EXPECT_ILEVAL));
        assert!(close(score(RamaClass::General, -179.5, 179.5), EXPECT_WRAP));
    }

    #[test]
    fn regions_follow_cctbx_thresholds() {
        assert_eq!(evaluate(RamaClass::General, -63.0, -42.0), R::Favored);
        assert_eq!(evaluate(RamaClass::General, -120.0, 130.0), R::Favored);
        assert_eq!(evaluate(RamaClass::General, 0.0, 0.0), R::Outlier);
        assert_eq!(evaluate(RamaClass::Glycine, 80.0, -170.0), R::Favored);
        assert_eq!(evaluate(RamaClass::TransPro, 60.0, 60.0), R::Outlier);
    }

    #[test]
    fn class_selection() {
        assert_eq!(RamaClass::classify("ALA", Some("PRO"), Some(180.0)), RamaClass::PrePro);
        assert_eq!(RamaClass::classify("GLY", Some("PRO"), Some(180.0)), RamaClass::Glycine);
        assert_eq!(RamaClass::classify("PRO", Some("ALA"), Some(178.0)), RamaClass::TransPro);
        assert_eq!(RamaClass::classify("PRO", Some("ALA"), Some(-5.0)), RamaClass::CisPro);
        assert_eq!(RamaClass::classify("ILE", None, None), RamaClass::IleVal);
        assert_eq!(RamaClass::classify("VAL", Some("PRO"), None), RamaClass::PrePro);
        assert_eq!(RamaClass::classify("XYZ", None, None), RamaClass::General);
    }

    #[test]
    fn crambin_is_almost_all_favored() {
        let (pdb, _) = pdbtbx::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"), pdbtbx::StrictnessLevel::Loose).unwrap();
        let d = crate::metrics::analyze_pdb_detailed(&pdb, None).unwrap();
        let s = d.metrics.ramachandran_stats.unwrap();
        assert_eq!(s.total_evaluated, 44);
        assert_eq!(s.outlier_count, 0, "{s:?}");
        assert!(s.favored_fraction >= 43.0 / 44.0 - 1e-9, "{s:?}");
    }
}
```
Precedence in `classify` (cctbx `ramalyze`): Gly → Glycine; Pro → Cis/TransPro by ω (cis if `|ω| < 30`; unknown ω → Trans); next is Pro → PrePro; Ile/Val → IleVal; else General.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p proteus-core rama8000`
Expected: compile error, module missing.

- [ ] **Step 4: Implement `rama8000.rs`**

```rust
//! MolProbity Top8000 Ramachandran evaluation (data and semantics from cctbx `rama_eval.h`).

use std::sync::OnceLock;

use crate::structure::RamachandranRegion;

const GRID: usize = 180;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum RamaClass {
    General,
    Glycine,
    CisPro,
    TransPro,
    PrePro,
    IleVal,
}

impl RamaClass {
    pub fn classify(name: &str, next_name: Option<&str>, omega: Option<f64>) -> RamaClass {
        match name {
            "GLY" => RamaClass::Glycine,
            "PRO" => match omega {
                Some(w) if w.abs() < 30.0 => RamaClass::CisPro,
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

const FAVORED_THRESHOLD: f64 = 0.02;

fn decode(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len(), GRID * GRID * 4, "rama8000 grid must be 180x180 f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

/// Neighbouring odd-degree bin centres around `v` and their bin indices (cctbx `get_bins_and_values`).
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

/// Interpolated Top8000 density at (φ, ψ) for the given class.
pub fn score(class: RamaClass, phi: f64, psi: f64) -> f64 {
    let t = class.table();
    let (p, s) = (wrap(phi), wrap(psi));
    let (p0, p1, x1, x2) = bins(p);
    let (s0, s1, y1, y2) = bins(s);
    let v = |a: usize, b: usize| t[a * GRID + b] as f64;
    // bilinear over the unit cell (x1,y1)-(x2,y2) — cctbx linear_interpolation_2d
    let (v11, v22, v12, v21) = (v(p0, s0), v(p1, s1), v(p0, s1), v(p1, s0));
    let fx = (p - x1) / (x2 - x1);
    let fy = (s - y1) / (y2 - y1);
    (1.0 - fx) * (1.0 - fy) * v11 + fx * fy * v22 + (1.0 - fx) * fy * v12 + fx * (1.0 - fy) * v21
}

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
```
If the `scores_match_cctbx_rama_eval` test disagrees at 1e-5 on any value, check cctbx's `linear_interpolation_2d(x1,y1,x2,y2,v1,v2,v3,v4,x,y)` argument order: `v1=(x1,y1)`, `v2=(x2,y2)`, `v3=(x1,y2)`, `v4=(x2,y1)` — that is what the `(v11, v22, v12, v21)` tuple above encodes.

- [ ] **Step 5: Rewrite the Ramachandran section of `structure.rs`**

Delete `ResidueContext`, `classify_ramachandran`, `classify_ramachandran_context`, `evaluate_ramachandran_angles`, `evaluate_ramachandran_with_context` and their tests. Replace `RamachandranRegion` with:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum RamachandranRegion {
    Favored,
    Allowed,
    Outlier,
}
```
and add:
```rust
/// Aggregate MolProbity-style Ramachandran statistics. Residues without both angles are not evaluated.
pub fn evaluate_ramachandran(
    points: &[(Option<f64>, Option<f64>, crate::rama8000::RamaClass)],
) -> RamachandranStats {
    let (mut fav, mut allow, mut out, mut total) = (0usize, 0usize, 0usize, 0usize);
    for (phi, psi, class) in points {
        let (Some(p), Some(s)) = (phi, psi) else { continue };
        total += 1;
        match crate::rama8000::evaluate(*class, *p, *s) {
            RamachandranRegion::Favored => fav += 1,
            RamachandranRegion::Allowed => allow += 1,
            RamachandranRegion::Outlier => out += 1,
        }
    }
    let n = total.max(1) as f64;
    RamachandranStats {
        favored_fraction: fav as f64 / n,
        allowed_fraction: allow as f64 / n,
        outlier_fraction: out as f64 / n,
        outlier_count: out,
        total_evaluated: total,
    }
}
```
Add `pub mod rama8000;` to `lib.rs` (no glob re-export).

- [ ] **Step 6: Update `metrics.rs` loop and downstream**

In `analyze_pdb_detailed` replace the φ/ψ loop from Task 2 with:
```rust
    let mut rama_points: Vec<(Option<f64>, Option<f64>, crate::rama8000::RamaClass)> = Vec::new();
    for i in 0..n_res {
        let phi = if i > 0 { crate::backbone::phi(&backbones[i - 1], &backbones[i]) } else { None };
        let psi = if i + 1 < n_res { crate::backbone::psi(&backbones[i], &backbones[i + 1]) } else { None };
        let omega = if i > 0 { crate::backbone::omega(&backbones[i - 1], &backbones[i]) } else { None };
        let next_name = if i + 1 < n_res && !backbones[i + 1].chain_break_before {
            Some(backbones[i + 1].name.as_str())
        } else {
            None
        };
        let class = crate::rama8000::RamaClass::classify(&backbones[i].name, next_name, omega);
        let region = match (phi, psi) {
            (Some(p), Some(s)) => crate::rama8000::evaluate(class, p, s),
            _ => crate::structure::RamachandranRegion::Outlier,
        };
        ramachandran_points.push((phi, psi, region));
        rama_points.push((phi, psi, class));
    }
    let rama_stats = crate::structure::evaluate_ramachandran(&rama_points);
```
Note: `ramachandran_points` entries with `None` angles keep `Outlier` only as a plot marker; they are excluded from stats (as before by `total_evaluated`). In `dashboard.rs:160-162` the match already handles `Outlier`/`Allowed`/`_`; change the sample data at `:498-499` to `RamachandranRegion::Favored`. Grep the workspace for `CoreHelix|CoreStrand|LeftHandedHelix|ResidueContext|classify_ramachandran|evaluate_ramachandran_` and fix every hit.

- [ ] **Step 7: Run workspace tests, clippy, fmt, commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add scripts/convert_rama8000.py crates/proteus-core crates/proteus-render
git commit -m "feat(core): MolProbity Top8000 Ramachandran contours replace box classifier

1CRN: 44 evaluated, 0 outliers (cctbx ramalyze: 43 favored / 1 allowed / 0 outliers).
Data converted from cctbx rama8000_tables.h (BSD-3), see data/rama8000/NOTICE."
```

---

### Task 5: pLDDT provenance (`confidence.rs`) and fitness renormalisation

**Files:**
- Create: `crates/proteus-core/src/confidence.rs`
- Modify: `crates/proteus-core/src/models.rs:77-99`, `crates/proteus-core/src/metrics.rs` (detection + struct init), `crates/proteus-core/src/ranking.rs:19-95`, `crates/proteus-cli/src/main.rs:433-470` (`Analyze` arm), `crates/proteus-storage/src/repository.rs:367,687`, `crates/proteus-render/src/tui/dashboard.rs:477`, `crates/proteus-server/src/api.rs` (component list), `crates/proteus-core/src/lib.rs`

**Interfaces:**
- Produces:
```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)] pub enum ConfidenceSource { Predicted, ExperimentalBFactor, #[default] Unknown }
pub fn detect_confidence_source(pdb: &pdbtbx::PDB, b_factors: &[f64]) -> ConfidenceSource;
// BiophysicalMetrics gains: #[serde(default)] pub confidence_source: ConfidenceSource
impl BiophysicalMetrics { pub fn plddt(&self) -> Option<&PlddtDistribution> } // None when ExperimentalBFactor
```

- [ ] **Step 1: Failing tests** (`confidence.rs` bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn pdb_with_header(header: &str) -> pdbtbx::PDB {
        let text = format!("{header}\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 12.00           C\nEND\n");
        pdbtbx::open_raw(std::io::BufReader::new(text.as_bytes()), pdbtbx::StrictnessLevel::Loose).unwrap().0
    }

    #[test]
    fn crambin_is_experimental() {
        let (pdb, _) = pdbtbx::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"), pdbtbx::StrictnessLevel::Loose).unwrap();
        let b: Vec<f64> = pdb.atoms().filter(|a| a.name() == "CA").map(|a| a.b_factor()).collect();
        assert_eq!(detect_confidence_source(&pdb, &b), ConfidenceSource::ExperimentalBFactor);
    }

    #[test]
    fn expdta_xray_wins_over_values() {
        let pdb = pdb_with_header("EXPDTA    X-RAY DIFFRACTION");
        assert_eq!(detect_confidence_source(&pdb, &[85.0, 90.0, 92.0]), ConfidenceSource::ExperimentalBFactor);
    }

    #[test]
    fn alphafold_remark_is_predicted() {
        let pdb = pdb_with_header("TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR P69905");
        assert_eq!(detect_confidence_source(&pdb, &[85.0, 90.0, 92.0]), ConfidenceSource::Predicted);
    }

    #[test]
    fn headerless_plddt_like_values_are_predicted() {
        let pdb = pdb_with_header("REMARK   1 NOTHING USEFUL");
        assert_eq!(detect_confidence_source(&pdb, &[71.0, 88.5, 93.2, 60.4]), ConfidenceSource::Predicted);
    }

    #[test]
    fn headerless_bfactor_like_values_are_experimental() {
        let pdb = pdb_with_header("REMARK   1 NOTHING USEFUL");
        assert_eq!(detect_confidence_source(&pdb, &[5.0, 8.1, 12.4, 6.6]), ConfidenceSource::ExperimentalBFactor);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p proteus-core confidence` → module missing.

- [ ] **Step 3: Implement**

```rust
//! Decide whether a structure's B-factor column carries a predictor's pLDDT or crystallographic B-factors.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSource {
    /// B-factor column holds pLDDT (AlphaFold, ESMFold, Boltz, OpenFold, …).
    Predicted,
    /// Experimental structure: B-factor column is a displacement parameter, not a confidence.
    ExperimentalBFactor,
    #[default]
    Unknown,
}

const EXPERIMENTAL_MARKERS: &[&str] = &["X-RAY", "NMR", "ELECTRON", "NEUTRON", "FIBER", "CRYSTALLOGRAPHY"];
const PREDICTED_MARKERS: &[&str] = &["ALPHAFOLD", "ESMFOLD", "ESM-FOLD", "BOLTZ", "OPENFOLD", "PLDDT", "COLABFOLD", "CHAI-1", "PREDICTED"];

/// Header text pdbtbx keeps: identifier + remarks (pdbtbx stores REMARK lines as `(number, text)`).
fn header_text(pdb: &pdbtbx::PDB) -> String {
    let mut s = String::new();
    if let Some(id) = pdb.identifier.as_deref() {
        s.push_str(id);
        s.push('\n');
    }
    for (_, line) in pdb.remarks() {
        s.push_str(line);
        s.push('\n');
    }
    s.to_uppercase()
}

pub fn detect_confidence_source(pdb: &pdbtbx::PDB, b_factors: &[f64]) -> ConfidenceSource {
    let header = header_text(pdb);
    if EXPERIMENTAL_MARKERS.iter().any(|m| header.contains(m)) {
        return ConfidenceSource::ExperimentalBFactor;
    }
    if PREDICTED_MARKERS.iter().any(|m| header.contains(m)) {
        return ConfidenceSource::Predicted;
    }
    if b_factors.is_empty() {
        return ConfidenceSource::Unknown;
    }
    let n = b_factors.len() as f64;
    let mean = b_factors.iter().sum::<f64>() / n;
    let var = b_factors.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let in_range = b_factors.iter().all(|v| (0.0..=100.0).contains(v));
    let plddt_like = in_range && mean > 30.0 && var.sqrt() < 40.0;
    if plddt_like {
        ConfidenceSource::Predicted
    } else {
        ConfidenceSource::ExperimentalBFactor
    }
}
```
pdbtbx 0.11 exposes `PDB::remarks()` as an iterator of `&(usize, String)` and `pdb.identifier: Option<String>`; adapt if the accessor names differ (`rg "pub fn remarks" ~/.cargo/registry/src/*/pdbtbx-0.11.0/src/structs/pdb.rs`). `EXPDTA` is not retained by pdbtbx — check with `rg -n "EXPDTA" ~/.cargo/registry/src/*/pdbtbx-0.11.0/src/`; if it is dropped, read the first 64 lines of the file in `io::open_structure` (Task 7) and pass the raw header through `analyze_pdb_file`; for now the value-heuristic path covers 1CRN (mean B ≈ 5.8 → Experimental) and the test `expdta_xray_wins_over_values` may need to feed the raw header string instead of a `PDB` — if so, change the signature to `detect_confidence_source(header: &str, b_factors: &[f64])` and have the caller build `header` from `pdb.identifier` + remarks + raw first lines.

- [ ] **Step 4: Plumb through `models.rs`, `metrics.rs`, `ranking.rs`**

`models.rs`: add `#[serde(default)] pub confidence_source: crate::confidence::ConfidenceSource,` to `BiophysicalMetrics` and
```rust
impl BiophysicalMetrics {
    /// pLDDT statistics, only meaningful for predicted structures.
    pub fn plddt(&self) -> Option<&PlddtDistribution> {
        match self.confidence_source {
            crate::confidence::ConfidenceSource::ExperimentalBFactor => None,
            _ => Some(&self.plddt_distribution),
        }
    }
}
```
`metrics.rs`: after collecting `plddts`, `let confidence_source = crate::confidence::detect_confidence_source(pdb, &plddts);` and set the field. Keep the 0–1 → 0–100 normalisation only when `confidence_source != ExperimentalBFactor`.
`ranking.rs`: replace the weighted sum with
```rust
    let has_plddt = metrics.plddt().is_some();
    let plddt_component = metrics.plddt().map(|p| p.mean.clamp(0.0, 100.0)).unwrap_or(0.0);
    // weights: pLDDT 0.30, compactness 0.20, rama 0.15, burial 0.15, network 0.20
    let (w_p, w_c, w_r, w_b, w_n) = if has_plddt {
        (0.30, 0.20, 0.15, 0.15, 0.20)
    } else {
        // redistribute the pLDDT weight proportionally over the remaining four (sum 0.70)
        (0.0, 0.20 / 0.70, 0.15 / 0.70, 0.15 / 0.70, 0.20 / 0.70)
    };
    let total_score = (w_p * plddt_component
        + w_c * compactness_component
        + w_r * ramachandran_component
        + w_b * hydrophobic_burial_component
        + w_n * interaction_network_component
        - clash_penalty)
        .clamp(0.0, 100.0);
```
Add a test in `ranking.rs`: same metrics with `confidence_source = ExperimentalBFactor` and `Predicted{mean 0}` → experimental score is strictly higher than predicted-with-zero-pLDDT and equals the four-term weighted mean.
Fix every `BiophysicalMetrics { … }` literal (`rg "BiophysicalMetrics \{" crates`) to add `confidence_source: Default::default()`. Add `pub mod confidence;` + `pub use confidence::ConfidenceSource;` in `lib.rs`; add `proteus_core::confidence::ConfidenceSource` to the OpenAPI components in `api.rs`.

- [ ] **Step 5: CLI**

In the `Analyze` arm replace the four pLDDT rows with:
```rust
            match metrics.plddt() {
                Some(p) => {
                    table.add_row(vec![Cell::new("Mean pLDDT"), Cell::new(format!("{:.2}", p.mean))]);
                    table.add_row(vec![Cell::new("Median pLDDT"), Cell::new(format!("{:.2}", p.median))]);
                    table.add_row(vec![Cell::new("Fraction High Conf (pLDDT >= 70)"), Cell::new(format!("{:.1}%", p.high_confidence_fraction * 100.0))]);
                    table.add_row(vec![Cell::new("Fraction Very High Conf (pLDDT >= 90)"), Cell::new(format!("{:.1}%", p.very_high_confidence_fraction * 100.0))]);
                }
                None => {
                    table.add_row(vec![Cell::new("pLDDT"), Cell::new("n/a (experimental structure; B-factor column is not a confidence)")]);
                }
            }
```
Add `--confidence-source <predicted|experimental|auto>` (default `auto`) to `Analyze`; when not `auto`, overwrite `metrics.confidence_source` after analysis and recompute `candidate_fitness_score` via `evaluate_candidate_fitness`.

- [ ] **Step 6: Run, verify, commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
./target/debug/proteus analyze --pdb crates/proteus-core/tests/data/1crn.pdb | rg -n "pLDDT|Fitness"
```
Expected: `pLDDT ┆ n/a (experimental …)` and a fitness score computed from four terms.
```bash
git add crates && git commit -m "feat(core,cli): detect pLDDT provenance; never report B-factors as confidence"
```

---

### Task 6: Relabel the heavy-atom clash metric

**Files:**
- Modify: `crates/proteus-core/src/clash.rs:22-33,124-126,235-245`, `crates/proteus-core/src/models.rs:1,86`, `crates/proteus-core/src/ranking.rs:80-83`, `crates/proteus-cli/src/main.rs:515-522,969,1118`, `crates/proteus-render/src/tui/dashboard.rs:415-427`, `crates/proteus-storage/src/export.rs:28,39,63,95,128,236,256,267`, `crates/proteus-server/src/api.rs:47`, `README.md` (the two "MolProbity Clashscore" mentions)

**Interfaces:**
- Produces: `pub struct StericOverlapStats { pub clash_count, pub heavy_atom_overlap_score: f64, pub worst_overlap, pub total_atoms_evaluated, pub clashes }`, `pub fn compute_steric_overlap(pdb) -> StericOverlapStats`; `BiophysicalMetrics.steric_overlap: Option<StericOverlapStats>` (serde alias `clash_stats`); export column `heavy_atom_overlap_score`; `pub const EXPORT_SCHEMA_VERSION: u32 = 2` in `export.rs` and written as Parquet key-value metadata `proteus.schema_version`.

- [ ] **Step 1: Failing test** (`export.rs` tests): assert the CSV header contains `heavy_atom_overlap_score` and not `clashscore`; assert Parquet file metadata has `proteus.schema_version = "2"` (read back with `parquet::file::reader::SerializedFileReader` → `metadata().file_metadata().key_value_metadata()`).

- [ ] **Step 2: Run** `cargo test -p proteus-storage export` → fails on header.

- [ ] **Step 3: Rename** with Serena `rename_symbol` where possible (`ClashStats` → `StericOverlapStats`, `compute_clash_stats` → `compute_steric_overlap`, field `clashscore` → `heavy_atom_overlap_score`, `clash_stats` → `steric_overlap`); add `#[serde(alias = "clash_stats")]` on the metrics field; doc comment on the struct:
```rust
/// Heavy-atom steric overlap statistics. **Not** the MolProbity clashscore: MolProbity adds
/// hydrogens (Reduce) before counting ≥ 0.4 Å overlaps; this metric uses heavy atoms only and
/// therefore under-counts. Score = overlaps per 1,000 heavy atoms evaluated.
```
CLI row label: `"Heavy-atom steric overlap (>0.4 Å, no H)"`; dashboard label `"Overlap/1k"`; CSV/Parquet column `heavy_atom_overlap_score`; add key-value metadata to the Parquet writer properties: `WriterProperties::builder().set_key_value_metadata(Some(vec![KeyValue::new("proteus.schema_version".into(), EXPORT_SCHEMA_VERSION.to_string())]))`. README: replace both "MolProbity Clashscore" headings with "Heavy-atom steric overlap (MolProbity-style, no hydrogens)" and add one sentence stating the difference.

- [ ] **Step 4: Run, commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add -A && git commit -m "refactor(core,storage,cli): relabel heavy-atom overlap score; it is not MolProbity clashscore

Export schema_version=2 (column heavy_atom_overlap_score)."
```

---

### Task 7: One structure-opening entry point with mmCIF and gzip

**Files:**
- Create: `crates/proteus-core/src/io.rs`, `crates/proteus-core/tests/data/1crn.cif` (download: `curl -sL https://files.rcsb.org/download/1CRN.cif -o crates/proteus-core/tests/data/1crn.cif`)
- Modify: `crates/proteus-core/src/metrics.rs:156-181` (`analyze_pdb_file` uses `io`), `crates/proteus-cli/src/main.rs` (every `open(`/`open_raw(` on user paths: `analyze`, `view`, `screen`, `submit`), `crates/proteus-render/src/lib.rs:127` (accept a `&Path` overload `load_structure_from_path`), `crates/proteus-core/src/lib.rs`

**Interfaces:**
- Produces: `pub fn open_structure(path: &Path) -> Result<pdbtbx::PDB, CoreError>` — dispatches on extension (`.pdb`, `.ent`, `.cif`, `.mmcif`, each optionally `.gz`); unknown extension → sniff first non-empty line: starts with `data_` → mmCIF else PDB. Also `pub fn open_structure_bytes(bytes: &[u8], hint: Option<&str>) -> Result<pdbtbx::PDB, CoreError>` for in-memory use (render/server).

- [ ] **Step 1: Failing test** (`io.rs` bottom)

```rust
    #[test]
    fn cif_and_pdb_give_same_metrics() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");
        let a = crate::metrics::analyze_pdb_file(std::path::Path::new(&format!("{dir}1crn.pdb")), None).unwrap();
        let b = crate::metrics::analyze_pdb_file(std::path::Path::new(&format!("{dir}1crn.cif")), None).unwrap();
        assert!((a.radius_of_gyration - b.radius_of_gyration).abs() < 1e-3);
        assert_eq!(a.ramachandran_stats.unwrap().outlier_count, b.ramachandran_stats.unwrap().outlier_count);
        assert_eq!(a.secondary_structure_summary.unwrap().dssp, b.secondary_structure_summary.unwrap().dssp);
    }

    #[test]
    fn gz_roundtrip() {
        // write 1crn.pdb.gz to a tempdir with flate2? — not a dependency: instead call `gzip -k` in the test only if available
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb");
        let dir = tempfile::tempdir().unwrap();
        let dst = dir.path().join("1crn.pdb.gz");
        let ok = std::process::Command::new("gzip").arg("-c").arg(src).output().map(|o| { std::fs::write(&dst, o.stdout).unwrap(); o.status.success() }).unwrap_or(false);
        if !ok { eprintln!("gzip not available; skipping"); return; }
        let pdb = open_structure(&dst).unwrap();
        assert_eq!(pdb.residue_count(), 46);
    }
```

- [ ] **Step 2: Run** `cargo test -p proteus-core io` → module missing.

- [ ] **Step 3: Implement**

```rust
//! Single entry point for reading structures: PDB / mmCIF, optionally gzip-compressed.

use std::path::Path;

use pdbtbx::StrictnessLevel;

use crate::error::CoreError;

fn parse_err(e: impl std::fmt::Debug) -> CoreError {
    CoreError::StructureParseError(format!("{e:?}"))
}

pub fn open_structure(path: &Path) -> Result<pdbtbx::PDB, CoreError> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    let known = name.ends_with(".pdb") || name.ends_with(".ent") || name.ends_with(".cif") || name.ends_with(".mmcif")
        || name.ends_with(".pdb.gz") || name.ends_with(".ent.gz") || name.ends_with(".cif.gz") || name.ends_with(".mmcif.gz");
    if known {
        let s = path.to_str().ok_or_else(|| CoreError::StructureParseError("non-UTF-8 path".into()))?;
        let r = if name.ends_with(".gz") { pdbtbx::open_gz(s, StrictnessLevel::Loose) } else { pdbtbx::open(s, StrictnessLevel::Loose) };
        return r.map(|(p, _)| p).map_err(parse_err);
    }
    let bytes = std::fs::read(path).map_err(|e| CoreError::StructureParseError(e.to_string()))?;
    open_structure_bytes(&bytes, None)
}

pub fn open_structure_bytes(bytes: &[u8], hint: Option<&str>) -> Result<pdbtbx::PDB, CoreError> {
    let text = std::str::from_utf8(bytes).map_err(|e| CoreError::StructureParseError(e.to_string()))?;
    let is_cif = match hint {
        Some(h) if h.ends_with("cif") => true,
        Some(_) => false,
        None => text.lines().find(|l| !l.trim().is_empty()).map(|l| l.starts_with("data_")).unwrap_or(false),
    };
    let reader = std::io::BufReader::new(bytes);
    let r = if is_cif { pdbtbx::open_mmcif_bufread(reader, StrictnessLevel::Loose) } else { pdbtbx::open_pdb_raw(reader, pdbtbx::Context::none(), StrictnessLevel::Loose) };
    r.map(|(p, _)| p).map_err(parse_err)
}
```
Check the exact pdbtbx 0.11 raw-reader names with `rg "pub fn open_" ~/.cargo/registry/src/*/pdbtbx-0.11.0/src/read/` and use those (`open_raw` auto-sniffs in 0.11 — if so, `open_structure_bytes` can simply call `open_raw`). `.ent`/`.mmcif` are not recognised by `pdbtbx::open`'s extension check, so for those call `open_pdb`/`open_mmcif` directly.
Replace `open(path_str, …)` in `metrics.rs::analyze_pdb_file` with `crate::io::open_structure(path)`; in the CLI, replace direct `pdbtbx::open`/`open_raw` on user-supplied paths with `proteus_core::io::open_structure`; in `proteus-render/src/lib.rs` add `pub fn load_structure_from_path(path: &Path) -> Result<…>` that calls `open_structure` and the existing builder, and make `view` use it.

- [ ] **Step 4: Verify with the CLI**

```bash
cargo build && ./target/debug/proteus analyze --pdb crates/proteus-core/tests/data/1crn.cif | head -5
./target/debug/proteus view crates/proteus-core/tests/data/1crn.cif --backend braille --width 40 --height 16 | head -3
```
Expected: same Rg (9.676 Å) from `.cif`; the viewer renders.

- [ ] **Step 5: Commit**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all
git add -A && git commit -m "feat(core,cli,render): mmCIF and gzip input through one open_structure entry point"
```

---

### Task 8: Validation harness (`validate/`) + integration test + CI job

**Files:**
- Create: `validate/README.md`, `validate/corpus.toml`, `validate/fetch.py`, `validate/reference.py`, `validate/tolerances.toml`, `validate/requirements.txt`, `validate/.gitignore` (`corpus/`, `.venv/`), `validate/reference/<id>.json` (generated, committed), `crates/proteus-core/tests/validation.rs`, `Makefile`, `.github/workflows/validate.yml`
- Modify: `crates/proteus-core/Cargo.toml` (`[dev-dependencies] toml = "0.8"`, `serde_json` already), root `Cargo.toml` (`toml = "0.8"` in workspace deps)

**Interfaces:**
- Reference JSON per structure:
```json
{"id":"1crn","format":"pdb","kind":"xray","n_residues":46,
 "rg_ca":9.676,"sasa_freesasa_lr":2999.3,"sasa_mdtraj_sr":2968.9,
 "dssp8":"-EE---HHHH…","dssp3":"CEECCCHHH…",
 "phi":[null,-107.8,…],"psi":[-147.7,…,null],
 "rama":{"favored":43,"allowed":1,"outliers":0,"labels":["F","F",…]}}
```
- `tolerances.toml`:
```toml
rg_ca_abs = 0.01
sasa_vs_freesasa_lr_rel = 0.02
sasa_vs_mdtraj_sr_rel = 0.01
phi_psi_abs_deg = 0.1
dssp3_min_agreement = 0.98
dssp8_min_agreement = 0.95
rama_label_min_agreement = 0.995
```

- [ ] **Step 1: Corpus manifest**

`validate/corpus.toml` (RCSB URLs are `https://files.rcsb.org/download/<ID>.<pdb|cif>`; AF-DB: `https://alphafold.ebi.ac.uk/files/AF-<UNIPROT>-F1-model_v4.cif`). Entries:
```toml
[[structure]]
id = "1crn"
url = "https://files.rcsb.org/download/1CRN.pdb"
format = "pdb"
kind = "xray"
```
Repeat for: X-ray `1UBQ 2LZM 4HHB 1A3N 3PTB 1TIM 2PTC 1BRS 1HHO 1L2Y 1AKI 1BNI 2CI2 1STN 1MBN 1PGB 1GB1 3SSI 1BPI` (pdb and, for `1UBQ 4HHB 1TIM 2PTC 1BRS`, also a second entry with `format = "cif"` and `.cif` URL); NMR `1D3Z 2KOD 1G6J 2L3B 1LB5`; cryo-EM `6VXX 7K00 6M0J 7BV2 6ZGE` (**check size < 2 MB; drop any larger**); AF-DB `P69905 P00698 P0DTD1 Q9Y6K9 P04637 P01308 P00533 P38398 P42212 P0A7Y4` (cif, kind `afdb`); multimers are covered by 4HHB/1BRS/2PTC/1TIM.

- [ ] **Step 2: `fetch.py`**

```python
#!/usr/bin/env python3
"""Download the validation corpus into validate/corpus/ (idempotent, sha256-verified when recorded)."""
import hashlib, pathlib, sys, tomllib, urllib.request
root = pathlib.Path(__file__).parent
cfg = tomllib.loads((root / "corpus.toml").read_text())
dst = root / "corpus"; dst.mkdir(exist_ok=True)
fail = 0
for s in cfg["structure"]:
    p = dst / f"{s['id']}.{s['format']}"
    if not p.exists():
        try:
            urllib.request.urlretrieve(s["url"], p)
        except Exception as e:
            print("FAIL", s["id"], e); fail += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if "sha256" in s and s["sha256"] != h:
        print("SHA MISMATCH", s["id"]); fail += 1
    print(s["id"], p.stat().st_size, h[:12])
sys.exit(1 if fail else 0)
```
After the first successful run, add `sha256 = "…"` to each entry (print them, paste).

- [ ] **Step 3: `reference.py`**

```python
#!/usr/bin/env python3
"""Generate validate/reference/<id>.json from mdtraj, freesasa and cctbx (ramalyze)."""
import json, pathlib, sys, tomllib
import numpy as np, mdtraj as md, freesasa
import iotbx.pdb
from mmtbx.validation.ramalyze import ramalyze

root = pathlib.Path(__file__).parent
cfg = tomllib.loads((root / "corpus.toml").read_text())
out = root / "reference"; out.mkdir(exist_ok=True)

def one(s):
    p = root / "corpus" / f"{s['id']}.{s['format']}"
    t = md.load(str(p))
    t = t.atom_slice(t.topology.select("protein and not element H"))
    ca = t.topology.select("name CA")
    rg_ca = float(md.compute_rg(t.atom_slice(ca))[0] * 10)
    sasa_sr = float(md.shrake_rupley(t, probe_radius=0.14, n_sphere_points=960).sum() * 100)
    fs = freesasa.Structure(str(p)) if s["format"] == "pdb" else None
    sasa_lr = float(freesasa.calc(fs).totalArea()) if fs else None
    ss8 = md.compute_dssp(t, simplified=False)[0]
    ss3 = md.compute_dssp(t, simplified=True)[0]
    phi_idx, phi = md.compute_phi(t); psi_idx, psi = md.compute_psi(t)
    n = t.n_residues
    phis = [None] * n; psis = [None] * n
    for k, quad in enumerate(phi_idx):
        phis[t.topology.atom(int(quad[2])).residue.index] = float(np.degrees(phi[0][k]))
    for k, quad in enumerate(psi_idx):
        psis[t.topology.atom(int(quad[1])).residue.index] = float(np.degrees(psi[0][k]))
    h = iotbx.pdb.input(file_name=str(p)).construct_hierarchy()
    r = ramalyze(pdb_hierarchy=h, outliers_only=False)
    labels = {}
    for res in r.results:
        labels[(res.chain_id.strip(), int(res.resseq), res.icode.strip())] = {"OUTLIER": "O", "Allowed": "A", "Favored": "F"}[res.ramalyze_type()]
    rama_labels = []
    for res in t.topology.residues:
        key = (res.chain.chain_id.strip() if hasattr(res.chain, "chain_id") else "", res.resSeq, "")
        rama_labels.append(labels.get(key))
    return {
        "id": s["id"], "format": s["format"], "kind": s["kind"], "n_residues": n,
        "rg_ca": rg_ca, "sasa_freesasa_lr": sasa_lr, "sasa_mdtraj_sr": sasa_sr,
        "dssp8": "".join(ss8).replace(" ", "-"), "dssp3": "".join(ss3),
        "phi": phis, "psi": psis,
        "rama": {"favored": r.n_favored, "allowed": r.n_allowed, "outliers": r.n_outliers, "labels": rama_labels},
    }

ids = sys.argv[1:] or [s["id"] for s in cfg["structure"]]
for s in cfg["structure"]:
    if s["id"] not in ids: continue
    try:
        d = one(s)
    except Exception as e:
        print("FAIL", s["id"], repr(e)); continue
    (out / f"{s['id']}_{s['format']}.json").write_text(json.dumps(d))
    print(s["id"], s["format"], "rg", round(d["rg_ca"], 3), "rama", d["rama"]["favored"], d["rama"]["allowed"], d["rama"]["outliers"])
```
`validate/requirements.txt`: `mdtraj\nfreesasa\nnumpy\ncctbx-base\n`. Residue-key alignment between mdtraj and cctbx may need adjusting for insertion codes/chain ids — verify on 4HHB (4 chains) and fix the key construction until `rama_labels` has no `None` for standard residues.

- [ ] **Step 4: `Makefile`**

```make
VENV := validate/.venv
PY := $(VENV)/bin/python

$(VENV):
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -r validate/requirements.txt

.PHONY: fetch reference validate
fetch: $(VENV)
	$(PY) validate/fetch.py
reference: fetch
	$(PY) validate/reference.py
validate: fetch
	cargo test -p proteus-core --release --test validation -- --ignored --nocapture
```

- [ ] **Step 5: Integration test `crates/proteus-core/tests/validation.rs`**

```rust
//! Corpus validation against mdtraj / freesasa / cctbx reference values.
//! Run: `make validate` (needs validate/corpus/ fetched). Ignored by default.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    id: String,
    format: String,
    n_residues: usize,
    rg_ca: f64,
    sasa_freesasa_lr: Option<f64>,
    sasa_mdtraj_sr: f64,
    dssp8: String,
    dssp3: String,
    phi: Vec<Option<f64>>,
    psi: Vec<Option<f64>>,
    rama: Rama,
}
#[derive(Deserialize)]
struct Rama { favored: usize, allowed: usize, outliers: usize, labels: Vec<Option<String>> }

#[derive(Deserialize)]
struct Tolerances {
    rg_ca_abs: f64,
    sasa_vs_freesasa_lr_rel: f64,
    sasa_vs_mdtraj_sr_rel: f64,
    phi_psi_abs_deg: f64,
    dssp3_min_agreement: f64,
    dssp8_min_agreement: f64,
    rama_label_min_agreement: f64,
}

fn root() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate") }

#[test]
#[ignore]
fn corpus_matches_reference_implementations() {
    let tol: Tolerances = toml::from_str(&std::fs::read_to_string(root().join("tolerances.toml")).unwrap()).unwrap();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(root().join("reference")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
        let r: Reference = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let structure = root().join("corpus").join(format!("{}.{}", r.id, r.format));
        if !structure.exists() {
            if std::env::var("PROTEUS_VALIDATE_OFFLINE").is_ok() { eprintln!("skip {} (offline)", r.id); continue; }
            panic!("missing {} — run `make fetch`", structure.display());
        }
        let pdb = proteus_core::io::open_structure(&structure).unwrap();
        let d = proteus_core::metrics::analyze_pdb_detailed(&pdb, None).unwrap();
        let bb = proteus_core::backbone::extract_backbone(&pdb);
        let m = &d.metrics;

        let mut bad = Vec::new();
        let rg_err = (m.radius_of_gyration - r.rg_ca).abs();
        if rg_err > tol.rg_ca_abs { bad.push(format!("rg {:.3} vs {:.3}", m.radius_of_gyration, r.rg_ca)); }
        let sasa = m.sasa_metrics.as_ref().unwrap().total_sasa;
        let sr_rel = (sasa - r.sasa_mdtraj_sr).abs() / r.sasa_mdtraj_sr;
        if sr_rel > tol.sasa_vs_mdtraj_sr_rel { bad.push(format!("sasa(sr) {:.1} vs {:.1}", sasa, r.sasa_mdtraj_sr)); }
        if let Some(lr) = r.sasa_freesasa_lr {
            let lr_rel = (sasa - lr).abs() / lr;
            if lr_rel > tol.sasa_vs_freesasa_lr_rel { bad.push(format!("sasa(lr) {:.1} vs {:.1}", sasa, lr)); }
        }
        // phi/psi
        let (mut n_ang, mut ang_bad) = (0usize, 0usize);
        if bb.len() == r.n_residues {
            for i in 0..bb.len() {
                let (p, s, _) = d.ramachandran_points[i];
                for (got, want) in [(p, r.phi[i]), (s, r.psi[i])] {
                    if let (Some(g), Some(w)) = (got, want) {
                        n_ang += 1;
                        let mut diff = (g - w).abs(); if diff > 180.0 { diff = 360.0 - diff; }
                        if diff > tol.phi_psi_abs_deg { ang_bad += 1; }
                    }
                }
            }
            if ang_bad > 0 { bad.push(format!("phi/psi {ang_bad}/{n_ang} beyond {}°", tol.phi_psi_abs_deg)); }
        } else {
            bad.push(format!("residue count {} vs {}", bb.len(), r.n_residues));
        }
        // dssp
        let ss = m.secondary_structure_summary.as_ref().unwrap();
        let agree = |a: &str, b: &str| a.chars().zip(b.chars()).filter(|(x, y)| x == y).count() as f64 / a.len().max(1) as f64;
        let d8 = agree(&ss.dssp, &r.dssp8);
        let three: String = ss.assignment.iter().map(|s| s.as_char()).collect();
        let d3 = agree(&three, &r.dssp3);
        if d8 < tol.dssp8_min_agreement { bad.push(format!("dssp8 {:.3}", d8)); }
        if d3 < tol.dssp3_min_agreement { bad.push(format!("dssp3 {:.3}", d3)); }
        // rama labels
        let (mut n_lab, mut lab_ok) = (0usize, 0usize);
        for (i, want) in r.rama.labels.iter().enumerate() {
            let Some(w) = want else { continue };
            let (p, s, region) = d.ramachandran_points[i];
            if p.is_none() || s.is_none() { continue; }
            n_lab += 1;
            let g = match region { proteus_core::structure::RamachandranRegion::Favored => "F", proteus_core::structure::RamachandranRegion::Allowed => "A", _ => "O" };
            if g == w { lab_ok += 1; }
        }
        let rama_agree = lab_ok as f64 / n_lab.max(1) as f64;
        if rama_agree < tol.rama_label_min_agreement { bad.push(format!("rama labels {:.3} ({}/{})", rama_agree, lab_ok, n_lab)); }
        let rs = m.ramachandran_stats.as_ref().unwrap();
        rows.push(format!("| {} | {} | {:.3} | {:.1}% | {:.1}% | {:.3} | {:.3} | {:.3} | {}/{}/{} vs {}/{}/{} |",
            r.id, r.format, rg_err, sr_rel * 100.0, r.sasa_freesasa_lr.map(|lr| (sasa - lr).abs() / lr * 100.0).unwrap_or(0.0),
            d8, d3, rama_agree, rs.total_evaluated - rs.outlier_count - (rs.allowed_fraction * rs.total_evaluated as f64).round() as usize, (rs.allowed_fraction * rs.total_evaluated as f64).round() as usize, rs.outlier_count,
            r.rama.favored, r.rama.allowed, r.rama.outliers));
        if !bad.is_empty() { failures.push(format!("{}: {}", r.id, bad.join("; "))); }
    }
    println!("| id | fmt | Δrg Å | ΔSASA sr | ΔSASA lr | dssp8 | dssp3 | rama | proteus F/A/O vs cctbx |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for row in &rows { println!("{row}"); }
    std::fs::write(root().join("last_run.md"), rows.join("\n")).ok();
    assert!(failures.is_empty(), "validation failures:\n{}", failures.join("\n"));
}
```
`SecondaryStructure::as_char` must yield `H`/`E`/`C` to match mdtraj's simplified codes — check `structure.rs:14` and adjust.

- [ ] **Step 6: CI workflow `.github/workflows/validate.yml`**

```yaml
name: validate
on:
  push: { branches: [main] }
  pull_request:
jobs:
  reference-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: astral-sh/setup-uv@v5
      - uses: actions/cache@v4
        with: { path: validate/corpus, key: corpus-${{ hashFiles('validate/corpus.toml') }} }
      - run: make fetch
      - run: make validate
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: validation-table, path: validate/last_run.md }
```

- [ ] **Step 7: Run end to end, tune, commit**

```bash
make reference 2>&1 | tail -60     # generates validate/reference/*.json — inspect FAIL lines, drop entries that fail to parse in mdtraj/cctbx from corpus.toml
make validate 2>&1 | tail -80
```
Expected: the markdown table prints for every structure and the test passes. If a metric fails across most of the corpus it is a Proteus bug — fix it in the relevant module (do **not** loosen `tolerances.toml` to make it pass; loosen only with a written justification in `validate/README.md`, e.g. SASA S&R vs L&R algorithmic difference). Typical first-run offenders: residues mdtraj drops (non-standard names, altlocs) → align by `(chain, resseq)` instead of index; AF-DB CIF residue numbering.
```bash
git add validate Makefile .github crates/proteus-core/tests/validation.rs crates/proteus-core/Cargo.toml Cargo.toml Cargo.lock
git commit -m "test(validate): reference-implementation harness (mdtraj, freesasa, cctbx) over a 50-structure corpus"
```

- [ ] **Step 8: `validate/README.md`**

Document: what each metric is compared against and why that tool; the S&R-vs-L&R SASA note; how to add a structure; how to regenerate references; that tolerances are the contract. Commit with the message `docs(validate): describe the reference harness`.

---

### Task 9: README truth pass and vault bookkeeping

**Files:**
- Modify: `README.md` (analyze table, "48 tests", "Pareto", clash label, add a "Validated against" table generated from `validate/last_run.md`), `~/forelsket/00_estus/2026-09-21-proteus-scientific-logical-e2e-audit.md`, `~/forelsket/00_estus/2026-09-21-proteus-performance-benchmarks.md`

- [ ] **Step 1: Regenerate the README `analyze` block**

```bash
cargo build --release && ./target/release/proteus analyze --pdb crates/proteus-core/tests/data/1crn.pdb
```
Paste the actual table into README (replace the old one). Replace "Run all 48 workspace unit and integration tests" with the number from `cargo test --workspace 2>&1 | rg "test result" | awk '{s+=$4} END {print s}'`. Replace "Multi-Objective Pareto Ranking" with "Weighted composite fitness score" and describe the renormalisation when pLDDT is absent. Under "Native Biophysical Validation Engines" replace the P-SEA/box-Ramachandran text with DSSP (Kabsch–Sander, `proteus-dssp`) and Top8000 contours (cctbx/MolProbity), and add:

```markdown
### Validated against reference implementations
Every release runs `make validate` over a 50-structure corpus (X-ray, NMR, cryo-EM, AlphaFold-DB) and compares against mdtraj (φ/ψ, DSSP, Rg, Shrake–Rupley SASA), FreeSASA (Lee–Richards SASA) and cctbx/MolProbity `ramalyze` (Top8000 Ramachandran). Tolerances: `validate/tolerances.toml`. Latest table: `validate/last_run.md`.
```
Paste the table from `validate/last_run.md` (first 10 rows + "… see full table").

- [ ] **Step 2: Supersede the two vault notes** — with the Write tool (never `cat >`), set `status: superseded` in their frontmatter and insert after the H1:
```
> [!warning] Superseded by [[2026-09-21-proteus-portfolio-flagship-research]] — the Ramachandran, secondary-structure and pLDDT claims below did not reproduce at commit 914e854 (dihedral sign inverted; see §1 of the superseding note). Benchmark comparisons against Python were never measured.
```

- [ ] **Step 3: Commit and push**

```bash
git add README.md && git commit -m "docs(readme): regenerate analyze output; describe DSSP/Top8000; add validation section"
git push origin main
```
(Push authorised by the user on 2026-09-21.)

---

## Self-review

- **Spec coverage:** 3.1→T1, 3.2→T3, 3.3→T4, 3.4→T5, 3.5→T6, 3.6→T7, 3.7→T8, 3.8→T9, §4 error handling → T2 (`None` across breaks), T3 (`n<3`), T8 (`PROTEUS_VALIDATE_OFFLINE`), §5 testing → each task's step 1.
- **Type consistency:** `BackboneResidue` (T2) consumed by T3 step 8, T4 step 6, T8; `RamaClass`/`evaluate`/`score` (T4) consumed by T8; `Ss::as_char` `-` for loop matches `reference.py` (`replace(" ", "-")`); `SecondaryStructureSummary.dssp` added in T3 and read in T7/T8; `StericOverlapStats` (T6) not referenced by later tasks; `metrics.plddt()` (T5) used in T5 CLI only.
- **Placeholders:** none except the cctbx oracle numbers in T4 step 2, which are deliberately to be pasted from the printed reference, and `<N>` in commit bodies to be filled from test output.
