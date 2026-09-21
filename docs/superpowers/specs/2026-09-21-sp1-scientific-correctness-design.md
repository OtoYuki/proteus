# SP1 — Scientific correctness and validation harness

**Status:** approved 2026-09-21 · **Scope:** `proteus-core`, new `proteus-dssp`, `proteus-cli`, `validate/`, CI job
**Supersedes:** the Ramachandran, secondary-structure, pLDDT and clashscore claims in `README.md` and in the 2026-09-21 vault audit.

## 1. Problem

At commit `914e854`, measured against `mdtraj 1.10` / `freesasa 2.2` on `1crn.pdb`:

| metric | proteus | reference | root cause |
|---|---|---|---|
| φ/ψ | every angle negated (ILE7 +63.6/+42.1 vs −63.6/−42.1) | mdtraj | `compute_dihedral` returns `atan2(m1·n2, n1·n2)` without the sign flip that the `m1 = n1 × b̂2` formulation requires |
| Ramachandran | 21 outliers / 47.7 % favored | ~0 outliers / ~98 % favored (MolProbity) | sign bug + hand-drawn rectangular regions instead of Top8000 distributions |
| secondary structure | H 69.6 / E 28.3 / C 2.2 % | DSSP H 47.8 / E 8.7 / C 43.5 % | P-SEA approximation: `has_i3 \|\| has_i4`, no dihedral test, no run length, no strand pairing |
| pLDDT | mean 5.81 | n/a — file is X-ray, column is B-factor | no provenance detection |
| "MolProbity clashscore" | 0.0 | MolProbity needs hydrogens (Reduce) | heavy-atom only, mislabelled |
| SASA | 2976.6 Å² | mdtraj 2968.9 / freesasa 2999.3 | OK (< 1 %) |
| Rg (Cα) | 9.676 | 9.676 | OK |

Nothing in `cargo test` catches any of the first five: the tests assert hand-picked symmetric values or the code's own output.

## 2. Goals

1. Every exported biophysical number is either equal to a reference implementation within a committed tolerance, or explicitly labelled as an approximation with the deviation stated.
2. The comparison runs in CI on a fixed corpus and fails on drift.
3. Secondary structure and Ramachandran use the field-standard algorithms (Kabsch–Sander DSSP; MolProbity Top8000 contours), not approximations.
4. Experimental structures never report a pLDDT.
5. mmCIF works everywhere PDB works.

Non-goals (later sub-projects): hydrogen placement for a true clashscore, CI/release infrastructure beyond the one validation job, README rewrite, benchmarks.

## 3. Design

### 3.1 `compute_dihedral` fix — `proteus-core/src/structure.rs`
Replace with the IUPAC-signed form:
```
b1 = p2−p1, b2 = p3−p2, b3 = p4−p3
n1 = b1×b2, n2 = b2×b3
y = |b2| · (b1 · n2),  x = n1 · n2
θ = atan2(y, x)
```
Test: table of all 44 (φ,ψ) pairs for 1CRN from mdtraj, asserted to ±0.05°; plus the four canonical checks (ideal α-helix −57/−47, β −120/+130, PPII −75/+145, αL +57/+47) built from synthetic backbone geometry and a chirality test (mirror the coordinates → angles negate).

### 3.2 `proteus-dssp` crate (new, `crates/proteus-dssp`)
Standalone, `no_std`-free but dependency-light (`nalgebra` only), publishable.

API:
```rust
pub struct BackboneResidue { pub n: [f64;3], pub ca: [f64;3], pub c: [f64;3], pub o: [f64;3], pub is_proline: bool, pub chain_break_before: bool }
pub enum Dssp { H, B, E, G, I, P, T, S, Loop }          // 8-state + PPII (P) as in DSSP 4
pub fn assign(residues: &[BackboneResidue]) -> Vec<Dssp>;
pub fn simplify(d: Dssp) -> Simple                       // Simple::{Helix, Strand, Coil} (H,G,I→Helix; E,B→Strand)
```
Algorithm (Kabsch & Sander 1983; DSSP 4 conventions):
- Backbone H: `H = N + (C_{i-1} − O_{i-1}) / |C_{i-1} − O_{i-1}| · 1.0 Å`; none for proline or chain start.
- H-bond energy `E = 0.084 · 332 · (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)` kcal/mol; bond if `E < −0.5`. Only pairs with Cα–Cα < 9 Å evaluated (cell list / sorted sweep).
- n-turns (n=3,4,5), minimal helices (two consecutive n-turns), bridges (parallel/antiparallel), ladders, sheets, bends (κ > 70°), PPII (φ,ψ within DSSP 4 window, run ≥ 2), priority H > B > E > G > I > P > T > S.
- Chain breaks: Cα(i)–Cα(i+1) > 4.5 Å or chain-id change → no bonds across.

Validation target: 8-state agreement ≥ 95 % and 3-state ≥ 98 % against `mdtraj.compute_dssp` across the corpus (mdtraj omits P; compare P as Loop for 8-state). Exact match is not expected — DSSP 4.x and mdtraj differ on edge residues themselves.

`proteus-core::structure::assign_secondary_structure` becomes a thin adapter over `proteus-dssp`; P-SEA code deleted.

### 3.3 Ramachandran — Top8000 contours
- Data: `crates/proteus-core/data/rama8000/{general,glycine,cispro,transpro,prepro,ileval}.f32` — six 180×180 little-endian `f32` grids (777 KB total) converted from cctbx `mmtbx/validation/ramachandran/rama8000_tables.h` by `scripts/convert_rama8000.py`; `data/rama8000/NOTICE` carries the cctbx BSD-3 licence and provenance (Top8000, Richardson lab). `include_bytes!` + `bytemuck`-free manual `f32::from_le_bytes` decode at first use (`OnceLock`).
- Lookup replicates `rama_eval.h`: bin centres at odd degrees (−179…179), index `phi_bin*180 + psi_bin`, bilinear interpolation between the four surrounding bins with wrap-around.
- Classes: `General, Glycine, CisPro, TransPro, PrePro, IleVal` — chosen from residue name, next residue (pre-Pro), and ω (cis if |ω| < 30°).
- Thresholds (cctbx): favored ≥ 0.02; allowed ≥ 0.0005 (general), ≥ 0.002 (cis-Pro), ≥ 0.001 (others); else outlier.
- `RamachandranRegion` collapses to `{Favored, Allowed, Outlier}` plus the raw score; `ResidueContext` renamed `RamaClass`. All rectangular-box code deleted. Downstream (`metrics.rs`, `ranking.rs`, `render/tui/dashboard.rs`, `storage/export.rs`, `server/dto.rs`) updated.
- Validation: MolProbity's own `molprobity.ramalyze` is not pip-installable; reference = cctbx (`pip install cctbx-base`) `mmtbx.validation.ramalyze` run in the harness. Target: identical favored/allowed/outlier labels for ≥ 99.5 % of residues (residual differences from float interpolation at boundaries).

### 3.4 pLDDT provenance
- `ConfidenceSource { Predicted, ExperimentalBFactor, Unknown }` decided by: `EXPDTA` record containing `X-RAY|NMR|ELECTRON|NEUTRON|FIBER` → Experimental; `REMARK` / title containing `ALPHAFOLD|ESMFOLD|BOLTZ|OPENFOLD|PLDDT` or mmCIF `_ma_qa_metric` → Predicted; else `Unknown` heuristics: all values in [0,100] with mean > 30 and stddev < 40 → Predicted; otherwise Experimental. CLI/API override `--confidence-source`.
- `BiophysicalMetrics.plddt: Option<PlddtStats>`; None for Experimental. `analyze` prints `pLDDT: n/a (experimental structure, B-factor column)`.
- `ranking.rs`: when pLDDT is None, the 0.30 weight is redistributed proportionally over the remaining terms; documented formula.

### 3.5 Clash relabel
`ClashStats` → `StericOverlapStats`, field `clashscore` → `heavy_atom_overlap_score`; CLI row "Heavy-atom steric overlap (>0.4 Å, no H)"; doc comment and README state it is *not* MolProbity clashscore. Parquet column renamed with schema version bump (`schema_version = 2`). Algorithm untouched.

### 3.6 mmCIF
`analyze`, `view`, `screen`, `submit` accept `.cif`/`.mmcif`/`.pdb`/`.pdb.gz`/`.cif.gz`; a single `proteus_core::io::open_structure(path)` wraps `pdbtbx::open`/`open_gz` and is the only entry point. Corpus includes ≥ 10 mmCIF files (AF-DB and Boltz outputs are mmCIF).

### 3.7 Validation harness — `validate/`
```
validate/
  corpus.toml            # id, source URL, format, kind (xray|nmr|cryoem|afdb|esmfold|multimer), sha256
  fetch.py               # downloads corpus into validate/corpus/ (git-ignored), verifies sha256
  reference.py           # uv script: mdtraj (dssp, phi/psi, sasa, rg), freesasa, cctbx ramalyze → validate/reference/<id>.json (committed)
  tolerances.toml        # per-metric tolerances
  README.md              # how to regenerate, what each metric is compared to
crates/proteus-core/tests/validation.rs   # reads reference/*.json + corpus, runs proteus, asserts within tolerance, prints a summary table
scripts/render_validation_table.py       # emits the markdown table for README from the test's JSON summary
```
Corpus (~50): 20 X-ray (1CRN, 1UBQ, 2LZM, 4HHB, 1A3N, 3PTB, 1TIM, 2PTC, 1BRS, 1HHO, …), 5 NMR (1D3Z, 2KOD, …), 5 cryo-EM, 10 AF-DB (`AF-P69905-F1`, …, mmCIF), 5 ESMFold outputs (from `proteus-engine` simulated/API tier if available, else AF-DB), 5 multimers (4HHB, 1BRS, 2PTC as complexes). Small files only (< 1 MB each).
Tolerances: SASA total ±2 % vs freesasa L&R (algorithmic difference S&R vs L&R), ±1 % vs mdtraj S&R; Rg ±0.01 Å; φ/ψ ±0.1°; DSSP 3-state ≥ 98 %, 8-state ≥ 95 %; Ramachandran labels ≥ 99.5 %; overlap score exact vs a Python re-implementation of our own rule (regression only).
CI: `validate` job runs `fetch.py` (cached) + `cargo test -p proteus-core --test validation -- --ignored` on Linux. Local: `make validate`.

### 3.8 Vault bookkeeping
`2026-09-21-proteus-scientific-logical-e2e-audit.md` and `2026-09-21-proteus-performance-benchmarks.md` get `status: superseded` + `> Superseded by [[2026-09-21-proteus-portfolio-flagship-research]]` (their claims were not reproduced). Not moved.

## 4. Error handling
- Missing backbone atoms → residue skipped for φ/ψ/DSSP, counted in `skipped_residues`; never a panic.
- Alt-locs: first altloc only (pdbtbx default), documented.
- Unknown residue names → `General` Rama class, `Loop`-eligible in DSSP.
- Corpus download failure → validation test skips with a clear message when `PROTEUS_VALIDATE_OFFLINE=1`, fails otherwise.

## 5. Testing
- Unit: dihedral (mdtraj table + chirality), Rama lookup (bin arithmetic at −180/180 wrap, four cctbx spot values copied from `rama_eval` output), DSSP on synthetic ideal helix / two-strand sheet, pLDDT source heuristics, mmCIF round-trip.
- Integration: `validation.rs` (ignored by default, run by `make validate` and CI).
- Existing 56 tests keep passing; tests that pinned wrong numbers are corrected, not deleted.

## 6. Out of scope, explicitly
Hydrogen placement / true clashscore; CI matrix, releases, benches, README rewrite (SP2); TES/Sprocket (SP3); ESM-2/ProteinGym/PyO3 (SP4).
