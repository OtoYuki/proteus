# proteus-dssp

Pure-Rust Kabsch–Sander DSSP secondary-structure assignment.

- Eight states (`H B E G I T S -`) plus a three-state reduction.
- Semantics follow DSSP 2.x as ported by [mdtraj](https://github.com/mdtraj/mdtraj)
  (α-helix overrides sheets, 3₁₀/π helices only fill unassigned residues, π preferred
  over α where both are possible, no PPII).
- Input is backbone N/CA/C/O per residue; amide hydrogens are placed from the preceding
  peptide plane as in the original algorithm.
- No `unsafe`, single dependency-free module; `serde` behind a feature flag.

Validated residue-by-residue against `mdtraj.compute_dssp` on a 43-structure corpus (X-ray,
NMR, cryo-EM, AlphaFold-DB) in the [Proteus](https://github.com/OtoYuki/proteus) repository's
`validate/` harness, which runs in CI: ≥ 98 % per-residue agreement on eight states.

```toml
[dependencies]
proteus-dssp = { git = "https://github.com/OtoYuki/proteus" }
```

```rust
use proteus_dssp::{assign, Residue};
let residues: Vec<Residue> = /* backbone atoms */ vec![];
let ss = assign(&residues);
```
