# proteus-dssp

Pure-Rust Kabsch–Sander DSSP secondary-structure assignment.

- Eight states (`H B E G I T S -`) plus a three-state reduction.
- Semantics follow DSSP 2.x as ported by [mdtraj](https://github.com/mdtraj/mdtraj)
  (α-helix overrides sheets, 3₁₀/π helices only fill unassigned residues, π preferred
  over α where both are possible, no PPII).
- Input is backbone N/CA/C/O per residue; amide hydrogens are placed from the preceding
  peptide plane as in the original algorithm.
- No `unsafe`, single dependency-free module; `serde` behind a feature flag.

Validated residue-by-residue against `mdtraj.compute_dssp` on a 53-structure corpus (X-ray,
NMR, cryo-EM, AlphaFold-DB) in the [Proteus](https://github.com/OtoYuki/proteus) repository's
`validate/` harness, which runs in CI: ≥ 98 % per-residue agreement on eight states.

## How this compares

[`molex`](https://github.com/foldit-org/molex) also implements Kabsch–Sander in Rust, as part of
a broader structure library with Python and C bindings — if you want parsing, density maps or
bindings, look there first. This crate differs in three ways, and they are the only reasons to
prefer it:

- **Eight states** (`H B E G I T S -`) plus a three-state reduction, where molex reports three.
  If you need to tell a 3₁₀ helix from an α-helix, or a bend from a turn, you need eight.
- **Zero dependencies and nothing else in the crate.** It takes backbone N/CA/C/O per residue
  and returns an assignment; it does not parse files or model chemistry.
- **Validated against an external implementation on real structures**, not against hand-written
  expectations — the agreement number above is reproducible with one command in the parent repo.

```toml
[dependencies]
proteus-dssp = { git = "https://github.com/OtoYuki/proteus" }
```

```rust
use proteus_dssp::{assign, Residue};
let residues: Vec<Residue> = /* backbone atoms */ vec![];
let ss = assign(&residues);
```
