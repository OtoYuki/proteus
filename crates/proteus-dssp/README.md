# proteus-dssp

Pure-Rust Kabsch–Sander DSSP secondary-structure assignment.

- Eight states (`H B E G I T S -`) plus a three-state reduction.
- Semantics follow DSSP 2.x as ported by [mdtraj](https://github.com/mdtraj/mdtraj)
  (α-helix overrides sheets; a 3₁₀ helix only fills residues nothing else claimed; a π-helix
  may overwrite an α-helix — "prefer-π", as mdtraj does; no PPII).
- Input is backbone N/CA/C/O per residue; amide hydrogens are placed from the preceding
  peptide plane as in the original algorithm.
- No `unsafe`, no dependencies by default; `serde` derives behind an optional feature.

Validated residue-by-residue against `mdtraj.compute_dssp` in the
[Proteus](https://github.com/OtoYuki/proteus) repository's `validate/` harness, which runs in
CI, on 53 files (48 PDB entries, five of them in both PDB and mmCIF; X-ray, NMR, cryo-EM,
AlphaFold DB). Pooled over all 30 335 residues the eight-state assignment agrees on 99.6 % and
the three-state reduction on 99.96 %. Per structure the CI floor is 95 % on eight states and
98 % on three; the lowest structures that are not exempted are 97.8 % (1TEN, 6VXX). One is
exempted, with its reason recorded in `validate/tolerances.toml`: 1ZNF, a 25-residue zinc
finger at 84 % on eight states, where mdtraj counts the acetyl cap as a residue (26 against 25)
and the turns around the metal site are genuinely ambiguous.

## How this compares

[`molex`](https://github.com/foldit-org/molex) also implements Kabsch–Sander in Rust, as part of
a broader structure library with Python and C bindings — if you want parsing, density maps or
bindings, look there first. This crate differs in three ways, and they are the only reasons to
prefer it:

- **Eight states** (`H B E G I T S -`) plus a three-state reduction, where molex reports three.
  If you need to tell a 3₁₀ helix from an α-helix, or a bend from a turn, you need eight.
- **No dependencies by default, and nothing else in the crate.** It takes backbone N/CA/C/O per residue
  and returns an assignment; it does not parse files or model chemistry.
- **Validated against an external implementation on real structures**, not against hand-written
  expectations — the agreement number above is reproducible with one command in the parent repo.

```toml
[dependencies]
proteus-dssp = "0.6"
```

```rust
use proteus_dssp::{assign, Residue};
let residues: Vec<Residue> = /* backbone atoms */ vec![];
let ss = assign(&residues);
```
