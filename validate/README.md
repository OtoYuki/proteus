# Reference validation

Every biophysical number Proteus emits is compared, structure by structure, against an
independent reference implementation. `make validate` runs it locally; the `validate`
GitHub Actions job runs it on every push and pull request and uploads the table.

| Proteus metric | reference | how compared | tolerance (`tolerances.toml`) |
|---|---|---|---|
| Cα radius of gyration | `mdtraj.compute_rg` | absolute Å | 0.01 |
| backbone φ/ψ | `mdtraj.compute_phi/psi` | every angle, degrees | 0.1 |
| DSSP 8-state / 3-state | `mdtraj.compute_dssp` (DSSP 2.x port) | per-residue agreement | ≥ 95 % / ≥ 98 % |
| Ramachandran Favored/Allowed/Outlier | cctbx `mmtbx.validation.ramalyze` (MolProbity Top8000) | per-residue label agreement | ≥ 99.5 % |
| SASA (Shrake–Rupley, 960 pts, Bondi radii) | `mdtraj.shrake_rupley` (same algorithm and radii) | relative | 1 % |
| SASA | FreeSASA Lee–Richards, ProtOr radii | relative | 4 % — different algorithm **and** radius set; an independent sanity check, not a tight bound (observed 0.2–3.4 %) |
| hydrogen-bond network | `mdtraj.baker_hubbard` on structures with explicit H | **recall** of mdtraj's non-local bonds | ≥ 92 % (observed 94.9–100 %) |

### Hydrogen bonds

Proteus detects H-bonds from **heavy atoms only** — donor/acceptor distance plus antecedent
angles — because predicted models never ship hydrogens. mdtraj's `baker_hubbard` uses the
explicit H (D–H···A distance and angle). The criteria are related but not the same, so the test
measures **recall** (how many of mdtraj's bonds Proteus also finds), not set equality. Proteus
legitimately reports more bonds; that is the price of working without hydrogens, and the extra
ones are not counted against it.

Only the four NMR entries in the corpus carry hydrogens (1D3Z, 2KOD, 1G6J, 2L3B); mdtraj's
i→i±1 bonds are excluded because Proteus requires |Δseq| ≥ 2 by design. Comparison is by
donor/acceptor **residue pair**, so multiple atom-level bonds between the same two residues
collapse to one.

Still not validated (no reference run): the heavy-atom overlap score (a Python re-implementation
of Proteus's own rule would only be a regression test), **salt bridges, π–π stacking and
cation–π** (no widely used reference implementation with the same definitions), and the
composite fitness score (a Proteus-defined heuristic).

## Corpus

`corpus.toml` lists 43 entries (~50 MB): 20 X-ray PDB files, 5 of them also as mmCIF, 5 NMR
ensembles (first model), 4 cryo-EM mmCIF, 9 AlphaFold-DB v6 mmCIF models. Files are
downloaded into `corpus/` (git-ignored) and verified by sha256. Add a structure by appending a
`[[structure]]` block, running `make fetch`, pasting the printed sha256, and `make reference`.

## Regenerating references

```
make reference     # validate/.venv (uv, Python 3.12) → validate/reference/<id>_<fmt>.json
```

`reference.py` uses mdtraj + freesasa; `ramalyze_ref.py` runs cctbx in a **separate
process** — importing cctbx into the same interpreter corrupts mdtraj's native SASA kernel
(47 374 vs 2 969 Å² on 1CRN, observed 2026-09-21) and can segfault.

Residues are aligned by `chain:resseq:icode`. mdtraj labels mmCIF chains by `label_asym_id`
while pdbtbx uses the author ids, so the test falls back to chain-ordinal keys when id keys do
not match (2PTC, 6M0J).

## What Proteus does before measuring

`proteus_core::io::protein_heavy_atoms`: first model only, protein residues only (an atom
named `CA` with element carbon), heavy atoms only, first alternate conformation only. This is
what mdtraj's `protein and not element H` selection and DSSP/MolProbity operate on. Two of
these rules were added because the harness caught the discrepancy (altloc duplicates inflated
1BPI's SASA by 1.5 %; waters and ions were being counted as atoms).
