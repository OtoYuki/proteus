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
| hydrogen-bond network | `mdtraj.baker_hubbard` on all six structures with explicit H | **recall** of mdtraj's non-local bonds; **precision** of Proteus' bonds | recall ≥ 85 % (observed 85.7–100 %); precision ≥ 50 % (observed 58–76 %) |

### Hydrogen bonds

Proteus detects H-bonds from **heavy atoms only** — donor/acceptor distance plus antecedent
angles — because predicted models never ship hydrogens. mdtraj's `baker_hubbard` uses the
explicit H (D–H···A distance and angle). The criteria are related but not the same, so the test
measures **recall** (how many of mdtraj's bonds Proteus also finds) and reports **precision**
(how many of Proteus' bonds mdtraj confirms), not set equality. Without hydrogens Proteus
reports 1.3–1.7× as many bonds as Baker–Hubbard — precision 58–76 % on the corpus — so treat
its H-bond counts as an upper bound with the ranking-relevant bonds inside it, not as a
Baker–Hubbard equivalent. A precision floor of 50 % is enforced so this cannot silently drift.

All six NMR entries in the corpus carry hydrogens (1D3Z, 2KOD, 1G6J, 2L3B, 1GB1, 1L2Y) and all six are checked; 1L2Y, a 20-residue mini-protein with 14 reference bonds, is the 85.7 % floor; mdtraj's
i→i±1 bonds are excluded because Proteus requires |Δseq| ≥ 2 by design. Comparison is by
donor/acceptor **residue pair**, so multiple atom-level bonds between the same two residues
collapse to one.

### The composite fitness score

There is no reference implementation to compare a Proteus-defined weighted sum against, so
`fitness_discrimination.rs` measures the claim the score actually makes — *a structure that
looks like a folded protein scores above one that does not* — on eight deposited X-ray
structures against decoys built from each: coordinate noise at σ = 0.5/1.0/3.0 Å, a 1.5×
expansion, and an ideal poly-alanine helix of the same length (the shape Proteus's own offline
simulator emits). All 40 native/decoy pairs separate correctly, the score is monotone in the
noise level, and the smallest margin is 10.4 points (1MBN, σ = 0.5 Å).

That is the whole of the claim. The score is **not** validated as a predictor of experimental
stability or activity, and nothing here says a higher score means a better protein — for
sequence-level fitness use `--scorer esm2`, whose ProteinGym Spearman numbers are in
`bench/README.md`. Treat it as a triage filter that rejects models which are not folded.

Still not validated (no reference run): the heavy-atom overlap score (a Python re-implementation
of Proteus's own rule would only be a regression test) and **salt bridges, π–π stacking and
cation–π** (no widely used reference implementation with the same definitions).

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
