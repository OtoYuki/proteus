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
| hydrogen-bond network | `mdtraj.baker_hubbard` on all six structures with explicit H | **recall** of mdtraj's non-local bonds; **precision** of Proteus' bonds | recall ≥ 85 % (observed 85.7–100 %); precision ≥ 50 % (observed 58–79 %) |
| salt bridges | PLIP, intra-chain, 15 structures | recall and precision by residue pair | recall ≥ 65 % (observed 72.0 %); precision ≥ 90 % (observed **97.7 %**) |
| π–π stacking | PLIP, intra-chain | recall and precision by residue pair | both ≥ 70 % (observed 81.8 % / 81.8 %) |
| cation–π | PLIP, intra-chain | recall and precision by residue pair | recall ≥ 55 % (observed 65.4 %); precision ≥ 60 % (observed 73.9 %) |
| covalent geometry: bond, angle, chirality and planarity restraints | cctbx `pdb_interpretation` with Phenix's default library (geostd, CDL v1.2, EH99 cis-Pro) + `mmtbx.validation.restraints` | restraint count; every > 4σ outlier; RMSZ | counts equal; outliers identical outside ±0.01σ of the cutoff (0 needed the allowance); RMSZ 1e-4 relative (observed 5e-6) |
| Cβ deviation | cctbx `mmtbx.validation.cbetadev` | every residue | 0.001 Å; outlier flags identical |
| peptide ω (cis / twisted) | cctbx `mmtbx.validation.omegalyze` | every peptide | 0.01°; types identical |
| side-chain rotamers | cctbx `mmtbx.validation.rotalyze` (MolProbity Top8000) | every residue: χ, percentile, evaluation, rotamer name | χ 0.01°, percentile 0.01 (observed 5e-4° and 5e-5); 26 469 / 26 469 residues identical |

### Hydrogen bonds

Proteus detects H-bonds from **heavy atoms only** — donor/acceptor distance plus antecedent
angles — because predicted models never ship hydrogens. mdtraj's `baker_hubbard` uses the
explicit H (D–H···A distance and angle). The criteria are related but not the same, so the test
measures **recall** (how many of mdtraj's bonds Proteus also finds) and reports **precision**
(how many of Proteus' bonds mdtraj confirms), not set equality. Without hydrogens Proteus
reports 1.3–1.7× as many bonds as Baker–Hubbard — precision 58–79 % on the corpus — so treat
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

### Salt bridges, π–π and cation–π

These three had no reference until 2026-09-22. [PLIP](https://github.com/pharmai/plip)
(Salentin et al. 2015) is one, run in **intra-chain** mode by `validate/plip_reference.py` over
15 X-ray structures. The criteria differ deliberately, so the harness measures recall and
precision by residue pair rather than asserting equality:

| interaction | PLIP | Proteus |
|---|---|---|
| salt bridge | ≤ 5.5 Å between charge centres | ≤ 4.0 Å between closest atoms |
| π–π | ≤ 5.5 Å centroids, ring offset ≤ 2.0 Å | ≤ 6.5 Å centroids, ring offset ≤ 2.0 Å |
| cation–π | ≤ 6.0 Å, offset ≤ 2.0 Å | ≤ 6.0 Å, ≤ 45° to the ring normal, offset ≤ 2.0 Å |

Salt-bridge recall is 72 % *by design*: a 4.0 Å atom-to-atom rule reports a subset of a 5.5 Å
centre-to-centre one. The 97.7 % precision is the evidence that it is the right subset.

**Adopting this reference found a real defect.** Proteus had no lateral-offset test on π–π or
cation–π, so two rings that were parallel and within range but slid sideways past each other
counted as stacked. Measured against PLIP that was 20 % precision on π–π (55 reported against
PLIP's 11). Adding the McGaughey (1998) 2.0 Å offset term — the same one PLIP uses — took π–π to
81.8 % precision at 11 reported, and cation–π from 33.3 % to 73.9 %.

Scope: intra-chain only, because PLIP's INTRA mode profiles one chain against itself.
Inter-chain contacts are covered by Proteus's own chain-awareness unit test.

Still not validated (no reference run): the heavy-atom overlap score — a Python
re-implementation of Proteus's own rule would only be a regression test.

### Covalent geometry and rotamers

`geometry_reference.py` runs `geometry_ref.py` (cctbx, in its own process) on all 53 corpus
entries and on the 13 committed ESMFold models in `predicted/`, and writes
`reference/geometry/<name>.json.gz`: restraint counts, RMSZ and every > 4σ outlier, plus
cbetadev, omegalyze and rotalyze for every residue. `tests/geometry_validation.rs` and
`tests/rotamer_validation.rs` reproduce all of it with `proteus_core::geometry` and
`proteus_core::rotamer`; tolerances are in `[geometry]` and `[rotamer]` of
`tolerances.toml`. `geometry_reference.py --full` also dumps every individual restraint
(target, σ, model value); pointing `PROTEUS_GEOMETRY_FULL` at such dumps makes the test compare
them one by one, which is how the port was brought to parity.

What had to be matched, beyond the monomer dictionaries:

* **The restraint library.** Phenix's default: geostd monomers (Engh & Huber 1991 values),
  the TRANS/PTRANS peptide links, the C-terminal COO modification when OXT is present,
  disulfides for SG–SG < 3 Å, and the Conformation-Dependent Library replacing the backbone
  targets of every residue linked on both sides (Engh & Huber 1999 values for a cis-proline;
  a cis non-proline keeps the library values). The data are embedded in the crate; see
  `crates/proteus-core/data/cdl/NOTICE`.
* **Symmetric-atom renaming.** cctbx swaps Arg NH1/NH2, Asp OD1/OD2, Glu OE1/OE2, Phe/Tyr
  ring atoms and Val/Leu methyls named against the IUPAC convention before restraining.
  Adding that step removed 1,325 of the first 1,343 disagreements: they were naming, not
  geometry. phenix.molprobity also runs rotalyze, cbetadev and omegalyze on the renamed
  model, which changes 13 rotamer evaluations on this corpus, so the reference does the same
  (found by review: the first reference ran rotalyze on the unrenamed atoms and agreed with
  the equally wrong port).
* **Linkage.** Two residues are linked when C(i)–N(i+1) ≤ 3 Å (restraints, CDL) or < 2 Å
  (omegalyze); the previous residue needs only its C.
* **The same atoms.** The reference reduces each structure exactly as
  `protein_heavy_atoms` does: first model, residues with a CA, no hydrogens, the *first*
  alternate conformation (cctbx's own default keeps the one with the highest occupancy), and
  one residue at a microheterogeneous position (1EJG 22 PRO/SER).
* **Which residues.** The 20 amino acids and selenomethionine (cctbx's `common_amino_acid`
  class) are restrained; Cβ deviation also covers modified and D-amino acids, as cbetadev does.

One deliberate difference: cctbx calls any chirality outlier beyond 20σ a "handedness swap".
Proteus calls a centre inverted only when its chiral volume is on the other side of zero by at
least half the ideal magnitude, so a badly distorted centre of the right hand (4HHB D:47 Asp CA,
+8.5 Å³ against +2.5 Å³) or a flattened one (4HHB B:12 Thr CB, −0.16 Å³) is a tetrahedral
outlier, not an inversion. The test compares the Z-scores, not these labels.

What the checks find on the corpus (from the committed references; X-ray entries also
present as mmCIF counted once):

| group | structures | with a bond > 4σ | bond outliers | median bond RMSZ | angle outliers | rotamer outliers | Cβ ≥ 0.25 Å |
|---|---|---|---|---|---|---|---|
| X-ray | 27 | 23 | 3023 / 111 551 (2.71 %) | 1.18 | 4.25 % | 7.1 % | 567 / 13 387 |
| NMR | 8 | 2 | 4 / 7 383 (0.05 %) | 0.68 | 0.92 % | 5.9 % | 0 / 844 |
| cryo-EM | 4 | 1 | 9 / 64 601 (0.01 %) | 0.37 | 0.20 % | 0.6 % | 0 / 7 532 |
| AlphaFold DB v6 | 9 | 3 | 94 / 37 300 (0.25 %) | 0.75 | 3.97 % | 5.9 % | 230 / 4 403 |
| ESMFold (unrelaxed) | 13 | **13** | 374 / 12 615 (2.96 %) | 1.25 | 1.12 % | 0.8 % | 0 / 1 454 |

Every ESMFold model has bond outliers, almost all of them peptide C–N bonds that are too short
(mean 1.312 Å against 1.336 Å in AlphaFold DB models), and mostly in residues ESMFold is
confident about; its 1CRN model puts two disulfide sulfurs 0.82 Å apart. ESMFold builds each
residue from ideal geometry, so it has no Cβ outliers and few rotamer outliers, but it learns the
peptide bond and does not relax the model. 4HHB (1984) accounts for most X-ray outliers,
including the only inverted Thr Cβ centres; the one inverted centre in a predicted model is an
Ile Cβ of AF-P00533 at pLDDT 46.

## Corpus

`corpus.toml` lists 53 files of 48 structures (~65 MB): 27 X-ray PDB files, 5 of them also as
mmCIF, 8 NMR ensembles (first model), 4 cryo-EM mmCIF, 9 AlphaFold-DB v6 mmCIF models. Files are
downloaded into `corpus/` (git-ignored) and verified by sha256. Add a structure by appending a
`[[structure]]` block, running `make fetch`, pasting the printed sha256, and `make reference`.

## Regenerating references

```
make reference     # validate/.venv (uv, Python 3.12) → validate/reference/<id>_<fmt>.json
validate/fetch_chem_data.sh                          # geostd + Top8000 grids for cctbx
PROTEUS_CHEM_DATA=~/.cache/proteus-validate/chem_data make geometry-reference
```

`reference.py` uses mdtraj + freesasa; `ramalyze_ref.py` runs cctbx in a **separate
process** — importing cctbx into the same interpreter corrupts mdtraj's native SASA kernel
(47 374 vs 2 969 Å² on 1CRN, observed 2026-09-21) and can segfault.

Residues are aligned by `chain:resseq:icode`. mdtraj labels mmCIF chains by `label_asym_id`
while pdbtbx uses the author ids, so the test falls back to chain-ordinal keys when id keys do
not match (2PTC, 6M0J).

## What Proteus does before measuring

`proteus_core::io::protein_heavy_atoms`: first model only, protein residues only (an atom
named `CA` with element carbon), heavy atoms only, first alternate conformation only (the first
in the file, not the highest occupancy). Two residues that share a chain, number and insertion
code are kept apart (the later one gets a free insertion code); no corpus file has one. This is
what mdtraj's `protein and not element H` selection and DSSP/MolProbity operate on. Two of
these rules were added because the harness caught the discrepancy (altloc duplicates inflated
1BPI's SASA by 1.5 %; waters and ions were being counted as atoms).
