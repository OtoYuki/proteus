# Predicted models in the validation corpus

`esmfold/` holds 13 unrelaxed ESMFold v1 predictions, one per corpus X-ray structure, folded on
2026-09-24 through the public ESMFold API (`POST https://api.esmatlas.com/foldSequence/v1/pdb/`)
from the chain's SEQRES sequence with non-standard residues dropped. The API cannot be
re-queried reproducibly, so the files are committed instead of fetched. The B-factor column is
ESMFold's per-residue pLDDT (0–100).

| file | source chain | residues |
|---|---|---|
| `1aki_A.pdb` | 1AKI A | 129 |
| `1bni_A.pdb` | 1BNI A | 110 |
| `1bpi_A.pdb` | 1BPI A | 58 |
| `1crn_A.pdb` | 1CRN A | 46 |
| `1hrc_A.pdb` | 1HRC A | 104 |
| `1mbn_A.pdb` | 1MBN A | 153 |
| `1pgb_A.pdb` | 1PGB A | 56 |
| `1stn_A.pdb` | 1STN A | 149 |
| `1tim_A.pdb` | 1TIM A | 247 |
| `1ubq_A.pdb` | 1UBQ A | 76 |
| `2ci2_I.pdb` | 2CI2 I | 83 |
| `2lzm_A.pdb` | 2LZM A | 164 |
| `3ptb_A.pdb` | 3PTB A | 223 |

They are here because ESMFold, like every AlphaFold2-style structure module, builds each residue
from ideal internal geometry but learns the peptide bond, and returns the model without an
energy minimisation. Its peptide C–N bonds are systematically short (mean 1.312 Å, sd 0.040,
against 1.336 Å, sd 0.0045 in AlphaFold DB models), which the covalent-geometry checks must see.

## `esmfold_relaxed/`

The same 13 models after an AlphaFold2-style restrained Amber minimisation (`relax.py`:
Amber99SB, vacuum, 10 kcal/mol/Å² heavy-atom restraints, OpenMM 8.6.1). Only coordinates
change; atom records and the pLDDT column are the input's. `tests/geometry_validation.rs`
(`relaxed_and_unrelaxed_models_separate`) checks that the geometry tells the two apart while
the fold stays put:

| model | Cα RMSD (Å) | bond RMSZ | bonds > 4σ | angle RMSZ | mean peptide C–N (Å) |
|---|---|---|---|---|---|
| 1aki_A | 0.08 | 1.01 → 0.73 | 16 → 0 | 1.01 → 1.10 | 1.321 → 1.336 |
| 1bni_A | 0.09 | 1.16 → 0.72 | 24 → 0 | 0.94 → 1.07 | 1.317 → 1.334 |
| 1bpi_A | 0.09 | 1.61 → 0.78 | 21 → 0 | 1.25 → 1.01 | 1.320 → 1.335 |
| 1crn_A | 0.13 | 4.05 → 0.81 | 15 → 0 | 3.15 → 1.35 | 1.325 → 1.334 |
| 1hrc_A | 0.09 | 1.26 → 0.76 | 27 → 0 | 1.08 → 1.08 | 1.320 → 1.336 |
| 1mbn_A | 0.09 | 1.09 → 0.75 | 24 → 0 | 0.90 → 1.14 | 1.306 → 1.337 |
| 1pgb_A | 0.09 | 1.57 → 0.64 | 24 → 0 | 0.94 → 0.96 | 1.302 → 1.331 |
| 1stn_A | 0.10 | 1.25 → 0.72 | 32 → 0 | 1.44 → 1.10 | 1.318 → 1.334 |
| 1tim_A | 0.09 | 1.24 → 0.72 | 47 → 0 | 0.91 → 1.14 | 1.310 → 1.335 |
| 1ubq_A | 0.09 | 1.14 → 0.70 | 17 → 0 | 1.15 → 1.00 | 1.315 → 1.335 |
| 2ci2_I | 0.14 | 1.65 → 0.74 | 25 → 0 | 3.04 → 1.11 | 1.324 → 1.335 |
| 2lzm_A | 0.08 | 1.13 → 0.71 | 35 → 0 | 0.90 → 1.09 | 1.314 → 1.336 |
| 3ptb_A | 0.08 | 1.64 → 0.68 | 67 → 0 | 1.08 → 1.08 | 1.307 → 1.333 |

Every unrelaxed model has bond outliers and no relaxed one has any. Angle RMSZ does not
separate them: Amber's angle targets differ from Engh & Huber's, so minimisation moves some
angles away from the library values. The 1CRN model starts with two disulfide sulfurs 0.82 Å
apart; the relaxed one is back at a bonded distance.
