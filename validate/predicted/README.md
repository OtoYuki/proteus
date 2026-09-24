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
