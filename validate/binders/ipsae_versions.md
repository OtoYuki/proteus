# ipsae.py before and after its d0 change, on the binder dataset

`ipsae.py` (DunbrackLab/IPSAE, no tagged releases) at 3480750 (2026-01-02) and 6174cf9 (2026-01-03, current), PAE and distance cutoffs 10 Å, over the AlphaFold 3 model of each design in Overath et al.'s dataset. Written by `validate/binders/ipsae_versions.py`; run with `make validate-ipsae-versions`. 3669 of 3669 labelled designs scored by both versions: 3532 with a single-chain target, 137 with several.

6174cf9 changed d0 for small interfaces: the vectorised `calc_d0_array` (ipSAE proper) now floors the residue count at 26 instead of 27, and the scalar `calc_d0` (d0chn, d0dom) returns 1.0 for counts up to 27. At a count of exactly 27 the two functions now disagree (1.0 against 1.0389).

## 1. What the change does to ipSAE_min (single-chain targets)

| quantity | value |
|---|---|
| designs whose ipSAE_min changed | 976 of 3532 (27.6%) |
| raised / lowered | 0 / 976 |
| change on those, mean / largest | -0.00109 / -0.00833 |
| Spearman ρ / Kendall τ, old vs new | 0.999958 / 0.999045 |
| AP per target, old / new / dataset | 0.5516 / 0.5516 / 0.5512 |
| kept by `ipSAE_min > 0.61`, old / new | 509 (203 bound) / 509 (203 bound) |
| designs crossing 0.61 | 0 |

AP per target here is over single-chain targets only, so it differs from last_run.md, which also scores the pMHC designs.

## 2. Which version made the dataset

| version | designs | median \|Δ\| | p99 | within the dataset's 3-decimal rounding (5e-4) |
|---|---|---|---|---|
| 3480750 | 3138 | 0.00025 | 0.00050 | 99.3% |
| 6174cf9 | 3138 | 0.00037 | 0.00181 | 69.6% |

Left out above: the 394 designs where one direction has no PAE under 10 Å, which both versions score 0 (394 of 394 under 3480750 too). The dataset holds 0 for 317; the other 77 hold values up to 0.017 that neither direction of the model reproduces.

## 3. The paper's multi-chain rule

The paper's Methods: "If the target had several subchains we took the average of the min and max values across both directions of binder → target comparisons, but only if there are interacting residues between the binder chain and a given target subchain." Read as: for each target subchain with residues within the distance cutoff of the binder (`ipsae.py`'s dist1 + dist2 > 0), take the min of its two directions; average those.

| version | designs | median \|Δ\| vs dataset | within 5e-4 |
|---|---|---|---|
| 3480750 | 137 | 0.00027 | 137 of 137 |
| 6174cf9 | 137 | 0.00776 | 4 of 137 |

