# Binder triage vs the Adaptyv Nipah binder competition

Dataset: ProteinBase collection `nipah-binder-competition-results` (ODC-By), table sha256 `e6399877a3228614`, and the Boltz-2 complex and full PAE of each design as ProteinBase publishes them. Proteus: `proteus analyze <models> --interface B`. Written by `validate/nipah/compare.py`.

1196 designs with a model and a lab result (5 positive controls left out), 111 binders (prevalence 0.093, the AP of a random ranking). One target, so per-target and pooled are the same number.

**These designs were chosen by ipSAE before they were tested.** The competition sent the 60 collections with the best average ipSAE (600 designs) to the lab (Adaptyv's `nipah_ipsae_pipeline` README), so the tested set is already filtered on the score being evaluated: 491 of 1196 (41%) have `ipsae_min > 0.61`. Ranking within a set that ipSAE has already enriched is a harder test than validate/binders, and AP here understates what ipSAE does on unfiltered designs.

## 1. ipSAE against ProteinBase's own values

ProteinBase's numbers come from Adaptyv's `nipah_ipsae_pipeline`, which runs `ipsae.py` with PAE and distance cutoffs of 15 Å (the standard, and Proteus, use 10 Å). Its `boltz2_ipsae` is the max over the two directions; its `boltz2_min_ipsae` is not a minimum but one direction alone, A→B (target-aligned, binder scored): the notebook takes the `asym` row whose chain order matches the `max` row, which `ipsae.py` always labels A,B. Checked once on 45 designs by running that pipeline's `ipsae.py`: both reproduce ProteinBase to 1e-6, and Proteus's `ipsae_min` matches the same script at 10 Å to 0.0009 (the d0 floor moved since; see validate/binders). So the values below agree in rank, not in value.

| ours | theirs | n | Pearson r | Spearman ρ | median \|Δ\| | gate: r ≥ | |
|---|---|---|---|---|---|---|---|
| `ipsae_max` | `boltz2_ipsae` | 1196 | 0.987 | 0.972 | 0.0127 | 0.98 | ✓ |
| `ipsae_min` | `boltz2_min_ipsae` | 1196 | 0.996 | 0.996 | 0.0000 | 0.98 | ✓ |

## 2. Separating designs that bound in the lab from those that did not

Higher AP and AUROC are better; each metric is oriented so that larger means more likely to bind. Missing values rank last. `proteus` rows are computed here from the model and PAE; `ProteinBase` rows are the values published with the dataset.

| metric | source | AP | AUROC |
|---|---|---|---|
| ipSAE_min | proteus | 0.191 | 0.658 |
| ipSAE_min | ProteinBase | 0.193 | 0.658 |
| ipSAE_max | proteus | 0.165 | 0.660 |
| ipSAE | ProteinBase | 0.169 | 0.651 |
| LIS | proteus | 0.122 | 0.630 |
| LIS | ProteinBase | 0.122 | 0.630 |
| −ipAE | proteus | 0.105 | 0.569 |
| ipTM | ProteinBase | 0.119 | 0.618 |
| interface pLDDT | ProteinBase | 0.190 | 0.707 |
| pLDDT (mean) | proteus | 0.140 | 0.644 |
| Sc | proteus | 0.179 | 0.675 |
| Sc | ProteinBase | 0.183 | 0.682 |
| dSASA | proteus | 0.116 | 0.606 |
| interface H-bonds | proteus | 0.132 | 0.634 |
| ESMFold pLDDT (binder alone) | ProteinBase | 0.114 | 0.614 |

ProteinBase's `boltz2_pdockq` and `boltz2_pdockq2` are left out: each holds one value for every design (0.0183 and 0.0073), which is what `ipsae.py` returns when it finds no pLDDT file beside the PAE.

Binder rate by Proteus's `ipsae_min`:

| ipsae_min | designs | binders | binder rate |
|---|---|---|---|
| 0.00–0.20 | 264 | 7 | 0.027 |
| 0.20–0.40 | 128 | 6 | 0.047 |
| 0.40–0.61 | 313 | 28 | 0.089 |
| 0.61–0.70 | 324 | 43 | 0.133 |
| 0.70–0.80 | 151 | 21 | 0.139 |
| 0.80–1.00 | 16 | 6 | 0.375 |

Filter `ipsae_min > 0.61` (the threshold from validate/binders): keeps 491 designs, of which 70 bound (precision 0.143, recall 0.631).

## Result

All gates pass.
