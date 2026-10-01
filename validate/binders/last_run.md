# Binder triage vs the Overath et al. 2025 meta-analysis

Dataset: Zenodo 10.5281/zenodo.15722219 (CC-BY-4.0), `final_dataset.csv` and the AlphaFold 3 top model of each design. Proteus: `proteus analyze <af3 dir> --interface A`, binder = chain A against every other chain. Written by `validate/binders/compare.py`.

3669 designs matched (3532 with a single-chain target).

## 1. PAE metrics against the dataset's own values (single-chain targets)

| ours | dataset | n | median \|Δ\| | p99 \|Δ\| | max \|Δ\| | gate: p99 ≤ | |
|---|---|---|---|---|---|---|---|
| `iptm` | `af3_iptm_model_0` | 3532 | 0.0000 | 0.0000 | 0.0000 | 0.001 | ✓ |
| `ipsae_max` | `af3_ipSAE_max` | 3532 | 0.0003 | 0.0018 | 0.0145 | 0.005 | ✓ |
| `ipsae_min` | `af3_ipSAE_min` | 3138 | 0.0004 | 0.0018 | 0.0105 | 0.005 | ✓ |
| `ipae` | `af3_ipae` | 3532 | 0.0002 | 0.0005 | 0.0005 | 0.005 | ✓ |
| `lis` | `af3_LIS` | 3532 | 0.0229 | 0.0868 | 0.1273 | 0.15 | ✓ |

- `ipsae_min` parity leaves out the 394 designs where it is 0 (see below); their largest `ipsae_max` is 0.016.
- ipSAE follows the current `ipsae.py` (DunbrackLab, which since 2026-01-03 floors d0's residue count at 26). The dataset matches the version before that change: run on all 3 532 single-chain designs, `ipsae.py` at 3480750 (2026-01-02) agrees with the dataset's ipSAE_min within its 3-decimal rounding on 99.3 % of designs (median |Δ| 0.00025), the current version on 69.6 % (median 0.00037). The change lowers 27.6 % of values by at most 0.0083 and moves no design across the 0.61 threshold (Spearman 0.99996). Both runs: `make validate-ipsae-versions`, validate/binders/ipsae_versions.md.
- Where one direction has no PAE under 10 Å, `ipsae.py` (old and current) scores that direction 0 and so does Proteus's ipSAE_min: 394 designs. The dataset holds 0 for 317 of them; the other 77 hold small values (at most 0.017) that neither direction of the model reproduces. Every one has ipSAE_max below 0.016, so they sit at the bottom of an ipSAE ranking either way.
- LIS is `ipsae.py`'s reported value: the mean of the two directions, over PAE < 12 Å. The dataset holds one direction of an earlier version (PAE ≤ 12 Å), and which direction varies by design; so LIS is gated loosely, as a regression floor, not as parity.
- Multi-chain targets (pMHC, 137 designs) are left out of parity because the definitions differ: Proteus's ipSAE_min is the minimum over every binder↔target-chain direction, while the paper (Methods) averages, over the target subchains the binder touches, the min of the two directions. That rule with `ipsae.py` at 3480750 reproduces the dataset on 137 of 137 within 5e-4 (validate/binders/ipsae_versions.md). The two agree when the target is one chain.

## 2. Structure-based interface metrics against the dataset's Rosetta values (AF3 models)

| ours | dataset (Rosetta) | n | Pearson r | Spearman ρ | mean Δ | gate: r ≥ | |
|---|---|---|---|---|---|---|---|
| `interface_dsasa` | `af3_rosetta_interface_dSASA` | 3669 | 0.960 | 0.905 | -29.57 | 0.95 | ✓ |
| `interface_sc` | `af3_rosetta_interface_sc` | 3669 | 0.569 | 0.551 | -0.03 | 0.5 | ✓ |
| `interface_hbonds` | `af3_rosetta_interface_interface_hbonds` | 3669 | 0.771 | 0.694 | +0.29 | 0.7 | ✓ |
| `interface_binder_residues` | `af3_rosetta_interface_nres_binder` | 3669 | 0.945 | 0.872 | -3.92 | 0.9 | ✓ |

- Shape complementarity is a port of sc-rs, checked against it to 1e-12 (crates/proteus-core/tests/interface.rs); FreeBindCraft's README describes sc-rs's values as nearly identical to PyRosetta's. The dataset's Rosetta Sc was computed with a protocol adapted from BindCraft whose details (relaxation before scoring, in particular) are not published, and correlates only moderately with Sc on the raw AF3 model.
- Rosetta counts H-bonds from its energy function with explicit hydrogens; Proteus uses heavy-atom geometry. dSASA uses different radii and probe conventions. Interface residues: Proteus counts heavy atoms within 4 Å (BindCraft's hotspot cutoff), Rosetta's InterfaceAnalyzer uses its own definition.

## 3. Separating designs that bound in the lab from those that did not

3669 designs with an outcome, 394 binders, 15 targets (15 with at least one binder, 11 with at least 5). Higher AP and AUROC are better; each metric is oriented so that larger means more likely to bind. Missing values rank last.

**Per target** is the mean over targets of the AP (AUROC) of ranking that target's designs, which is how a design campaign uses the score. A random ranking's AP is the binder rate: 0.131 averaged over the 15 targets, 0.158 over the 11. **Pooled** ranks all designs together; its random AP is 0.107. Pooled numbers mostly measure which targets are easy, so read the per-target columns.

| metric | source | AP per target | AUROC per target | AP per target (≥5 binders) | AP pooled | AUROC pooled |
|---|---|---|---|---|---|---|
| ipSAE_min | proteus | 0.513 | 0.803 | 0.546 | 0.358 | 0.792 |
| ipSAE_min | dataset (AF3) | 0.513 | 0.815 | 0.543 | 0.350 | 0.787 |
| ipSAE d0chn | dataset (AF3) | 0.499 | 0.811 | 0.527 | 0.308 | 0.771 |
| ipSAE_max | proteus | 0.479 | 0.806 | 0.499 | 0.252 | 0.766 |
| LIS | proteus | 0.477 | 0.808 | 0.524 | 0.313 | 0.779 |
| −ipAE | proteus | 0.444 | 0.791 | 0.483 | 0.298 | 0.764 |
| pDockQ2_min | dataset (AF3) | 0.436 | 0.779 | 0.464 | 0.248 | 0.761 |
| ipTM | proteus (from AF3's file) | 0.425 | 0.791 | 0.467 | 0.236 | 0.748 |
| pLDDT (mean) | proteus | 0.409 | 0.730 | 0.451 | 0.208 | 0.739 |
| ipSAE_min | dataset (Boltz-1) | 0.402 | 0.736 | 0.430 | 0.323 | 0.768 |
| interface pLDDT | dataset (Boltz-1) | 0.301 | 0.690 | 0.338 | 0.221 | 0.739 |
| Sc | proteus | 0.381 | 0.714 | 0.305 | 0.267 | 0.722 |
| Sc | dataset (Rosetta) | 0.267 | 0.656 | 0.265 | 0.178 | 0.649 |
| actifpTM | dataset (ColabFold) | 0.346 | 0.735 | 0.425 | 0.205 | 0.719 |
| −interface ΔG | dataset (Rosetta) | 0.332 | 0.725 | 0.398 | 0.187 | 0.694 |
| dSASA | proteus | 0.248 | 0.657 | 0.271 | 0.126 | 0.579 |
| interface H-bonds | proteus | 0.246 | 0.669 | 0.281 | 0.140 | 0.595 |
| LIS × Sc | proteus | 0.464 | 0.803 | 0.475 | 0.356 | 0.785 |

Per target, ranked by Proteus's `ipsae_min`:

| target | designs | binders | random AP | AP | AUROC |
|---|---|---|---|---|---|
| FGFR2 | 2123 | 193 | 0.091 | 0.384 | 0.753 |
| EGFR | 434 | 28 | 0.065 | 0.245 | 0.792 |
| IL7Ra | 171 | 38 | 0.222 | 0.526 | 0.798 |
| TrkA | 128 | 9 | 0.070 | 0.306 | 0.754 |
| InsulinR | 117 | 20 | 0.171 | 0.430 | 0.741 |
| VirB8 | 99 | 9 | 0.091 | 0.546 | 0.827 |
| SARS_CoV2_RBD | 99 | 9 | 0.091 | 0.692 | 0.851 |
| pMHC_SILSY1 | 96 | 2 | 0.021 | 0.089 | 0.809 |
| Mdm2 | 96 | 55 | 0.573 | 0.648 | 0.573 |
| Pdl1 | 95 | 12 | 0.126 | 0.557 | 0.877 |
| IL2Ra | 66 | 6 | 0.091 | 0.804 | 0.972 |
| sntx | 49 | 7 | 0.143 | 0.872 | 0.973 |
| pMHC_NY1 | 41 | 1 | 0.024 | 0.500 | 0.975 |
| LTK | 33 | 3 | 0.091 | 1.000 | 1.000 |
| IL10Ra | 22 | 2 | 0.091 | 0.091 | 0.350 |

Filter `ipsae_min > 0.61` (the paper's single-metric threshold): keeps 509 designs, of which 203 bound (precision 0.399, recall 0.515).

## Result

All gates pass.
