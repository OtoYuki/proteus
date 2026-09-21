# Non-Covalent Interaction Network (NCIN) Architectural Specification

**Date:** 2026-09-21  
**Status:** Approved  
**Author:** Proteus Core Team  
**Scope:** `crates/proteus-core`, `crates/proteus-storage`, `crates/proteus-cli`, `crates/proteus-server`

---

## 1. Executive Summary

Proteus automates synthetic protein engineering and structural analytics in pure Rust. While macroscopic biophysical descriptors (SASA, radius of gyration, secondary structure, and MolProbity steric clashes) quantify gross geometric validity, the primary drivers of thermodynamic fold stability, tertiary cooperativity, and ligand binding affinity are **non-covalent interaction networks**.

This specification defines the architecture, stereochemical geometry, and algorithmic pipeline for the **All-Atom Non-Covalent Interaction Network (NCIN)** engine (`proteus-core::interactions`), integrating:
1. **Baker-Hubbard Hydrogen Bonds**: Backbone-Backbone, Backbone-Sidechain, and Sidechain-Sidechain with antecedent angle geometry.
2. **Ionic Salt Bridges**: Cation-anion ion pairs across Lys, Arg, Asp, and Glu.
3. **$\pi$-$\pi$ Aromatic Stacking**: Parallel displaced (face-to-face) and T-shaped (edge-to-face) geometries across Phe, Tyr, Trp, and His.
4. **Cation-$\pi$ Interactions**: Basic amine/guanidinium cations oriented over aromatic $\pi$-electron clouds.
5. **Downstream Integration**: Candidate fitness scoring in `proteus-core::ranking`, Apache Parquet v60 columnar data lake export in `proteus-storage`, and terminal inspection in `proteus-cli`.

---

## 2. Stereochemical Foundations & Mathematical Formulations

### 2.1 Hydrogen Bonds ($D - \text{H}\cdots A$)
For crystallographic structures lacking explicit hydrogens, donor-acceptor interactions are evaluated between heavy atoms using antecedent vectors:

* **Donors ($D$) and Antecedents ($D_{\text{ante}}$)**:
  * Backbone Amide $N$ ($D_{\text{ante}} = C_\alpha$).
  * Sidechain nitrogens: Trp $N^{\epsilon1}$ ($C^{\delta1}$), Lys $N^\zeta$ ($C^\epsilon$), Arg $N^\epsilon$ ($C^\delta$) & $N^{\eta1}, N^{\eta2}$ ($C^\zeta$), His $N^{\delta1}$ ($C^\gamma$) & $N^{\epsilon2}$ ($C^{\epsilon1}$), Asn $N^{\delta2}$ ($C^\gamma$), Gln $N^{\epsilon2}$ ($C^\delta$).
  * Sidechain hydroxyls: Ser $O^\gamma$ ($C^\beta$), Thr $O^{\gamma1}$ ($C^\beta$), Tyr $O^\eta$ ($C^\zeta$).
* **Acceptors ($A$) and Antecedents ($A_{\text{ante}}$)**:
  * Backbone Carbonyl $O$ ($A_{\text{ante}} = C$).
  * Sidechain carboxylates: Asp $O^{\delta1}, O^{\delta2}$ ($C^\gamma$), Glu $O^{\epsilon1}, O^{\epsilon2}$ ($C^\delta$).
  * Sidechain carbonyls: Asn $O^{\delta1}$ ($C^\gamma$), Gln $O^{\epsilon1}$ ($C^\delta$).
  * Sidechain hydroxyls / imidazole: Ser $O^\gamma$ ($C^\beta$), Thr $O^{\gamma1}$ ($C^\beta$), Tyr $O^\eta$ ($C^\zeta$), His $N^{\delta1}$ ($C^\gamma$) & $N^{\epsilon2}$ ($C^{\epsilon1}$).
* **Formulations**:
  $$2.4\,\text{Å} \le \|\mathbf{r}_A - \mathbf{r}_D\| \le 3.5\,\text{Å}$$
  $$\theta(D_{\text{ante}} - D \cdots A) = \arccos\left(\frac{(\mathbf{r}_{D_{\text{ante}}} - \mathbf{r}_D) \cdot (\mathbf{r}_A - \mathbf{r}_D)}{\|\mathbf{r}_{D_{\text{ante}}} - \mathbf{r}_D\| \|\mathbf{r}_A - \mathbf{r}_D\|}\right) \ge 90^\circ$$
  $$\theta(A_{\text{ante}} - A \cdots D) = \arccos\left(\frac{(\mathbf{r}_{A_{\text{ante}}} - \mathbf{r}_A) \cdot (\mathbf{r}_D - \mathbf{r}_A)}{\|\mathbf{r}_{A_{\text{ante}}} - \mathbf{r}_A\| \|\mathbf{r}_D - \mathbf{r}_A\|}\right) \ge 90^\circ$$
  with sequence separation $|res_D - res_A| \ge 2$.

### 2.2 Ionic Salt Bridges
* **Cations**: Lys $N^\zeta$, Arg $N^\epsilon, N^{\eta1}, N^{\eta2}$.
* **Anions**: Asp $O^{\delta1}, O^{\delta2}$, Glu $O^{\epsilon1}, O^{\epsilon2}$.
* **Formulation**:
  $$\|\mathbf{r}_{\text{cation}} - \mathbf{r}_{\text{anion}}\| \le 4.0\,\text{Å} \quad\text{with}\quad res_{\text{cation}} \neq res_{\text{anion}}$$

### 2.3 $\pi$-$\pi$ Aromatic Stacking
For aromatic rings (Phe, Tyr, Trp, His), ring centroids $\mathbf{c}$ and unit normal vectors $\mathbf{n}$ are derived:
$$\mathbf{c} = \frac{1}{k}\sum_{i=1}^k \mathbf{r}_i, \quad \mathbf{n} = \frac{(\mathbf{r}_1 - \mathbf{c}) \times (\mathbf{r}_2 - \mathbf{c})}{\|(\mathbf{r}_1 - \mathbf{c}) \times (\mathbf{r}_2 - \mathbf{c})\|}$$
* **Geometric Formulations**:
  $$\|\mathbf{c}_1 - \mathbf{c}_2\| \le 6.5\,\text{Å} \quad\text{with}\quad res_1 \neq res_2$$
  $$\cos\theta = |\mathbf{n}_1 \cdot \mathbf{n}_2|$$
  * **Parallel Displaced (Face-to-Face)**: $\theta \le 30^\circ$ or $\theta \ge 150^\circ \iff \cos\theta \ge \cos(30^\circ) \approx 0.866$.
  * **T-Shaped (Edge-to-Face)**: $60^\circ \le \theta \le 120^\circ \iff \cos\theta \le \cos(60^\circ) = 0.500$.

### 2.4 Cation-$\pi$ Interactions
* **Cations**: Lys $N^\zeta$, Arg $N^\epsilon, N^{\eta1}, N^{\eta2}$.
* **Aromatic Centroids**: Phe, Tyr, Trp.
* **Formulations**:
  $$\mathbf{v} = \mathbf{r}_{\text{cation}} - \mathbf{c}, \quad \|\mathbf{v}\| \le 6.0\,\text{Å}$$
  $$\cos\alpha = \frac{|\mathbf{n} \cdot \mathbf{v}|}{\|\mathbf{v}\|} \ge \cos(45^\circ) \approx 0.707$$

---

## 3. High-Throughput Spatial Engine Design

To preserve sub-millisecond screening throughput across 10,000+ variants:
* **Spatial Grid Hashing ($R_{\text{cell}} = 7.0\,\text{Å}$)**: A 3D bounding-box spatial hash grid indexes target entities. Because $7.0\,\text{Å} \ge \max(d_{\text{cutoff}})$, querying the 27 neighboring cells ($(dx, dy, dz) \in \{-1, 0, 1\}^3$) captures all interacting pairs in $O(N)$ expected time.
* **Squared Distance Thresholding**: Distances are filtered using $d^2 \le d_{\text{max}}^2$ prior to computing square roots, inverse trigonometric functions, or cross products.

---

## 4. Downstream Integration

### 4.1 Candidate Fitness Function Enhancement
In `proteus-core::ranking`, non-covalent networks contribute positively to predicted thermodynamic stability:
$$S_{\text{stability}} = 0.5 \cdot N_{\text{bb-bb}} + 1.0 \cdot N_{\text{tertiary-hbond}} + 2.0 \cdot N_{\text{salt-bridge}} + 1.5 \cdot N_{\pi-\pi} + 1.5 \cdot N_{\text{cation}-\pi}$$
Normalized per 100 residues and bounded to prevent saturation:
$$B_{\text{network}} = \min\left(10.0, \frac{S_{\text{stability}}}{N_{\text{res}}} \times 10.0\right)$$
The composite fitness score becomes:
$$S_{\text{fitness}} = 0.30 \cdot \text{pLDDT} + 0.20 \cdot f_{\text{favored}} + 0.15 \cdot f_{\text{burial}} + 0.15 \cdot f_{\text{helix}+\text{strand}} + B_{\text{network}} - P$$

### 4.2 Columnar Data Lake Exporter (`proteus-storage`)
The Apache Arrow schema in `crates/proteus-storage/src/export.rs` expands from 14 to 18 biophysical columns:
* `hbond_count` (`UInt32`)
* `salt_bridge_count` (`UInt32`)
* `pi_stacking_count` (`UInt32`)
* `cation_pi_count` (`UInt32`)

---

## 5. Verification & Test Plan

1. **Crambin Crystallographic Benchmark (`1crn.pdb`, 1.5 Å resolution)**:
   - Verify secondary structure hydrogen bonds along the two amphipathic $\alpha$-helices (residues 7–19 and 23–30).
   - Verify core disulfide bridge preservation alongside non-covalent contacts.
   - Verify aromatic packing interactions involving Phe13, Tyr29, Tyr44.
2. **Synthetic Peptides with Known Geometry**:
   - Ideal poly-alanine $\alpha$-helix: confirm expected $i \to i+4$ backbone-backbone hydrogen bonds.
   - Arg-Asp synthetic pair at $3.2\,\text{Å}$: confirm salt bridge identification.
   - Phe-Phe parallel and perpendicular dimers: confirm classification into Parallel and T-shaped stacking.
3. **Workspace Performance & Quality**:
   - Sub-millisecond execution on Crambin.
   - 0 clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings`).
   - Clean formatting (`cargo fmt --check`).
   - 100% test pass rate across all workspace crates.
