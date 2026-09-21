# proteus

High-throughput bio-compute orchestration engine and terminal biophysics workbench written in Rust.

Proteus automates the synthetic protein engineering design loop: sequence mutagenesis, multi-tiered structural prediction (ESMFold, Boltz-1, ColabFold), OCI/Podman container scheduling, pure-Rust all-atom biophysical validation (SASA, DSSP, MolProbity Ramachandran, heavy-atom steric overlap), software 3D terminal rasterization, and Apache Parquet data lake exports.

---

## Architecture

The project is organized as a Cargo workspace across six decoupled crates:

```
crates/
├── proteus-core/       Domain models, FASTA parser, DMS mutagenesis, and native biophysics
├── proteus-storage/    Embedded SQLite repository (SQLx WAL) and Apache Parquet data lake exporter
├── proteus-engine/     Async DAG task scheduler and OCI/Podman container runner (bollard)
├── proteus-render/     Software 3D rasterizer, Bishop ribbon extruder, and TUI dashboard
├── proteus-server/     Headless Axum daemon (proteusd) with SSE event streams and OpenAPI docs
└── proteus-cli/        Unified CLI binary (proteus) for screening, inspection, and daemon hosting
```

---

## Core Capabilities

### 1. In-Silico Deep Mutational Scanning (DMS)
Generates high-density mutant variant libraries directly from wildtype scaffolds:
- **Alanine Scanning:** Systematic single-point mutations to Alanine across selected sequence windows to map critical functional epitopes.
- **Site-Saturation Mutagenesis:** Exhaustive substitution of all 20 canonical amino acids across target active sites or binding interfaces.
- **Pipeline Streaming:** Native UNIX pipeline support (`mutate | screen -`) for zero-disk intermediate streaming.

### 2. High-Throughput Screening Funnel & Parquet Data Lake
Evaluates variant libraries across multi-threaded computational workers:
- **Multi-Tier Inference:** Dispatches structural prediction jobs across fast ESMFold heuristics, Boltz-1/ColabFold OCI containers, or local simulation fallback.
- **Multi-Objective Pareto Ranking:** Evaluates candidates against composite fitness functions incorporating pLDDT confidence, compactness ($R_g$), hydrophobic core burial, secondary structure stability, and steric clashes.
- **Columnar Data Lake Export:** Serializes screened variant batches into ZSTD-compressed Apache Parquet files using canonical Apache Arrow schemas for direct query execution in DuckDB, Polars, or PyArrow.

### 3. Pure-Rust Terminal 3D Rasterizer & Live Telemetry Dashboard
Enables full structural inspection over SSH without X11 forwarding, WebGL browser dependencies, or headless display servers:
- **Cartoon Ribbon Mesh Generation:** Interpolates $C_\alpha$ backbones via cubic Hermite splines with Bishop parallel-transport frames, elliptic cross-sections, and Richardson $\beta$-arrowheads.
- **Lighting & Post-Processing:** Software $z$-buffer rasterizer with directional Blinn-Phong shading, Screen-Space Ambient Occlusion (SSAO), and edge-detection cel-outlines.
- **Covalent Disulfide Bridges:** Automatically detects and renders cystine covalent bonds ($S_\gamma - S_\gamma$) as high-visibility sidechain cylinders.
- **Multi-Structure Superposition:** Visualizes pairwise structural alignments in distinct dual-color palettes with Kabsch RMSD metrics.
- **Terminal Compositing:** High-resolution sub-pixel Braille (2x4 dots/cell), ANSI 24-bit half-block (1x2 pixels/cell), and Kitty graphics protocol for raw 24-bit RGB pixel blitting.
- **Live TUI Dashboard:** Split-screen layout displaying real-time 3D rotation alongside an ASCII Ramachandran ($\phi, \psi$) conformational scatter plot, per-residue pLDDT spectrum, and biophysical metrics.

### 4. Native Biophysical Validation Engines
Executes all-atom biophysical calculations in sub-milliseconds:
- **Non-Covalent Interaction Networks (NCIN):** Evaluates all-atom hydrogen bonds (Baker-Hubbard heavy-atom antecedent criteria across backbone and sidechains), ionic salt bridges ($\le 4.0\text{ \AA}$ between basic cations and acidic anions), $\pi$-$\pi$ aromatic stacking (parallel displaced and T-shaped edge-to-face), and cation-$\pi$ interactions over $O(N)$ spatial bounding-box cell lists.
- **Shrake-Rupley SASA:** Computes solvent-accessible surface area and hydrophobic core burial ratios using a 92-point Fibonacci sphere tessellation and an $O(N)$ spatial grid cell-list.
- **MolProbity Ramachandran Distributions:** Calculates backbone dihedral angles ($\phi, \psi$) and classifies conformations across four residue-specific stereochemical contexts (General, Glycine, Proline, Pre-Proline).
- **Heavy-Atom Steric Overlap (MolProbity-style, no hydrogens):** Counts severe heavy-atom overlaps ($> 0.40\text{ \AA}$) per 1,000 atoms using Bondi van der Waals radii, cell-list spatial hashing, and covalent exclusions (intra-residue bonding, peptide backbone linkages, proline pyrrolidine ring geometry, and disulfide bridges). This is **not** the MolProbity clashscore, which adds hydrogens with Reduce first; it under-counts on deposited structures and is intended as a relative screen for grossly overlapping predicted models.
- **Kabsch Coordinate Superposition:** Computes optimal rotational alignment and minimum RMSD via Singular Value Decomposition (SVD) on $3 \times 3$ covariance matrices (`nalgebra`).

---

## Quickstart

### Prerequisites
- **Rust Toolchain:** 1.85+ (tested on Rust 1.94)
- **Container Runtime (Optional):** Podman rootless socket (`systemctl --user enable --now podman.socket`) or Docker daemon for live OCI container execution.

### Build
```bash
cargo build --release
```
The compiled binary will be located at `target/release/proteus`.

### Verification & Test Suite
```bash
# Run all 48 workspace unit and integration tests
cargo test --workspace

# Strict lint check
cargo clippy --workspace --all-targets -- -D warnings

# Code formatting check
cargo fmt --check
```

---

## CLI Reference

### 1. In-Silico Mutagenesis (`proteus mutate`)
Generate an Alanine scanning variant library:
```bash
proteus mutate --scaffold scaffold.fasta --mode alanine --output alanine_library.fasta
```

Generate a site-saturation library restricted to residues 10–18:
```bash
proteus mutate --scaffold scaffold.fasta --mode saturation --start 10 --end 18 --max-variants 50
```

### 2. High-Throughput Screening (`proteus screen`)
Screen a variant library through the compute funnel and export candidate biophysics to Apache Parquet:
```bash
proteus screen \
  --library alanine_library.fasta \
  --tier fast \
  --workers 8 \
  --min-plddt 75.0 \
  --top 10 \
  --export results.parquet
```

Pipe mutations directly into the screening funnel without saving intermediate FASTA files:
```bash
proteus mutate --scaffold wildtype.fasta --mode alanine | proteus screen --library - --export results.parquet
```

### 3. Terminal 3D Structure Viewer (`proteus view`)
Launch the interactive 3D viewer with the live split-screen biophysical dashboard:
```bash
proteus view structure.pdb --interactive --dashboard
```

Interactive keyboard controls:
- `Arrow Keys` / `HJKL`: Rotate structure pitch and yaw.
- `+` / `-`: Zoom in and zoom out.
- `Space`: Toggle automatic rotation.
- `C`: Cycle color schemes (pLDDT spectrum, secondary structure, cyan, green, amber).
- `D`: Toggle biophysical telemetry dashboard.
- `Q` / `Esc`: Exit viewer.

Superimpose two structures to visually inspect conformational changes:
```bash
proteus view mutant.pdb --compare wildtype.pdb --interactive
```

Render high-fidelity terminal snapshots to stdout for scripting and CI logs:
```bash
# High-resolution Braille rendering
proteus view structure.pdb --backend braille --width 80 --height 36

# Full-color ANSI half-block rendering
proteus view structure.pdb --backend halfblock --color sst --width 80 --height 36

# Native Kitty graphics protocol (kitty, wezterm, ghostty)
proteus view structure.pdb --backend kitty --width 100 --height 40
```

### 4. Offline Biophysical Analysis (`proteus analyze`)
Inspect all-atom biophysical metrics for any local PDB structure:
```bash
proteus analyze --pdb structure.pdb
```
Output:
```
┌───────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Biophysical Metric                    ┆ Value                                           │
╞═══════════════════════════════════════╪═════════════════════════════════════════════════╡
│ Radius of Gyration (Rg)               ┆ 9.676 Å                                         │
│ Contact Density (C-alpha <= 8Å)       ┆ 9.86% (Cα pairs)                                │
│ Mean pLDDT                            ┆ 82.50 (Confident)                               │
│ Secondary Structure Composition       ┆ α-Helix: 69.6% | β-Strand: 28.3% | Coil: 2.2%   │
│ Ramachandran Conformation             ┆ Favored: 95.5% | Allowed: 4.5% | Outliers: 0    │
│ Solvent Accessible Surface Area       ┆ Total: 2976.6 Å² (Hydrophobic Burial: 92.8%)    │
│ MolProbity Clashscore (>0.4Å)         ┆ 0.0 (0 severe steric overlaps)                  │
│ Hydrogen Bonds (H-Bonds)              ┆ 54 total (43 BB-BB, 10 BB-SC, 1 SC-SC)          │
│ Ionic Salt Bridges (≤4.0Å)            ┆ 1 detected (closest: ARG17:NH2-GLU23:OE2 3.97Å) │
│ Aromatic π-π Stacking                 ┆ 0 conjugated pairs (0 parallel, 0 T-shaped)     │
│ Cation-π Interactions                 ┆ 1 active interactions                           │
│ Non-Covalent Network Density          ┆ 121.7 contacts / 100 res                        │
│ Candidate Fitness Score               ┆ 84.8 / 100                                      │
└───────────────────────────────────────┴─────────────────────────────────────────────────┘
```

### 5. Headless Daemon (`proteus serve`)
Run the headless background service:
```bash
proteus serve --port 8080 --host 0.0.0.0
```
Interactive Swagger UI documentation is served at `http://localhost:8080/swagger-ui`.

---

## Mathematical Formulations

### Radius of Gyration ($R_g$)
$$\mathbf{r}_{\text{cm}} = \frac{1}{N}\sum_{i=1}^N \mathbf{r}_i \quad\text{over all } C_\alpha\text{ atoms}$$
$$R_g = \sqrt{\frac{1}{N}\sum_{i=1}^N \|\mathbf{r}_i - \mathbf{r}_{\text{cm}}\|^2}$$

### Kabsch Optimal Superposition ($C_\alpha$ RMSD)
For centered coordinate matrices $P, Q \in \mathbb{R}^{N \times 3}$:
1. Compute the cross-covariance matrix: $H = P^T Q$.
2. Compute Singular Value Decomposition: $H = U \Sigma V^T$.
3. Correct for improper rotation (reflection):
   $$R = V \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(V U^T) \end{pmatrix} U^T$$
4. Compute aligned root-mean-square deviation:
   $$\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|R \mathbf{p}_i - \mathbf{q}_i\|^2}$$

### Heavy-Atom Steric Overlap Score (MolProbity-style, no hydrogens)
$$\text{Overlap}_{1k} = \frac{\sum_{i < j} \mathbb{I}\left(r_i^{\text{vdW}} + r_j^{\text{vdW}} - d_{ij} > 0.40\text{ \AA}\right)}{N_{\text{atoms}}} \times 1000$$
Subject to topological exclusions:
- Atoms within the same residue ($res_i = res_j$).
- Backbone peptide linkages and proline pyrrolidine ring geometry ($|res_i - res_j| = 1$ within the same chain).
- Covalent disulfide-bonded cysteine sulfur pairs ($d(S_\gamma, S_\gamma) \in [1.70, 2.60]\text{ \AA}$).

### Non-Covalent Interaction Network (NCIN)
- **Baker-Hubbard Hydrogen Bonds:**
  $$2.4\,\text{Å} \le d(D, A) \le 3.5\,\text{Å}, \quad \theta(D_{\text{ante}}-D\cdots A) \ge 90^\circ, \quad \theta(A_{\text{ante}}-A\cdots D) \ge 90^\circ$$
- **Ionic Salt Bridges:**
  $$d(\text{cation}, \text{anion}) \le 4.0\,\text{Å} \quad\text{with}\quad res_{\text{cat}} \neq res_{\text{ani}}$$
- **Aromatic $\pi$-$\pi$ Stacking:**
  $$d(\mathbf{c}_1, \mathbf{c}_2) \le 6.5\,\text{Å}, \quad \theta = \arccos(|\mathbf{n}_1 \cdot \mathbf{n}_2|) \implies \begin{cases} \text{Parallel} & \theta \le 30^\circ \\ \text{T-Shaped} & 60^\circ \le \theta \le 120^\circ \end{cases}$$
- **Cation-$\pi$ Interactions:**
  $$d(\text{cation}, \mathbf{c}) \le 6.0\,\text{Å}, \quad \cos\alpha = \frac{|\mathbf{n} \cdot (\mathbf{r}_{\text{cat}} - \mathbf{c})|}{\|\mathbf{r}_{\text{cat}} - \mathbf{c}\|} \ge \frac{1}{\sqrt{2}}$$

### Composite Candidate Fitness Score
$$S_{\text{fitness}} = 0.30 \cdot \text{pLDDT} + 0.20 \cdot S_{\text{compactness}} + 0.15 \cdot f_{\text{favored}} + 0.15 \cdot f_{\text{burial}} + 0.20 \cdot B_{\text{network}} - P_{\text{clash}}$$
where non-covalent tertiary network density $B_{\text{network}}$ rewards secondary/tertiary hydrogen bonds, salt bridges, and aromatic contacts:
$$B_{\text{network}} = \min\left(100.0, \frac{0.5 N_{\text{bb}} + 1.0 N_{\text{sc-hbond}} + 2.5 N_{\text{salt}} + 2.0 N_{\pi\text{-}\pi} + 2.0 N_{\text{cat-}\pi}}{0.60 \cdot N_{\text{res}}}\right)$$

---

## Provenance

The legacy Python/Django/Celery undergraduate thesis prototype is preserved under git tag `v0.1.0-thesis`. The repository root and active codebase are 100% Rust.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT)

at your option.
