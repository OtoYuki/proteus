# proteus

[![ci](https://github.com/OtoYuki/proteus/actions/workflows/ci.yml/badge.svg)](https://github.com/OtoYuki/proteus/actions/workflows/ci.yml)
[![validate](https://github.com/OtoYuki/proteus/actions/workflows/validate.yml/badge.svg)](https://github.com/OtoYuki/proteus/actions/workflows/validate.yml)
[![release](https://img.shields.io/github/v/release/OtoYuki/proteus?include_prereleases)](https://github.com/OtoYuki/proteus/releases)
[![MSRV 1.88](https://img.shields.io/badge/MSRV-1.88-blue)](Cargo.toml)
[![license MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#license)

High-throughput bio-compute orchestration engine and terminal biophysics workbench, in Rust.

![proteus demo: analyze, interactive dashboard, pLDDT provenance, validation table](docs/media/demo.gif)

Proteus runs the protein-engineering design loop end to end — sequence mutagenesis → structure
prediction (ESMFold API, Boltz/ColabFold containers via Podman/Docker, GA4GH TES v1.1) → all-atom
biophysical validation → ranking → Apache Parquet — and lets you look at the result in the terminal
you are already SSH'd into. The biophysics is pure Rust and **checked against mdtraj, FreeSASA and
cctbx/MolProbity on 43 structures in CI**; the speed claims are measured, not asserted.

| what | how it is checked |
|---|---|
| φ/ψ, Cα radius of gyration, Kabsch RMSD | mdtraj, every angle within 0.1° |
| Kabsch–Sander DSSP (`proteus-dssp`, standalone crate) | mdtraj, ≥ 98 % per-residue |
| MolProbity Ramachandran (Top8000 contours from cctbx) | cctbx `ramalyze`, 100 % label agreement |
| Shrake–Rupley SASA (Bondi radii, 960 pts) | mdtraj ≤ 1 %, FreeSASA ≤ 4 % (L&R, ProtOr radii) |
| heavy-atom steric overlap, H-bond / salt-bridge / π network | Proteus-defined; labelled as such |

**Speed** (same metric, same file, median wall-clock; full table in [`bench/README.md`](bench/README.md)):
SASA 2.5–4.3× faster than mdtraj's C++ kernel at equal point count and ~35× faster than
Biopython; DSSP 1.2–14× vs mdtraj; φ/ψ + Ramachandran 33–167× vs mdtraj φ/ψ. 6VXX
(22 812 atoms) full profile: 0.63 s.

## Install

```bash
# release binaries (Linux x86_64/aarch64, macOS x86_64/arm64)
curl -L https://github.com/OtoYuki/proteus/releases/latest/download/proteus-x86_64-unknown-linux-gnu.tar.gz | tar xz
# from source (Rust 1.88+)
cargo install --git https://github.com/OtoYuki/proteus proteus-cli
# container
podman run --rm -v "$PWD:/w" ghcr.io/otoyuki/proteus analyze --pdb /w/structure.pdb
```

---

## Architecture

The project is organized as a Cargo workspace across seven decoupled crates:

```
crates/
├── proteus-core/       Domain models, FASTA parser, DMS mutagenesis, and native biophysics
├── proteus-dssp/       Standalone pure-Rust Kabsch–Sander DSSP secondary-structure assignment
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
- **Weighted Composite Fitness Score:** Ranks candidates by a weighted sum of pLDDT confidence (predicted models only), compactness ($R_g$ vs Flory scaling), MolProbity Ramachandran quality, hydrophobic core burial and non-covalent network density, minus a steric-overlap penalty. For experimental structures the pLDDT weight is redistributed over the other terms.
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
- **MolProbity Ramachandran Evaluation:** Backbone $\phi/\psi$ are scored against the six Top8000 percentile contour grids (general, Gly, cis-Pro, trans-Pro, pre-Pro, Ile/Val) converted from cctbx, with MolProbity's Favored ≥ 2 % / Allowed ≥ 0.05–0.2 % thresholds. Labels agree with cctbx `ramalyze` on 100 % of residues across the validation corpus.
- **Kabsch–Sander DSSP (`proteus-dssp`):** Eight-state secondary structure from backbone H-bond energies (α/3₁₀/π helices, bridges, ladders, bends, turns), a standalone pure-Rust crate validated residue-by-residue against mdtraj.
- **Heavy-Atom Steric Overlap (MolProbity-style, no hydrogens):** Counts severe heavy-atom overlaps ($> 0.40\text{ \AA}$) per 1,000 atoms using Bondi van der Waals radii, cell-list spatial hashing, and covalent exclusions (intra-residue bonding, peptide backbone linkages, proline pyrrolidine ring geometry, and disulfide bridges). This is **not** the MolProbity clashscore, which adds hydrogens with Reduce first; it under-counts on deposited structures and is intended as a relative screen for grossly overlapping predicted models.
- **Kabsch Coordinate Superposition:** Computes optimal rotational alignment and minimum RMSD via Singular Value Decomposition (SVD) on $3 \times 3$ covariance matrices (`nalgebra`).

### 5. Validated Against Reference Implementations
Every push runs `make validate` (`.github/workflows/validate.yml`) over a 43-structure corpus (X-ray, NMR, cryo-EM, AlphaFold-DB; PDB and mmCIF) and compares each metric to an independent implementation: **mdtraj** (φ/ψ, DSSP, $R_g$, Shrake–Rupley SASA), **FreeSASA** (Lee–Richards SASA) and **cctbx/MolProbity `ramalyze`** (Top8000 Ramachandran). Tolerances are the contract in `validate/tolerances.toml`; the full table for the last run is written to `validate/last_run.md`. Excerpt:

| id | fmt | kind | res | Δrg Å | SASA vs mdtraj | SASA vs freesasa | φ/ψ ≤tol | DSSP-8 | DSSP-3 | Rama labels | F/A/O proteus | F/A/O cctbx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1crn | pdb | xray | 46 | 0.000 | 0.15% | 0.86% | 90/90 | 100.0% | 100.0% | 100.00% | 43/1/0 | 43/1/0 |
| 1tim | pdb | xray | 494 | 0.000 | 0.06% | 3.42% | 984/984 | 100.0% | 100.0% | 100.00% | 306/123/61 | 306/123/61 |
| 1ubq | pdb | xray | 76 | 0.000 | 0.05% | 1.40% | 150/150 | 100.0% | 100.0% | 100.00% | 74/0/0 | 74/0/0 |
| 2kod | pdb | nmr | 176 | 0.000 | 0.00% | 1.06% | 348/348 | 100.0% | 100.0% | 100.00% | 155/8/9 | 155/8/9 |
| 4hhb | pdb | xray | 574 | 0.000 | 0.07% | 2.03% | 1140/1140 | 100.0% | 100.0% | 100.00% | 505/54/7 | 505/54/7 |
| 6vxx | cif | cryoem | 2916 | 0.000 | 0.02% | – | 5760/5760 | 97.8% | 100.0% | 100.00% | 2775/69/0 | 2775/69/0 |
| af-p38398 | cif | afdb | 1863 | 0.000 | 0.01% | – | 3724/3724 | 100.0% | 100.0% | 100.00% | 826/413/622 | 826/413/622 |
| af-p69905 | cif | afdb | 142 | 0.000 | 0.05% | – | 282/282 | 100.0% | 100.0% | 100.00% | 138/2/0 | 138/2/0 |

Δrg is the absolute Cα radius-of-gyration error in Å; SASA columns are relative errors; φ/ψ counts angles within 0.1°; F/A/O are MolProbity Favored/Allowed/Outlier counts. See `validate/README.md` for what is and is not covered.

---

## Quickstart

### Prerequisites
- **Rust Toolchain:** 1.88+ (MSRV, checked in CI; developed on 1.94)
- **Container Runtime (Optional):** Podman rootless socket (`systemctl --user enable --now podman.socket`) or Docker daemon for live OCI container execution.

### Build
```bash
cargo build --release
```
The compiled binary will be located at `target/release/proteus`.

### Verification & Test Suite
```bash
# Run all 80 workspace unit, integration and doc tests
cargo test --workspace

# Strict lint check
cargo clippy --workspace --all-targets -- -D warnings

# Code formatting check
cargo fmt --check

# Reference validation (downloads a ~50 MB corpus once; needs uv)
make validate

# Benchmarks (criterion + mdtraj/FreeSASA/Biopython baselines)
bench/run.sh
```

---

## CLI Reference

### 1. In-Silico Mutagenesis (`proteus mutate`)
Generate an Alanine scanning variant library:
```bash
proteus mutate scaffold.fasta --mode alanine --output alanine_library.fasta
```

Generate a site-saturation library restricted to residues 10–18:
```bash
proteus mutate scaffold.fasta --mode saturation --start 10 --end 18 --max-variants 50
```

### 2. High-Throughput Screening (`proteus screen`)
Screen a variant library through the compute funnel and export candidate biophysics to Apache Parquet:
```bash
proteus screen alanine_library.fasta \
  --tier fast \
  --workers 8 \
  --min-plddt 75.0 \
  --top 10 \
  --export results.parquet
```

Pipe mutations directly into the screening funnel without saving intermediate FASTA files:
```bash
proteus mutate wildtype.fasta --mode alanine | proteus screen - --export results.parquet
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
proteus view structure.pdb --backend halfblock --color ss --width 80 --height 36

# Native Kitty graphics protocol (kitty, wezterm, ghostty)
proteus view structure.pdb --backend kitty --width 100 --height 40
```

### 4. Offline Biophysical Analysis (`proteus analyze`)
Inspect all-atom biophysical metrics for any local PDB structure:
```bash
proteus analyze --pdb structure.pdb
```
Output (1CRN, crambin — an X-ray structure, so no pLDDT is reported):
```
┌──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
│ Biophysical Metric                       ┆ Value                                                             │
╞══════════════════════════════════════════╪═══════════════════════════════════════════════════════════════════╡
│ Radius of Gyration (Rg)                  ┆ 9.676 Å                                                           │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Contact Density (C-alpha <= 8Å)          ┆ 9.86%                                                             │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ pLDDT                                    ┆ n/a (experimental structure; B-factor column is not a confidence) │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Secondary Structure Composition          ┆ α-Helix: 47.8% | β-Strand: 8.7% | Coil: 43.5%                     │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Ramachandran Conformation                ┆ Favored: 97.7% | Allowed: 2.3% | Outliers: 0                      │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Solvent Accessible Surface Area          ┆ Total: 2973.4 Å² (Hydrophobic Burial: 92.8%)                      │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Heavy-atom steric overlap (>0.4 Å, no H) ┆ 0.0 per 1k atoms (0 overlaps)                                     │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Hydrogen Bonds (H-Bonds)                 ┆ 54 total (43 BB-BB, 10 BB-SC, 1 SC-SC)                            │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Ionic Salt Bridges (≤4.0Å)               ┆ 1 detected (closest: ARG17:NH2-GLU23:OE2 3.97Å)                   │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Aromatic π-π Stacking                    ┆ 0 conjugated pairs (0 parallel, 0 T-shaped)                       │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Cation-π Interactions                    ┆ 1 active interactions                                             │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Non-Covalent Network Density             ┆ 121.7 contacts / 100 res                                          │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Candidate Fitness Score                  ┆ 98.2 / 100                                                        │
└──────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘
```

### 5. Headless Daemon (`proteus serve`)
Run the headless background service:
```bash
proteus serve --port 8080 --host 0.0.0.0
```
Interactive Swagger UI documentation is served at `http://localhost:8080/swagger-ui`.

> The daemon has no authentication and, with a container socket available, runs the image named in each TES task. Bind it to localhost or put it behind an authenticating proxy — see [SECURITY.md](SECURITY.md).

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
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
