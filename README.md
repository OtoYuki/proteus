# Proteus: High-Throughput Bio-Compute Orchestrator & Structural Pipeline

[![Rust](https://img.shields.io/badge/Rust-1.94+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Cargo%20Workspace-green.svg)](docs/superpowers/specs/2026-09-19-proteus-rust-architecture-design.md)

**Proteus is an asynchronous, high-throughput bio-compute pipeline and orchestration engine written in Rust (`proteus-rs`). It orchestrates structural biology computational workloads across tiered models (ESMFold, Boltz-1, ColabFold, GROMACS MD) via native OCI/Podman container sockets, persists execution state in embedded SQLite (WAL mode), and calculates biophysical descriptors ($R_g$, Kabsch RMSD, contact density, pLDDT distribution) in microseconds using SIMD-accelerated linear algebra.**

Originally conceptualized as an undergraduate thesis project, Proteus has evolved into a production-grade systems engineering demonstration for high-performance bio-compute infrastructure.

---

## Key Features (`proteus-rs`)

* **⚡ Native Biophysical Analytics:** Direct Rust computation of **Center-of-Mass Radius of Gyration ($R_g$)**, **Kabsch optimal superposition $C_\alpha$ RMSD** via SVD (`nalgebra`), **tertiary contact topology**, and **per-residue pLDDT statistics** using `pdbtbx`. Runs in sub-milliseconds without Python runtime or GIL bottlenecks.
* **🐳 Direct OCI / Podman Container Management:** Talks directly to Podman rootless sockets (`/run/user/1000/podman/podman.sock`) or Docker sockets via `bollard` over UNIX domain sockets. No fragile shell scripts or unvalidated subprocess execution.
* **📦 Zero-Config Embedded Storage:** Embedded SQLite database with SQLx running in WAL mode with connection pooling and embedded migrations. Works out of the box with zero external infrastructure setup (no PostgreSQL or RabbitMQ required).
* **🔄 Deterministic Fallback (`SimulatedRunner`):** Automatically detects if a container runtime socket is active; if offline, falls back gracefully to a deterministic local simulation runner for reproducible testing and CI environments.
* **🌐 Headless Daemon (`proteusd`):** High-concurrency Axum server exposing RESTful endpoints, live progress streaming via Server-Sent Events (SSE), and interactive Swagger UI at `/swagger-ui`.
* **💻 Rich Terminal Tooling (`proteus-cli`):** Production-grade CLI built with `clap`, `comfy-table`, and `indicatif` progress spinners for batch submission, job status polling, and offline PDB inspection.

---

## Workspace Architecture

```
proteus/
├── Cargo.toml                  # Workspace root manifest
├── crates/
│   ├── proteus-core/           # Domain models, canonical FASTA validation, biophysical calculations (pdbtbx)
│   ├── proteus-storage/        # Embedded SQLite with SQLx, embedded DDL migrations, repository abstractions
│   ├── proteus-engine/         # Async DAG task scheduler, OCI container runner (bollard), simulated runner
│   ├── proteus-server/         # Headless Axum daemon (proteusd), SSE streams, OpenAPI docs (utoipa)
│   └── proteus-cli/            # CLI binary (proteus) with rich table output & progress indicators
├── docs/
│   └── superpowers/specs/      # Architectural specifications & technical design documents
└── core/, proteus/             # Legacy Django/Celery thesis implementation (archived for provenance)
```

---

## Local Quickstart

### 1. Prerequisites
* **Rust toolchain:** 1.85+ (tested on Rust 1.94 on Arch Linux)
* **Container Runtime (Optional):** Podman (`podman.socket`) or Docker for live container inference.

```bash
# Optional: Enable Podman user socket for live OCI container execution on Arch/Linux
systemctl --user enable --now podman.socket
```

### 2. Build Release Binaries
```bash
cargo build --workspace --release
```
The unified binary will be located at `./target/release/proteus`.

### 3. Run Test Suite & Quality Checks
```bash
# Run 12 unit & integration tests across all workspace crates
cargo test --workspace

# Run strict Clippy linter
cargo clippy --workspace --all-targets -- -D warnings

# Check code formatting
cargo fmt --check
```

---

## CLI Usage

### Submit a Sequence to the Bio-Compute Pipeline
```bash
./target/release/proteus submit --fasta ">test_insulin\nGIVEQCCTSICSLYQLENYCN" --tier fast
```
**Output:**
```
Sequence validated: 'test_insulin' (21 residues)
Job created: c74a7bdd-b19f-43ba-bd4d-43eeaa279497
Pipeline completed successfully!
┌───────────────────────────┬───────────────────────────────────────────────────┬───────────────────────────┐
│ Metric                    ┆ Value                                             ┆ Confidence Assessment     │
╞═══════════════════════════╪═══════════════════════════════════════════════════╪═══════════════════════════╡
│ Predicted PDB Path        ┆ .../artifacts/.../c74a7bdd..._predicted.pdb       ┆ Artifact on disk          │
│ Global Confidence (pLDDT) ┆ 82.50                                             ┆ Confident (Good backbone) │
│ Radius of Gyration (Rg)   ┆ 9.369 Å                                           ┆ Compactness metric        │
│ Tertiary Contact Density  ┆ 11.11%                                            ┆ C-alpha <= 8.0Å pairs     │
│ High Conf Residues (>=70) ┆ 100.0%                                            ┆ Reliable backbone         │
└───────────────────────────┴───────────────────────────────────────────────────┴───────────────────────────┘
```

### Query Existing Job Status & Inspection
```bash
# Query job lifecycle status
./target/release/proteus status <job-id>

# Inspect detailed biophysical breakdown
./target/release/proteus inspect <job-id>
```

### Direct Offline PDB Structure Analysis
Perform instant mathematical structure analysis on any PDB file without running a server or database:
```bash
./target/release/proteus analyze --pdb /path/to/structure.pdb
```

### Launch Headless API Server (`proteusd`)
```bash
./target/release/proteus serve --port 8080
```
Open **http://localhost:8080/swagger-ui** in your browser to interact with the OpenAPI documentation.

---

## Mathematical Biophysics Formulations in Rust

### 1. Center of Mass & Radius of Gyration ($R_g$)
$$\mathbf{r}_{\text{cm}} = \frac{1}{N}\sum_{i=1}^N \mathbf{r}_i \quad\text{over all } C_\alpha\text{ atoms}$$
$$R_g = \sqrt{\frac{1}{N}\sum_{i=1}^N \|\mathbf{r}_i - \mathbf{r}_{\text{cm}}\|^2}$$

### 2. Kabsch Optimal Superposition Algorithm ($C_\alpha$ RMSD)
Given centered coordinate sets $P$ and $Q$:
* Covariance matrix: $H = P^T Q$
* Singular Value Decomposition (SVD): $H = U \Sigma V^T$
* Optimal rotation matrix: $R = V \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(V U^T) \end{pmatrix} U^T$
* Optimal RMSD: $\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|R \mathbf{p}_i - \mathbf{q}_i\|^2}$

---

## Historical Context: 2024–2025 Thesis Implementation

The original thesis implementation was written in Python 3.12 using Django 4.2, Celery 5.4, PostgreSQL, RabbitMQ, and Mol*Star. Application code resides in [`core/`](core/) and [`proteus/`](proteus/). For historical execution:
```bash
# Set up Python virtualenv
pip install -r requirements.txt
python manage.py migrate
celery -A proteus worker -l info
python manage.py runserver
```

---

## Documentation & Architecture Specs

* **Architecture Specification:** [`docs/superpowers/specs/2026-09-19-proteus-rust-architecture-design.md`](docs/superpowers/specs/2026-09-19-proteus-rust-architecture-design.md)
* **Obsidian Vault Reports:**
  * [`00_estus/2026-09-19-proteus-rust-architecture-design.md`](file:///home/s1re/forelsket/00_estus/2026-09-19-proteus-rust-architecture-design.md)
  * [`00_estus/2026-09-19-proteus-state-and-contracting-evolution.md`](file:///home/s1re/forelsket/00_estus/2026-09-19-proteus-state-and-contracting-evolution.md)
  * [`00_estus/2026-09-19-proteus-commercial-valuation-systems-rust.md`](file:///home/s1re/forelsket/00_estus/2026-09-19-proteus-commercial-valuation-systems-rust.md)
