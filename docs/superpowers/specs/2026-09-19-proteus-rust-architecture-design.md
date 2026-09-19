# Architecture Design: Proteus-rs (High-Throughput Bio-Compute Orchestrator)

- **Date:** 2026-09-19
- **Status:** Approved
- **Target:** Transition Proteus from 2024–2025 Django thesis to modern, high-throughput Rust bio-compute pipeline and orchestrator for contractor portfolio evidence.

---

## 1. System Context & High-Level Architecture

Proteus is an asynchronous bio-compute engine designed to orchestrate state-of-the-art computational biology tasks: single-sequence fast folding (ESMFold), high-fidelity biomolecular structure prediction (Boltz-1 / ColabFold), molecular dynamics physics validation (GROMACS), and native SIMD-accelerated structural analysis in Rust.

```
                  ┌──────────────────────────────────────────────┐
                  │                 proteus-cli                  │
                  │   (Terminal client: submit, status, bench)   │
                  └──────────────────────┬───────────────────────┘
                                         │ HTTP / SSE
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          proteus-server (proteusd)                             │
│       (Axum REST API, SSE Live Progress Streaming, utoipa OpenAPI docs)        │
└────────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
┌──────────────────────────────────┐            ┌──────────────────────────────────┐
│          proteus-engine          │            │         proteus-storage          │
│  - Async DAG Task Scheduler      │            │  - SQLite in WAL Mode            │
│  - bollard OCI Container Runner  │            │  - SQLx compile-time queries     │
│  - Podman / Docker socket detect │            │  - Embedded migrations           │
│  - Mock runner for offline tests │            │  - Repository abstractions       │
└────────────────┬─────────────────┘            └──────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                  proteus-core                                  │
│  - Domain Models: Sequence, PipelineJob, StructureOutput, BiophysicalMetrics  │
│  - FASTA Parser & Validator (canonical 20 AA verification)                     │
│  - Biophysical Engine (`pdbtbx` + `nalgebra`):                                 │
│      * Center-of-mass Radius of Gyration (Rg)                                  │
│      * Kabsch C-alpha RMSD alignment                                           │
│      * Residue-level pLDDT confidence distribution                             │
│      * C-beta contact map and density matrix                                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Workspace Crate Layout

A Cargo workspace containing 5 targeted crates:

### 2.1 `crates/proteus-core`
- **Purpose:** Pure domain models, zero-dependency validation, and mathematical/biophysical calculations.
- **Dependencies:** `uuid`, `serde`, `thiserror`, `pdbtbx`, `nalgebra`.
- **Key Modules:**
  - `sequence`: Amino acid validation, FASTA streaming/parsing.
  - `job`: Typed job definitions, priorities, states.
  - `metrics`: Center of mass, Radius of Gyration ($R_g$), Kabsch RMSD alignment algorithm, contact map matrices, pLDDT statistics.

### 2.2 `crates/proteus-storage`
- **Purpose:** Embedded, zero-configuration local persistence.
- **Dependencies:** `sqlx` (features: `sqlite`, `runtime-tokio`, `uuid`, `chrono`), `proteus-core`.
- **Database:** SQLite with WAL enabled (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`).
- **Migrations:** Embedded via `sqlx::migrate!`.

### 2.3 `crates/proteus-engine`
- **Purpose:** Asynchronous DAG task orchestration and OCI container management.
- **Dependencies:** `bollard`, `tokio`, `futures-util`, `proteus-core`, `proteus-storage`, `tracing`.
- **Execution Model:**
  - `ComputeRunner` trait abstraction.
  - `OciRunner`: Connects to Podman (`/run/user/1000/podman/podman.sock`) or Docker (`/var/run/docker.sock`) via `bollard`. Mounts isolated directories, injects GPU flags when available, and streams logs.
  - `SimulatedRunner`: Deterministic local runner for testing and offline development without needing Docker/Podman or external model downloads.

### 2.4 `crates/proteus-server`
- **Purpose:** Headless daemon (`proteusd`) exposing APIs and real-time feeds.
- **Dependencies:** `axum`, `tower-http`, `utoipa`, `utoipa-swagger-ui`, `tokio`, `proteus-core`, `proteus-engine`, `proteus-storage`.
- **Endpoints:**
  - `POST /api/v1/sequences` — Submit sequence.
  - `POST /api/v1/jobs` — Enqueue a pipeline job.
  - `GET /api/v1/jobs/{id}` — Get job status.
  - `GET /api/v1/jobs/{id}/events` — SSE stream of job progress and container output.
  - `GET /api/v1/predictions/{id}` — Retrieve structural output.
  - `GET /api/v1/metrics/{id}` — Retrieve biophysical metrics.
  - `GET /health` — Liveness & readiness check.

### 2.5 `crates/proteus-cli`
- **Purpose:** Polished developer and researcher CLI tool (`proteus`).
- **Dependencies:** `clap` (derive), `indicatif`, `comfy-table`, `reqwest`, `proteus-core`.
- **Commands:**
  - `proteus submit --file input.fasta --tier fast|sota|full`
  - `proteus status <job-id>`
  - `proteus inspect <prediction-id>`
  - `proteus analyze --pdb <file.pdb>` (direct offline analysis via `proteus-core`)
  - `proteus serve --port 8080` (starts the background daemon)

---

## 3. Biophysical Analysis Algorithms in Rust

Replaces the missing Python MLRanking with verified mathematical algorithms:

1. **Radius of Gyration ($R_g$):**
   $$R_g = \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{r}_i - \mathbf{r}_{\text{cm}}\|^2}$$
   where $\mathbf{r}_{\text{cm}} = \frac{1}{N}\sum_{i=1}^N \mathbf{r}_i$ computed over all $C_\alpha$ atoms.

2. **Kabsch Algorithm (C-alpha RMSD):**
   Given coordinate sets $P$ and $Q$ centered at origin:
   - Covariance matrix $H = P^T Q$.
   - SVD decomposition $H = U \Sigma V^T$.
   - Rotation matrix $R = V \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(V U^T) \end{pmatrix} U^T$.
   - $\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|R \mathbf{p}_i - \mathbf{q}_i\|^2}$.

3. **Contact Density:**
   Fraction of non-consecutive residue pairs ($|i - j| \ge 4$) whose $C_\beta$ ($C_\alpha$ for Gly) distance is $\le 8.0 \text{ \AA}$.

---

## 4. Verification & Quality Gates

- **Compiler & Lints:** Zero warnings on `cargo clippy --workspace --all-targets -- -D warnings`.
- **Formatting:** Clean check on `cargo fmt --check`.
- **Tests:** `cargo test --workspace` covers:
  - FASTA parser edge cases and validation failures.
  - Numerical parity of $R_g$ and Kabsch RMSD against reference structures.
  - SQLite migrations, inserts, queries, transactions.
  - Mock engine pipeline execution end-to-end.
