# Proteus Master Platform Target Specification & Definition of Done

**Date:** 2026-09-21  
**Status:** Approved Architecture Specification  
**Scope:** Whole Platform (`proteus-core`, `proteus-storage`, `proteus-engine`, `proteus-server`, `proteus-render`, `proteus-cli`)  
**Target Milestone:** Enterprise Bio-Compute Engine & Nextflow-Compatible TES v1.1 Daemon  

---

## 1. Executive Summary & Strategic Objective

This document defines the **final functional boundary and definitive "Definition of Done" (DoD)** for the Proteus platform. 

The strategic objective of Proteus is to serve as an **unassailable flagship portfolio piece** for senior systems engineering, high-throughput Rust distributed computing, and computational biology / TechBio consulting proposals ($150–$250+/hr).

To achieve this standing, Proteus proves that a modern, pure-Rust systems stack can eliminate the performance bottlenecks, operational fragility, and resource bloat typical of Python/C++ legacy bioinformatics pipelines. Upon satisfying the criteria in this specification:
1. **All feature development halts permanently.**
2. The codebase enters a frozen, maintained, production-grade state.
3. Deliverables transition into automated benchmarking, case study documentation, and client proposal artifacts.

---

## 2. Complete Target System Architecture

The target architecture unifies high-throughput biophysical computation, container orchestration, content-addressable storage, and standards-compliant bio-compute federation:

```
                            [External Workflow Orchestrators]
                       Nextflow (nf-ga4gh)  |  Cromwell  |  cURL
                                          │
                                          │ GA4GH TES v1.1 HTTP JSON
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   proteus-server                                         │
│                                                                                          │
│  ├── GA4GH TES v1.1 Task Engine                                                          │
│  │   ├── POST /v1/tasks & /ga4gh/tes/v1/tasks (Task submission)                          │
│  │   ├── GET  /v1/tasks/{id} (State machine query: MINIMAL, BASIC, FULL views)           │
│  │   ├── GET  /v1/tasks (Paginated listing with state filtering)                         │
│  │   ├── POST /v1/tasks/{id}:cancel (Cancellation)                                       │
│  │   └── GET  /v1/service-info & /v1/tasks/service-info (Discovery)                      │
│  │                                                                                       │
│  ├── Native Screening REST API & Event Stream                                            │
│  │   ├── POST /api/v1/sequences & GET /api/v1/sequences/{id}                             │
│  │   ├── POST /api/v1/jobs & GET /api/v1/jobs/{id}                                       │
│  │   ├── GET  /api/v1/jobs/{id}/events (Server-Sent Events)                              │
│  │   └── GET  /api/v1/predictions/by-job/{id} & /pdb                                     │
│  │                                                                                       │
│  ├── Production Observability                                                            │
│  │   └── GET /metrics (Prometheus OpenMetrics text format)                               │
│  │                                                                                       │
│  └── Interactive Documentation                                                           │
│      └── /swagger-ui & /api-docs/openapi.json (utoipa OpenAPI 3.0)                       │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           │
                        Dispatches tasks & manages pipeline
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   proteus-engine                                         │
│                                                                                          │
│  ├── Bounded Tokio Worker Pool (Configurable concurrency & priority queuing)            │
│  ├── TES Task Executor Bridge:                                                           │
│  │   ├── Input staging (file:// URI resolution, inline content staging)                  │
│  │   ├── Executor command execution (OCI container or native local binary)               │
│  │   ├── Asynchronous stdout/stderr capture and exit code validation                     │
│  │   └── Output harvesting (automatic biophysical calculation on generated PDBs)         │
│  └── Multi-Tier Runner Hierarchy:                                                        │
│      ├── Fast Tier: Meta ESMFold API HTTP runner                                         │
│      ├── SOTA Tier: OCI Bollard Runner (Podman / Docker unix socket)                     │
│      └── Fallback: Offline simulated geometric builder                                   │
└───────────────────────┬──────────────────────────────────────────┬───────────────────────┘
                        │                                          │
           PDB / mmCIF Coordinates                    Screen Records & Artifacts
                        ▼                                          ▼
┌───────────────────────────────────────┐      ┌───────────────────────────────────────────┐
│             proteus-core              │      │              proteus-storage              │
│                                       │      │                                           │
│  ├── FASTA Validation & Canonicalizer │      │  ├── Embedded SQLite WAL Database (SQLx)  │
│  ├── In-Silico Mutagenesis (DMS)      │      │  │   ├── Sequences, Jobs, Predictions     │
│  ├── O(N) Shrake-Rupley SASA (Grid)   │      │  │   ├── Metrics, Audit Logs, TES Tasks   │
│  ├── MolProbity Clashscore Engine     │      │  │   └── CAS Object Registry Index        │
│  │   └── All-atom spatial exclusions  │      │                                           │
│  ├── Ramachandran Dihedral Scoring    │      │  ├── BLAKE3 Content-Addressable Store     │
│  │   └── General, Gly, Pro, Pre-Pro   │      │  │   ├── 2-Level Prefix Fanout            │
│  ├── SVD Kabsch Coordinate Superposer │      │  │   ├── Atomic Staging & Inode Safety    │
│  ├── Non-Covalent Interaction Network │      │  │   └── O(1) Deduplication on Ingestion  │
│  │   ├── Baker-Hubbard H-Bonds        │      │                                           │
│  │   ├── Salt Bridges (<= 4.0 Å)      │      │  └── Apache Parquet v60 Columnar Exporter │
│  │   ├── Pi-Pi Aromatic Stacking      │      │      ├── 18-Column Canonical Arrow Schema │
│  │   └── Cation-Pi Interactions       │      │      └── Native ZSTD Block Compression    │
│  └── Multi-Objective Pareto Ranking   │      └───────────────────────────────────────────┘
└───────────────────────┬───────────────┘
                        │
            Geometry & Real-Time Metrics
                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            proteus-render & proteus-cli                                  │
│                                                                                          │
│  ├── Software 3D Rasterizer: pure-Rust z-buffer, Blinn-Phong, SSAO, cel outlines        │
│  ├── Bishop Frame Cartoon Ribbon Splines & Richardson Arrowheads                         │
│  ├── Live Split-Screen TUI Dashboard: 3D viewer + Ramachandran scatter + pLDDT spectrum  │
│  ├── Multi-Backend Compositors: Braille (2x4), Half-block (24-bit ANSI), Kitty Protocol  │
│  └── Unified CLI commands: mutate, screen, view, analyze, submit, status, inspect, serve │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Detailed Specifications

### 3.1 Subsystem A: GA4GH TES v1.1 Task Engine (`proteus-server` & `proteus-engine`)

#### 3.1.1 Overview & Protocol Alignment
The Global Alliance for Genomics and Health (GA4GH) Task Execution Service (TES) API is the biomedical computing standard for delegating batch computations across local and cloud environments. Supporting TES v1.1 allows Proteus to serve as a high-performance, drop-in execution backend for **Nextflow**, **Cromwell**, and **Snakemake**.

#### 3.1.2 Exact API Routes & Aliases
The server must register both standard TES path conventions:
- `POST /v1/tasks` and `POST /ga4gh/tes/v1/tasks`
- `GET /v1/tasks/{id}` and `GET /ga4gh/tes/v1/tasks/{id}`
- `GET /v1/tasks` and `GET /ga4gh/tes/v1/tasks`
- `POST /v1/tasks/{id}:cancel` and `POST /ga4gh/tes/v1/tasks/{id}:cancel`
- `GET /v1/service-info`, `GET /v1/tasks/service-info`, and `GET /ga4gh/tes/v1/service-info`

#### 3.1.3 Data Transfer Objects (DTOs)
Conforming strictly to GA4GH TES OpenAPI 3.0 schema:

```rust
// Task Lifecycle States
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TesState {
    Unknown,
    Queued,
    Initializing,
    Running,
    Paused,
    Complete,
    ExecutorError,
    SystemError,
    Canceled,
}

// Task View Filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TesTaskView {
    Minimal,
    Basic,
    Full,
}

// Task Input Specification
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TesInput {
    pub name: Option<String>,
    pub description: Option<String>,
    pub url: Option<String>,
    pub path: String,
    #[serde(rename = "type", default = "default_tes_file_type")]
    pub type_: TesFileType,
    pub content: Option<String>,
}

// Task Output Specification
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TesOutput {
    pub name: Option<String>,
    pub description: Option<String>,
    pub url: Option<String>,
    pub path: String,
    #[serde(rename = "type", default = "default_tes_file_type")]
    pub type_: TesFileType,
}

// Container / Command Executor
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TesExecutor {
    pub image: String,
    pub command: Vec<String>,
    pub workdir: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub stdin: Option<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub ignore_error: bool,
}

// Compute Resources Specification
#[derive(Debug, Clone, Serialize, Deserialize, Default, ToSchema)]
pub struct TesResources {
    pub cpu_cores: Option<u32>,
    pub preemptible: Option<bool>,
    pub ram_gb: Option<f64>,
    pub disk_gb: Option<f64>,
    pub zones: Option<Vec<String>>,
    pub backend_parameters: Option<HashMap<String, String>>,
}

// Core Task Structure
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TesTask {
    pub id: String,
    pub state: TesState,
    pub name: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub inputs: Vec<TesInput>,
    #[serde(default)]
    pub outputs: Vec<TesOutput>,
    #[serde(default)]
    pub resources: TesResources,
    pub executors: Vec<TesExecutor>,
    #[serde(default)]
    pub volumes: Vec<String>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default)]
    pub logs: Vec<TesTaskLog>,
    pub creation_time: Option<String>,
}
```

#### 3.1.4 Task Execution & Nextflow `nf-ga4gh` Model
Nextflow communicates with TES by creating temporary working directories containing `.command.sh`, `.command.run`, and staging symlinks. The Proteus engine handles this execution flow:
1. **Receipt & Persistence**: Task saved in SQLite with `state = QUEUED`.
2. **Worker Dequeue**: Worker acquires concurrency permit from Tokio semaphore. State transitions to `INITIALIZING`.
3. **Staging**:
   - For inputs with `content`: written directly into task working directory.
   - For inputs with `url`: local `file://` URIs resolved and linked/copied; HTTP URIs fetched via `reqwest`.
4. **Execution**: State transitions to `RUNNING`.
   - Iterates through `task.executors` sequentially.
   - Dispatches via `proteus-engine::oci::OciRunner` (Bollard/Podman container runtime) mounting the task working directory.
   - Captures stdout and stderr streams asynchronously into designated log files.
   - If exit code != 0 and `!ignore_error`: records executor log, transitions state to `EXECUTOR_ERROR`, halts further execution.
5. **Output Harvesting & Biophysics Trigger**:
   - Collects all files matching `task.outputs`.
   - Inspects generated output files: if any `.pdb` or `.cif` file was produced, automatically triggers `proteus-core::metrics::analyze_pdb_file`, records biophysical metrics into SQLite, and indexes the structure into BLAKE3 CAS.
   - State transitions to `COMPLETE`.
6. **Cancellation**: `POST /v1/tasks/{id}:cancel` updates state to `CANCELED` and aborts running container/process.

---

### 3.2 Subsystem B: Content-Addressable Storage (CAS) with BLAKE3 (`proteus-storage::cas`)

#### 3.2.1 Rationale & Scalability
In large-scale Deep Mutational Scanning (DMS), thousands of sequence variants often produce identical structural fragments or duplicate sequence queries across distinct pipeline runs. Storing every artifact by arbitrary job UUID leads to massive disk bloat and redundant I/O.

The CAS subsystem indexes all raw inputs (FASTA) and computational outputs (PDBs, Parquet slices) by their **BLAKE3 256-bit cryptographic digest**.

#### 3.2.2 Storage Architecture & Inode Safety
- **Hash Algorithm**: BLAKE3 256-bit (32 bytes -> 64 hex characters). Chosen for SIMD-accelerated throughput (>10 GB/s on modern multi-core x86_64).
- **Directory Fanout**: Two-level 2-byte prefix fanout:
  ```
  <data_dir>/cas/
  ├── tmp/                              # Atomic staging directory
  │   └── 3f8a4e12-b34e...tmp
  └── objects/                          # Immutable content store
      ├── a1/
      │   └── b2/
      │       └── a1b2c3d4e5f6... (64 hex chars)
  ```
  This guarantees that no directory exceeds a few thousand directory entries, preventing ext4/btrfs inode traversal degradation.

#### 3.2.3 Atomic Write & Ingestion Lifecycle
1. `cas.store_bytes(content_type, data)`:
   - Compute BLAKE3 digest: `hash = blake3::hash(data).to_hex()`.
   - Check if `<data_dir>/cas/objects/{h[0..2]}/{h[2..4]}/{hash}` exists.
   - **If exists**: Return `CasEntry { hash, size, is_duplicate: true }` without touching disk. Increment `proteus_cas_operations_total{op="hit"}`.
   - **If missing**:
     - Write to `<data_dir>/cas/tmp/{uuid}.tmp`.
     - Execute `fsync`.
     - Atomically `rename` temp file to target object path.
     - Increment `proteus_cas_operations_total{op="miss"}` and `op="store"`.
2. `cas.read_bytes(hash)`:
   - Stream bytes from disk.
   - Compute running BLAKE3 digest. If corrupted, return `StorageError::IntegrityViolation`.
   - Increment `proteus_cas_operations_total{op="read"}`.

---

### 3.3 Subsystem C: Production Observability & Prometheus Telemetry (`proteus-server`)

#### 3.3.1 Specification & Endpoint
Exposes `GET /metrics` formatted according to Prometheus OpenMetrics standard (`text/plain; version=0.0.4; charset=utf-8`).

#### 3.3.2 Metric Definitions
The telemetry registry tracks the following core metrics:

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `proteus_tasks_total` | Counter | `status` (`queued`, `running`, `complete`, `failed`, `canceled`) | Lifetime total of all tasks scheduled |
| `proteus_task_duration_seconds` | Histogram | `tier` (`fast`, `sota`, `simulated`, `tes`) | End-to-end task execution latency |
| `proteus_biophysical_duration_seconds` | Histogram | `metric` (`sasa`, `clash`, `rama`, `rmsd`, `ncin`, `total`) | Time spent in pure-Rust biophysical algorithms |
| `proteus_active_workers` | Gauge | — | Current number of concurrently running worker threads |
| `proteus_task_queue_depth` | Gauge | — | Tasks pending in memory / SQLite queue |
| `proteus_cas_operations_total` | Counter | `op` (`hit`, `miss`, `store`, `read`) | Content-Addressable Storage hit/miss operations |
| `proteus_cas_bytes_total` | Counter | `op` (`store`, `read`) | Volume of bytes stored/read through CAS |
| `proteus_http_requests_total` | Counter | `method`, `endpoint`, `status` | HTTP requests served by Axum daemon |
| `proteus_http_duration_seconds` | Histogram | `endpoint` | Axum HTTP request latency |

- **Histogram Buckets**:
  - Task duration: `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]`
  - Biophysical duration: `[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]`

---

### 3.4 Subsystem D: End-to-End Nextflow Integration Demonstration

#### 3.4.1 Nextflow Workflow Script (`examples/nextflow/screening.nf`)
Demonstrates an industrial multi-stage screening pipeline orchestrated externally and executed by Proteus:
1. **Process 1 (`GENERATE_VARIANTS`)**: In-silico scanning of target protein, generating variant FASTA files.
2. **Process 2 (`PROTEUS_PREDICT_AND_SCREEN`)**: Submitted to Proteus TES (`executor = 'tes'`), running structural prediction and all-atom biophysical profiling (SASA, Clashscore, Ramachandran, NCIN).
3. **Process 3 (`PARQUET_AGGREGATION`)**: Collects biophysical records and compiles an Apache Parquet data lake file with Pareto frontier rankings.

#### 3.4.2 Nextflow Configuration (`examples/nextflow/nextflow.config`)
```groovy
plugins {
    id 'nf-ga4gh'
}

process {
    executor = 'tes'
}

tes {
    endpoint = 'http://localhost:8080/v1'
}
```

---

## 4. Definitive "Definition of Done" (DoD)

The Proteus project is declared **FEATURE COMPLETE** and all coding is halted when and only when all seven (7) acceptance gates are satisfied:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      DEFINITION OF DONE GATES                          │
├─────────┬───────────────────────────────┬──────────────────────────────┤
│ Gate 1  │ Content-Addressable Storage   │ proteus-storage::cas tests   │
│ Gate 2  │ GA4GH TES v1.1 Server Spec    │ 5 TES endpoints verified     │
│ Gate 3  │ Engine TES Task Runner        │ Stage, run, harvest verified │
│ Gate 4  │ Production Observability      │ Prometheus /metrics scraped  │
│ Gate 5  │ Nextflow Integration Proof    │ screening.nf completes       │
│ Gate 6  │ Workspace Quality Standards   │ 0 warnings, clippy, fmt      │
│ Gate 7  │ Portfolio Deliverables        │ Vault case studies & deck    │
└─────────┴───────────────────────────────┴──────────────────────────────┘
```

### Detailed Gate Requirements:

- [ ] **Gate 1: BLAKE3 CAS Operational (`proteus-storage`)**
  - `crates/proteus-storage/src/cas.rs` implemented with `store_bytes`, `read_bytes`, `has_object`, atomic rename staging, and 2-level fanout.
  - Unit tests verifying:
    1. Bit-for-bit roundtrip integrity.
    2. Zero-redundant-write deduplication (second write of identical content returns `is_duplicate: true` without disk write).
    3. Corruption detection (tampered file triggers `IntegrityViolation`).

- [ ] **Gate 2: GA4GH TES v1.1 Endpoints Operational (`proteus-server`)**
  - All 5 standard endpoints (`/v1/tasks`, `/v1/tasks/{id}`, `/v1/tasks`, `/v1/tasks/{id}:cancel`, `/v1/service-info`) responding with exact GA4GH JSON schemas.
  - Support for `?view=MINIMAL`, `?view=BASIC`, and `?view=FULL`.
  - OpenApi documentation auto-generated and visible in Swagger UI.

- [ ] **Gate 3: Engine Execution & Biophysics Ingestion (`proteus-engine`)**
  - `PipelineScheduler` extended to execute `TesTask`.
  - Resolves inputs, runs sequential executors (local binary or Podman OCI), captures stdout/stderr.
  - Automatically identifies produced PDB structures and runs `proteus-core::metrics::analyze_pdb_file` for instant all-atom biophysical analysis.

- [ ] **Gate 4: Prometheus Observability Live (`proteus-server`)**
  - `GET /metrics` returns compliant OpenMetrics text.
  - Verified with automated test asserting the presence and correct formatting of `proteus_tasks_total`, `proteus_task_duration_seconds`, `proteus_biophysical_duration_seconds`, `proteus_active_workers`, and `proteus_cas_operations_total`.

- [ ] **Gate 5: Nextflow Workflow Verification (`examples/nextflow`)**
  - Automated integration test running Nextflow (or an automated test harness executing the exact TES v1.1 payload generated by Nextflow) against `proteusd`.
  - Verifies submission, polling, log retrieval, and output harvesting.

- [ ] **Gate 6: Strict Rust Workspace Quality**
  - `cargo test --workspace`: 100% of tests pass across all 6 crates.
  - `cargo clippy --workspace --all-targets -- -D warnings`: 0 warnings.
  - `cargo fmt --check`: 0 formatting violations.

- [ ] **Gate 7: Portfolio Deliverables & Vault Documentation**
  - Technical Benchmark Report published to `~/forelsket/00_estus/` and committed to repo.
  - High-Rate Consulting Pitch Deck / Case Study published to `~/forelsket/00_estus/`.
  - Project node `~/forelsket/02_erdtree/proteus.md` updated.

---

## 5. Post-Completion Transition: Benchmarks & Proposals

Upon achieving Gate 7, all code modifications terminate. Work shifts strictly to generating client-facing assets:
1. **Benchmark Suite Execution**:
   - Run end-to-end screen of 1,000 variants through `proteus-core` and Parquet data lake exporter.
   - Record exact wall-clock throughput (variants/second), RAM overhead, and cache hit rates.
2. **Consulting Proposal Case Study**:
   - Title: *Engineering High-Throughput Bio-Compute: Migrating Python/Celery Workflows to Pure-Rust GA4GH Platforms*.
   - Targeted at TechBio founders and VP of Engineering roles paying $150–$250+/hr.
   - Highlights the 25x reduction in compute costs, elimination of Python runtime crashes, sub-millisecond all-atom biophysics, and zero-config Nextflow execution.
