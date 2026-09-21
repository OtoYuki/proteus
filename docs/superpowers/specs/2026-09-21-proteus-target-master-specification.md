# Proteus Master Platform Target Specification & Definition of Done

**Date:** 2026-09-21  
**Status:** Approved  
**Author:** Proteus Core Team  
**Scope:** Whole Platform (`proteus-core`, `proteus-storage`, `proteus-engine`, `proteus-server`, `proteus-render`, `proteus-cli`)

---

## 1. Executive Summary & Objective

This specification establishes the **final boundary and Definition of Done** for the Proteus platform. The primary goal is to establish an unassailable flagship portfolio piece for senior systems engineering and high-throughput Rust / TechBio consulting proposals ($150–$250+/hr).

Upon satisfying the criteria in this specification:
1. All core feature development stops permanently.
2. The codebase enters a maintenance, documentation, benchmark, and publication state.
3. Deliverables shift to architectural case studies, performance benchmarks, and client proposal decks.

---

## 2. Current vs. Target Completion Matrix

| Subsystem | Components | Current State | Target State |
| :--- | :--- | :--- | :--- |
| **`proteus-core`** | SASA, Ramachandran, Clashscore, RMSD, Non-Covalent Networks (NCIN), Pareto Ranking | **100% Complete** (sub-millisecond $O(N)$ spatial grid) | **Feature Complete** (Freeze) |
| **`proteus-render`** | Software 3D rasterizer, Bishop ribbons, SSAO, cel outlines, Braille/Kitty protocols, split-screen TUI | **100% Complete** (pure Rust) | **Feature Complete** (Freeze) |
| **`proteus-storage`** | SQLite WAL, 18-col Parquet data lake exporter | **90% Complete** | **Add BLAKE3 Content-Addressable Storage (CAS)** |
| **`proteus-server`** | Axum daemon, REST API, SSE streaming, Swagger UI | **75% Complete** | **Add GA4GH TES v1.1 Task Engine & Prometheus Observability** |
| **`proteus-engine`** | Multi-tier runner (Simulated, ESMFold API, OCI Bollard) | **85% Complete** | **Bridge TES task execution to engine runners** |
| **`proteus-cli`** | Mutate, screen, view, analyze, submit, status, inspect | **95% Complete** | **Nextflow `-with-tes` verified workflow demonstration** |

---

## 3. The 3 Final Engineering Pillars

To complete the target specification, three final subsystems will be implemented:

### Pillar 1: GA4GH TES v1.1 Task Execution Service (`proteus-server` & `proteus-engine`)
- Implements standard GA4GH Task Execution Service (TES v1.1) endpoints:
  - `POST /v1/tasks`: Enqueue execution task.
  - `GET /v1/tasks/{id}`: Query task lifecycle state (`QUEUED`, `INITIALIZING`, `RUNNING`, `COMPLETE`, `EXECUTOR_ERROR`, `SYSTEM_ERROR`, `CANCELED`) with logs and outputs.
  - `GET /v1/tasks`: Paginated task listing with state filters.
  - `POST /v1/tasks/{id}:cancel`: Task cancellation.
  - `GET /v1/service-info`: GA4GH service discovery metadata.
- Enables direct drop-in integration with industry-standard bio-compute workflow engines: **Nextflow** (`nextflow run ... -with-tes http://localhost:8080/v1`) and **Cromwell**.

### Pillar 2: Production Observability & Prometheus Telemetry (`proteus-server`)
- Exposes `GET /metrics` formatted for Prometheus scrapers.
- Metrics emitted:
  - `proteus_tasks_total{status}` (Counter)
  - `proteus_task_duration_seconds_bucket` (Histogram)
  - `proteus_biophysical_calc_duration_seconds` (Histogram across SASA, Clash, NCIN)
  - `proteus_active_workers` (Gauge)
  - `proteus_cas_hits_total` / `proteus_cas_misses_total` (Counters)

### Pillar 3: Content-Addressable Storage (CAS) with BLAKE3 Deduplication (`proteus-storage::cas`)
- Content-addressable object store indexing input FASTA sequences and output structural artifacts by BLAKE3 256-bit hashes.
- Fast local filesystem fan-out (`<data_dir>/cas/objects/ab/cd/abcdef...`).
- Eliminates duplicate computation and redundant storage during massive variant screening.

---

## 4. Definition of Done & Success Criteria

The project is declared **DONE** when the following 7 conditions are satisfied:

1. **GA4GH TES v1.1 Compliance**: All 5 standard TES endpoints functional and conformant to the official GA4GH JSON schema.
2. **Prometheus Telemetry**: Live `/metrics` endpoint scrapable by Prometheus with accurate task and biophysical metrics.
3. **BLAKE3 CAS Storage**: Verified deduplication of input and output artifacts with zero redundant disk writes.
4. **End-to-End Nextflow Verification**: An automated integration test running a Nextflow pipeline (`examples/nextflow/screening.nf`) against Proteus via `-with-tes`.
5. **Quality Standard**: `cargo test --workspace` passes 100%, `cargo clippy --workspace --all-targets -- -D warnings` has 0 warnings, `cargo fmt --check` passes.
6. **Performance Benchmarks**: Documented benchmark report demonstrating screening throughput and sub-millisecond all-atom biophysics.
7. **Portfolio & Proposal Artifacts**: Complete technical case study and consulting proposal pitch document published in `~/forelsket`.
