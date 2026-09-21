# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **TES executors run inside their container image** (bollard; Podman/Docker socket): argv
  overrides the image entrypoint, declared paths are bind-mounted from the task work dir,
  `cpu_cores`/`ram_gb` become container limits, network off by default, per-executor timeout,
  cancellation kills the container, missing images are pulled (`--no-pull` to forbid).
- `--auth-token` / `PROTEUS_AUTH_TOKEN` bearer authentication (constant-time) on the TES and
  native APIs; `--allow-image GLOB` allow-list advertised in `service-info.tags`.
- GA4GH TES 1.1 compliance: `tag_key`/`tag_value` filters, `page_token` pagination,
  `backend_parameters_strict`, idempotent cancel on finished tasks, `size_bytes` as a string.
  The ELIXIR/GA4GH suite passes 23/23 and runs in CI (`tes-conformance.yml`).
- Outputs are delivered to their declared `file://` URLs (files and directories).
- `examples/wdl/`: Sprocket (WDL) → TES → proteusd → container, verified in CI.
- `--executor host|container`, `--executor-timeout`, `--executor-network`.

### Changed
- `--executor host` (the 0.3.x behaviour) is refused unless bound to loopback.
- Tasks with relative paths, empty images or mounts over system directories are rejected (400).

## [0.3.0] — 2026-09-21

The scientific-correctness release. Every biophysical metric is now checked against a reference
implementation on a 43-structure corpus in CI (`make validate`), and the numbers in the README are
the ones the binary prints.

### Fixed
- **Backbone dihedral sign was inverted**: every φ/ψ was negated, so helices classified as
  left-handed and 1CRN reported 21 Ramachandran outliers (MolProbity: 0). Pinned to mdtraj.
- Secondary structure over-assignment (P-SEA approximation reported 2 % coil on crambin; DSSP: 43 %).
- B-factors of experimental structures were reported as pLDDT and fed into the fitness score.
- Alternate conformations were all kept (duplicate atoms inflated SASA); waters, ions, ligands and
  hydrogens were included in metrics; NMR ensembles analysed every model as one structure.
- Headerless predictor PDB files (ESMFold, ColabFold, Boltz) failed to parse; deposited files with
  malformed `SEQADV` records (e.g. 1TIM) were rejected.
- `RUST_LOG` is honoured; logs go to stderr.

### Added
- `proteus-dssp`: standalone pure-Rust Kabsch–Sander DSSP crate (DSSP 2.x / mdtraj semantics).
- MolProbity Top8000 Ramachandran evaluation from the cctbx `rama8000` contour grids (BSD-3),
  replicating `ramalyze` thresholds; six residue classes incl. cis/trans-Pro and Ile/Val.
- `ConfidenceSource` provenance detection (predicted vs experimental); `proteus analyze
  --confidence-source`.
- mmCIF and gzip input everywhere (`proteus_core::io::open_structure`).
- `validate/`: reference harness (mdtraj, FreeSASA, cctbx) with committed reference values,
  tolerances as a contract, and a CI job.
- `bench/`: criterion benchmarks plus measured mdtraj / FreeSASA / Biopython baselines and a
  rendered table.
- CI: Linux/macOS + MSRV test matrix, fmt/clippy/rustdoc gates, cargo-deny, cargo-audit, coverage.
- Release workflow: Linux/macOS binaries (x86_64, aarch64) and a `ghcr.io` image on tags.
- `LICENSE-MIT`, `LICENSE-APACHE`, `CONTRIBUTING.md`, `SECURITY.md`, this changelog.
- Terminal demo GIF (`docs/media/`).

### Changed
- `ClashStats` → `StericOverlapStats`, `clashscore` → `heavy_atom_overlap_score`; documented as a
  heavy-atom approximation, **not** the MolProbity clashscore. Export `schema_version = 2`.
- SASA default sphere points 96 → 960 (mdtraj parity; ≤ 0.4 % difference on the corpus).
- Fitness score: pLDDT weight redistributed for experimental structures; "Pareto" wording dropped
  (it is a weighted sum).
- `reqwest` uses rustls (no OpenSSL at build time). MSRV 1.88. `indicatif` 0.18.
- `PROTEUS_DATA_DIR` overrides the data directory.

## [0.2.0] — 2026-09-20

Rust rewrite of the Django/Celery thesis prototype: six-crate workspace, GA4GH TES v1.1 daemon,
BLAKE3 CAS, Parquet export, software terminal rasterizer, DMS screening funnel.

## [0.1.0-thesis]

Original Python/Django thesis implementation (git tag `v0.1.0-thesis`).

[Unreleased]: https://github.com/OtoYuki/proteus/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/OtoYuki/proteus/compare/v0.1.0-thesis...v0.3.0
