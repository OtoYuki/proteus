# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.5.0] — 2026-09-22

The correctness-and-provenance release. Every structure now says where it came from and whether
it ran at the tier you asked for; the interaction network, the task listing, the Nextflow path
and the terminal renderer had real bugs fixed, each with a regression test. MSRV is 1.94 and
every dependency is at its current major.

**Upgrading:** fitness scores change for non-compact models (the compactness term was ~30 % too
generous); Parquet exports are `schema_version = 4` and carry an `engine` column; `--executor
host` is refused off loopback; building from source needs Rust 1.94.

### Added
- Exports carry an `engine` column (`esmfold-api`, `oci`, `simulated`); `schema_version = 4`.
  `proteus inspect` shows the engine. Every runner records `metadata.engine`.
- `/metrics`: the task-duration and biophysics histograms and the CAS counters are now driven
  by engine events (`BiophysicsAnalyzed`, `CasStored`); `proteus_task_queue_depth` and the CAS
  `read` series, which nothing ever wrote, are gone.
- `proteus submit --wait=false` enqueues and returns.
- The auto runner records `tier_requested` / `tier_honoured` / `fallback_reason` in the
  prediction metadata. `proteus inspect` shows a Tier row, `proteus screen` warns when ranked
  structures did not run at the requested tier (e.g. `--tier sota` with no Boltz image ran the
  ESMFold API), and the viewers' titles carry the engine.
- `proteus-esm`: `MarginalScorer` caches the wild-type (and per-position masked) forward passes,
  so scoring a variant library is one forward pass for wild-type marginals instead of one per
  variant; `proteus screen --scorer esm2` on 875 variants takes 0.6 s instead of minutes.
- The terminal viewer opens on the model's principal-axis frame (longest axis across the
  screen, viewer looking down the shortest) and fits the oriented extents to the viewport;
  `r` resets to that frame.
- `scripts/smoke.sh`: an assertion-based end-to-end check of the release binary (analyze,
  mutate|screen, exports, job lifecycle, every viewer back-end, daemon auth, TES listing; with
  `--tes IMAGE` a container task). Runs in CI. Replaces the two narrated demo scripts.
- Every crate has a README (crates.io landing page); `docs/design/README.md` indexes the
  design records and states where the shipped code differs from each.
- Install instructions corrected: the CLI crate README said `cargo install proteus-cli`, which
  would install an unrelated package — that name, and `proteus-engine`, are taken on crates.io.
  The binary installs from the GitHub releases, the ghcr image, or `cargo install --git`;
  `proteus-dssp` and `proteus-esm` are the crates intended for reuse and package cleanly
  (`cargo package`) under names that are free.

### Changed
- **MSRV 1.88 → 1.94.** Dependencies at their current majors: bollard 0.18 → 0.21
  (query-parameter API), pdbtbx 0.11 → 0.12 (`ReadOptions`), reqwest 0.12 → 0.13 (`rustls` +
  `webpki-roots` features), sqlx 0.8 → 0.9 (drops `rsa` from the lockfile, so the
  RUSTSEC-2023-0071 audit ignore is gone), nalgebra 0.33 → 0.35 (drops eleven `glam`
  versions from the graph), comfy-table 7 → 8, crossterm 0.28 → 0.29, criterion 0.5 → 0.8,
  tower-http 0.6 → 0.7, base64 0.22 → 0.23, toml 0.8 → 1. `make validate`, the container
  executor and `scripts/smoke.sh` were re-verified on the new versions.
- Compactness term of the fitness score recalibrated to the empirical folded-protein law
  `Rg ≈ 2.2·N^0.38 Å` (was `2.82·N^0.392`, ~30 % too wide, which scored every model ≤ 1.4× the
  folded Rg as fully compact). Full credit ≤ 1.10×, none ≥ 2.0×. **Fitness scores change** for
  non-compact models; compact ones are unaffected.
- Prediction-tier container images are configurable (`PROTEUS_IMAGE_FAST|SOTA|RELAX`) and
  documented as bring-your-own; the Boltz tier writes Boltz-format FASTA (`>A|protein|empty`);
  the relax tier is refused up front instead of failing inside the container.
- H-bond validation covers all six hydrogen-bearing corpus entries (was four) and reports
  precision (58–76 %) next to recall (86–100 %), with floors on both.

### Fixed
- `proteus screen --runner auto` ranked the offline simulator's placeholder helices as if
  they were predictions whenever the ESMFold API was unreachable. Simulated structures are now
  excluded from the leaderboard and export with a warning, unless `--runner simulated` is
  explicit, in which case the leaderboard is labelled.
- `proteus analyze --reference` failed with a coordinate-length mismatch when the reference
  carried alternate conformations or a calcium ion named `CA`; the reference is normalised
  like the query.
- The Prometheus collector stopped counting for good after the first broadcast lag; cancelling
  a running TES task decremented `proteus_active_workers` twice.
- A cancel that arrived while a TES task was `INITIALIZING` could be overwritten by the worker's
  next state write and the task ran to `COMPLETE`. The cancel token is now registered before
  the first write and non-terminal writes are refused once the row is `CANCELED`.
- `proteus inspect` labelled secondary structure "P-SEA"; it is DSSP.
- A `nan`/`inf` coordinate in a PDB or mmCIF atom record panicked inside the parser; it is now a
  parse error naming the line.
- The OCI runner derived the rootless-Podman socket path from `$UID`, which shells do not
  export; it now reads `/proc/self/status` like the TES executor. Containers are removed
  explicitly after `wait` instead of relying on `AutoRemove`.
- `proteus-esm` refuses configs with `emb_layer_norm_before = true` instead of loading them
  and producing wrong logits.
- `validate/corpus.toml`: 1L2Y is NMR and 1LB5 is X-ray (labels were swapped).
- `bench/README.md`: the φ/ψ row is labelled as mdtraj's per-call Python API, not a C++ kernel;
  `bench/render.py` no longer discards the hand-written sections when regenerating.
- CI workflows run with a read-only `GITHUB_TOKEN`.
- Interaction network: the "bonded neighbour" exclusions compared residue numbers without the
  chain, so an inter-chain contact between equally numbered residues (A5–B5) was dropped; the
  exclusions are now chain-aware (#2).
- TES `GET /v1/tasks` fetched and deserialised every task on each poll; `state` and
  `name_prefix` filters and paging now run in SQL (`%`/`_` in a prefix are literals). Tag
  filters still scan the state/prefix-filtered rows.
- TES output URLs given as bare absolute paths (what Nextflow's nf-ga4gh plugin sends) passed
  validation but failed at delivery with "output URL scheme not supported", so every Nextflow
  task ended in `SYSTEM_ERROR` after running. Bare paths are accepted for outputs as they
  already were for inputs, under the same `--allow-dir` check.
- C-alpha-only models (the simulated runner's output, coarse-grained traces) rendered as an
  empty frame: with no C/N atoms every residue counted as a chain break. Continuity now falls
  back to the CA–CA distance (≤ 4.2 Å). Sub-pixel-thin geometry that straddled a pixel boundary
  was also skipped by the rasteriser; each triangle now lights at least its centroid pixel.
- The recorded demo printed a hard-coded "43/43 structures within tolerance" line; it now
  shows the harness's own result line. Doc comments and help text no longer claim "60 FPS",
  ">10 GB/s" or "SOTA"; they say what the code does.
- `examples/nextflow`: the config pinned `nf-ga4gh@0.3.0` (never published; current is 1.5.0)
  and an endpoint with `/v1`, which the plugin appends itself, and the pipeline never invoked
  proteus. It now runs `proteus mutate`/`proteus analyze` in the proteus container through TES.
  README no longer claims the Nextflow example runs in CI (only the Sprocket one does).

## [0.4.0] — 2026-09-22

The TES-hardening release: executors run in their container image, the API takes a bearer
token, `file://` access is confined to `--allow-dir`, and ESM-2 inference ships in pure Rust.

### Added
- **`proteus-esm`: ESM-2 masked-LM inference in pure Rust** (candle) — Hub or local safetensors
  checkpoints, wild-type/masked marginal mutation scores, 20×L deep mutational scans. Parity with
  `transformers` pinned by tests (logits ≤ 1e-2, amino-acid log-probs ≤ 5e-3). ProteinGym v1.1
  Spearman ρ mean |ρ| 0.42 (35M) / 0.24 (8M) on five small assays.
- Hydrogen-bond network validated against `mdtraj.baker_hubbard` on the four NMR structures
  that carry explicit hydrogens: 94.9–100 % recall of mdtraj's non-local bonds, checked by
  `make validate`.
- `proteus esm score|scan`; `proteus screen --scorer esm2|hybrid` with an `esm2_score` export
  column (schema_version 3) and a terminal DMS heat map.
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

- `--allow-dir DIR` (repeatable): `file://` input/output URLs must resolve inside an allowed
  host directory (default: the artifacts directory); advertised as `proteus.file_allowlist`.

### Changed
- `--executor host` (the 0.3.x behaviour) is refused unless bound to loopback.
- Tasks with relative paths, `..` components, empty images or mounts over system directories
  are rejected (400); executor `stdout`/`stderr` paths are validated like every other path.

### Fixed
- **Security:** `inputs[].path` (and every other task path) could contain `..` and escape the
  task work dir on the host; `file://` URLs could read and write any host path the daemon can.
- A task whose output failed to upload to its declared URL was reported `COMPLETE`; it is now
  `SYSTEM_ERROR`, as is an output that resolves outside the work dir through a symlink.
- A task whose input could not be staged (missing `file://` source, unsupported scheme) stayed
  `INITIALIZING` forever; it now ends in `SYSTEM_ERROR` with the reason in `system_logs`.
- `proteus analyze`/`screen` panicked (`index out of bounds` in the SASA cell list) on
  structures whose bounding box exceeds ~800 Å per axis; the grid now widens its cells instead.

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

[Unreleased]: https://github.com/OtoYuki/proteus/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/OtoYuki/proteus/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/OtoYuki/proteus/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OtoYuki/proteus/compare/v0.1.0-thesis...v0.3.0
