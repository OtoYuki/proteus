# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **`proteus analyze` takes many structures.** Files and directories (searched recursively
  for `.pdb`, `.ent`, `.cif`, `.mmcif`, optionally gzipped) are analysed in parallel (`-j`) into
  one row per structure: file, model, chain and residue counts, one-letter sequence, confidence
  source, pLDDT statistics (null for experimental structures), Rg with the folded-protein
  expectation and their ratio, DSSP fractions and string, Ramachandran, SASA, heavy-atom
  overlaps, interaction counts, optional RMSD to `--reference`, and the triage score.
  `--export` writes `.parquet` (tagged `proteus.qc_schema_version = 1`), `.csv` or `.json`;
  `--json` prints JSON Lines. An unreadable file is reported and makes the exit status
  non-zero without stopping the others. One file with neither flag still prints the full
  report; `--pdb` still works.

### Fixed
- **Claims corrected after a line-by-line fact audit** (every number re-measured, external
  facts re-checked at the source):
  - `proteus-dssp`: agreement with mdtraj was stated as "≥ 98 % on eight states"; the 98 %
    floor is for three states. Now: 99.6 % of 30 335 residues on eight states, 99.96 % on three,
    worst non-exempt file 97.8 %, one documented exemption. The docs said π-helices only fill
    unassigned residues; they may overwrite α (prefer-π, as mdtraj), which is what the code does.
  - `proteus-esm`: loading "any" `facebook/esm2_*` checkpoint from the Hub was wrong for 3B and
    15B, which publish no safetensors. "ESM-3 weights are non-commercial" is out of date — the
    open ESM3 and ESM C weights are MIT since mid-2026. The unsourced "weak on long multi-domain
    sequences" claim is replaced by ProteinGym's per-taxon numbers (human 0.457, virus 0.261).
    Parity is checked in CI for the 8M checkpoint only; the README had said both.
  - README: Kabsch RMSD was listed as validated against mdtraj (it is unit-tested only); corpus
    is 53 files / 48 entries; `make validate` installs ~650 MB of Python tools, not ~50 MB;
    crambin full profile is ~9 ms; the lateral π-offset test is PLIP's criterion, not
    McGaughey's; H-bonds use heavy-atom criteria compared against Baker–Hubbard rather than
    being Baker–Hubbard; Sixel's 6–21× is against Proteus's uncompressed kitty output; other
    terminal viewers do infer secondary structure, so that is no longer claimed as a difference.
  - Both crates now ship `LICENSE-MIT` and `LICENSE-APACHE` in the package.
- `view --html` and `--web` were described as a Mol\* page in `--help`, the README and the
  smoke test; the page has been 3Dmol.js since 0.5.0. The smoke assertion had kept passing
  only because a licence comment inside the vendored 3Dmol.js mentions molstar; it now checks
  for the page's own viewer call.

### Changed
- **Positioning corrected against a proper landscape search.** Three "only in Rust" claims were
  wrong, and are now stated accurately with the neighbours named and linked in the README:
  - [`molex`](https://github.com/foldit-org/molex) implements Kabsch–Sander DSSP in Rust.
    `proteus-dssp` is the 8-state one, standalone and dependency-free, and the one validated
    against mdtraj on real structures — not the only one.
  - [`esm-rs`](https://github.com/tcztzy/esm-rs) runs ESM on candle with CUDA/MLX backends.
    `proteus-esm` is aimed at variant-effect scoring rather than embeddings — not the only one.
  - [`planetary`](https://github.com/stjude-rust-labs/planetary) serves GA4GH TES from Rust on
    Kubernetes. Proteus is the single-binary, local-container-socket deployment — a different
    model, not a better one.
  Also: mdtraj validates its own DSSP against stored `mkdssp` references, so our harness is
  *unusually thorough*, not categorically different. The phrasing everywhere now reflects that.

### Fixed
- **Secondary structure in the render is now pinned to the coordinates, not the file's
  annotations.** Predicted structures carry no `HELIX`/`SHEET` records — an ESMFold response has
  none — so a viewer that reads them has nothing to read for exactly the files this tool exists
  to look at. Proteus already ran DSSP, but nothing stopped that regressing; a test now strips
  the annotations and asserts the per-class vertex counts are unchanged. Measured head-to-head,
  another terminal viewer's β-strand coverage nearly halves on the same pair of files where ours
  does not move at all, and on an ESMFold prediction of protein G it renders helix where the
  four-stranded sheet is.
- **The cartoon ribbon's flat face now follows the backbone instead of an arbitrary axis.**
  Frames came from pure parallel transport seeded on a fixed reference vector, so the ribbon
  was smooth and twist-free but its wide face bore no relation to the peptide planes — β-strands
  did not lie flat in their sheet and did not show the sheet's real twist. The wide axis is now
  the backbone carbonyl, flip-corrected per residue (Carson & Bugg 1986, the construction PyMOL,
  Mol\* and Chimera use), falling back to parallel transport for Cα-only traces that have no
  carbonyl. Checked against the **interaction network** rather than against the ribbon's own
  inputs: residues that are H-bond partners across a β-sheet should present near-parallel
  faces, and now do at a consistent **21–31°** across the corpus (the sheet's genuine twist),
  where parallel transport gave an erratic 25–84° depending on where its seed landed.
- **`proteus view` says when the viewport cannot resolve what you asked for.** Rendering 8 015
  residues into 80×24 cells gives 3.4 Å per pixel while consecutive Cα atoms are 3.8 Å apart —
  the picture is the fold's outline and nothing per-residue survives, and nothing said so. It
  now prints the Å-per-pixel and points at a larger terminal or a finer backend, and a test
  pins that the advice is true (braille really does resolve more than half-block at the same
  cell count, kitty more than braille).
- **Errors name the mistake instead of the symptom.** Handing a FASTA to a structure command
  produced pdbtbx's "No Atoms in the given PDB struct while validating", which tells a
  first-time user neither what they did nor what to do. It now says the file looks like a FASTA
  and points at `proteus submit`/`screen`; a file with no coordinates at all says so; and the
  mirror mistake — a PDB handed to a sequence command — points at `proteus analyze --pdb`.
  Both directions are tested.
- `proteus esm` prints, once, where zero-shot ESM-2 scores are known to be weak (long
  multi-domain sequences), because the published benchmarks say so and a reader should not have
  to find that in a paper after acting on a ranking. The README and the crate README carry the
  viral-protein caveat and state why ESM-2 rather than ESM-3 (ESM-3's weights are
  non-commercial; ESM C 300M is MIT and is the natural next checkpoint).
- **π–π stacking and cation–π were over-reported, and the counts change.** Neither test included
  a lateral-offset term, so two aromatic rings that were parallel and within the distance cutoff
  but slid sideways past each other counted as stacked, and a cation beyond the ring edge counted
  as sitting over its face. Measured against PLIP across 15 structures that was **20 % precision
  on π–π** (55 reported where PLIP finds 11) and 33 % on cation–π. Both now apply the McGaughey
  (1998) 2.0 Å offset criterion that PLIP uses: **π–π precision 20 % → 81.8 %** (11 reported,
  matching PLIP's 11) and **cation–π 33.3 % → 73.9 %** (51 → 23). Found by adopting the reference,
  not by inspection.
  **This changes reported counts, the non-covalent network density and therefore fitness scores**
  for structures with aromatics — 1CRN's network density goes 121.7 → 119.6 contacts/100 res.
  A unit test that asserted crambin has ≥ 1 aromatic interaction was itself wrong: PLIP finds
  none there, and the assertion now requires agreement with the reference.

### Added
- **A Sixel backend** (`--backend sixel`). Sixel reaches terminals the kitty protocol does not —
  xterm (`-ti vt340`), mlterm, foot, contour, WezTerm, Windows Terminal — and on several of them
  it is the only true-pixel path there is. It is also **6–21× cheaper on the wire** than kitty
  at the same resolution (1CRN 20.9×, 1PGB 16.9×, 1TEN 14.3×, 4HHB 6.2×), because kitty sends
  raw RGB while Sixel run-length encodes and a protein render is mostly background — which
  matters when the terminal is at the far end of an SSH session.
  The encoder is checked by **libsixel's own `sixel2png`**, not by our idea of the format: the
  round-trip must reproduce the framebuffer pixel for pixel, and `scripts/smoke.sh` decodes a
  real render with it on every run. Sixel is palette-indexed, so the 24-bit framebuffer is
  median-cut quantised to ≤ 256 colours; the cost is *measured* rather than assumed
  (`encode_with_stats` reports palette size and worst channel error — 0 for a cartoon render,
  which stays inside the budget).
- **The validation corpus grows 43 → 53**, chosen for coverage rather than count: crambin at
  0.54 Å (alternate conformations everywhere), Top7 (a de novo designed fold, which is what
  Proteus screens), collagen (polyproline II, which DSSP assigns to no canonical state), a
  membrane GPCR (where hydrophobic burial is inverted), an all-β domain, an intact IgG (insertion
  codes), cytochrome c (covalent heme), a zinc finger (fold held by a metal, not a core), an
  amyloid fibril (inter-chain β stacking) and GroEL/GroES (21 chains, ~58 000 atoms).
  Each entry carries a comment saying which failure mode it exists to catch.
  **Six of the ten failed on arrival**, which is the point:
  - An insertion-code bug **in the harness**: mdtraj's Python API drops insertion codes, so
    residues 52 and 52A both key as "52" and eight φ/ψ angles in 1IGT were compared against the
    wrong residue. The reference now lists colliding keys and the harness refuses to compare
    them, reporting how many it skipped.
  - Five documented convention differences, now recorded per structure in `tolerances.toml`
    with the exact checks exempted and the reason — and **printed on every run**, so an
    exemption can never quietly hide a regression the way a widened global tolerance would.
    In two of them Proteus is the stricter and more correct side: mdtraj's `is_protein` counts
    ACE acetyl caps, and counts a residue that has no Cα at all.
- **Salt bridges, π–π and cation–π have an external reference for the first time.** These were
  the rows in the README that said "no widely used reference implementation with the same
  definitions". [PLIP](https://github.com/pharmai/plip) is one, run in intra-chain mode over 15
  X-ray structures by `validate/plip_reference.py`, compared by residue pair with recall *and*
  precision, in `make validate` and therefore in CI. Observed: salt bridges **97.7 % precision**
  at 72 % recall (a stricter cutoff by design), π–π 81.8 / 81.8 %, cation–π 73.9 / 65.4 %.
- Every reported interaction now carries the **chain id** of both partners. `ARG17` in a
  four-chain structure was ambiguous; it is `A:ARG17` now, and it is what makes the per-chain
  PLIP comparison possible at all.
- **The composite fitness score is now measured, not just labelled.** It ranks everything
  `proteus screen` outputs and had no external check, because no reference implementation of a
  Proteus-defined weighted sum exists. `fitness_discrimination.rs` instead measures the claim
  the score actually makes — a folded structure outranks a broken one — on eight deposited
  X-ray structures against decoys built from each (coordinate noise at σ = 0.5/1.0/3.0 Å, a
  1.5× expansion, and an ideal poly-alanine helix, the shape the offline simulator emits).
  All 40 pairs separate, the score is monotone in the noise level, smallest margin 10.4/100.
  Runs in `make validate`, so in CI on every push.
- The README and `validate/README.md` now say what that does **not** mean: the score is a
  triage filter, not a predictor of experimental stability or activity. For sequence-level
  fitness the answer is `--scorer esm2`, whose ProteinGym numbers are measured separately.
- The ESM-2 parity tests run in CI. `esm2_t6_8m_matches_transformers` and
  `library_scoring_reuses_forward_passes` were `#[ignore]`d for want of checkpoints and nothing
  ran them; the 8M checkpoint is 31 MB, so the `validate` workflow caches it and checks the
  headline feature against `transformers` on every push instead of trusting committed JSON.

### Changed
- `proteus-cli` is one module per subcommand. `main.rs` was 1 654 lines with a 1 067-line
  `main()` holding every command body in one `match`; it is now 31 lines that parse and
  dispatch. Each subcommand owns its clap `Args` struct and its `run` in `src/cmd/<name>.rs`,
  the parser surface lives in `cli.rs`, the `inspect` table in `report.rs`, and the parser tests
  sit next to the commands they parse. The CLI surface is unchanged — every subcommand's
  `--help` output is byte-identical to before the split.
- **The browser page is self-contained and no longer uses Mol\*.** `proteus view --html/--web`
  and the daemon's `/view/{job}` embedded a `<script src="https://unpkg.com/molstar@3.30.0">`
  tag — a 2023 release, two majors behind, fetched from a CDN at open time. That page did not
  work on an HPC login node, an airgapped cluster or a plane, which is where Proteus is meant
  to be used, and it tied a scientific artifact to a third party's uptime. The viewer is now
  3Dmol.js 2.5.5 (BSD-3, 525 KB vs Mol\*'s 4.9 MB) vendored into the binary, so the page is a
  single file that opens offline. Mol\* remains the better tool for interactive analysis and
  the structure file is always on disk for it; Proteus's page shows one structure, one
  representation, one colouring, and none of Mol\*'s machinery was used.
- The page is rendered by one module (`proteus_core::webview`) instead of two near-identical
  copies in `proteus-server` and `proteus-cli`, and **carries Proteus's own DSSP** rather than
  the viewer's built-in guess: on 1CRN the browser now shows the assignment that agrees with
  `mdtraj.compute_dssp` on 46/46 residues, where 3Dmol.js's heuristic misses the 3₁₀ helix at
  42–44. The web page can no longer disagree with `analyze`, the terminal viewer or the export.

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
