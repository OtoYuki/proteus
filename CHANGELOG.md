# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.8.0] — 2026-09-23

The release where Proteus gets a face.
- **Browser viewer:** the page is now ours. WebGL2 draws the same ribbon, colours and
  measurements as the terminal, verified in Chromium, Firefox and WebKit, and it replaces the
  vendored 3Dmol.js.
- **Home screen:** `proteus` with no command opens one in a terminal, with your jobs, a folder
  of structures measured as you move, and forms that show the command they run.
- **Identity:** everything wears one identity, a s1re.sh sub-brand. The structure colours now
  stay apart for colour-blind readers.
- **Bug hunt:** a review of all of this before release fixed 26 defects. Among them, a closed
  terminal could leave `proteus` spinning at full CPU, and structures with disulfide bonds drew
  an empty browser page.

### Added
- A home screen. `proteus` with no command, in a terminal, opens a full-screen TUI (ratatui)
  with three tabs:
  - **Jobs:** your jobs, which no command could list before. Refreshed every 2 s; opens the
    viewer, the browser page or the full report.
  - **Structures:** a file browser that measures the structure file you stop on, with the same
    numbers as `analyze`.
  - **Run:** forms for `submit` and for a `mutate | screen` scan.

  Every action runs `proteus` itself and the forms show the command line first. Outside a
  terminal, bare `proteus` behaves exactly as before: usage on stderr, exit 2.
- `ProteusRepository::list_jobs` and `proteus_core::qc::summary_rows`. The browser page and the
  home screen format measurements with the same function.

### Changed
- Proteus has an identity, as a sub-brand of s1re.sh: a mark (a chain of residues that folds
  into a **p**), a dot-matrix wordmark, and a palette with roles for dark and light grounds,
  defined once in `proteus_render::brand` and tested for contrast (WCAG) and for colour-blind
  separation. The assets are in `docs/brand/`.
- The home screen and the terminal 3-D viewer wear the identity:
  - The launch folds the chain into the mark.
  - Hairline sections and `(named)` panels.
  - The `~ $` prompt as the status line.
  - Colours at the terminal's depth (24-bit, 256, 16, or none under `NO_COLOR`).
  - A job state is always a glyph and a word.

  The viewer's status bar shows a legend for the current colours. On an experimental file it
  says the pLDDT scheme is the B-factor column, not a confidence.
- The browser viewer page wears the identity:
  - The mark and dot-matrix wordmark in the header.
  - The structure's name as the heading.
  - `(named)` hairline panels.
  - Ramachandran points with a shape per region.
  - A legend that names the AlphaFold colours, or says a B-factor is not a confidence.
  - Geist Mono and Figtree embedded (SIL OFL 1.1), so it looks the same offline.
- Brand assets and CLI output:
  - The README opens with the lockup (light and dark).
  - A 1280 × 640 social card is in `docs/brand/`.
  - Progress bars, spinners and the `--compare` summary use the brand colours at the terminal's
    depth, and none under `NO_COLOR` or in a pipe.
- The dashboard's pLDDT strip uses the AlphaFold colours. It had four colours of its own, so
  the strip and the ribbon beside it disagreed. Without colour, the strip shows the bands as
  shade density.
- `view --compare` draws the target in Tide and the reference in Clay (was cyan and ruby).
- Secondary structure is now coloured Clay (helix), Tide (strand) and pale cream (coil), and
  disulfides Chartreuse, in the terminal and in the browser. The smallest difference between
  of the three secondary-structure colours as seen with red–green colour blindness rose from ΔE 23 to 32
  (Machado 2009 simulation).
- The browser viewer (`proteus view --web/--html`, the daemon's `/view/{job}`) is now our own:
  about 800 lines of our own JavaScript (WebGL2) drawing the same ribbon mesh, DSSP and colours as the terminal
  viewer, with the same keys (`c`, `o`, `d`, `Space`, `r`, plus `s` to save a PNG), hover labels
  (chain, residue, 8-state DSSP, pLDDT or B-factor), and panels for the measurements, a
  Ramachandran plot and the per-residue pLDDT. It replaces the vendored 3Dmol.js 2.5.5, which
  drew a different cartoon from the terminal, rendered some twisted strands black, and put
  525 KB into every page. The page needs WebGL2 and `DecompressionStream` (Chrome 80, Firefox
  113, Safari 16.4 or later) and still loads nothing from the network. It no longer embeds the
  structure file; that stays next to the page or at `/api/v1/predictions/by-job/{id}/pdb`.

### Fixed
- Found by a review of everything since 0.7.0 (four reviewers, each finding reproduced) and a
  WebKit run.

  **Browser page:**
  - A structure with a disulfide bond (1CRN, for one) drew an empty 3-D view. A variable was
    shadowed; this came in with the identity foundation. A real-browser test now runs in CI
    (`scripts/browser-check.mjs` in Chromium, Firefox and WebKit), and it fails on that page.
  - Hover labels named residues that do not exist in WebKit, the engine behind Safari. WebKit
    misread the integer residue attribute, `uint` shader constants of 65536 and above, and the
    blue and alpha bytes of the pick target. Picking now uses float arithmetic and the red and
    green bytes only.
  - The page had no keyboard rotate or zoom. It now has arrows/hjkl and `+`/`−`, as in the
    terminal.
  - GPU targets leaked on every resize.
  - The legend overlapped the key help.

  **Home screen:**
  - Closing the terminal could leave `proteus` spinning at full CPU and ignoring SIGTERM:
    crossterm retries a dead tty inside `event::poll`. The signal handler now stops any child,
    restores the terminal and exits by itself.
  - `q` could hang on an analysis still running, for good on a FIFO. It now exits at once, and
    only regular files are listed.
  - An escape sequence in a file name reached the terminal through the echoed command. Control
    characters are now quoted as `$'\xNN'`.
  - Measurements went stale when a file changed; they are now keyed to its modification time
    and size.
  - Typing on a focused Run field without Enter ran commands (`q` quit, digits switched tabs).
    Typing now edits; `F1` opens the help from a field.
  - The jobs filter did not match the "done" it displays.
  - Esc did not quit; it now quits when nothing is open.
  - A one-line `>header SEQ` record folded header words into the sequence, and rejected a
    lower-case sequence.
  - `SIGTERM` was ignored while a child command ran.
  - The jobs list and `view`/`inspect` could disagree about which prediction a job has. Both
    now use the newest.

  **Terminal colour:**
  - In 16 colours, helix, strand and the `--compare` target and reference all turned the same
    grey. Data colours now go by hue.
  - kitty, ghostty, alacritty, foot and wezterm fell back to 16 colours over SSH, which does not
    pass `COLORTERM` on. They are now known by `TERM`, as are `-direct` and `truecolor` names.
  - `TERM=dumb` received escape codes.
  - In 16 colours, the empty part of a progress bar was invisible black.

  **Dashboard:**
  - Rows were cut mid-number ("coil 4" for 43 %). Whole items are now dropped instead.
  - The Ramachandran φ = 0 tick sat one column left of its axis (older than 0.7.0).
  - A name with an emoji sequence could make a line one column too wide.
- `proteus view --web` always ran `xdg-open`. On macOS, which has no `xdg-open`, the failure was
  ignored and no browser opened. It now uses `open` on macOS and says so when no opener can be
  started.
- `proteus esm scan`'s heat map was one shade of red with wild-type marginals (almost every score
  is below 0, and the colour scale was fixed around 0). The scale is now centred on the scan's own
  median and spread, printed in the legend, so blue marks the substitutions this protein
  tolerates best.
- `inspect`, `status`, `analyze` and `esm` tables were not fitted to the terminal and hard-wrapped
  in narrow windows; every table now uses the same fitting as `screen` (wrap inside cells on a
  terminal, never wrap into a pipe).
- The Prometheus HELP text says that `proteus_tasks_total` counts transitions into each state,
  not tasks currently in it.
- `tower-http`'s unused `cors` feature is no longer enabled.

## [0.7.0] — 2026-09-23

The bug-hunt release, and a security release for anyone running `proteus serve`. Five
independent code reviews of 0.6.0 reported 75 defects; each was reproduced before it was fixed
and has a regression test, and a second review of the fixes found eight regressions, also fixed.
**If you run the TES daemon, upgrade**: in 0.6.0 a client could read and write host files
through a chosen task id, a container could overwrite host files through its stdout path, and
any web page could submit tasks to a daemon on localhost. Also: ESM-2 now matches
`transformers` to within 5·10⁻⁵ at every tested length (a rotary-frequency bug grew to 7·10⁻² at 1022
residues); malformed structure files are errors, not crashes; AlphaFold models are no longer
classed experimental by a substring match.

### Security
A code-level bug hunt before publishing the crates (five independent reviews: engine and
server, core science, CLI and storage, renderer, ESM-2) found the following in the TES server.
Each has a regression test that reproduces the reported attack.
- **A client-chosen task `id` became a host path.** `id` names the task's work dir, so
  `"id": "<abs path>"` or `"../.."` let inputs, outputs and volumes read and write anywhere
  the daemon's user could, and reusing an id gave a 500. The id is now always server-assigned,
  as TES 1.1 specifies; client `logs` and `creation_time` are ignored too.
- **An executor could write any host file through its stdout/stderr path** by planting a
  symlink there (`ln -s ~/.bashrc /data/log.txt`). These files are now created fresh inside
  the work dir; a planted directory link fails the task.
- **Symlinks inside copied directories were followed**, on the way out (a DIRECTORY output
  holding `leak.txt -> /etc/…` delivered the host file) and on the way in (a directory input
  from an allowed dir). A symlink below the top level now fails the copy.
- **`Access-Control-Allow-Origin: *`** let any web page in the operator's browser submit tasks
  to a daemon on localhost. The CORS layer is gone.
- A task tag `proteus.network=true` turned container networking on without the operator's
  `--executor-network`; the network is now the operator's decision only.
- Captured stdout/stderr was unbounded (200 MB of output took the daemon to 859 MB and wrote a
  200 MB row); it is capped at 8 MiB per stream with the drop logged.
- The bearer token is compared as fixed-length digests, so its length no longer shows in
  timing.

### Changed
- **`proteus view --compare` pairs residues instead of positions**: by identity (chain ID,
  residue number, insertion code) when the files share a numbering, by residue number when
  each is a single chain with a different chain ID (a predicted model against a deposited
  entry), and by sequence alignment when the numbering differs; the pairing that matches the
  most identical residues wins and is named in the summary. Only paired residues are
  superposed; fewer than three is an error. The summary gives the pair count against each structure's total, is green only when
  the sequences match residue for residue, and warns what differs otherwise. `--color` and
  `--dashboard`, which `--compare` silently ignored, are now rejected with it.
  `render_superposition_snapshot` returns `SuperpositionStats` rather than a bare RMSD.
- `--width`/`--height` accept 1–4096 cells; the renderer also refuses a framebuffer over
  2²⁵ pixels (`RenderError::InvalidViewport`).

### Fixed
- **TES lifecycle.** A cancel answered 200 could be overwritten by COMPLETE; a terminal state
  is now written once. Early exits and mid-task I/O or database errors left tasks non-terminal
  and the worker gauge raised; every path now ends in one terminal state and one event. A daemon
  restart left tasks RUNNING and their containers running; they are marked SYSTEM_ERROR at
  start-up and their labelled containers removed. `stdin` piped the path string instead of the
  file's contents. The OCI runner left a container behind when it failed to start.
- **TES conformance.** `POST /v1/tasks/{id}` without `:cancel` cancelled the task (now 405);
  MINIMAL carried `resources`/`executors` and BASIC carried input `content`; `view` was
  case-sensitive on single tasks with a plain-text 400; an unknown `state` filter returned an
  empty 200 and an out-of-range `page_token` the first page (both 400 now); `PREEMPTED` and
  `CANCELING` were missing.
- **Native jobs** that failed after starting (unwritable work dir, database error) stayed
  Running with an open event stream; they now fail. `submit --wait=false` jobs were never run by
  anything; they are now marked Pending and a running `proteus serve` picks them up
  (`--queue-workers`, default 2, at a time). Jobs a `screen` or `submit --wait` inserted belong
  to that process and are never taken, and every start claims its job atomically.
- **One daemon per data directory.** `proteus serve` takes a lock on the data directory and
  binds its port before closing out tasks a previous run left unfinished, so a second daemon
  can no longer mark the first one's live tasks SYSTEM_ERROR.
- **Malformed structure files crashed the process.** pdbtbx panics on an mmCIF `?` in a
  required field and on multi-byte characters in PDB records; both are now parse errors.
- **Hostile chain ids ran as script** in `view --html`/`--web` and `/view/{job}`
  (`</script><svg onload=…>` in an mmCIF `auth_asym_id`). JS escaping now `\u`-escapes
  everything outside a small safe set.
- **Confidence source.** "NMR" matched inside protein names ("NmrA-like"), hiding pLDDT 96.6 on
  real AlphaFold DB models; `_exptl.method` beyond the first 16 KiB of an mmCIF was never read;
  a header-less ESMFold model with low confidence was classed experimental (which raised its
  score); a declared `--confidence-source predicted` skipped the 0–1 rescale. Detection now reads
  machine-readable provenance from the whole file first and matches words as words.
- **A blank element column turned `CA` into calcium**, dropping every residue (element symbols
  read from atom names are now corrected in the standard amino acids only, so mercury in CMH
  or bromine in a modified residue is kept); deuterium was kept as a heavy atom; ATOM lines cut
  after the B-factor were refused; an SSBOND or CONECT record naming a residue no longer in the
  file made the whole file unreadable (bond records are not needed and are skipped).
- **Hydrogen bonds.** Proline's N was counted as a donor and a Ser/Thr/Tyr hydroxyl pair was
  counted twice. Against mdtraj, precision on 1D3Z rises 76.1 → 78.5 % and on 1L2Y
  63.2 → 66.7 %; 2L3B recall falls 97.1 → 95.7 % (one bond is now kept in the other direction).
- Median pLDDT of an even count took the upper middle value. `--max-variants` counted the wild
  type and overshot (0 gave one or two sequences). FASTA upper-casing let `ß` and `ı` through as
  residues, and a literal `\n` in a multi-line file was turned into a line break. A C-alpha-only
  trace scored 0 on the Ramachandran term instead of the neutral baseline. A ligand named `NAN`
  was refused as a non-finite coordinate; coordinates past 4.3e9 wrapped silently.
- **CLI.** A closed stdout (`| head -1`) panicked, and `analyze` lost its `--export`; a closed
  stdout now ends the process quietly (status 141) and the export is written first. SIGPIPE
  itself stays ignored, so a closed executor pipe cannot take down `proteus serve`. `screen --scorer esm2` ranked
  unscored entries above every scored one. `analyze` walked a symlink loop 41 times, aborted on
  one unreadable subdirectory, dropped every row when `--reference` had another length, and
  checked the export target only after the work. Export extensions were case-sensitive (and
  `x.CSV` got JSON). `--host localhost` and `::1` were refused; a `?` or `%` in the data
  directory broke the database URL; the progress bar never moved; piped leaderboards wrapped
  job ids over two lines.
- **ESM-2 scores drifted from `transformers` with sequence length.** `proteus-esm` recomputed
  the rotary inverse frequencies exactly; the checkpoints store them rounded to fp16, and that
  is what the model was trained with and what `transformers` loads. At 1022 residues the 8M
  model's amino-acid log-probabilities were off by up to 0.07 (0.1 on a random sequence); the
  2.5e-3 on the short parity proteins, documented as fp32 accumulation noise, was the same bug.
  They are now read from the checkpoint. Parity is 3.6e-5 in logits and 1.3e-5 in amino-acid
  log-probabilities on the short proteins, 4.6e-5 / 3.3e-5 on a new 1022-residue fixture
  (`validate/esm_reference.py --long`); tolerances are tightened from 1e-2 / 5e-3 to 2e-4 / 1e-4.
- **Mutation positions counted whitespace that the tokenizer skipped**, so on
  `MKTAYIAKQR QISF…` a mutation after the space scored the next residue, and one at the last
  position scored the `<eos>` row. The wild type is now normalised once and both positions and
  tokens come from it.
- **Non-amino-acid input was scored.** Wild-type characters such as `J`, digits, `-`, `.` or an
  inner `*` became `<unk>` or gap tokens, and substitutions to `J`/`X`/`B`/`Z`/`U`/`O` were
  scored against them. Now: the wild type may hold the 20 standard amino acids and ESM's
  `X`/`B`/`Z`/`U`/`O` tokens (a final `*` is dropped), anything else is an error; both sides of
  a substitution must be standard; `scan` leaves out non-standard wild-type positions.
- `proteus esm score wt.fast …` (a mistyped file name) scored the "protein" `WT.FAST`. An
  argument that is neither a file nor a valid sequence is now an error.
- The length limit was 1024 residues; ESM-2 was trained on 1022 (1024 tokens with `<cls>` and
  `<eos>`), and longer input is refused with that explanation.
- A `config.json` with the wrong layer or head count for its weights loaded silently (a
  truncated network, or wrongly split heads); it is refused.
- `[mutation=…]` tags: ProteinGym's `A10G:C4S` aborted the whole screen; `A+10G` parsed; a
  position mutated twice in one variant was scored twice and summed. All fixed.
- An interrupted checkpoint download left a `.part` file in the cache; a failed request left an
  empty cache directory. Hub errors now say what to do: the 3B/15B repositories have no
  safetensors (with the conversion command), an unknown or gated repository needs a correct id
  or `HF_TOKEN`, and a path given to `--esm-model` that is not a directory is reported as such.
  Missing or unreadable local files are named.
- `--compare` paired the i-th Cα of one file with the i-th of the other, so a single missing
  residue shifted every pairing (1UBQ against itself minus residue 1: 3.77 Å instead of 0),
  and the longer structure was truncated without a word.
- The interactive viewer below 118 columns: the HUD was padded but never cut, wrapped, and
  scrolled the screen every frame — at 80×24 the structure scrolled out of view. Every HUD and
  dashboard line is now cut to the terminal width by display width.
- The viewer's layout is recomputed on resize and on the dashboard toggle (the separator,
  dashboard and HUD stayed where the first frame put them), and a terminal under 12 rows is
  no longer overdrawn: the HUD shrinks first, and the dashboard is hidden when it does not fit.
  Dashboard section rules were one column too wide and measured in bytes.
- SIGTERM, SIGHUP or SIGINT during the interactive viewer, or a panic in it, left the terminal
  raw, on the alternate screen and with the cursor hidden. The terminal is now restored first
  (the process then exits 128 + the signal number).
- An oversized `--width/--height` aborted on allocation (100000×100000), overflowed to an empty
  kitty image printed with exit status 0, or printed nothing for 0.
- Sixel's median-cut palette ignored how many pixels used each colour, so the background was
  averaged with dark shading tones: 6VXX's black backdrop decoded as (15,15,13). Colours are
  now weighted by pixel count and the most common one keeps an exact register.
- The ribbon face held still inside each residue and snapped by the whole carbonyl-to-carbonyl
  angle at the next Cα. Each guide is now anchored mid-peptide and interpolated; on 1CRN the
  worst boundary snap went from 77° to 17°.
- `--web` wrote to a fixed name in the shared temp directory and followed symlinks; the page
  is now a new file with a random name, created exclusively.
- Half-block cells with one empty half painted it explicit black instead of leaving the
  terminal's own background, a black fringe on any other background.
- `proteus view typo.pdb` created the job database (`proteus.db`, `-wal`, `-shm`) before
  failing; it is now opened only for an argument shaped like a job ID, and only if it exists.

## [0.6.0] — 2026-09-23

The structure-QC release. `proteus analyze` takes a folder of models and writes one row per
structure, which is the first command that serves someone who already has models from another
tool. Every claim in the README and in the two standalone crates was re-checked against a fresh
measurement or its source before this release: 11 were wrong and 12 overstated, and all are
corrected below. Also new since 0.5.0: a Sixel backend, a carbonyl-oriented ribbon, PLIP as a
reference for salt bridges and π interactions (which found and fixed a 5× π–π over-count), and a
53-file validation corpus.

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
  smoke test; the page moved to 3Dmol.js after 0.5.0 (unreleased). The smoke assertion kept
  passing only because a licence comment inside the vendored 3Dmol.js mentions molstar; it now
  checks for the page's own viewer call.
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
  mirror mistake — a PDB handed to a sequence command — points at `proteus analyze`.
  Both directions are tested; the message no longer carries a run of stray spaces.
- `proteus esm` suggests, once, scoring known domains separately when a sequence runs past 400
  residues. The README and the crate README carry ProteinGym's own per-taxon numbers for ESM-2
  650M (Spearman ρ 0.457 on human assays, 0.261 on viral ones), so the weak spot is stated with
  its source rather than discovered after acting on a ranking.
- **π–π stacking and cation–π were over-reported, and the counts change.** Neither test included
  a lateral-offset term, so two aromatic rings that were parallel and within the distance cutoff
  but slid sideways past each other counted as stacked, and a cation beyond the ring edge counted
  as sitting over its face. Measured against PLIP across 15 structures that was **20 % precision
  on π–π** (55 reported where PLIP finds 11) and 33 % on cation–π. Both now apply PLIP's 2.0 Å
  lateral-offset criterion (benzene radius + 0.5 Å): **π–π precision 20 % → 81.8 %** (11 reported,
  matching PLIP's 11) and **cation–π 33.3 % → 73.9 %** (51 → 23). Found by adopting the reference,
  not by inspection.
  **This changes reported counts, the non-covalent network density and therefore fitness scores**
  for structures with aromatics — 1CRN's network density goes 121.7 → 119.6 contacts/100 res.
  A unit test that asserted crambin has ≥ 1 aromatic interaction was itself wrong: PLIP finds
  none there, and the assertion now requires agreement with the reference.

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

[Unreleased]: https://github.com/OtoYuki/proteus/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/OtoYuki/proteus/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/OtoYuki/proteus/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/OtoYuki/proteus/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/OtoYuki/proteus/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/OtoYuki/proteus/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OtoYuki/proteus/compare/v0.1.0-thesis...v0.3.0
