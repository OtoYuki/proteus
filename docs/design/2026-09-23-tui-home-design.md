# A home screen: bare `proteus` in a terminal opens a TUI

2026-09-23.

## Why

Typing `proteus` prints clap's usage error and exits 2. Every capability is there, but a new
user has to learn nine subcommands before seeing anything, and there is no way to see one's own
jobs at all: the repository has no query that lists them (`status` needs an id you already
have). The terminal is where Proteus is meant to live (SSH, HPC login nodes), so the first
screen should be there too.

## Facts the design rests on

- **ratatui 0.30.2** (crates.io, 2026-09-23): MIT, `rust-version = 1.88` (ours is 1.94), with a
  `crossterm_0_29` feature matching the workspace's crossterm 0.29. Its `TestBackend` renders
  a frame into a buffer, so screens can be unit-tested; the existing interactive viewer is
  hand-written crossterm and is not.
- The interactive viewer (`proteus view --interactive`) already handles raw mode, the
  alternate screen, panics and SIGTERM/SIGHUP/SIGINT (`view.rs: run_viewer`). Reusing it by
  calling it in-process would mean sharing one terminal between two owners.
- `jobs` joins to `sequences` (header, length) and to `predictions` (pLDDT, engine in
  `metadata`); there is no index on `created_at`, which is fine for the thousands of rows a
  single user produces.

## Design

**Entry.** `Commands` becomes optional. No subcommand and both stdin and stdout are terminals →
the TUI. No subcommand otherwise (a pipe, CI, a script) → clap's usage error, exit 2, exactly
as today. Every subcommand is unchanged.

**Screens** (tabs, `1`–`3` or `Tab`):

1. **Jobs** — the newest 500 jobs: short id, sequence name, length, tier, status, engine, mean
   pLDDT, age. The detail pane shows the full id, timestamps, the error log for a failed job and
   the tier downgrade if any. Refreshed every 2 s, so a job running under `proteus serve`
   visibly progresses.
2. **Structures** — a file browser rooted at the current directory showing folders and
   structure files (`.pdb`, `.cif`, `.mmcif`, `.ent`, and those gzipped). Selecting a file runs
   the same analysis as `analyze` on a background thread and shows the same summary rows as the
   browser page (one function, `proteus_core::qc::summary_rows`, feeds both).
3. **Run** — forms for the two workflows that start from nothing: fold a sequence (`submit`)
   and scan a protein (`mutate … | screen -`). The form shows the exact command line it will
   run, and updates it as you type.

**Actions run as child processes of the same binary** (`std::env::current_exe()`): the TUI
leaves the alternate screen, runs `proteus view X --interactive`, `proteus view X --web`,
`proteus submit …` or the `mutate | screen` pipe (two processes joined by a pipe, no shell),
and comes back when it ends — after "press Enter" for commands whose output should be read.
So nothing is implemented twice, signal and terminal handling stay where they are tested, and
every action teaches its command line.

**Keys:** `↑↓`/`jk` move, `Enter` opens, `v` terminal viewer, `w` browser page, `r` refresh,
`/` filter the jobs list, `?` help, `q`/`Esc`/`Ctrl-C` quit. Mouse is not required.

## Tests

- `ProteusRepository::list_jobs`: order, limit, and the joined fields, on an in-memory database.
- Rendering with `TestBackend`: each tab at 80×24 and at 40×12 (nothing panics, the key hints
  fit, an empty database says how to start).
- Key handling as a pure state machine: navigation clamps, the filter narrows, the forms build
  the documented command lines (asserted as argument vectors, so quoting cannot hide a bug).
- `proteus` with no subcommand and no terminal still exits 2 with the usage text (smoke).

## Non-goals

Running jobs inside the TUI process, editing or deleting jobs, and a mouse-first interface.
