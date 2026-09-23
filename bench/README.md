# Benchmarks

Measured on `archlinux` (x86_64), `rustc 1.98.1 (48a229cea 2026-09-01)`, Python 3.12.13, mdtraj 1.11.1.post2.
Rust: criterion median (`cargo bench -p proteus-core`). Python: median of 7 calls after a warm-up, metric only
(parsing excluded on both sides), on protein heavy atoms. Same structure files, same metric definition.
Both sides are measured back-to-back in one `bench/run.sh` on a desktop machine, so absolute times move
a few per cent with machine load; the **ratios** are the stable quantity and are what the README quotes.
Regenerate with `bench/run.sh`; raw numbers in `bench/results/`.

### SASA, Shrake–Rupley 960 pts/atom

| structure | atoms | residues | proteus | mdtraj (C++) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 8.39 ms | 23.52 ms | 2.8× |
| 1ubq | 602 | 76 | 14.93 ms | 34.46 ms | 2.3× |
| 4hhb | 4384 | 574 | 117.48 ms | 295.40 ms | 2.5× |
| 6vxx | 22812 | 2916 | 608.46 ms | 2.85 s | 4.7× |

### SASA, Shrake–Rupley 96 vs 100 pts/atom

| structure | atoms | residues | proteus | Biopython (pure Python) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 1.25 ms | 44.80 ms | 35.9× |
| 1ubq | 602 | 76 | 2.33 ms | 78.09 ms | 33.6× |
| 4hhb | 4384 | 574 | 18.58 ms | 616.58 ms | 33.2× |
| 6vxx | 22812 | 2916 | 99.06 ms | 3.30 s | 33.3× |

### SASA, Shrake–Rupley 96 vs 100 pts/atom

| structure | atoms | residues | proteus | FreeSASA (C) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 1.25 ms | 1.75 ms | 1.4× |
| 1ubq | 602 | 76 | 2.33 ms | 3.01 ms | 1.3× |
| 4hhb | 4384 | 574 | 18.58 ms | 24.12 ms | 1.3× |
| 6vxx | 22812 | 2916 | 99.06 ms | – | – |

### SASA, Shrake–Rupley 960 vs Lee–Richards

| structure | atoms | residues | proteus | FreeSASA L&R (C) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 8.39 ms | 9.92 ms | 1.2× |
| 1ubq | 602 | 76 | 14.93 ms | 16.70 ms | 1.1× |
| 4hhb | 4384 | 574 | 117.48 ms | 133.80 ms | 1.1× |
| 6vxx | 22812 | 2916 | 608.46 ms | – | – |

### DSSP 8-state

| structure | atoms | residues | proteus | mdtraj (C++) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 12 µs | 154 µs | 12.3× |
| 1ubq | 602 | 76 | 28 µs | 255 µs | 9.2× |
| 4hhb | 4384 | 574 | 1.28 ms | 2.62 ms | 2.0× |
| 6vxx | 22812 | 2916 | 31.94 ms | 40.33 ms | 1.3× |

### φ/ψ + MolProbity Ramachandran vs φ/ψ only

The mdtraj row measures its public per-call API, most of which is Python-side index building at these sizes; it is not a comparison of dihedral kernels.

| structure | atoms | residues | proteus | mdtraj `compute_phi`/`compute_psi` (Python API; rebuilds atom indices per call) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 5 µs | 734 µs | 159.2× |
| 1ubq | 602 | 76 | 7 µs | 629 µs | 84.8× |
| 4hhb | 4384 | 574 | 62 µs | 2.72 ms | 43.9× |
| 6vxx | 22812 | 2916 | 349 µs | 11.53 ms | 33.1× |

### φ/ψ + MolProbity Ramachandran vs φ/ψ only

| structure | atoms | residues | proteus | Biopython (pure Python) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 5 µs | 5.50 ms | 1192.6× |
| 1ubq | 602 | 76 | 7 µs | 8.76 ms | 1181.4× |
| 4hhb | 4384 | 574 | 62 µs | 67.90 ms | 1095.8× |
| 6vxx | 22812 | 2916 | 349 µs | 342.58 ms | 983.0× |

### Proteus-only kernels (no like-for-like baseline)

| structure | heavy-atom overlap | interaction network | Kabsch RMSD (Cα) | full profile |
|---|---|---|---|---|
| 1crn | 450 µs | 138 µs | 1 µs | 9.09 ms |
| 1ubq | 715 µs | 271 µs | 1 µs | 16.09 ms |
| 4hhb | 5.45 ms | 2.22 ms | 6 µs | 127.83 ms |
| 6vxx | 35.78 ms | 11.26 ms | 30 µs | 694.46 ms |

Ratios > 1 mean Proteus is faster. Where the point counts differ (96 vs 100, 960 vs Lee–Richards) or
Proteus does more work (φ/ψ **plus** Top8000 Ramachandran scoring vs φ/ψ only), the row label says so.
The interaction network and overlap score have no drop-in equivalent in mdtraj/Biopython (mdtraj's
`baker_hubbard` needs explicit hydrogens), so they are reported without a ratio.

## ProteinGym zero-shot fitness (ESM-2 in pure Rust)

`bench/proteingym.py` downloads ProteinGym v1.1 substitution assays, runs `proteus esm scan`,
and correlates the predicted scores with the measured fitness (Spearman ρ). Five smallest
single-mutant assays (≤ 60 residues), all mutants scored, CPU:

| assay | residues | mutants | 8M wt-marg. | 8M masked | 35M wt-marg. |
|---|---|---|---|---|---|
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | 40 | 707 | +0.190 | +0.186 | **+0.431** |
| VG08_BPP22_Tsuboyama_2023_2GP8 | 40 | 723 | +0.370 | +0.360 | **+0.510** |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | 42 | 635 | +0.212 | +0.205 | **+0.543** |
| DN7A_SACS2_Tsuboyama_2023_1JIC | 55 | 1008 | +0.079 | +0.074 | **+0.210** |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | 55 | 1040 | +0.323 | +0.285 | **+0.398** |
| **mean \|ρ\|** | | | 0.235 | 0.222 | **0.418** |

For scale, the published ESM-2 650M zero-shot average over all 217 ProteinGym substitution
assays is 0.414 (Spearman; ProteinGym's own summary table). These five are stability assays on very short proteins, so the
numbers are not comparable to that average — they show the Rust implementation reproduces the
expected behaviour (bigger model ≫ smaller model; masked ≈ wild-type marginals on short
sequences) on real experimental data, not that it beats anything.

Reproduce: `bench/proteingym.py --assays 5 --max-len 60 --model facebook/esm2_t12_35M_UR50D`.

## Storage under concurrent writers

SQLite in WAL mode permits many readers but **one writer at a time**, and Proteus fans screening
jobs across a worker pool, so every worker's status update contends for that writer. Measured by
`cargo test -p proteus-storage --release --test concurrency -- --nocapture` on the host above:

| scenario | result |
|---|---|
| 400 job inserts from 16 concurrent tasks | 29.7 ms → **13 493 writes/s**, zero rows lost |
| 240 status updates to **one row** from 12 tasks | 20.7 ms → **11 594 updates/s**, converged, no lock errors surfaced |
| readers during 200 concurrent inserts | **703 reads** completed in 22.9 ms — readers are not starved |

This is a single-process, single-file measurement on NVMe; it says the storage layer is not the
bottleneck at the scale Proteus currently schedules, and it is **not** a claim about tens of
thousands of concurrent tasks across processes or hosts, which has not been tested.

