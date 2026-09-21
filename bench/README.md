# Benchmarks

Measured on `archlinux` (x86_64), `rustc 1.94.1 (e408947bf 2026-03-25)`, Python 3.12.13, mdtraj 1.11.1.post2.
Rust: criterion median (`cargo bench -p proteus-core`). Python: median of 5 calls after a warm-up, metric only
(parsing excluded on both sides), on protein heavy atoms. Same structure files, same metric definition.
Regenerate with `bench/run.sh`; raw numbers in `bench/results/`.

### SASA, Shrake–Rupley 960 pts/atom

| structure | atoms | residues | proteus | mdtraj (C++) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 7.18 ms | 18.53 ms | 2.6× |
| 1ubq | 602 | 76 | 12.82 ms | 31.74 ms | 2.5× |
| 4hhb | 4384 | 574 | 100.43 ms | 257.96 ms | 2.6× |
| 6vxx | 22812 | 2916 | 527.21 ms | 2.26 s | 4.3× |

### SASA, Shrake–Rupley 96 vs 100 pts/atom

| structure | atoms | residues | proteus | Biopython (pure Python) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 1.06 ms | 38.21 ms | 35.9× |
| 1ubq | 602 | 76 | 2.00 ms | 68.73 ms | 34.4× |
| 4hhb | 4384 | 574 | 16.07 ms | 551.88 ms | 34.3× |
| 6vxx | 22812 | 2916 | 83.26 ms | 2.94 s | 35.3× |

### SASA, Shrake–Rupley 96 vs 100 pts/atom

| structure | atoms | residues | proteus | FreeSASA (C) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 1.06 ms | 1.48 ms | 1.4× |
| 1ubq | 602 | 76 | 2.00 ms | 2.66 ms | 1.3× |
| 4hhb | 4384 | 574 | 16.07 ms | 20.42 ms | 1.3× |
| 6vxx | 22812 | 2916 | 83.26 ms | – | – |

### SASA, Shrake–Rupley 960 vs Lee–Richards

| structure | atoms | residues | proteus | FreeSASA L&R (C) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 7.18 ms | 7.40 ms | 1.0× |
| 1ubq | 602 | 76 | 12.82 ms | 14.55 ms | 1.1× |
| 4hhb | 4384 | 574 | 100.43 ms | 117.43 ms | 1.2× |
| 6vxx | 22812 | 2916 | 527.21 ms | – | – |

### DSSP 8-state

| structure | atoms | residues | proteus | mdtraj (C++) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 11 µs | 153 µs | 13.9× |
| 1ubq | 602 | 76 | 26 µs | 246 µs | 9.6× |
| 4hhb | 4384 | 574 | 1.20 ms | 2.94 ms | 2.4× |
| 6vxx | 22812 | 2916 | 30.15 ms | 35.20 ms | 1.2× |

### φ/ψ + MolProbity Ramachandran vs φ/ψ only

| structure | atoms | residues | proteus | mdtraj (C++) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 4 µs | 686 µs | 166.9× |
| 1ubq | 602 | 76 | 7 µs | 605 µs | 87.0× |
| 4hhb | 4384 | 574 | 59 µs | 2.52 ms | 42.5× |
| 6vxx | 22812 | 2916 | 322 µs | 10.47 ms | 32.5× |

### φ/ψ + MolProbity Ramachandran vs φ/ψ only

| structure | atoms | residues | proteus | Biopython (pure Python) | ratio |
|---|---|---|---|---|---|
| 1crn | 327 | 46 | 4 µs | 4.64 ms | 1128.5× |
| 1ubq | 602 | 76 | 7 µs | 7.66 ms | 1100.9× |
| 4hhb | 4384 | 574 | 59 µs | 58.90 ms | 995.1× |
| 6vxx | 22812 | 2916 | 322 µs | 301.14 ms | 935.9× |

### Proteus-only kernels (no like-for-like baseline)

| structure | heavy-atom overlap | interaction network | Kabsch RMSD (Cα) | full profile |
|---|---|---|---|---|
| 1crn | 517 µs | 119 µs | 1 µs | 7.95 ms |
| 1ubq | 740 µs | 235 µs | 1 µs | 14.74 ms |
| 4hhb | 5.56 ms | 1.87 ms | 6 µs | 110.12 ms |
| 6vxx | 39.75 ms | 9.26 ms | 27 µs | 613.88 ms |

Ratios > 1 mean Proteus is faster. Where the point counts differ (96 vs 100, 960 vs Lee–Richards) or
Proteus does more work (φ/ψ **plus** Top8000 Ramachandran scoring vs φ/ψ only), the row label says so.
The interaction network and overlap score have no drop-in equivalent in mdtraj/Biopython (mdtraj's
`baker_hubbard` needs explicit hydrogens), so they are reported without a ratio.
