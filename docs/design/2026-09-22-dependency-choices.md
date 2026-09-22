# Dependency choices, and where the popular answer was the wrong one

**Date:** 2026-09-22 · **Status:** current · **Scope:** the whole workspace

Each row is a place where there was a default everyone reaches for, and what Proteus uses
instead. The pattern is not contrarianism: it is that Proteus ships **one static binary that
must work offline**, and several popular choices quietly break that.

| need | the popular answer | what Proteus does | why |
|---|---|---|---|
| browser structure viewer | **Mol\*** (RCSB, PDBe, AlphaFold DB all use it) | 3Dmol.js 2.5.5, **vendored into the binary** | Mol\* is the better tool for interactive analysis and we do not try to replace it — but our page shows one structure, one representation, one colouring, and used none of its sequence panel, measurements or plugins. It was loaded from a CDN, so the page did not open on an HPC login node or an airgapped cluster, which is where Proteus runs. 3Dmol.js is 525 KB against 4.9 MB, small enough to embed, which is what buys the offline guarantee. The structure file is always on disk for whoever wants Mol\*, PyMOL or ChimeraX. |
| secondary structure in the browser page | whatever the viewer guesses | Proteus's own DSSP, pushed into the page | 3Dmol.js and Mol\* both assign secondary structure heuristically. Ours is validated against `mdtraj.compute_dssp` at ≥ 98 % per residue, and on 1CRN it is 46/46 where 3Dmol.js misses the 3₁₀ helix at 42–44. Without this the web page can disagree with `analyze`, the terminal viewer and the Parquet export. |
| ML runtime for ESM-2 | **ONNX Runtime** (`ort`, ~19 M downloads) or libtorch (`tch`) | candle | Both alternatives link a C++ runtime, so the "download one binary and run it" promise dies and the container triples in size. candle is pure Rust; `ldd` on the release binary shows no torch, onnx or python. |
| DSSP | call out to the `mkdssp` binary, or port mdtraj's C | `proteus-dssp`, written here | A shell-out means a runtime dependency the user has to install, and the field had no pure-Rust Kabsch–Sander crate. 610 lines, no dependencies, validated per residue. |
| 3D rendering | X11/WebGL, or "just use PyMOL" | software rasteriser in the terminal | The stated use case is an SSH session on a cluster. Nothing else draws cartoon ribbons with SSAO into a terminal, which is the reason this project is interesting at all. |
| container execution | shell out to `podman run` | bollard over the Docker Engine socket | Typed API for create/start/wait/logs/remove, cgroup limits straight from TES `resources`, no argv quoting bugs. Podman serves the same API, so rootless Podman is the first socket tried; Docker is a fallback, not the assumption. |
| task API | a bespoke REST API | GA4GH TES 1.1 | Here the standard *is* the right answer: Nextflow, Sprocket and Cromwell already speak it, and the ELIXIR compliance suite gives an external pass/fail. Being unusual would have cost users and gained nothing. |
| benchmark harness | hand-rolled timing loops | criterion, plus Python baselines measured in the same run | The comparison is only meaningful if both sides run back-to-back on the same machine and the same files. |

## The rule behind the table

Prefer the standard when the standard is what users already speak (TES, Parquet, mmCIF). Depart
from it when it breaks the one property the project is built around — a single binary that runs
offline over SSH — and then say so, in writing, with the size and licence of what replaced it.

Every departure here is reversible and documented: `crates/proteus-core/assets/README.md` records
the vendored viewer's source URL, version, sha256 and licence, with `scripts/vendor_3dmol.sh` to
refresh it.
