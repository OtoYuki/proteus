# Design records

Dated design documents written before each piece of work. They are kept as the record of *why*
things are shaped the way they are; they are not maintained as current documentation. Where the
shipped code differs from a document, the table below says so — the code, `CHANGELOG.md` and
`validate/` are the ground truth.

| document | status | what shipped differently |
|---|---|---|
| [2026-09-19 Rust architecture](2026-09-19-proteus-rust-architecture-design.md) | implemented, with substitutions | GROMACS/ColabFold never became tiers: prediction is ESMFold (API or your own image) and Boltz (your own image); the "relax" tier is refused as unimplemented. Nothing uses explicit SIMD; the kernels are scalar Rust with cell lists. |
| [2026-09-21 non-covalent interaction network](2026-09-21-non-covalent-interaction-network-design.md) | implemented | Bonded-neighbour exclusions became chain-aware (2026-09-22). "MolProbity steric clashes" in the text is the heavy-atom overlap score, which is *not* the MolProbity clashscore (no hydrogens). |
| [2026-09-21 target master specification](2026-09-21-proteus-target-master-specification.md) | gates 1–6 met, 7 is outside the repo | Gate 5 is met by its second form: `tes_e2e.rs` drives a Nextflow-shaped task in CI, and the real `examples/nextflow` run is done by hand; Sprocket/WDL is the CI-checked engine. Parquet is ZSTD, `schema_version = 4`. |
| [2026-09-21 SP1 scientific correctness](2026-09-21-sp1-scientific-correctness-design.md) | implemented | Everything in §3 landed (`proteus-dssp`, Top8000 grids, pLDDT provenance, `validate/`); the P-SEA label is gone. |
| [2026-09-21 SP3 TES execution](2026-09-21-sp3-tes-execution-and-ecosystem-design.md) | implemented, with substitutions | No `--i-know-what-i-am-doing` escape hatch: `--executor host` is simply refused off loopback. `--task-timeout` became `--executor-timeout`. Nextflow gets no CI smoke test (Java on the runner was not worth it; the WDL job covers the same server path). |
| [2026-09-21 SP4 ESM-2 scoring](2026-09-21-sp4-esm2-fitness-scoring-design.md) | implemented | Library scoring caches the wild-type forward pass (`MarginalScorer`, 2026-09-22). |
| [2026-09-23 WebGL viewer](2026-09-23-webgl-viewer-design.md) | implemented | Faces are flipped toward the viewer by the view-space normal, not `gl_FrontFacing` (the document says why). |

[2026-09-22 dependency choices](2026-09-22-dependency-choices.md) records where the popular
answer was the wrong one for a binary that has to run offline, and what replaced it.

The SP2 (release infrastructure) work has no design document; it is `.github/workflows/release.yml`
and the `v0.3.0`/`v0.4.0` entries in the changelog.
