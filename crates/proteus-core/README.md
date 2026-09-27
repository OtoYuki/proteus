<img src="https://raw.githubusercontent.com/OtoYuki/proteus/main/docs/brand/proteus-mark.png" width="48" align="right" alt="Proteus">

# proteus-core

Domain models and the biophysics of [Proteus](https://github.com/OtoYuki/proteus), in pure
Rust: structure I/O, backbone geometry, DSSP-based secondary structure, Top8000 Ramachandran
scoring, Shrake–Rupley SASA, the non-covalent interaction network, heavy-atom steric overlap,
MolProbity/Phenix covalent geometry (bonds, angles, chirality, planarity against geostd and the
Conformation-Dependent Library; Cβ deviation; cis and twisted peptides) and Top8000 rotamers,
Kabsch superposition, binder–target interface metrics (contacts, dSASA, shape complementarity,
cross-interface H-bonds and salt bridges; ipAE, ipSAE and LIS from a predictor's PAE),
in-silico mutagenesis and the composite fitness score. Every number here
is compared with mdtraj, FreeSASA, cctbx and PLIP on a 53-structure corpus (plus 13 ESMFold
models for the geometry) in the repository's `validate/` harness.

```rust
use proteus_core::metrics::analyze_pdb_file;
let m = analyze_pdb_file(std::path::Path::new("1crn.pdb"), None)?;
println!("Rg {:.2} Å, {} H-bonds, fitness {:?}",
    m.radius_of_gyration,
    m.interaction_network.as_ref().map_or(0, |n| n.summary.total_hbonds),
    m.candidate_fitness_score);
```

Modules: `io` (PDB/mmCIF/gzip, first model, first altloc, protein heavy atoms), `backbone`,
`structure` (φ/ψ/ω, DSSP via `proteus-dssp`), `rama8000`, `geometry` (covalent restraints, Cβ,
ω), `rotamer`, `sasa`, `interactions`, `clash`, `interface` (with `interface::sc`), `pae`, `qc`,
`confidence` (is the B-factor column a pLDDT?), `metrics`, `ranking`, `mutagenesis`, `sequence`,
`models`, `tes` (GA4GH TES 1.1 types).

The heavy-atom overlap score is *not* the MolProbity clashscore (no hydrogens are added); salt
bridges, π–π and cation–π use Proteus-defined geometric criteria. Labels say so.

Embedded reference data: the Top8000 Ramachandran and rotamer grids (Richardson laboratory,
CC-BY-4.0 and cctbx BSD-3) and the geostd monomer restraints and CDL v1.2 (Phenix/cctbx,
BSD-3). The shape-complementarity code is ported from sc-rs (MIT). Provenance and licence
texts are in `data/*/NOTICE`.
