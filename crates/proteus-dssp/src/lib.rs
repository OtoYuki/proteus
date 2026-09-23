//! Kabsch–Sander DSSP secondary-structure assignment in pure Rust.
//!
//! Semantics follow DSSP 2.x (as ported by mdtraj): α-helix (H) overrides sheet
//! assignments, a 3₁₀ helix (G) only fills unassigned residues, a π-helix (I) may overwrite
//! an α-helix ("prefer-π", as mdtraj does), and there is no PPII.
//!
//! ```
//! use proteus_dssp::{assign, Residue, Ss};
//! let residues: Vec<Residue> = Vec::new(); // backbone N, CA, C, O per residue
//! let ss: Vec<Ss> = assign(&residues);
//! assert!(ss.is_empty());
//! ```
#![forbid(unsafe_code)]

mod assign;
mod hbond;

pub use assign::{assign, Residue, Simple, Ss};
