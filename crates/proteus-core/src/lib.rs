//! Core domain models and structural bioinformatics for Proteus.

pub mod error;
pub mod metrics;
pub mod models;
pub mod ranking;
pub mod sasa;
pub mod sequence;
pub mod structure;

pub use error::CoreError;
pub use models::*;
pub use ranking::*;
pub use sasa::*;
pub use structure::*;
