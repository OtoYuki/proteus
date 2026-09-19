//! Core domain models and structural bioinformatics for Proteus.

pub mod error;
pub mod metrics;
pub mod models;
pub mod sequence;

pub use error::CoreError;
pub use models::*;
