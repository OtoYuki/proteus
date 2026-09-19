//! Asynchronous bio-compute DAG pipeline and OCI execution engine for Proteus.

pub mod error;
pub mod oci;
pub mod runner;
pub mod scheduler;
pub mod simulated;

pub use error::EngineError;
pub use runner::ComputeRunner;
pub use scheduler::{EngineEvent, PipelineScheduler};
