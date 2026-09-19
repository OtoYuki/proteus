//! Asynchronous bio-compute DAG pipeline and OCI execution engine for Proteus.

pub mod auto;
pub mod error;
pub mod esm_api;
pub mod oci;
pub mod runner;
pub mod scheduler;
pub mod simulated;

pub use auto::AutoRunner;
pub use error::EngineError;
pub use esm_api::EsmApiRunner;
pub use runner::ComputeRunner;
pub use scheduler::{EngineEvent, PipelineScheduler};
