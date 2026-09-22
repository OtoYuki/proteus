//! Asynchronous bio-compute DAG pipeline and OCI execution engine for Proteus.

pub mod auto;
pub mod error;
pub mod esm_api;
pub mod oci;
pub mod runner;
pub mod scheduler;
pub mod simulated;
pub mod tes_exec;

pub use auto::AutoRunner;
pub use error::EngineError;
pub use esm_api::EsmApiRunner;
pub use runner::{engine_name, ComputeRunner, ENGINE_ESMFOLD_API, ENGINE_OCI, ENGINE_SIMULATED};
pub use scheduler::{CancelOutcome, EngineEvent, PipelineScheduler, TesExecutionConfig};
pub use tes_exec::{ContainerExecutor, ExecutorRequest, ExecutorResult, HostExecutor, TesExecutor};
