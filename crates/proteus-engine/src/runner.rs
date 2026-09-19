use crate::error::EngineError;
use proteus_core::models::{PipelineJob, Sequence};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio_stream::Stream;

pub type LogStream = Pin<Box<dyn Stream<Item = Result<String, EngineError>> + Send>>;

pub struct RunResult {
    pub pdb_path: PathBuf,
    pub plddt: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

#[async_trait::async_trait]
pub trait ComputeRunner: Send + Sync {
    /// Execute computation for a given sequence and job, returning structure output.
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError>;
}
