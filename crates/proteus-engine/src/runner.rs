use crate::error::EngineError;
use proteus_core::models::{PipelineJob, Sequence};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio_stream::Stream;

pub type LogStream = Pin<Box<dyn Stream<Item = Result<String, EngineError>> + Send>>;

pub struct RunResult {
    pub pdb_path: PathBuf,
    pub plddt: Option<f64>,
    /// Runner-specific provenance. Every runner sets `"engine"` to one of the `ENGINE_*`
    /// names so that downstream tables and exports can say where a structure came from.
    pub metadata: Option<serde_json::Value>,
}

/// `metadata.engine` of the offline simulator: a synthetic ideal helix, not a prediction.
pub const ENGINE_SIMULATED: &str = "simulated";
/// `metadata.engine` of the Meta ESM Atlas fold API.
pub const ENGINE_ESMFOLD_API: &str = "esmfold-api";
/// `metadata.engine` of a local OCI container run.
pub const ENGINE_OCI: &str = "oci";

/// The `engine` recorded in a prediction's metadata, or `"unknown"`.
pub fn engine_name(metadata: Option<&serde_json::Value>) -> &str {
    metadata
        .and_then(|m| m.get("engine"))
        .and_then(|e| e.as_str())
        .unwrap_or("unknown")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulated::SimulatedRunner;
    use proteus_core::models::{JobStatus, PipelineTier};

    #[tokio::test]
    async fn simulated_runner_names_its_engine() {
        let dir = tempfile::tempdir().unwrap();
        let job = PipelineJob {
            id: uuid::Uuid::new_v4(),
            sequence_id: uuid::Uuid::new_v4(),
            tier: PipelineTier::FastScreening,
            status: JobStatus::Queued,
            priority: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error_log: None,
        };
        let seq = Sequence {
            id: job.sequence_id,
            header: "x".into(),
            fasta: "ACDEFGHIK".into(),
            length: 9,
            created_at: chrono::Utc::now(),
        };
        let r = SimulatedRunner::new()
            .execute_job(&job, &seq, dir.path())
            .await
            .unwrap();
        assert_eq!(engine_name(r.metadata.as_ref()), ENGINE_SIMULATED);
        assert_eq!(engine_name(None), "unknown");
        assert_eq!(
            engine_name(Some(&serde_json::json!({ "engine": "oci" }))),
            "oci"
        );
    }
}
