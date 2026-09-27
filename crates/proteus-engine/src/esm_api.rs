use crate::error::EngineError;
use crate::runner::{ComputeRunner, RunResult};
use proteus_core::models::{PipelineJob, Sequence};
use std::path::{Path, PathBuf};
use tracing::info;

/// A bio-compute runner that invokes Meta's public ESMFold API endpoint
/// to obtain real atomic 3D protein structure predictions without requiring
/// multi-gigabyte local container weight downloads.
pub struct EsmApiRunner {
    endpoint: String,
    client: reqwest::Client,
}

impl EsmApiRunner {
    pub fn new() -> Self {
        Self {
            endpoint: "https://api.esmatlas.com/foldSequence/v1/pdb/".to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn with_endpoint(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }
}

impl Default for EsmApiRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ComputeRunner for EsmApiRunner {
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError> {
        if proteus_core::complex::ComplexSpec::is_spec(&sequence.fasta) {
            return Err(EngineError::Pipeline(
                "complexes, ligands, MSA options and several samples need the sota tier with \
                 --runner oci; LABEL"
                    .replace("LABEL", "the ESMFold API folds one chain"),
            ));
        }
        info!(
            "EsmApiRunner submitting job {} to Meta ESMFold API for sequence '{}' ({} residues)",
            job.id, sequence.header, sequence.length
        );

        tokio::fs::create_dir_all(work_dir).await?;
        let pdb_filename = format!("{}_esmfold.pdb", job.id);
        let pdb_path: PathBuf = work_dir.join(&pdb_filename);

        let response = self
            .client
            .post(&self.endpoint)
            .body(sequence.fasta.clone())
            .send()
            .await
            .map_err(|e| EngineError::RemoteApi(format!("ESMFold API network error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(EngineError::RemoteApi(format!(
                "ESMFold API returned error HTTP {status}: {text}"
            )));
        }

        let pdb_content = response.text().await.map_err(|e| {
            EngineError::RemoteApi(format!("Failed to read ESMFold PDB response: {e}"))
        })?;

        tokio::fs::write(&pdb_path, &pdb_content).await?;

        info!(
            "ESMFold API completed successfully for job {}. Saved PDB to {:?}",
            job.id, pdb_path
        );

        Ok(RunResult {
            pdb_path,
            plddt: None,
            metadata: Some(serde_json::json!({
                "engine": crate::runner::ENGINE_ESMFOLD_API,
                "model": "meta-esmfold-v1",
                "endpoint": self.endpoint,
                "residues": sequence.length,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::models::{JobStatus, PipelineTier};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// An HTTP error from the API is the API's failure, not a container runtime's.
    #[tokio::test]
    async fn an_api_http_error_is_not_called_a_container_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let _ = sock
                .write_all(b"HTTP/1.1 504 Gateway Timeout\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .await;
        });
        let runner = EsmApiRunner::with_endpoint(format!("http://{addr}/"));
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
            fasta: "ACDEFGHIKLMNPQ".into(),
            length: 14,
            created_at: chrono::Utc::now(),
        };
        let dir = tempfile::tempdir().unwrap();
        let Err(err) = runner.execute_job(&job, &seq, dir.path()).await else {
            panic!("a 504 must not produce a structure");
        };
        let err = err.to_string();
        assert!(err.contains("504"), "{err}");
        assert!(!err.contains("Container"), "{err}");
        assert!(err.starts_with("Remote folding API error"), "{err}");
    }
}
