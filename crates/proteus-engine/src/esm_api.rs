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
            .map_err(|e| EngineError::Container(format!("ESMFold API network error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(EngineError::Container(format!(
                "ESMFold API returned error HTTP {status}: {text}"
            )));
        }

        let pdb_content = response.text().await.map_err(|e| {
            EngineError::Container(format!("Failed to read ESMFold PDB response: {e}"))
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
