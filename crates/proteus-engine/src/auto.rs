use crate::error::EngineError;
use crate::esm_api::EsmApiRunner;
use crate::oci::OciRunner;
use crate::runner::{ComputeRunner, RunResult};
use crate::simulated::SimulatedRunner;
use proteus_core::models::{PipelineJob, PipelineTier, Sequence};
use std::path::Path;
use tracing::info;

/// High-availability compute runner that intelligently dispatches bio-compute tasks:
/// 1. Tries local OCI container runtime if image is present.
/// 2. Falls back to Meta ESMFold API for zero-setup live deep-learning folding.
/// 3. Falls back to offline SimulatedRunner if airgapped or network is unreachable.
pub struct AutoRunner {
    oci: Option<OciRunner>,
    esm_api: EsmApiRunner,
    simulated: SimulatedRunner,
}

impl AutoRunner {
    pub fn new() -> Self {
        let oci = OciRunner::new().ok();
        Self {
            oci,
            esm_api: EsmApiRunner::new(),
            simulated: SimulatedRunner::new(),
        }
    }
}

impl Default for AutoRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ComputeRunner for AutoRunner {
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError> {
        let image = match job.tier {
            PipelineTier::FastScreening => "ghcr.io/proteus/esmfold:latest",
            PipelineTier::HighFidelity => "ghcr.io/jwohlwend/boltz:latest",
            PipelineTier::FullValidation => "ghcr.io/proteus/openmm:latest",
        };

        // 1. Try local OCI container if runner connected and image is present
        if let Some(ref oci) = self.oci {
            if oci.has_image(image).await {
                info!("AutoRunner: Dispatched to local OCI container image '{image}'");
                return oci.execute_job(job, sequence, work_dir).await;
            }
        }

        // 2. Try ESMFold API for fast/sota prediction if online
        if job.tier == PipelineTier::FastScreening || job.tier == PipelineTier::HighFidelity {
            info!(
                "AutoRunner: Local container '{image}' not present. Attempting Meta ESMFold API..."
            );
            match self.esm_api.execute_job(job, sequence, work_dir).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    info!("AutoRunner: ESMFold API unavailable ({e}). Falling back to SimulatedRunner.");
                }
            }
        }

        // 3. Fallback to SimulatedRunner
        info!("AutoRunner: Using offline SimulatedRunner fallback.");
        self.simulated.execute_job(job, sequence, work_dir).await
    }
}
