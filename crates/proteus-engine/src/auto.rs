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

/// Stamps `tier_requested` / `tier_honoured` (and, on a downgrade, `fallback_reason`) onto a
/// runner's metadata so the provenance survives into the database and the CLI can report it.
fn annotate_tier(mut result: RunResult, job: &PipelineJob, downgrade: Option<String>) -> RunResult {
    let meta = result.metadata.get_or_insert_with(|| serde_json::json!({}));
    if let Some(obj) = meta.as_object_mut() {
        obj.insert("tier_requested".into(), job.tier_slug().into());
        obj.insert("tier_honoured".into(), downgrade.is_none().into());
        if let Some(reason) = downgrade {
            obj.insert("fallback_reason".into(), reason.into());
        }
    }
    result
}

#[async_trait::async_trait]
impl ComputeRunner for AutoRunner {
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError> {
        let image = crate::oci::tier_image(&job.tier);
        let image = image.as_str();
        // Each step that is skipped or fails adds a clause; the final reason is their join.
        let mut reasons: Vec<String> = Vec::new();

        // 1. Try local OCI container if runner connected and image is present
        match crate::oci::tier_supported(&job.tier) {
            Err(e) => reasons.push(e.to_string()),
            Ok(()) => match &self.oci {
                None => reasons.push(format!(
                    "no container runtime reachable for image '{image}'"
                )),
                Some(oci) if !oci.has_image(image).await => {
                    reasons.push(format!("container image '{image}' not present locally"))
                }
                Some(oci) => {
                    info!("AutoRunner: Dispatched to local OCI container image '{image}'");
                    let result = oci.execute_job(job, sequence, work_dir).await?;
                    return Ok(annotate_tier(result, job, None));
                }
            },
        }

        // A complex, a ligand or an MSA/sampling option only means something to Boltz: falling
        // back to ESMFold or a helix would answer a different question.
        if proteus_core::complex::ComplexSpec::is_spec(&sequence.fasta) {
            return Err(EngineError::Pipeline(format!(
                "this input (a complex, a ligand, or MSA/sampling options) needs the local \
                 Boltz image: {}",
                reasons.join("; ")
            )));
        }

        // 2. Try ESMFold API for fast/sota prediction if online. The API *is* ESMFold, so the
        //    fast tier is honoured by it; the sota (Boltz) tier is not.
        if job.tier == PipelineTier::FastScreening || job.tier == PipelineTier::HighFidelity {
            info!(
                "AutoRunner: Local container '{image}' not present. Attempting Meta ESMFold API..."
            );
            match self.esm_api.execute_job(job, sequence, work_dir).await {
                Ok(result) => {
                    let downgrade = (job.tier == PipelineTier::HighFidelity).then(|| {
                        format!(
                            "{}; ran the ESMFold API instead of {}",
                            reasons.join("; "),
                            job.tier_slug()
                        )
                    });
                    return Ok(annotate_tier(result, job, downgrade));
                }
                Err(e) => {
                    info!("AutoRunner: ESMFold API unavailable ({e}). Falling back to SimulatedRunner.");
                    reasons.push(format!("ESMFold API unavailable ({e})"));
                }
            }
        }

        // 3. Fallback to SimulatedRunner
        info!("AutoRunner: Using offline SimulatedRunner fallback.");
        let result = self.simulated.execute_job(job, sequence, work_dir).await?;
        let reason = format!(
            "{}; ran the offline simulator (synthetic helix)",
            reasons.join("; ")
        );
        Ok(annotate_tier(result, job, Some(reason)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{engine_name, tier_downgrade, ENGINE_SIMULATED};
    use proteus_core::models::JobStatus;

    fn job(tier: PipelineTier) -> (PipelineJob, Sequence) {
        let job = PipelineJob {
            id: uuid::Uuid::new_v4(),
            sequence_id: uuid::Uuid::new_v4(),
            tier,
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
        (job, seq)
    }

    /// No container runtime and an unreachable API: the simulator runs, and the metadata
    /// says which tier was asked for and why it was not honoured.
    #[tokio::test]
    async fn fallback_to_simulator_records_the_downgrade() {
        let runner = AutoRunner {
            oci: None,
            esm_api: EsmApiRunner::with_endpoint("http://127.0.0.1:9/".into()),
            simulated: SimulatedRunner::new(),
        };
        for tier in [PipelineTier::FastScreening, PipelineTier::HighFidelity] {
            let (job, seq) = job(tier);
            let dir = tempfile::tempdir().unwrap();
            let r = runner.execute_job(&job, &seq, dir.path()).await.unwrap();
            assert_eq!(engine_name(r.metadata.as_ref()), ENGINE_SIMULATED);
            let d = tier_downgrade(r.metadata.as_ref()).expect("downgrade recorded");
            assert_eq!(d.requested, job.tier_slug());
            assert!(
                d.reason.contains("ESMFold API"),
                "reason should name the failed API step: {}",
                d.reason
            );
            assert!(
                d.reason.contains("not present") || d.reason.contains("no container runtime"),
                "reason should name the missing image/runtime: {}",
                d.reason
            );
        }
    }

    /// The relax tier has no API fallback, so the simulator reason must say the tier itself
    /// is unsupported rather than blaming the network.
    #[tokio::test]
    async fn relax_tier_fallback_names_the_unsupported_tier() {
        let runner = AutoRunner {
            oci: None,
            esm_api: EsmApiRunner::with_endpoint("http://127.0.0.1:9/".into()),
            simulated: SimulatedRunner::new(),
        };
        let (job, seq) = job(PipelineTier::FullValidation);
        let dir = tempfile::tempdir().unwrap();
        let r = runner.execute_job(&job, &seq, dir.path()).await.unwrap();
        let d = tier_downgrade(r.metadata.as_ref()).expect("downgrade recorded");
        assert_eq!(d.requested, "relax");
        assert!(d.reason.contains("not implemented"), "{}", d.reason);
        assert!(!d.reason.contains("ESMFold API"), "{}", d.reason);
    }
}
