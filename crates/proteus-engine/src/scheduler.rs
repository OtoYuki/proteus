use crate::error::EngineError;
use crate::runner::ComputeRunner;
use proteus_core::metrics::analyze_pdb_file;
use proteus_core::models::{JobStatus, Prediction};
use proteus_storage::repository::ProteusRepository;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{error, info};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineEvent {
    JobQueued {
        job_id: Uuid,
        sequence_id: Uuid,
    },
    JobStarted {
        job_id: Uuid,
    },
    JobProgress {
        job_id: Uuid,
        step: String,
        percent: u8,
    },
    JobCompleted {
        job_id: Uuid,
        prediction_id: Uuid,
    },
    JobFailed {
        job_id: Uuid,
        error: String,
    },
}

#[derive(Clone)]
pub struct PipelineScheduler {
    repo: ProteusRepository,
    runner: Arc<dyn ComputeRunner>,
    events_tx: broadcast::Sender<EngineEvent>,
    artifacts_dir: PathBuf,
}

impl PipelineScheduler {
    pub fn new(
        repo: ProteusRepository,
        runner: Arc<dyn ComputeRunner>,
        artifacts_dir: PathBuf,
    ) -> Self {
        let (events_tx, _) = broadcast::channel(128);
        Self {
            repo,
            runner,
            events_tx,
            artifacts_dir,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.events_tx.subscribe()
    }

    pub fn repo(&self) -> &ProteusRepository {
        &self.repo
    }

    /// Process a job through the full computational pipeline.
    pub async fn process_job(&self, job_id: Uuid) -> Result<(), EngineError> {
        let job = match self.repo.get_job(job_id).await? {
            Some(j) => j,
            None => {
                return Err(EngineError::Pipeline(format!("Job {} not found", job_id)));
            }
        };

        let seq = match self.repo.get_sequence(job.sequence_id).await? {
            Some(s) => s,
            None => {
                return Err(EngineError::Pipeline(format!(
                    "Sequence {} for job {} not found",
                    job.sequence_id, job_id
                )));
            }
        };

        info!("Starting pipeline execution for job: {}", job_id);
        self.repo
            .update_job_status(job_id, JobStatus::Running, None)
            .await?;
        let _ = self.events_tx.send(EngineEvent::JobStarted { job_id });
        let _ = self
            .repo
            .log_event(Some(job_id), "INFO", "Job started")
            .await;

        let work_dir = self.artifacts_dir.join(job_id.to_string());
        tokio::fs::create_dir_all(&work_dir).await?;

        let _ = self.events_tx.send(EngineEvent::JobProgress {
            job_id,
            step: "Running compute container".into(),
            percent: 30,
        });

        // Execute runner
        let run_result = match self.runner.execute_job(&job, &seq, &work_dir).await {
            Ok(res) => res,
            Err(e) => {
                let err_msg = e.to_string();
                error!("Runner failed for job {}: {}", job_id, err_msg);
                self.repo
                    .update_job_status(job_id, JobStatus::Failed, Some(err_msg.clone()))
                    .await?;
                let _ = self.events_tx.send(EngineEvent::JobFailed {
                    job_id,
                    error: err_msg.clone(),
                });
                let _ = self
                    .repo
                    .log_event(Some(job_id), "ERROR", &format!("Compute failed: {err_msg}"))
                    .await;
                return Err(e);
            }
        };

        let _ = self.events_tx.send(EngineEvent::JobProgress {
            job_id,
            step: "Analyzing biophysical properties".into(),
            percent: 75,
        });

        // Compute biophysical metrics in Rust via pdbtbx
        let mut metrics = match analyze_pdb_file(&run_result.pdb_path, None) {
            Ok(m) => m,
            Err(e) => {
                let err_msg = format!("Biophysical analysis failed: {e}");
                error!("{}", err_msg);
                self.repo
                    .update_job_status(job_id, JobStatus::Failed, Some(err_msg.clone()))
                    .await?;
                let _ = self.events_tx.send(EngineEvent::JobFailed {
                    job_id,
                    error: err_msg,
                });
                return Err(EngineError::Core(e));
            }
        };

        // Persist prediction
        let prediction_id = Uuid::new_v4();
        metrics.prediction_id = prediction_id;

        let prediction = Prediction {
            id: prediction_id,
            job_id,
            pdb_path: run_result.pdb_path.to_string_lossy().to_string(),
            plddt: run_result.plddt.or(Some(metrics.plddt_distribution.mean)),
            confidence_category: Some(categorize_plddt(
                run_result.plddt.unwrap_or(metrics.plddt_distribution.mean),
            )),
            metadata: run_result.metadata,
        };

        self.repo.insert_prediction(&prediction).await?;
        self.repo.insert_metrics(&metrics).await?;
        self.repo
            .update_job_status(job_id, JobStatus::Completed, None)
            .await?;

        let _ = self
            .repo
            .log_event(Some(job_id), "INFO", "Job completed successfully")
            .await;
        let _ = self.events_tx.send(EngineEvent::JobCompleted {
            job_id,
            prediction_id,
        });

        info!("Job {} completed successfully", job_id);
        Ok(())
    }
}

fn categorize_plddt(score: f64) -> String {
    if score >= 90.0 {
        "Very High (Good for detailed modeling)".to_string()
    } else if score >= 70.0 {
        "Confident (Good backbone)".to_string()
    } else if score >= 50.0 {
        "Low (Consider flexible/disordered)".to_string()
    } else {
        "Very Low (Unstructured)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulated::SimulatedRunner;
    use chrono::Utc;
    use proteus_core::models::{PipelineTier, Sequence};
    use proteus_storage::pool::create_in_memory_pool;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_pipeline_scheduler_end_to_end() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let runner = Arc::new(SimulatedRunner::new());
        let tmp = tempdir().unwrap();

        let scheduler = PipelineScheduler::new(repo.clone(), runner, tmp.path().to_path_buf());
        let mut rx = scheduler.subscribe();

        // 1. Create Sequence
        let seq = Sequence {
            id: Uuid::new_v4(),
            header: "test_alpha_helix".into(),
            fasta: "ACDEFGHIKLMNPQRSTVWY".into(),
            length: 20,
            created_at: Utc::now(),
        };
        repo.insert_sequence(&seq).await.unwrap();

        // 2. Create Job
        let job = proteus_core::models::PipelineJob {
            id: Uuid::new_v4(),
            sequence_id: seq.id,
            tier: PipelineTier::FastScreening,
            status: JobStatus::Queued,
            priority: 1,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error_log: None,
        };
        repo.insert_job(&job).await.unwrap();

        // 3. Process Job
        scheduler.process_job(job.id).await.unwrap();

        // 4. Assert Job is completed
        let updated_job = repo.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(updated_job.status, JobStatus::Completed);
        assert!(updated_job.started_at.is_some());
        assert!(updated_job.completed_at.is_some());

        // 5. Assert Prediction exists
        let pred = repo.get_prediction_by_job(job.id).await.unwrap().unwrap();
        assert!(std::path::Path::new(&pred.pdb_path).exists());
        assert!(pred.plddt.is_some());

        // 6. Assert Biophysical Metrics exist
        let metrics = repo
            .get_metrics_by_prediction(pred.id)
            .await
            .unwrap()
            .unwrap();
        assert!(metrics.radius_of_gyration > 0.0);
        assert!(metrics.plddt_distribution.mean > 70.0);

        // 7. Verify Events received
        let mut event_types = Vec::new();
        while let Ok(event) = rx.try_recv() {
            event_types.push(match event {
                EngineEvent::JobStarted { .. } => "Started",
                EngineEvent::JobProgress { .. } => "Progress",
                EngineEvent::JobCompleted { .. } => "Completed",
                EngineEvent::JobFailed { .. } => "Failed",
                _ => "Other",
            });
        }
        assert!(event_types.contains(&"Started"));
        assert!(event_types.contains(&"Progress"));
        assert!(event_types.contains(&"Completed"));
    }
}
