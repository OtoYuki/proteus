use crate::error::EngineError;
use crate::runner::ComputeRunner;
use chrono::Utc;
use proteus_core::metrics::analyze_pdb_file;
use proteus_core::models::{JobStatus, Prediction};
use proteus_core::tes::{TesExecutorLog, TesOutputFileLog, TesState, TesTask, TesTaskLog};
use proteus_storage::cas::CasStore;
use proteus_storage::repository::ProteusRepository;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::tes_exec::{mount_root, ExecutorRequest, HostExecutor, TesExecutor};
use tracing::{debug, error, info, warn};
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
    TesTaskStarted {
        task_id: String,
    },
    TesTaskCompleted {
        task_id: String,
    },
    TesTaskFailed {
        task_id: String,
        error: String,
    },
}

/// Result of a TES cancel request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// The task was running or queued and is now `CANCELED`.
    Canceled,
    /// The task had already finished; nothing changed.
    AlreadyTerminal,
    /// No task with this id exists.
    NotFound,
}

/// How GA4GH TES tasks are executed.
#[derive(Clone)]
pub struct TesExecutionConfig {
    pub executor: Arc<dyn TesExecutor>,
    /// Glob patterns; when non-empty an executor image must match one of them.
    pub allow_images: Vec<glob::Pattern>,
    /// Wall-clock limit per executor.
    pub executor_timeout: Duration,
    /// Give containers outbound network access.
    pub network: bool,
}

impl Default for TesExecutionConfig {
    /// Host executor, no allow-list, 1 h timeout — the loopback-development default.
    fn default() -> Self {
        Self {
            executor: Arc::new(HostExecutor),
            allow_images: Vec::new(),
            executor_timeout: Duration::from_secs(3600),
            network: false,
        }
    }
}

impl TesExecutionConfig {
    /// Error message when `image` is not covered by the allow-list, else `None`.
    pub fn image_rejection(&self, image: &str) -> Option<String> {
        if self.allow_images.is_empty() || self.allow_images.iter().any(|p| p.matches(image)) {
            None
        } else {
            let allowed: Vec<&str> = self.allow_images.iter().map(|p| p.as_str()).collect();
            Some(format!(
                "executor image '{image}' is not allowed on this server; allowed patterns: {}",
                allowed.join(", ")
            ))
        }
    }
}

#[derive(Clone)]
pub struct PipelineScheduler {
    repo: ProteusRepository,
    runner: Arc<dyn ComputeRunner>,
    events_tx: broadcast::Sender<EngineEvent>,
    artifacts_dir: PathBuf,
    cas: Arc<CasStore>,
    tes: TesExecutionConfig,
    cancel_tokens: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl PipelineScheduler {
    pub fn new(
        repo: ProteusRepository,
        runner: Arc<dyn ComputeRunner>,
        artifacts_dir: PathBuf,
    ) -> Self {
        let (events_tx, _) = broadcast::channel(128);
        let cas_dir = artifacts_dir.join("cas");
        let cas =
            Arc::new(CasStore::new(&cas_dir).unwrap_or_else(|_| {
                CasStore::new(std::env::temp_dir().join("proteus-cas")).unwrap()
            }));
        Self {
            repo,
            runner,
            events_tx,
            artifacts_dir,
            cas,
            tes: TesExecutionConfig::default(),
            cancel_tokens: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Replace the TES execution backend/policy (container executor, allow-list, timeout).
    pub fn with_tes_config(mut self, tes: TesExecutionConfig) -> Self {
        self.tes = tes;
        self
    }

    pub fn tes_config(&self) -> &TesExecutionConfig {
        &self.tes
    }

    pub fn with_cas(
        repo: ProteusRepository,
        runner: Arc<dyn ComputeRunner>,
        artifacts_dir: PathBuf,
        cas: Arc<CasStore>,
    ) -> Self {
        let (events_tx, _) = broadcast::channel(128);
        Self {
            repo,
            runner,
            events_tx,
            artifacts_dir,
            cas,
            tes: TesExecutionConfig::default(),
            cancel_tokens: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn cas(&self) -> &CasStore {
        &self.cas
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

    /// Process a batch of jobs concurrently with a bounded worker pool.
    pub async fn process_batch(
        &self,
        job_ids: &[Uuid],
        max_concurrency: usize,
    ) -> Vec<Result<(), EngineError>> {
        let concurrency = max_concurrency.max(1);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut handles = Vec::with_capacity(job_ids.len());

        for &id in job_ids {
            let sched = self.clone();
            let sem = semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.map_err(|e| {
                    EngineError::Pipeline(format!("Semaphore acquisition failed: {e}"))
                })?;
                sched.process_job(id).await
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(res) => results.push(res),
                Err(e) => results.push(Err(EngineError::Pipeline(format!(
                    "Worker task panicked or was cancelled: {e}"
                )))),
            }
        }
        results
    }

    /// Enqueue a GA4GH TES task for asynchronous background processing.
    /// Reject tasks the server will not run: disallowed images, relative or system paths.
    pub fn validate_tes_task(&self, task: &TesTask) -> Result<(), EngineError> {
        if task.executors.is_empty() {
            return Err(EngineError::Tes("task has no executors".into()));
        }
        for ex in &task.executors {
            if ex.image.trim().is_empty() {
                return Err(EngineError::Tes("executor image must not be empty".into()));
            }
            if let Some(msg) = self.tes.image_rejection(&ex.image) {
                return Err(EngineError::Tes(msg));
            }
        }
        Self::mount_roots(task)?;
        Ok(())
    }

    /// Container directories that must be bind-mounted for this task.
    fn mount_roots(task: &TesTask) -> Result<BTreeSet<String>, EngineError> {
        let mut roots = BTreeSet::new();
        for p in task
            .inputs
            .iter()
            .map(|i| i.path.as_str())
            .chain(task.outputs.iter().map(|o| o.path.as_str()))
            .chain(task.volumes.iter().map(|v| v.as_str()))
            .chain(task.executors.iter().filter_map(|e| e.workdir.as_deref()))
        {
            roots.insert(mount_root(p)?);
        }
        Ok(roots)
    }

    pub async fn submit_tes_task(&self, mut task: TesTask) -> Result<String, EngineError> {
        self.validate_tes_task(&task)?;
        if task.id.is_empty() {
            task.id = format!("task-{}", Uuid::new_v4());
        }
        task.state = TesState::Queued;
        if task.creation_time.is_none() {
            task.creation_time = Some(Utc::now().to_rfc3339());
        }

        let task_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Failed to serialize TES task: {e}")))?;

        self.repo
            .insert_tes_task(
                &task.id,
                &task.state.to_string(),
                task.name.as_deref(),
                task.description.as_deref(),
                &task_json,
            )
            .await?;

        let sched = self.clone();
        let task_id = task.id.clone();
        tokio::spawn(async move {
            if let Err(e) = sched.process_tes_task(&task_id).await {
                error!("Error executing TES task {}: {}", task_id, e);
            }
        });

        Ok(task.id)
    }

    /// Cancels a running or queued TES task.
    /// Cancel a TES task. Returns `Ok(CancelOutcome::NotFound)` only for unknown ids;
    /// cancelling a task that already reached a terminal state is an idempotent no-op
    /// (`AlreadyTerminal`), as clients such as Nextflow and Sprocket retry cancels freely.
    pub async fn cancel_tes_task(&self, task_id: &str) -> Result<CancelOutcome, EngineError> {
        if let Some(record) = self.repo.get_tes_task(task_id).await? {
            if record.state == "COMPLETE"
                || record.state == "EXECUTOR_ERROR"
                || record.state == "SYSTEM_ERROR"
                || record.state == "CANCELED"
            {
                return Ok(CancelOutcome::AlreadyTerminal);
            }
            let mut task: TesTask =
                serde_json::from_str(&record.task_json).unwrap_or_else(|_| TesTask {
                    id: task_id.to_string(),
                    ..Default::default()
                });
            task.state = TesState::Canceled;
            let updated_json = serde_json::to_string(&task)
                .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
            self.repo
                .update_tes_task_state(task_id, "CANCELED", &updated_json)
                .await?;
            if let Some(token) = self.cancel_tokens.lock().unwrap().get(task_id) {
                token.cancel();
            }
            let _ = self.events_tx.send(EngineEvent::TesTaskFailed {
                task_id: task_id.to_string(),
                error: "Task canceled by user".into(),
            });
            Ok(CancelOutcome::Canceled)
        } else {
            Ok(CancelOutcome::NotFound)
        }
    }

    /// Process a GA4GH TES task through the execution lifecycle.
    pub async fn process_tes_task(&self, task_id: &str) -> Result<(), EngineError> {
        let record = match self.repo.get_tes_task(task_id).await? {
            Some(r) => r,
            None => {
                return Err(EngineError::Tes(format!("TES task {task_id} not found")));
            }
        };

        if record.state == "CANCELED" {
            return Ok(());
        }

        let mut task: TesTask = serde_json::from_str(&record.task_json)
            .map_err(|e| EngineError::Tes(format!("Invalid task JSON for {task_id}: {e}")))?;

        info!("Starting TES task execution for {}", task_id);
        let _ = self.events_tx.send(EngineEvent::TesTaskStarted {
            task_id: task_id.to_string(),
        });

        // 1. Initializing state
        task.state = TesState::Initializing;
        let mut task_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
        self.repo
            .update_tes_task_state(task_id, &task.state.to_string(), &task_json)
            .await?;

        let work_dir = self.artifacts_dir.join("tes").join(task_id);
        tokio::fs::create_dir_all(&work_dir).await?;

        // 2. Stage inputs
        for input in &task.inputs {
            let rel_path = input.path.trim_start_matches('/');
            let target_path = work_dir.join(rel_path);
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            if let Some(ref content) = input.content {
                tokio::fs::write(&target_path, content).await?;
            } else if let Some(ref url) = input.url {
                if let Some(src) = url.strip_prefix("file://") {
                    if std::path::Path::new(src).exists() {
                        tokio::fs::copy(src, &target_path).await?;
                    }
                } else if url.starts_with("http://") || url.starts_with("https://") {
                    let resp = reqwest::get(url).await.map_err(|e| {
                        EngineError::Tes(format!("Failed to fetch input URL {url}: {e}"))
                    })?;
                    let bytes = resp.bytes().await.map_err(|e| {
                        EngineError::Tes(format!("Failed to read input response from {url}: {e}"))
                    })?;
                    tokio::fs::write(&target_path, bytes).await?;
                } else if std::path::Path::new(url).exists() {
                    tokio::fs::copy(url, &target_path).await?;
                }
            }
        }

        // Pre-create parent directories for declared outputs
        for output in &task.outputs {
            let rel_path = output.path.trim_start_matches('/');
            let target_path = work_dir.join(rel_path);
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        // 3. Running state
        task.state = TesState::Running;
        let task_start_time = Utc::now().to_rfc3339();
        let mut task_log = TesTaskLog {
            start_time: Some(task_start_time),
            ..Default::default()
        };
        task_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
        self.repo
            .update_tes_task_state(task_id, &task.state.to_string(), &task_json)
            .await?;

        let mut executor_failed = false;
        let cancel = CancellationToken::new();
        self.cancel_tokens
            .lock()
            .unwrap()
            .insert(task_id.to_string(), cancel.clone());
        let mount_roots = Self::mount_roots(&task)?;
        task_log
            .system_logs
            .push(format!("executor backend: {}", self.tes.executor.kind()));

        for executor in &task.executors {
            // Check for cancellation
            if cancel.is_cancelled() {
                task.state = TesState::Canceled;
                break;
            }
            if let Some(cur) = self.repo.get_tes_task(task_id).await? {
                if cur.state == "CANCELED" {
                    self.cancel_tokens.lock().unwrap().remove(task_id);
                    return Ok(());
                }
            }

            if executor.command.is_empty() {
                continue;
            }

            let exec_start = Utc::now().to_rfc3339();
            let request = ExecutorRequest {
                image: &executor.image,
                command: &executor.command,
                workdir: executor.workdir.as_deref(),
                env: &executor.env,
                stdin: executor.stdin.as_deref(),
                work_dir: &work_dir,
                mount_roots: &mount_roots,
                cpu_cores: task.resources.cpu_cores,
                ram_gb: task.resources.ram_gb,
                network: self.tes.network
                    || task
                        .tags
                        .get("proteus.network")
                        .map(|v| v == "true")
                        .unwrap_or(false),
                timeout: self.tes.executor_timeout,
                cancel: cancel.clone(),
            };
            let (stdout_str, stderr_str, exit_code) = match self.tes.executor.run(request).await {
                Ok(r) => {
                    task_log.system_logs.extend(r.system_logs);
                    (r.stdout, r.stderr, r.exit_code)
                }
                Err(e) => {
                    task_log
                        .system_logs
                        .push(format!("executor backend error: {e}"));
                    task.state = TesState::SystemError;
                    executor_failed = true;
                    (String::new(), e.to_string(), -1)
                }
            };

            // Write stdout/stderr to files if requested
            if let Some(ref stdout_file) = executor.stdout {
                let p = work_dir.join(stdout_file.trim_start_matches('/'));
                if let Some(parent) = p.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                let _ = tokio::fs::write(&p, &stdout_str).await;
            }
            if let Some(ref stderr_file) = executor.stderr {
                let p = work_dir.join(stderr_file.trim_start_matches('/'));
                if let Some(parent) = p.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                let _ = tokio::fs::write(&p, &stderr_str).await;
            }

            task_log.logs.push(TesExecutorLog {
                start_time: Some(exec_start),
                end_time: Some(Utc::now().to_rfc3339()),
                stdout: Some(stdout_str),
                stderr: Some(stderr_str),
                exit_code: Some(exit_code as i32),
            });

            if task.state == TesState::SystemError {
                break;
            }
            if cancel.is_cancelled() {
                task.state = TesState::Canceled;
                break;
            }
            if exit_code != 0 && !executor.ignore_error {
                executor_failed = true;
                task.state = TesState::ExecutorError;
                break;
            }
        }
        self.cancel_tokens.lock().unwrap().remove(task_id);

        // 4. Output harvesting & Biophysics trigger
        for output in &task.outputs {
            let out_target = work_dir.join(output.path.trim_start_matches('/'));
            if out_target.exists() {
                let size = tokio::fs::metadata(&out_target)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                task_log.outputs.push(TesOutputFileLog {
                    url: format!("file://{}", out_target.to_string_lossy()),
                    path: output.path.clone(),
                    size_bytes: Some(size),
                });

                // Auto-analyze PDB outputs through core biophysics
                let path_str = out_target.to_string_lossy().to_string();
                if path_str.ends_with(".pdb") || path_str.ends_with(".cif") {
                    if let Ok(entry) = self.cas.store_file(&out_target) {
                        let _ = self
                            .repo
                            .record_cas_object(&entry.hash, entry.size_bytes as i64)
                            .await;
                        debug!(hash = %entry.hash, "Harvested PDB structure to CAS");
                    }
                    if let Ok(biophysics) = analyze_pdb_file(&out_target, None) {
                        let _ = self.repo.insert_metrics(&biophysics).await;
                        debug!(
                            "Harvested and analyzed biophysical metrics for TES output: {}",
                            path_str
                        );
                    }
                }
            }
        }

        if !executor_failed && task.state != TesState::Canceled {
            task.state = TesState::Complete;
        }

        task_log.end_time = Some(Utc::now().to_rfc3339());
        task.logs.push(task_log);

        let final_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
        self.repo
            .update_tes_task_state(task_id, &task.state.to_string(), &final_json)
            .await?;

        if task.state == TesState::Complete {
            let _ = self.events_tx.send(EngineEvent::TesTaskCompleted {
                task_id: task_id.to_string(),
            });
            info!("TES task {} completed successfully", task_id);
        } else {
            let _ = self.events_tx.send(EngineEvent::TesTaskFailed {
                task_id: task_id.to_string(),
                error: format!("TES task finished with state {:?}", task.state),
            });
            warn!("TES task {} finished with state {:?}", task_id, task.state);
        }

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

    #[tokio::test]
    async fn test_pipeline_scheduler_concurrent_batch() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let runner = Arc::new(SimulatedRunner::new());
        let tmp = tempdir().unwrap();

        let scheduler = PipelineScheduler::new(repo.clone(), runner, tmp.path().to_path_buf());

        let mut job_ids = Vec::new();
        for i in 0..4 {
            let seq = Sequence {
                id: Uuid::new_v4(),
                header: format!("batch_seq_{i}"),
                fasta: "ACDEFGHIKLMNPQRSTVWY".into(),
                length: 20,
                created_at: Utc::now(),
            };
            repo.insert_sequence(&seq).await.unwrap();

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
            job_ids.push(job.id);
        }

        let results = scheduler.process_batch(&job_ids, 2).await;
        assert_eq!(results.len(), 4);
        for res in results {
            assert!(res.is_ok());
        }

        for id in job_ids {
            let job = repo.get_job(id).await.unwrap().unwrap();
            assert_eq!(job.status, JobStatus::Completed);
        }
    }

    #[tokio::test]
    async fn test_pipeline_scheduler_tes_task_execution() {
        use proteus_core::tes::{TesExecutor, TesInput, TesOutput};

        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let runner = Arc::new(SimulatedRunner::new());
        let tmp = tempdir().unwrap();

        let scheduler = PipelineScheduler::new(repo.clone(), runner, tmp.path().to_path_buf());

        let task = TesTask {
            id: "tes-task-unit-test".into(),
            name: Some("test_job".into()),
            description: Some("Testing TES task execution".into()),
            inputs: vec![TesInput {
                name: Some("input_fasta".into()),
                description: None,
                url: None,
                path: "/data/input.txt".into(),
                type_: proteus_core::tes::TesFileType::File,
                content: Some("Proteus Bio-Compute Engine".into()),
            }],
            executors: vec![TesExecutor {
                image: "docker.io/library/alpine:3.20".into(),
                command: vec![
                    "sh".into(),
                    "-c".into(),
                    "cat input.txt > output.txt && echo 'Executor finished'".into(),
                ],
                workdir: Some("/data".into()),
                stdout: Some("/data/exec.stdout".into()),
                stderr: Some("/data/exec.stderr".into()),
                stdin: None,
                env: std::collections::HashMap::new(),
                ignore_error: false,
            }],
            outputs: vec![TesOutput {
                name: Some("output_file".into()),
                description: None,
                url: None,
                path: "/data/output.txt".into(),
                type_: proteus_core::tes::TesFileType::File,
            }],
            ..Default::default()
        };

        // Submit task
        let task_id = scheduler.submit_tes_task(task).await.unwrap();
        assert_eq!(task_id, "tes-task-unit-test");

        // Wait a short moment for background execution
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let record = repo.get_tes_task(&task_id).await.unwrap().unwrap();
        assert_eq!(record.state, "COMPLETE");

        let finished_task: TesTask = serde_json::from_str(&record.task_json).unwrap();
        assert_eq!(finished_task.state, TesState::Complete);
        assert_eq!(finished_task.logs.len(), 1);
        assert_eq!(finished_task.logs[0].logs.len(), 1);
        assert_eq!(finished_task.logs[0].logs[0].exit_code, Some(0));
        assert!(finished_task.logs[0].logs[0]
            .stdout
            .as_ref()
            .unwrap()
            .contains("Executor finished"));
        assert_eq!(finished_task.logs[0].outputs.len(), 1);
        assert_eq!(finished_task.logs[0].outputs[0].path, "/data/output.txt");
    }
}
