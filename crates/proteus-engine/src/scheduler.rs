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
    /// A cancel request was accepted for a queued or running task.
    TesTaskCanceled {
        task_id: String,
    },
    /// A harvested artefact was written to (or found in) the content-addressable store.
    CasStored {
        duplicate: bool,
        bytes: u64,
    },
    /// One full biophysical profile was computed.
    BiophysicsAnalyzed {
        duration_ms: u64,
        residues: usize,
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
    /// Host directories that `file://` input and output URLs may reference. Empty = none.
    pub allow_dirs: Vec<PathBuf>,
}

impl Default for TesExecutionConfig {
    /// Host executor, no image allow-list, no `file://` directories, 1 h timeout — the
    /// loopback-development default.
    fn default() -> Self {
        Self {
            executor: Arc::new(HostExecutor),
            allow_images: Vec::new(),
            executor_timeout: Duration::from_secs(3600),
            network: false,
            allow_dirs: Vec::new(),
        }
    }
}

impl TesExecutionConfig {
    /// Error message when a `file://` URL (or bare absolute host path) used as a task input or
    /// output lies outside `allow_dirs`, else `None`. URLs with other schemes are not host
    /// paths and always return `None`.
    pub fn host_path_rejection(&self, url: &str) -> Option<String> {
        let host_path = host_path_of(url)?;
        let allowed = resolve_host_path(host_path).is_some_and(|real| {
            self.allow_dirs
                .iter()
                .filter_map(|d| d.canonicalize().ok())
                .any(|d| real.starts_with(&d))
        });
        if allowed {
            None
        } else {
            let dirs: Vec<String> = self
                .allow_dirs
                .iter()
                .map(|d| d.display().to_string())
                .collect();
            Some(format!(
                "'{url}' is outside the directories this server allows for file:// URLs ({})",
                if dirs.is_empty() {
                    "none configured; see --allow-dir".to_string()
                } else {
                    dirs.join(", ")
                }
            ))
        }
    }

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

/// The host path a task URL refers to: `file://<path>` or a bare absolute path. `None` for
/// every other scheme.
fn host_path_of(url: &str) -> Option<&std::path::Path> {
    let p = url
        .strip_prefix("file://")
        .or_else(|| url.starts_with('/').then_some(url))?;
    let p = std::path::Path::new(p);
    p.is_absolute().then_some(p)
}

/// Resolve a host path through the filesystem: canonicalise the deepest existing ancestor
/// (following symlinks) and re-append the not-yet-existing tail. `None` when the tail
/// contains `..` or nothing on the path exists.
fn resolve_host_path(path: &std::path::Path) -> Option<PathBuf> {
    let mut existing = path;
    let mut tail = Vec::new();
    while !existing.exists() {
        // `file_name()` is `None` for `..` and for the root, both of which end the search.
        tail.push(existing.file_name()?.to_os_string());
        existing = existing.parent()?;
    }
    let mut real = existing.canonicalize().ok()?;
    for name in tail.into_iter().rev() {
        real.push(name);
    }
    Some(real)
}

/// How [`PipelineScheduler::run_tes_lifecycle`] ended.
enum Lifecycle {
    /// Ran to a terminal state, which is in `task.state`.
    Finished,
    /// A cancel was recorded first; the task must not be written again.
    LostToCancel,
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

    pub fn cas(&self) -> &CasStore {
        &self.cas
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.events_tx.subscribe()
    }

    pub fn repo(&self) -> &ProteusRepository {
        &self.repo
    }

    /// `analyze_pdb_file` plus a `BiophysicsAnalyzed` event carrying the wall-clock time.
    fn analyze_timed(
        &self,
        path: &std::path::Path,
    ) -> Result<proteus_core::models::BiophysicalMetrics, proteus_core::CoreError> {
        let started = std::time::Instant::now();
        let metrics = analyze_pdb_file(path, None)?;
        let _ = self.events_tx.send(EngineEvent::BiophysicsAnalyzed {
            duration_ms: started.elapsed().as_millis() as u64,
            residues: metrics
                .secondary_structure_summary
                .as_ref()
                .map_or(0, |s| s.assignment.len()),
        });
        Ok(metrics)
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

        // Claim, don't overwrite: the daemon's queue poller, its API handler and a CLI on the
        // same data dir may all reach this for one job.
        if !self.repo.claim_job(job_id).await? {
            debug!("job {job_id} is not queued (already claimed or finished); skipping");
            return Ok(());
        }
        info!("Starting pipeline execution for job: {}", job_id);
        let _ = self.events_tx.send(EngineEvent::JobStarted { job_id });
        let _ = self
            .repo
            .log_event(Some(job_id), "INFO", "Job started")
            .await;

        // Every failure from here on — the runner, the analysis, but also an unwritable work
        // dir or a database error — must leave the job Failed and end its event stream; a job
        // left Running keeps `/events` open and the worker gauge up forever.
        match self.run_started_job(&job, &seq).await {
            Ok(prediction_id) => {
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
            Err(e) => {
                let err_msg = e.to_string();
                error!("Job {} failed: {}", job_id, err_msg);
                let _ = self
                    .repo
                    .update_job_status(job_id, JobStatus::Failed, Some(err_msg.clone()))
                    .await;
                let _ = self.events_tx.send(EngineEvent::JobFailed {
                    job_id,
                    error: err_msg.clone(),
                });
                let _ = self
                    .repo
                    .log_event(Some(job_id), "ERROR", &format!("Job failed: {err_msg}"))
                    .await;
                Err(e)
            }
        }
    }

    /// The part of [`process_job`](Self::process_job) after the job is marked Running: fold,
    /// analyse, check, persist. Returns the prediction id.
    async fn run_started_job(
        &self,
        job: &proteus_core::models::PipelineJob,
        seq: &proteus_core::models::Sequence,
    ) -> Result<Uuid, EngineError> {
        let job_id = job.id;
        let work_dir = self.artifacts_dir.join(job_id.to_string());
        tokio::fs::create_dir_all(&work_dir).await?;

        let _ = self.events_tx.send(EngineEvent::JobProgress {
            job_id,
            step: "Running compute container".into(),
            percent: 30,
        });
        let run_result = self.runner.execute_job(job, seq, &work_dir).await?;

        let _ = self.events_tx.send(EngineEvent::JobProgress {
            job_id,
            step: "Analyzing biophysical properties".into(),
            percent: 75,
        });
        let mut metrics = self
            .analyze_timed(&run_result.pdb_path)
            .map_err(|e| EngineError::Pipeline(format!("Biophysical analysis failed: {e}")))?;

        // A structure that does not cover the sequence (truncated or wrong output from a
        // predictor) must not be scored as if it did.
        let residues = metrics
            .secondary_structure_summary
            .as_ref()
            .map_or(0, |s| s.assignment.len());
        if residues != seq.length {
            return Err(EngineError::Pipeline(format!(
                "predicted structure has {residues} residues but the sequence has {} \
                 (runner output {})",
                seq.length,
                run_result.pdb_path.display()
            )));
        }

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
        Ok(prediction_id)
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
        for url in task
            .inputs
            .iter()
            .filter_map(|i| i.url.as_deref())
            .chain(task.outputs.iter().filter_map(|o| o.url.as_deref()))
        {
            if let Some(msg) = self.tes.host_path_rejection(url) {
                return Err(EngineError::Tes(msg));
            }
        }
        // Proteus defines no backend parameters; under `backend_parameters_strict` any key is unknown.
        if task.resources.backend_parameters_strict == Some(true) {
            if let Some(params) = &task.resources.backend_parameters {
                if let Some(key) = params.keys().next() {
                    return Err(EngineError::Tes(format!(
                        "unknown backend parameter '{key}' (backend_parameters_strict is set; this server accepts none)"
                    )));
                }
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
            .chain(task.executors.iter().filter_map(|e| e.stdout.as_deref()))
            .chain(task.executors.iter().filter_map(|e| e.stderr.as_deref()))
        {
            roots.insert(mount_root(p)?);
        }
        Ok(roots)
    }

    pub async fn submit_tes_task(&self, mut task: TesTask) -> Result<String, EngineError> {
        self.validate_tes_task(&task)?;
        // `id`, `state`, `logs` and `creation_time` are output-only in TES 1.1. The id in
        // particular names the task's work dir on the host, so a client-chosen one (`../x`,
        // `/abs/path`, or another task's id) must never reach the filesystem.
        task.id = format!("task-{}", Uuid::new_v4());
        task.state = TesState::Queued;
        task.logs.clear();
        task.creation_time = Some(Utc::now().to_rfc3339());

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

    /// Copy or fetch every declared input into the task work dir. `file://` sources (and bare
    /// host paths) are checked against `allow_dirs` again here, so a task record that never went
    /// through [`validate_tes_task`](Self::validate_tes_task) cannot read outside them either.
    async fn stage_inputs(&self, task: &TesTask, work_dir: &std::path::Path) -> Result<(), String> {
        for input in &task.inputs {
            let target_path = work_dir.join(input.path.trim_start_matches('/'));
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let Some(url) = input.url.as_deref() else {
                if let Some(content) = &input.content {
                    tokio::fs::write(&target_path, content)
                        .await
                        .map_err(|e| format!("{}: {e}", input.path))?;
                }
                continue;
            };
            if let Some(content) = &input.content {
                tokio::fs::write(&target_path, content)
                    .await
                    .map_err(|e| format!("{}: {e}", input.path))?;
            } else if let Some(src) = host_path_of(url) {
                if let Some(msg) = self.tes.host_path_rejection(url) {
                    return Err(msg);
                }
                if !src.exists() {
                    return Err(format!("input {url} does not exist on the server host"));
                }
                copy_recursive(src, &target_path)
                    .await
                    .map_err(|e| format!("copy {url}: {e}"))?;
            } else if url.starts_with("http://") || url.starts_with("https://") {
                let resp = reqwest::get(url)
                    .await
                    .map_err(|e| format!("fetch {url}: {e}"))?;
                let bytes = resp.bytes().await.map_err(|e| format!("read {url}: {e}"))?;
                tokio::fs::write(&target_path, bytes)
                    .await
                    .map_err(|e| format!("{}: {e}", input.path))?;
            } else {
                return Err(format!(
                    "input URL scheme not supported: {url} (file:// and http(s):// only)"
                ));
            }
        }
        Ok(())
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
            // The worker may have finished between the read above and this write; its terminal
            // state stands and the cancel becomes the idempotent no-op it would have been.
            if !self
                .repo
                .update_tes_task_state_unless_terminal(task_id, "CANCELED", &updated_json)
                .await?
            {
                return Ok(CancelOutcome::AlreadyTerminal);
            }
            if let Some(token) = self.cancel_tokens.lock().unwrap().get(task_id) {
                token.cancel();
            }
            let _ = self.events_tx.send(EngineEvent::TesTaskCanceled {
                task_id: task_id.to_string(),
            });
            Ok(CancelOutcome::Canceled)
        } else {
            Ok(CancelOutcome::NotFound)
        }
    }

    /// Close out TES tasks a previous daemon process left unfinished. Their workers died with
    /// that process, so nothing will ever move them on: each becomes SYSTEM_ERROR with a log
    /// line saying why, and the container executor removes any container still labelled with
    /// the task. Call once at daemon start, before serving; assumes one daemon per data dir.
    /// Host-executor processes cannot be found again and are not stopped.
    pub async fn recover_interrupted_tes_tasks(&self) -> Result<usize, EngineError> {
        let ids = self.repo.unfinished_tes_task_ids().await?;
        for id in &ids {
            let Some(record) = self.repo.get_tes_task(id).await? else {
                continue;
            };
            let mut task: TesTask =
                serde_json::from_str(&record.task_json).unwrap_or_else(|_| TesTask {
                    id: id.clone(),
                    ..Default::default()
                });
            if let Err(e) = self.tes.executor.stop_task(id).await {
                warn!("could not remove containers of interrupted task {id}: {e}");
            }
            task.state = TesState::SystemError;
            task.logs.push(TesTaskLog {
                end_time: Some(Utc::now().to_rfc3339()),
                system_logs: vec![format!(
                    "the daemon restarted while this task was {}; it was not resumed",
                    record.state
                )],
                ..Default::default()
            });
            let json = serde_json::to_string(&task)
                .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
            self.repo
                .update_tes_task_state_unless_terminal(id, "SYSTEM_ERROR", &json)
                .await?;
            warn!(
                "TES task {id} was {} when the daemon stopped; now SYSTEM_ERROR",
                record.state
            );
        }
        Ok(ids.len())
    }

    /// Process a GA4GH TES task through the execution lifecycle.
    ///
    /// Owns the task's end: exactly one terminal state write (refused if a cancel got there
    /// first — a terminal state never changes again) and exactly one `TesTaskCompleted` /
    /// `TesTaskFailed` event after `TesTaskStarted`, whatever path the lifecycle took, including
    /// an I/O or database error half-way through.
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
        // The cancel token is registered before the first state write so that a cancel arriving
        // at any later point is seen; non-terminal writes are refused once the row is CANCELED.
        let cancel = CancellationToken::new();
        self.cancel_tokens
            .lock()
            .unwrap()
            .insert(task_id.to_string(), cancel.clone());

        let outcome = self.run_tes_lifecycle(task_id, &mut task, &cancel).await;
        self.cancel_tokens.lock().unwrap().remove(task_id);

        let mut error = None;
        match outcome {
            Ok(Lifecycle::Finished) => {}
            Ok(Lifecycle::LostToCancel) => {
                task.state = TesState::Canceled;
            }
            Err(e) => {
                task.state = TesState::SystemError;
                task.logs.push(TesTaskLog {
                    end_time: Some(Utc::now().to_rfc3339()),
                    system_logs: vec![format!("internal error: {e}")],
                    ..Default::default()
                });
                error = Some(e);
            }
        }
        if cancel.is_cancelled() {
            task.state = TesState::Canceled;
        }

        let final_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
        let written = task.state != TesState::Canceled
            && self
                .repo
                .update_tes_task_state_unless_terminal(
                    task_id,
                    &task.state.to_string(),
                    &final_json,
                )
                .await?;
        if written && task.state == TesState::Complete {
            let _ = self.events_tx.send(EngineEvent::TesTaskCompleted {
                task_id: task_id.to_string(),
            });
            info!("TES task {} completed successfully", task_id);
        } else {
            let state = if written {
                task.state
            } else {
                TesState::Canceled
            };
            let _ = self.events_tx.send(EngineEvent::TesTaskFailed {
                task_id: task_id.to_string(),
                error: format!("TES task finished with state {state:?}"),
            });
            warn!(
                "TES task {} finished with state {:?}; system_logs: {:?}",
                task_id,
                state,
                task.logs.last().map(|l| &l.system_logs)
            );
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Stage, execute, harvest. Leaves the terminal state in `task.state` and returns
    /// [`Lifecycle::LostToCancel`] when a cancel was recorded before the task could move on.
    async fn run_tes_lifecycle(
        &self,
        task_id: &str,
        task: &mut TesTask,
        cancel: &CancellationToken,
    ) -> Result<Lifecycle, EngineError> {
        // 1. Initializing state
        task.state = TesState::Initializing;
        let mut task_json = serde_json::to_string(&task)
            .map_err(|e| EngineError::Pipeline(format!("Serialization failed: {e}")))?;
        if !self
            .repo
            .update_tes_task_state_unless_canceled(task_id, &task.state.to_string(), &task_json)
            .await?
        {
            return Ok(Lifecycle::LostToCancel);
        }

        let work_dir = self.artifacts_dir.join("tes").join(task_id);
        tokio::fs::create_dir_all(&work_dir).await?;

        // 2. Stage inputs. A staging failure is the server's problem (SYSTEM_ERROR).
        if let Err(msg) = self.stage_inputs(task, &work_dir).await {
            task.state = TesState::SystemError;
            task.logs.push(TesTaskLog {
                start_time: Some(Utc::now().to_rfc3339()),
                end_time: Some(Utc::now().to_rfc3339()),
                system_logs: vec![format!("input staging failed: {msg}")],
                ..Default::default()
            });
            return Ok(Lifecycle::Finished);
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
        if !self
            .repo
            .update_tes_task_state_unless_canceled(task_id, &task.state.to_string(), &task_json)
            .await?
        {
            return Ok(Lifecycle::LostToCancel);
        }

        let mut executor_failed = false;
        let mount_roots = Self::mount_roots(task)?;
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
                    return Ok(Lifecycle::LostToCancel);
                }
            }

            if executor.command.is_empty() {
                continue;
            }

            let exec_start = Utc::now().to_rfc3339();
            // TES `stdin` names a file in the task's filesystem; its contents are piped in.
            let stdin_bytes = match executor.stdin.as_deref() {
                None => None,
                Some(path) => match read_inside(&work_dir, path).await {
                    Ok(b) => Some(b),
                    Err(e) => {
                        task_log
                            .system_logs
                            .push(format!("cannot read stdin {path}: {e}"));
                        task.state = TesState::SystemError;
                        executor_failed = true;
                        break;
                    }
                },
            };
            let request = ExecutorRequest {
                image: &executor.image,
                command: &executor.command,
                workdir: executor.workdir.as_deref(),
                env: &executor.env,
                stdin: stdin_bytes.as_deref(),
                task_id,
                work_dir: &work_dir,
                mount_roots: &mount_roots,
                cpu_cores: task.resources.cpu_cores,
                ram_gb: task.resources.ram_gb,
                // The operator's `--executor-network` alone decides; a task cannot opt itself in.
                network: self.tes.network,
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

            // Write stdout/stderr to files if requested. The executor has just had write access
            // to the work dir, so these paths may now be symlinks it planted: write only to a
            // freshly created regular file whose real parent is inside the work dir.
            for (file, text) in [
                (&executor.stdout, &stdout_str),
                (&executor.stderr, &stderr_str),
            ] {
                if let Some(file) = file {
                    if let Err(e) = write_inside(&work_dir, file, text.as_bytes()).await {
                        task_log
                            .system_logs
                            .push(format!("could not write {file}: {e}"));
                        task.state = TesState::SystemError;
                        executor_failed = true;
                    }
                }
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
        // 4. Output harvesting: upload to the declared URL (file:// supported), record the log,
        //    and analyse structure files.
        let mut delivery_failed = false;
        let work_dir_real = tokio::fs::canonicalize(&work_dir).await?;
        for output in &task.outputs {
            let out_target = work_dir.join(output.path.trim_start_matches('/'));
            if out_target.exists() {
                // An executor can plant a symlink inside the work dir; follow it and refuse
                // anything that resolves outside (it would be a host file, not a task output).
                match tokio::fs::canonicalize(&out_target).await {
                    Ok(real) if real.starts_with(&work_dir_real) => {}
                    _ => {
                        task_log.system_logs.push(format!(
                            "output {} resolves outside the task work dir; not delivered",
                            output.path
                        ));
                        delivery_failed = true;
                        continue;
                    }
                }
                let is_dir = out_target.is_dir();
                let size = if is_dir {
                    dir_size(&out_target).await
                } else {
                    tokio::fs::metadata(&out_target)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0)
                };
                let mut recorded_url = format!("file://{}", out_target.to_string_lossy());
                if let Some(url) = output.url.as_deref().filter(|u| !u.is_empty()) {
                    let upload = match self.tes.host_path_rejection(url) {
                        Some(msg) => Err(EngineError::Tes(msg)),
                        None => upload_output(&out_target, url).await,
                    };
                    match upload {
                        Ok(()) => recorded_url = url.to_string(),
                        Err(e) => {
                            task_log
                                .system_logs
                                .push(format!("output upload to {url} failed: {e}"));
                            delivery_failed = true;
                        }
                    }
                }
                task_log.outputs.push(TesOutputFileLog {
                    url: recorded_url,
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
                        let _ = self.events_tx.send(EngineEvent::CasStored {
                            duplicate: entry.is_duplicate,
                            bytes: entry.size_bytes,
                        });
                        debug!(hash = %entry.hash, "Harvested PDB structure to CAS");
                    }
                    if let Ok(biophysics) = self.analyze_timed(&out_target) {
                        let _ = self.repo.insert_metrics(&biophysics).await;
                        debug!(
                            "Harvested and analyzed biophysical metrics for TES output: {}",
                            path_str
                        );
                    }
                }
            }
        }

        if delivery_failed {
            task.state = TesState::SystemError;
        } else if !executor_failed && task.state != TesState::Canceled {
            task.state = TesState::Complete;
        }
        task_log.end_time = Some(Utc::now().to_rfc3339());
        task.logs.push(task_log);
        Ok(Lifecycle::Finished)
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

/// Total size of a directory tree in bytes.
async fn dir_size(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(mut rd) = tokio::fs::read_dir(&dir).await else {
            continue;
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            match entry.metadata().await {
                Ok(m) if m.is_dir() => stack.push(entry.path()),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

/// Create `rel` (a container path such as `/data/log.txt`) under `work_dir` and write `bytes`
/// to it, without following a symlink anywhere below `work_dir`: the parent directory must
/// resolve inside `work_dir`, and the file itself is created fresh (`O_CREAT|O_EXCL` does not
/// follow a symlink at the final component; an existing link or file is unlinked first).
async fn write_inside(
    work_dir: &std::path::Path,
    rel: &str,
    bytes: &[u8],
) -> Result<(), std::io::Error> {
    use std::io::{Error, ErrorKind};
    let root = tokio::fs::canonicalize(work_dir).await?;
    let target = work_dir.join(rel.trim_start_matches('/'));
    let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
        return Err(Error::new(ErrorKind::InvalidInput, "not a file path"));
    };
    tokio::fs::create_dir_all(parent).await?;
    let real_parent = tokio::fs::canonicalize(parent).await?;
    if !real_parent.starts_with(&root) {
        return Err(Error::new(
            ErrorKind::PermissionDenied,
            "path resolves outside the task work dir",
        ));
    }
    let target = real_parent.join(name);
    match tokio::fs::symlink_metadata(&target).await {
        Ok(m) if m.is_dir() => {
            return Err(Error::new(ErrorKind::AlreadyExists, "is a directory"));
        }
        Ok(_) => tokio::fs::remove_file(&target).await?,
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    use tokio::io::AsyncWriteExt;
    let mut f = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target)
        .await?;
    f.write_all(bytes).await?;
    f.flush().await
}

/// Read `rel` (a container path) from under `work_dir`, refusing anything that resolves
/// outside it — an earlier executor may have replaced the file with a symlink to a host file.
async fn read_inside(work_dir: &std::path::Path, rel: &str) -> Result<Vec<u8>, std::io::Error> {
    let root = tokio::fs::canonicalize(work_dir).await?;
    let real = tokio::fs::canonicalize(work_dir.join(rel.trim_start_matches('/'))).await?;
    if !real.starts_with(&root) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "path resolves outside the task work dir",
        ));
    }
    tokio::fs::read(real).await
}

/// Copy `src` (file or directory) to `dst`, creating parents.
///
/// `src` itself may be a symlink (callers resolve and check it first), but nothing *inside* a
/// copied directory may be: an executor or another task can plant `d/x -> /etc/shadow`, and
/// following it would carry a host file across the allowed-dir boundary. A symlink found
/// below the top level, on either side, fails the copy.
async fn copy_recursive(
    src: &std::path::Path,
    dst: &std::path::Path,
) -> Result<(), std::io::Error> {
    use std::io::{Error, ErrorKind};
    let refuse = |p: &std::path::Path| {
        Error::new(
            ErrorKind::PermissionDenied,
            format!("refusing to copy through symbolic link {}", p.display()),
        )
    };
    async fn is_symlink(p: &std::path::Path) -> bool {
        tokio::fs::symlink_metadata(p)
            .await
            .is_ok_and(|m| m.file_type().is_symlink())
    }
    if src.is_dir() {
        if is_symlink(dst).await {
            return Err(refuse(dst));
        }
        tokio::fs::create_dir_all(dst).await?;
        let mut stack = vec![(src.to_path_buf(), dst.to_path_buf())];
        while let Some((s, d)) = stack.pop() {
            let mut rd = tokio::fs::read_dir(&s).await?;
            while let Some(entry) = rd.next_entry().await? {
                let target = d.join(entry.file_name());
                let kind = entry.file_type().await?;
                if kind.is_symlink() {
                    return Err(refuse(&entry.path()));
                }
                if is_symlink(&target).await {
                    return Err(refuse(&target));
                }
                if kind.is_dir() {
                    tokio::fs::create_dir_all(&target).await?;
                    stack.push((entry.path(), target));
                } else {
                    tokio::fs::copy(entry.path(), &target).await?;
                }
            }
        }
    } else {
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        if is_symlink(dst).await {
            return Err(refuse(dst));
        }
        tokio::fs::copy(src, dst).await?;
    }
    Ok(())
}

/// Deliver a task output to its declared URL. `file://` URLs and bare absolute host paths
/// (what Nextflow's nf-ga4gh plugin sends) are supported, on a local or shared filesystem;
/// other schemes are reported.
async fn upload_output(local: &std::path::Path, url: &str) -> Result<(), EngineError> {
    if let Some(dest) = host_path_of(url) {
        copy_recursive(local, dest).await.map_err(EngineError::Io)
    } else {
        Err(EngineError::Tes(format!(
            "output URL scheme not supported: {url} (file:// only in this release)"
        )))
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

    /// Scheduler over an in-memory DB and the host executor, rooted at `artifacts`.
    async fn host_scheduler(artifacts: &std::path::Path) -> (PipelineScheduler, ProteusRepository) {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let runner = Arc::new(SimulatedRunner::new());
        let scheduler = PipelineScheduler::new(repo.clone(), runner, artifacts.to_path_buf());
        (scheduler, repo)
    }

    fn sh_task(id: &str, script: &str) -> proteus_core::tes::TesTask {
        proteus_core::tes::TesTask {
            id: id.into(),
            executors: vec![proteus_core::tes::TesExecutor {
                image: "ignored".into(),
                command: vec!["sh".into(), "-c".into(), script.into()],
                workdir: Some("/data".into()),
                stdout: None,
                stderr: None,
                stdin: None,
                env: std::collections::HashMap::new(),
                ignore_error: false,
            }],
            ..Default::default()
        }
    }

    async fn finished(repo: &ProteusRepository, id: &str) -> proteus_core::tes::TesTask {
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let rec = repo.get_tes_task(id).await.unwrap().unwrap();
            if !matches!(rec.state.as_str(), "QUEUED" | "INITIALIZING" | "RUNNING") {
                return serde_json::from_str(&rec.task_json).unwrap();
            }
        }
        panic!("task {id} did not finish");
    }

    #[tokio::test]
    async fn executor_stdout_and_stderr_paths_are_validated_like_other_paths() {
        let tmp = tempdir().unwrap();
        let (scheduler, _) = host_scheduler(tmp.path()).await;
        let mut task = sh_task("t-stdout", "echo hi");
        task.executors[0].stdout = Some("/data/../../escaped.txt".into());
        assert!(scheduler.validate_tes_task(&task).is_err());
        let mut task = sh_task("t-stderr", "echo hi");
        task.executors[0].stderr = Some("relative.txt".into());
        assert!(scheduler.validate_tes_task(&task).is_err());
    }

    #[tokio::test]
    async fn output_symlink_pointing_outside_the_work_dir_is_not_delivered() {
        let tmp = tempdir().unwrap();
        let secret = tmp.path().join("host-secret.txt");
        std::fs::write(&secret, "TOP SECRET").unwrap();
        let dest = tmp.path().join("delivered.txt");
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        });
        let mut task = sh_task("t-symlink", &format!("ln -s {} out.txt", secret.display()));
        task.outputs.push(proteus_core::tes::TesOutput {
            name: None,
            description: None,
            url: Some(format!("file://{}", dest.display())),
            path: "/data/out.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
        });
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert!(
            !dest.exists(),
            "symlinked host file was copied to the output URL"
        );
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
    }

    #[tokio::test]
    async fn a_client_chosen_task_id_never_becomes_a_host_path() {
        // Reported: `"id": "../../outside"` created a work dir above the data dir, and
        // `"id": "<abs>/hostroot"` let inputs and outputs read and write arbitrary host paths.
        let tmp = tempdir().unwrap();
        let artifacts = tmp.path().join("data").join("artifacts");
        let (scheduler, repo) = host_scheduler(&artifacts).await;
        let mut task = sh_task("../../outside", "echo hi > out.txt");
        task.inputs.push(proteus_core::tes::TesInput {
            name: None,
            description: None,
            url: None,
            path: "/data/planted.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
            content: Some("x".into()),
        });
        task.logs.push(TesTaskLog {
            system_logs: vec!["FORGED".into()],
            ..Default::default()
        });
        task.creation_time = Some("1999-01-01T00:00:00Z".into());
        let id = scheduler.submit_tes_task(task).await.unwrap();
        assert!(id.starts_with("task-") && !id.contains('/'), "{id}");
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        assert!(!tmp.path().join("outside").exists());
        assert!(artifacts
            .join("tes")
            .join(&id)
            .join("data/planted.txt")
            .exists());
        // Output-only fields are the server's too.
        assert!(done
            .logs
            .iter()
            .all(|l| !l.system_logs.contains(&"FORGED".to_string())));
        assert_ne!(done.creation_time.as_deref(), Some("1999-01-01T00:00:00Z"));
    }

    #[tokio::test]
    async fn stdout_file_planted_as_a_symlink_does_not_write_through_to_the_host() {
        // Reported: an executor ran `ln -s <host file> /data/log.txt` with `stdout:
        // /data/log.txt`, and the daemon then overwrote the host file with the executor's output.
        let tmp = tempdir().unwrap();
        let victim = tmp.path().join("victim.txt");
        std::fs::write(&victim, "original").unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let mut task = sh_task(
            "",
            &format!(
                "ln -s {} log.txt; echo ATTACKER-CONTROLLED",
                victim.display()
            ),
        );
        task.executors[0].stdout = Some("/data/log.txt".into());
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "original");
        // The planted link is replaced by a real file inside the work dir.
        assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        let log = tmp
            .path()
            .join("artifacts/tes")
            .join(&id)
            .join("data/log.txt");
        assert!(!std::fs::symlink_metadata(&log)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "ATTACKER-CONTROLLED\n"
        );

        // A planted *directory* link cannot be written through either; that fails the task.
        let outside = tmp.path().join("outside-dir");
        std::fs::create_dir_all(&outside).unwrap();
        let mut task = sh_task("", &format!("ln -s {} sub; echo x", outside.display()));
        task.executors[0].stdout = Some("/data/sub/log.txt".into());
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert!(
            !outside.join("log.txt").exists(),
            "wrote through a planted dir link"
        );
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);

        // An ordinary stdout file is still written.
        let mut task = sh_task("", "echo plain");
        task.executors[0].stdout = Some("/data/log.txt".into());
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        let written = tmp
            .path()
            .join("artifacts/tes")
            .join(&id)
            .join("data/log.txt");
        assert_eq!(std::fs::read_to_string(written).unwrap(), "plain\n");
    }

    #[tokio::test]
    async fn symlinks_inside_directory_inputs_and_outputs_are_not_followed() {
        // Reported: a DIRECTORY output containing `leak.txt -> <host file>` delivered the host
        // file, and a DIRECTORY input from an allowed dir did the same on the way in.
        let tmp = tempdir().unwrap();
        let secret = tmp.path().join("outside-secret.txt");
        std::fs::write(&secret, "outside-secret").unwrap();
        let allowed = tmp.path().join("allow");
        std::fs::create_dir_all(allowed.join("indir")).unwrap();
        std::fs::write(allowed.join("indir/ok.txt"), "ok").unwrap();
        std::os::unix::fs::symlink(&secret, allowed.join("indir/s")).unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![allowed.clone()],
            ..Default::default()
        });

        let mut task = sh_task("", "cat in/s");
        task.inputs.push(proteus_core::tes::TesInput {
            name: None,
            description: None,
            url: Some(format!("file://{}", allowed.join("indir").display())),
            path: "/data/in".into(),
            type_: proteus_core::tes::TesFileType::Directory,
            content: None,
        });
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
        assert!(done.logs[0].logs.is_empty(), "the executor must not run");

        let dest = allowed.join("delivered");
        let mut task = sh_task(
            "",
            &format!("mkdir out && ln -s {} out/leak.txt", secret.display()),
        );
        task.outputs.push(proteus_core::tes::TesOutput {
            name: None,
            description: None,
            url: Some(format!("file://{}", dest.display())),
            path: "/data/out".into(),
            type_: proteus_core::tes::TesFileType::Directory,
        });
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
        assert!(!dest.join("leak.txt").exists(), "host file delivered");
    }

    #[tokio::test]
    async fn a_terminal_state_is_never_overwritten() {
        // Reported: a cancel that returned 200 was later overwritten by COMPLETE, because the
        // worker's final write was unconditional.
        let tmp = tempdir().unwrap();
        let (_scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        repo.insert_tes_task("t", "RUNNING", None, None, "{}")
            .await
            .unwrap();
        assert!(repo
            .update_tes_task_state_unless_terminal("t", "CANCELED", "{}")
            .await
            .unwrap());
        assert!(!repo
            .update_tes_task_state_unless_terminal("t", "COMPLETE", "{}")
            .await
            .unwrap());
        assert_eq!(
            repo.get_tes_task("t").await.unwrap().unwrap().state,
            "CANCELED"
        );
    }

    #[tokio::test]
    async fn a_restarted_daemon_closes_out_tasks_the_old_process_left_running() {
        // Reported: after a restart, tasks stayed RUNNING forever.
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        for (id, state) in [("a", "RUNNING"), ("b", "QUEUED"), ("c", "COMPLETE")] {
            let t = TesTask {
                id: id.into(),
                ..Default::default()
            };
            repo.insert_tes_task(id, state, None, None, &serde_json::to_string(&t).unwrap())
                .await
                .unwrap();
        }
        assert_eq!(scheduler.recover_interrupted_tes_tasks().await.unwrap(), 2);
        for (id, want) in [
            ("a", "SYSTEM_ERROR"),
            ("b", "SYSTEM_ERROR"),
            ("c", "COMPLETE"),
        ] {
            let rec = repo.get_tes_task(id).await.unwrap().unwrap();
            assert_eq!(rec.state, want, "{id}");
        }
        let a: TesTask =
            serde_json::from_str(&repo.get_tes_task("a").await.unwrap().unwrap().task_json)
                .unwrap();
        assert_eq!(a.state, TesState::SystemError);
        assert!(a.logs[0].system_logs[0].contains("daemon restarted"));
    }

    #[tokio::test]
    async fn stdin_is_the_file_contents_not_its_path() {
        // Reported: `stdin: /data/in.txt` piped the string "/data/in.txt" into the command.
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let mut task = sh_task("", "cat");
        task.executors[0].stdin = Some("/data/in.txt".into());
        task.inputs.push(proteus_core::tes::TesInput {
            name: None,
            description: None,
            url: None,
            path: "/data/in.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
            content: Some("file body".into()),
        });
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        assert_eq!(done.logs[0].logs[0].stdout.as_deref(), Some("file body"));
    }

    #[tokio::test]
    async fn executor_output_is_capped_and_says_so() {
        // Reported: 200 MB of stdout took the daemon to 859 MB RSS and a 200 MB DB row.
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        // A plain byte count: BSD head (macOS) does not take `-c 10M`.
        let bytes = crate::tes_exec::MAX_CAPTURED_BYTES + 2 * 1024 * 1024;
        let task = sh_task("", &format!("head -c {bytes} /dev/zero | tr '\\0' x"));
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        let out = done.logs[0].logs[0].stdout.as_deref().unwrap();
        assert_eq!(out.len(), crate::tes_exec::MAX_CAPTURED_BYTES);
        assert!(done.logs[0]
            .system_logs
            .iter()
            .any(|l| l.contains("stdout truncated")));
    }

    fn file_input(url: &str) -> proteus_core::tes::TesInput {
        proteus_core::tes::TesInput {
            name: None,
            description: None,
            url: Some(url.into()),
            path: "/data/in.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
            content: None,
        }
    }

    fn file_output(url: &str) -> proteus_core::tes::TesOutput {
        proteus_core::tes::TesOutput {
            name: None,
            description: None,
            url: Some(url.into()),
            path: "/data/out.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
        }
    }

    #[tokio::test]
    async fn file_urls_outside_the_allowed_dirs_are_rejected_at_submit() {
        let tmp = tempdir().unwrap();
        let allowed = tmp.path().join("allowed");
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("secret.txt"), "TOP SECRET").unwrap();
        let (scheduler, _) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![allowed.clone()],
            ..Default::default()
        });

        let mut task = sh_task("t-in", "cat in.txt");
        task.inputs.push(file_input(&format!(
            "file://{}/secret.txt",
            elsewhere.display()
        )));
        let err = scheduler.validate_tes_task(&task).unwrap_err().to_string();
        assert!(err.contains("allowed"), "{err}");

        let mut task = sh_task("t-in-bare", "cat in.txt");
        task.inputs
            .push(file_input(&format!("{}/secret.txt", elsewhere.display())));
        assert!(scheduler.validate_tes_task(&task).is_err());

        let mut task = sh_task("t-out", "echo hi > out.txt");
        task.outputs.push(file_output(&format!(
            "file://{}/leak.txt",
            elsewhere.display()
        )));
        let err = scheduler.validate_tes_task(&task).unwrap_err().to_string();
        assert!(err.contains("allowed"), "{err}");

        // Inside the allow-list: accepted, staged, and delivered.
        std::fs::write(allowed.join("in.txt"), "hello").unwrap();
        let mut task = sh_task("t-ok", "cat in.txt > out.txt");
        task.inputs
            .push(file_input(&format!("file://{}/in.txt", allowed.display())));
        task.outputs.push(file_output(&format!(
            "file://{}/out.txt",
            allowed.display()
        )));
        scheduler.validate_tes_task(&task).unwrap();
    }

    #[tokio::test]
    async fn file_urls_are_denied_when_no_allowed_dirs_are_configured() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("in.txt"), "hello").unwrap();
        let (scheduler, _) = host_scheduler(&tmp.path().join("artifacts")).await;
        let mut task = sh_task("t-default", "cat in.txt");
        task.inputs.push(file_input(&format!(
            "file://{}/in.txt",
            tmp.path().display()
        )));
        assert!(scheduler.validate_tes_task(&task).is_err());
    }

    #[tokio::test]
    async fn missing_file_input_is_a_system_error_not_a_stuck_task() {
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        });
        let mut task = sh_task("t-missing", "cat in.txt");
        task.inputs.push(file_input(&format!(
            "file://{}/missing.txt",
            tmp.path().display()
        )));
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
        let log = &done.logs[0];
        assert!(log.logs.is_empty(), "executor must not run: {:?}", log.logs);
        assert!(log.system_logs.iter().any(|l| l.contains("missing.txt")));
    }

    #[tokio::test]
    async fn staging_rechecks_allowed_dirs_for_tasks_that_bypassed_validation() {
        let tmp = tempdir().unwrap();
        let allowed = tmp.path().join("allowed");
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("secret.txt"), "TOP SECRET").unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![allowed],
            ..Default::default()
        });
        let mut task = sh_task("t-bypass", "cat in.txt");
        task.inputs.push(file_input(&format!(
            "file://{}/secret.txt",
            elsewhere.display()
        )));
        task.state = TesState::Queued;
        // Straight into the store, as a row written by an older build would be.
        repo.insert_tes_task(
            &task.id,
            "QUEUED",
            None,
            None,
            &serde_json::to_string(&task).unwrap(),
        )
        .await
        .unwrap();
        scheduler.process_tes_task("t-bypass").await.unwrap();
        let done = finished(&repo, "t-bypass").await;
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
        assert!(done.logs[0].logs.is_empty(), "executor must not run");
    }

    /// Nextflow's nf-ga4gh plugin hands the server bare absolute paths (no `file://`) for
    /// inputs and outputs. Both directions must accept them under the same allow-list.
    #[tokio::test]
    async fn bare_absolute_paths_work_as_input_and_output_urls() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("in.txt");
        std::fs::write(&src, "payload").unwrap();
        let dest = tmp.path().join("delivered").join("out.txt");
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        });
        let mut task = sh_task("t-bare", "cp in.txt out.txt");
        task.inputs.push(file_input(&src.display().to_string()));
        task.outputs.push(file_output(&dest.display().to_string()));
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "payload");
    }

    #[tokio::test]
    async fn delivery_rechecks_allowed_dirs_for_tasks_that_bypassed_validation() {
        let tmp = tempdir().unwrap();
        let allowed = tmp.path().join("allowed");
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![allowed],
            ..Default::default()
        });
        let leak = elsewhere.join("leak.txt");
        let mut task = sh_task("t-bypass-out", "echo hi > out.txt");
        task.outputs
            .push(file_output(&format!("file://{}", leak.display())));
        task.state = TesState::Queued;
        repo.insert_tes_task(
            &task.id,
            "QUEUED",
            None,
            None,
            &serde_json::to_string(&task).unwrap(),
        )
        .await
        .unwrap();
        scheduler.process_tes_task("t-bypass-out").await.unwrap();
        let done = finished(&repo, "t-bypass-out").await;
        assert!(
            !leak.exists(),
            "output was written outside the allowed dirs"
        );
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
    }

    #[tokio::test]
    async fn harvesting_a_structure_emits_cas_and_biophysics_events() {
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let mut rx = scheduler.subscribe();
        let crambin = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        );
        // Run the same task twice: the second harvest of identical bytes is a CAS hit.
        for id in ["t-cas-1", "t-cas-2"] {
            let mut task = sh_task(id, &format!("cp {crambin} out.pdb"));
            task.outputs.push(proteus_core::tes::TesOutput {
                name: None,
                description: None,
                url: None,
                path: "/data/out.pdb".into(),
                type_: proteus_core::tes::TesFileType::File,
            });
            let id = scheduler.submit_tes_task(task).await.unwrap();
            let done = finished(&repo, &id).await;
            assert_eq!(done.state, TesState::Complete, "{:?}", done.logs);
        }
        let mut cas = Vec::new();
        let mut bio = 0;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                EngineEvent::CasStored { duplicate, bytes } => cas.push((duplicate, bytes)),
                EngineEvent::BiophysicsAnalyzed { duration_ms, .. } => {
                    bio += 1;
                    assert!(duration_ms < 60_000);
                }
                _ => {}
            }
        }
        assert_eq!(cas.len(), 2, "{cas:?}");
        assert!(
            !cas[0].0 && cas[0].1 > 1000,
            "first store is a miss: {cas:?}"
        );
        assert!(cas[1].0, "second store is a hit: {cas:?}");
        assert_eq!(bio, 2);
    }

    #[tokio::test]
    async fn cancel_emits_a_canceled_event_and_the_worker_reports_the_end_once() {
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let mut rx = scheduler.subscribe();
        let id = scheduler
            .submit_tes_task(sh_task("t-cancel", "sleep 30"))
            .await
            .unwrap();
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            if repo.get_tes_task(&id).await.unwrap().unwrap().state == "RUNNING" {
                break;
            }
        }
        assert_eq!(
            scheduler.cancel_tes_task(&id).await.unwrap(),
            CancelOutcome::Canceled
        );
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::Canceled);
        let (mut started, mut canceled, mut failed) = (0, 0, 0);
        while let Ok(ev) = rx.try_recv() {
            match ev {
                EngineEvent::TesTaskStarted { .. } => started += 1,
                EngineEvent::TesTaskCanceled { .. } => canceled += 1,
                EngineEvent::TesTaskFailed { .. } => failed += 1,
                _ => {}
            }
        }
        // One start, one cancel notification, and exactly one end-of-work report.
        assert_eq!((started, canceled, failed), (1, 1, 1));
    }

    #[tokio::test]
    async fn cancel_during_initialisation_is_not_overwritten_by_the_worker() {
        let tmp = tempdir().unwrap();
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        // An input served by a local HTTP endpoint that answers only after 400 ms keeps the
        // worker between its INITIALIZING and RUNNING writes for a known window.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            use tokio::io::AsyncWriteExt;
            let _ = sock
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi")
                .await;
        });
        let mut task = sh_task("t-race", "sleep 30");
        task.inputs
            .push(file_input(&format!("http://127.0.0.1:{port}/in.txt")));
        repo.insert_tes_task(
            &task.id,
            "QUEUED",
            None,
            None,
            &serde_json::to_string(&task).unwrap(),
        )
        .await
        .unwrap();
        let worker = {
            let s = scheduler.clone();
            tokio::spawn(async move { s.process_tes_task("t-race").await })
        };
        // Cancel while the worker is INITIALIZING (staging the slow input).
        for _ in 0..2000 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if repo.get_tes_task("t-race").await.unwrap().unwrap().state == "INITIALIZING" {
                break;
            }
        }
        assert_eq!(
            repo.get_tes_task("t-race").await.unwrap().unwrap().state,
            "INITIALIZING"
        );
        assert_eq!(
            scheduler.cancel_tes_task("t-race").await.unwrap(),
            CancelOutcome::Canceled
        );
        worker.await.unwrap().unwrap();
        let done = finished(&repo, "t-race").await;
        assert_eq!(done.state, TesState::Canceled, "{:?}", done.logs);
        assert!(
            done.logs
                .iter()
                .all(|l| l.logs.iter().all(|e| e.exit_code != Some(0))),
            "the executor must not have run to completion: {:?}",
            done.logs
        );
    }

    #[tokio::test]
    async fn output_upload_failure_is_a_system_error() {
        let tmp = tempdir().unwrap();
        // A regular file where the destination's parent directory would have to be.
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();
        let dest = blocker.join("out.txt");
        let (scheduler, repo) = host_scheduler(&tmp.path().join("artifacts")).await;
        let scheduler = scheduler.with_tes_config(TesExecutionConfig {
            allow_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        });
        let mut task = sh_task("t-upload", "echo hi > out.txt");
        task.outputs.push(proteus_core::tes::TesOutput {
            name: None,
            description: None,
            url: Some(format!("file://{}", dest.display())),
            path: "/data/out.txt".into(),
            type_: proteus_core::tes::TesFileType::File,
        });
        let id = scheduler.submit_tes_task(task).await.unwrap();
        let done = finished(&repo, &id).await;
        assert_eq!(done.state, TesState::SystemError, "{:?}", done.logs);
        assert!(done.logs[0]
            .system_logs
            .iter()
            .any(|l| l.contains("output upload") && l.contains("failed")));
    }

    /// A runner that returns a structure with a fixed number of residues, whatever the input.
    struct TruncatedRunner(usize);

    #[async_trait::async_trait]
    impl ComputeRunner for TruncatedRunner {
        async fn execute_job(
            &self,
            job: &proteus_core::models::PipelineJob,
            _sequence: &Sequence,
            work_dir: &std::path::Path,
        ) -> Result<crate::runner::RunResult, EngineError> {
            tokio::fs::create_dir_all(work_dir).await?;
            let path = work_dir.join(format!("{}.pdb", job.id));
            let mut pdb = String::new();
            for i in 0..self.0 {
                pdb.push_str(&format!(
                    "ATOM  {:5}  CA  ALA A{:4}    {:8.3}{:8.3}{:8.3}  1.00 90.00           C\n",
                    i + 1,
                    i + 1,
                    i as f64 * 3.8,
                    0.0,
                    0.0
                ));
            }
            tokio::fs::write(&path, pdb).await?;
            Ok(crate::runner::RunResult {
                pdb_path: path,
                plddt: Some(90.0),
                metadata: Some(serde_json::json!({"engine": "stub"})),
            })
        }
    }

    #[tokio::test]
    async fn structure_with_the_wrong_residue_count_fails_the_job() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let tmp = tempdir().unwrap();
        let scheduler = PipelineScheduler::new(
            repo.clone(),
            Arc::new(TruncatedRunner(3)),
            tmp.path().to_path_buf(),
        );
        let seq = Sequence {
            id: Uuid::new_v4(),
            header: "twenty-two".into(),
            fasta: "MKTAYIAKQRQISFVKSHFSRQ".into(),
            length: 22,
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
        let err = scheduler.process_job(job.id).await.unwrap_err().to_string();
        assert!(err.contains("3") && err.contains("22"), "{err}");
        let job = repo.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert!(repo.get_prediction_by_job(job.id).await.unwrap().is_none());
    }

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
        // The id is the server's, never the client's: it names a directory on the host.
        assert_ne!(task_id, "tes-task-unit-test");
        assert!(task_id.starts_with("task-"), "{task_id}");

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
