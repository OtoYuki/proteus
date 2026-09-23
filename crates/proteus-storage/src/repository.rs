use crate::error::StorageError;
use chrono::{DateTime, Utc};
use proteus_core::models::{
    BiophysicalMetrics, JobStatus, PipelineJob, PipelineTier, PlddtDistribution, Prediction,
    Sequence,
};
use sqlx::{Row, SqlitePool};
use std::str::FromStr;
use uuid::Uuid;

#[derive(Clone)]
pub struct ProteusRepository {
    pool: SqlitePool,
}

impl ProteusRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    // Sequence methods
    pub async fn insert_sequence(&self, seq: &Sequence) -> Result<(), StorageError> {
        let id_str = seq.id.to_string();
        let created_str = seq.created_at.to_rfc3339();
        let len_i64 = seq.length as i64;

        sqlx::query(
            "INSERT INTO sequences (id, header, fasta, length, created_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(id_str)
        .bind(&seq.header)
        .bind(&seq.fasta)
        .bind(len_i64)
        .bind(created_str)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_sequence(&self, id: Uuid) -> Result<Option<Sequence>, StorageError> {
        let id_str = id.to_string();
        let row =
            sqlx::query("SELECT id, header, fasta, length, created_at FROM sequences WHERE id = ?")
                .bind(id_str)
                .fetch_optional(&self.pool)
                .await?;

        if let Some(r) = row {
            let seq_id_str: String = r.get("id");
            let header: String = r.get("header");
            let fasta: String = r.get("fasta");
            let length: i64 = r.get("length");
            let created_at_str: String = r.get("created_at");

            let seq_id = Uuid::from_str(&seq_id_str)
                .map_err(|e| StorageError::NotFound(format!("Corrupt sequence UUID: {e}")))?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| StorageError::NotFound(format!("Corrupt timestamp: {e}")))?
                .with_timezone(&Utc);

            Ok(Some(Sequence {
                id: seq_id,
                header,
                fasta,
                length: length as usize,
                created_at,
            }))
        } else {
            Ok(None)
        }
    }

    // Job methods
    pub async fn insert_job(&self, job: &PipelineJob) -> Result<(), StorageError> {
        let id_str = job.id.to_string();
        let seq_id_str = job.sequence_id.to_string();
        let tier_str = match job.tier {
            PipelineTier::FastScreening => "FastScreening",
            PipelineTier::HighFidelity => "HighFidelity",
            PipelineTier::FullValidation => "FullValidation",
        };
        let status_str = match job.status {
            JobStatus::Pending => "Pending",
            JobStatus::Queued => "Queued",
            JobStatus::Running => "Running",
            JobStatus::Completed => "Completed",
            JobStatus::Failed => "Failed",
            JobStatus::Cancelled => "Cancelled",
        };
        let created_str = job.created_at.to_rfc3339();
        let started_str = job.started_at.map(|t| t.to_rfc3339());
        let completed_str = job.completed_at.map(|t| t.to_rfc3339());

        sqlx::query(
            "INSERT INTO jobs (id, sequence_id, tier, status, priority, created_at, started_at, completed_at, error_log)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(id_str)
        .bind(seq_id_str)
        .bind(tier_str)
        .bind(status_str)
        .bind(job.priority)
        .bind(created_str)
        .bind(started_str)
        .bind(completed_str)
        .bind(&job.error_log)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Move a job from Queued (or Pending) to Running, atomically. Returns whether this caller
    /// got it: the CLI, the daemon's API and its queue poller may all try to start the same
    /// job, and exactly one must run it.
    pub async fn claim_job(&self, id: Uuid) -> Result<bool, StorageError> {
        let result = sqlx::query(
            "UPDATE jobs SET status = 'Running', started_at = ?, error_log = NULL \
             WHERE id = ? AND status IN ('Queued', 'Pending')",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    /// Ids of jobs handed to the daemon (status Pending), oldest first. Queued jobs belong to
    /// the process that inserted them and are not listed.
    pub async fn pending_job_ids(&self, limit: i64) -> Result<Vec<Uuid>, StorageError> {
        let rows =
            sqlx::query("SELECT id FROM jobs WHERE status = 'Pending' ORDER BY created_at LIMIT ?")
                .bind(limit)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .iter()
            .filter_map(|r| Uuid::parse_str(&r.get::<String, _>("id")).ok())
            .collect())
    }

    pub async fn update_job_status(
        &self,
        id: Uuid,
        status: JobStatus,
        error: Option<String>,
    ) -> Result<(), StorageError> {
        let id_str = id.to_string();
        let status_str = match status {
            JobStatus::Pending => "Pending",
            JobStatus::Queued => "Queued",
            JobStatus::Running => "Running",
            JobStatus::Completed => "Completed",
            JobStatus::Failed => "Failed",
            JobStatus::Cancelled => "Cancelled",
        };

        let now_str = Utc::now().to_rfc3339();

        if status == JobStatus::Running {
            sqlx::query("UPDATE jobs SET status = ?, started_at = ?, error_log = ? WHERE id = ?")
                .bind(status_str)
                .bind(now_str)
                .bind(error)
                .bind(id_str)
                .execute(&self.pool)
                .await?;
        } else if matches!(
            status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            sqlx::query("UPDATE jobs SET status = ?, completed_at = ?, error_log = ? WHERE id = ?")
                .bind(status_str)
                .bind(now_str)
                .bind(error)
                .bind(id_str)
                .execute(&self.pool)
                .await?;
        } else {
            sqlx::query("UPDATE jobs SET status = ?, error_log = ? WHERE id = ?")
                .bind(status_str)
                .bind(error)
                .bind(id_str)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    pub async fn get_job(&self, id: Uuid) -> Result<Option<PipelineJob>, StorageError> {
        let id_str = id.to_string();
        let row = sqlx::query(
            "SELECT id, sequence_id, tier, status, priority, created_at, started_at, completed_at, error_log FROM jobs WHERE id = ?",
        )
        .bind(id_str)
        .fetch_optional(&self.pool)
        .await?;

        row.as_ref().map(job_from_row).transpose()
    }

    /// The newest `limit` jobs, each with its sequence and the newest prediction if there is
    /// one: what a list of one's own work needs, in one query.
    pub async fn list_jobs(&self, limit: i64) -> Result<Vec<JobSummary>, StorageError> {
        let rows = sqlx::query(
            "SELECT j.id, j.sequence_id, j.tier, j.status, j.priority, j.created_at, \
                    j.started_at, j.completed_at, j.error_log, \
                    s.header, s.length, p.pdb_path, p.plddt, p.metadata \
             FROM jobs j \
             LEFT JOIN sequences s ON s.id = j.sequence_id \
             LEFT JOIN predictions p ON p.rowid = \
                 (SELECT rowid FROM predictions WHERE job_id = j.id ORDER BY rowid DESC LIMIT 1) \
             ORDER BY j.created_at DESC, j.rowid DESC \
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| {
                let metadata: Option<String> = r.get("metadata");
                Ok(JobSummary {
                    job: job_from_row(r)?,
                    header: r.get::<Option<String>, _>("header").unwrap_or_default(),
                    length: r.get::<Option<i64>, _>("length").unwrap_or(0),
                    pdb_path: r.get("pdb_path"),
                    plddt: r.get("plddt"),
                    metadata: metadata.and_then(|m| serde_json::from_str(&m).ok()),
                })
            })
            .collect()
    }

    // Prediction methods
    pub async fn insert_prediction(&self, pred: &Prediction) -> Result<(), StorageError> {
        let id_str = pred.id.to_string();
        let job_id_str = pred.job_id.to_string();
        let meta_str = pred.metadata.as_ref().map(|m| m.to_string());

        sqlx::query(
            "INSERT INTO predictions (id, job_id, pdb_path, plddt, confidence_category, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(id_str)
        .bind(job_id_str)
        .bind(&pred.pdb_path)
        .bind(pred.plddt)
        .bind(&pred.confidence_category)
        .bind(meta_str)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Job IDs whose hex form starts with `prefix`, for git-style short IDs.
    ///
    /// Returns at most `limit` matches so the caller can tell "one" from "ambiguous"
    /// without loading the whole table. The prefix is matched case-insensitively against
    /// the canonical lowercase-hyphenated form the rest of the codebase stores.
    pub async fn find_job_ids_by_prefix(
        &self,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<Uuid>, StorageError> {
        // `%` and `_` are LIKE wildcards; a caller-supplied prefix must not smuggle them in.
        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            return Ok(Vec::new());
        }
        let rows = sqlx::query("SELECT id FROM jobs WHERE id LIKE ? ORDER BY id LIMIT ?")
            .bind(format!("{}%", prefix.to_ascii_lowercase()))
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .filter_map(|r| {
                let id: String = r.get("id");
                Uuid::from_str(&id).ok()
            })
            .collect())
    }

    pub async fn get_prediction_by_job(
        &self,
        job_id: Uuid,
    ) -> Result<Option<Prediction>, StorageError> {
        let job_id_str = job_id.to_string();
        let row = sqlx::query(
            // The newest prediction, as `list_jobs` shows it (a job can gain a second one).
            "SELECT id, job_id, pdb_path, plddt, confidence_category, metadata FROM predictions WHERE job_id = ? ORDER BY rowid DESC LIMIT 1",
        )
        .bind(job_id_str)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            let id_str: String = r.get("id");
            let j_id_str: String = r.get("job_id");
            let pdb_path: String = r.get("pdb_path");
            let plddt: Option<f64> = r.get("plddt");
            let confidence_category: Option<String> = r.get("confidence_category");
            let metadata_str: Option<String> = r.get("metadata");

            let id = Uuid::from_str(&id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
            let j_id =
                Uuid::from_str(&j_id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

            Ok(Some(Prediction {
                id,
                job_id: j_id,
                pdb_path,
                plddt,
                confidence_category,
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    // Metrics methods
    pub async fn insert_metrics(&self, metrics: &BiophysicalMetrics) -> Result<(), StorageError> {
        let id_str = metrics.id.to_string();
        let pred_id_str = metrics.prediction_id.to_string();
        let json_data = serde_json::to_string(metrics)?;

        sqlx::query(
            "INSERT INTO metrics (
                id, prediction_id, radius_of_gyration, rmsd_to_reference,
                contact_density, mean_plddt, median_plddt, high_conf_fraction,
                very_high_conf_fraction, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(id_str)
        .bind(pred_id_str)
        .bind(metrics.radius_of_gyration)
        .bind(metrics.rmsd_to_reference)
        .bind(metrics.contact_density)
        .bind(metrics.plddt_distribution.mean)
        .bind(metrics.plddt_distribution.median)
        .bind(metrics.plddt_distribution.high_confidence_fraction)
        .bind(metrics.plddt_distribution.very_high_confidence_fraction)
        .bind(json_data)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_metrics_by_prediction(
        &self,
        prediction_id: Uuid,
    ) -> Result<Option<BiophysicalMetrics>, StorageError> {
        let pred_id_str = prediction_id.to_string();
        let row = sqlx::query(
            "SELECT id, prediction_id, radius_of_gyration, rmsd_to_reference,
                   contact_density, mean_plddt, median_plddt, high_conf_fraction,
                   very_high_conf_fraction, metrics_json
            FROM metrics
            WHERE prediction_id = ?",
        )
        .bind(pred_id_str)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            let metrics_json: Option<String> = r.get("metrics_json");
            if let Some(ref json_str) = metrics_json {
                if let Ok(deserialized) = serde_json::from_str::<BiophysicalMetrics>(json_str) {
                    return Ok(Some(deserialized));
                }
            }

            let id_str: String = r.get("id");
            let p_id_str: String = r.get("prediction_id");
            let radius_of_gyration: f64 = r.get("radius_of_gyration");
            let rmsd_to_reference: Option<f64> = r.get("rmsd_to_reference");
            let contact_density: f64 = r.get("contact_density");
            let mean_plddt: f64 = r.get("mean_plddt");
            let median_plddt: f64 = r.get("median_plddt");
            let high_conf_fraction: f64 = r.get("high_conf_fraction");
            let very_high_conf_fraction: f64 = r.get("very_high_conf_fraction");

            let id = Uuid::from_str(&id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
            let p_id =
                Uuid::from_str(&p_id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;

            Ok(Some(BiophysicalMetrics {
                id,
                prediction_id: p_id,
                radius_of_gyration,
                rmsd_to_reference,
                contact_density,
                plddt_distribution: PlddtDistribution {
                    mean: mean_plddt,
                    median: median_plddt,
                    high_confidence_fraction: high_conf_fraction,
                    very_high_confidence_fraction: very_high_conf_fraction,
                },
                confidence_source: Default::default(),
                secondary_structure_summary: None,
                ramachandran_stats: None,
                steric_overlap: None,
                sasa_metrics: None,
                interaction_network: None,
                candidate_fitness_score: None,
            }))
        } else {
            Ok(None)
        }
    }

    // Audit logs
    pub async fn log_event(
        &self,
        job_id: Option<Uuid>,
        level: &str,
        message: &str,
    ) -> Result<(), StorageError> {
        let id_str = Uuid::new_v4().to_string();
        let job_id_str = job_id.map(|j| j.to_string());
        let now_str = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO audit_logs (id, job_id, level, message, timestamp) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(id_str)
        .bind(job_id_str)
        .bind(level)
        .bind(message)
        .bind(now_str)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_logs_for_job(
        &self,
        job_id: Uuid,
    ) -> Result<Vec<(String, String, String)>, StorageError> {
        let job_id_str = job_id.to_string();
        let rows = sqlx::query("SELECT level, message, timestamp FROM audit_logs WHERE job_id = ? ORDER BY timestamp ASC")
            .bind(job_id_str)
            .fetch_all(&self.pool)
            .await?;

        let mut res = Vec::new();
        for r in rows {
            let level: String = r.get("level");
            let message: String = r.get("message");
            let timestamp: String = r.get("timestamp");
            res.push((level, message, timestamp));
        }

        Ok(res)
    }

    // CAS Object methods
    pub async fn record_cas_object(&self, hash: &str, size_bytes: i64) -> Result<(), StorageError> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"INSERT INTO cas_objects (hash, size_bytes, created_at, reference_count)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(hash) DO UPDATE SET reference_count = reference_count + 1"#,
        )
        .bind(hash)
        .bind(size_bytes)
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_cas_object(
        &self,
        hash: &str,
    ) -> Result<Option<CasObjectRecord>, StorageError> {
        let row = sqlx::query(
            "SELECT hash, size_bytes, created_at, reference_count FROM cas_objects WHERE hash = ?",
        )
        .bind(hash)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            Ok(Some(CasObjectRecord {
                hash: r.get("hash"),
                size_bytes: r.get("size_bytes"),
                created_at: r.get("created_at"),
                reference_count: r.get("reference_count"),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn count_cas_objects(&self) -> Result<i64, StorageError> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM cas_objects")
            .fetch_one(&self.pool)
            .await?;
        let count: i64 = row.get("count");
        Ok(count)
    }

    pub async fn total_cas_bytes(&self) -> Result<i64, StorageError> {
        let row = sqlx::query("SELECT COALESCE(SUM(size_bytes), 0) as total FROM cas_objects")
            .fetch_one(&self.pool)
            .await?;
        let total: i64 = row.get("total");
        Ok(total)
    }

    // TES Task methods
    pub async fn insert_tes_task(
        &self,
        id: &str,
        state: &str,
        name: Option<&str>,
        description: Option<&str>,
        task_json: &str,
    ) -> Result<(), StorageError> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO tes_tasks (id, state, name, description, task_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(id)
        .bind(state)
        .bind(name)
        .bind(description)
        .bind(task_json)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_tes_task_state(
        &self,
        id: &str,
        state: &str,
        task_json: &str,
    ) -> Result<(), StorageError> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE tes_tasks SET state = ?, task_json = ?, updated_at = ? WHERE id = ?")
            .bind(state)
            .bind(task_json)
            .bind(now)
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// As [`update_tes_task_state`](Self::update_tes_task_state), but refused when the task
    /// has meanwhile been cancelled. Returns whether the row was written; `false` means the
    /// caller lost a race with a cancel and must stop.
    pub async fn update_tes_task_state_unless_canceled(
        &self,
        id: &str,
        state: &str,
        task_json: &str,
    ) -> Result<bool, StorageError> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE tes_tasks SET state = ?, task_json = ?, updated_at = ? WHERE id = ? AND state != 'CANCELED'",
        )
        .bind(state)
        .bind(task_json)
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    /// As [`update_tes_task_state`](Self::update_tes_task_state), but refused once the task is
    /// in any terminal state (COMPLETE, EXECUTOR_ERROR, SYSTEM_ERROR, CANCELED). A cancel and a
    /// finishing worker race for the same row; whichever writes a terminal state first wins and
    /// the other is told so (`false`).
    pub async fn update_tes_task_state_unless_terminal(
        &self,
        id: &str,
        state: &str,
        task_json: &str,
    ) -> Result<bool, StorageError> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE tes_tasks SET state = ?, task_json = ?, updated_at = ? WHERE id = ? \
             AND state NOT IN ('COMPLETE', 'EXECUTOR_ERROR', 'SYSTEM_ERROR', 'CANCELED', 'PREEMPTED')",
        )
        .bind(state)
        .bind(task_json)
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    /// Ids of TES tasks that are not in a terminal state (queued, initializing, running,
    /// paused). At daemon start these are tasks whose worker died with the previous process.
    pub async fn unfinished_tes_task_ids(&self) -> Result<Vec<String>, StorageError> {
        let rows = sqlx::query(
            "SELECT id FROM tes_tasks WHERE state NOT IN \
             ('COMPLETE', 'EXECUTOR_ERROR', 'SYSTEM_ERROR', 'CANCELED', 'PREEMPTED')",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    pub async fn get_tes_task(&self, id: &str) -> Result<Option<TesTaskRecord>, StorageError> {
        let row = sqlx::query(
            "SELECT id, state, name, description, task_json, created_at, updated_at FROM tes_tasks WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            Ok(Some(TesTaskRecord {
                id: r.get("id"),
                state: r.get("state"),
                name: r.get("name"),
                description: r.get("description"),
                task_json: r.get("task_json"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn list_tes_tasks(
        &self,
        state: Option<&str>,
        limit: i64,
    ) -> Result<Vec<TesTaskRecord>, StorageError> {
        let rows = if let Some(s) = state {
            sqlx::query(
                "SELECT id, state, name, description, task_json, created_at, updated_at FROM tes_tasks WHERE state = ? ORDER BY created_at DESC LIMIT ?"
            )
            .bind(s)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, state, name, description, task_json, created_at, updated_at FROM tes_tasks ORDER BY created_at DESC LIMIT ?"
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        let mut tasks = Vec::with_capacity(rows.len());
        for r in rows {
            tasks.push(TesTaskRecord {
                id: r.get("id"),
                state: r.get("state"),
                name: r.get("name"),
                description: r.get("description"),
                task_json: r.get("task_json"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            });
        }

        Ok(tasks)
    }
}

/// Escape `%`, `_` and `\` for a `LIKE … ESCAPE '\'` prefix match.
fn like_prefix(prefix: &str) -> String {
    let mut out = String::with_capacity(prefix.len() + 1);
    for c in prefix.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('%');
    out
}

impl ProteusRepository {
    /// One page of TES tasks, newest first, filtered by state and name prefix in SQL.
    pub async fn list_tes_tasks_page(
        &self,
        state: Option<&str>,
        name_prefix: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<TesTaskRecord>, StorageError> {
        let rows = sqlx::query(
            "SELECT id, state, name, description, task_json, created_at, updated_at FROM tes_tasks \
             WHERE (?1 IS NULL OR state = ?1) AND (?2 IS NULL OR name LIKE ?2 ESCAPE '\\') \
             ORDER BY created_at DESC LIMIT ?3 OFFSET ?4",
        )
        .bind(state)
        .bind(name_prefix.map(like_prefix))
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| TesTaskRecord {
                id: r.get("id"),
                state: r.get("state"),
                name: r.get("name"),
                description: r.get("description"),
                task_json: r.get("task_json"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    /// Number of TES tasks matching the same filters as [`list_tes_tasks_page`](Self::list_tes_tasks_page).
    pub async fn count_tes_tasks(
        &self,
        state: Option<&str>,
        name_prefix: Option<&str>,
    ) -> Result<i64, StorageError> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM tes_tasks \
             WHERE (?1 IS NULL OR state = ?1) AND (?2 IS NULL OR name LIKE ?2 ESCAPE '\\')",
        )
        .bind(state)
        .bind(name_prefix.map(like_prefix))
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get("n"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasObjectRecord {
    pub hash: String,
    pub size_bytes: i64,
    pub created_at: String,
    pub reference_count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TesTaskRecord {
    pub id: String,
    pub state: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub task_json: String,
    pub created_at: String,
    pub updated_at: String,
}

/// One row of [`ProteusRepository::list_jobs`].
#[derive(Debug, Clone)]
pub struct JobSummary {
    pub job: PipelineJob,
    /// FASTA header of the job's sequence (empty if the sequence row is gone).
    pub header: String,
    pub length: i64,
    /// Newest prediction's structure file, mean pLDDT and engine metadata, if it has one.
    pub pdb_path: Option<String>,
    pub plddt: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

/// A `jobs` row (columns as selected by `get_job` and `list_jobs`) as a [`PipelineJob`].
fn job_from_row(r: &sqlx::sqlite::SqliteRow) -> Result<PipelineJob, StorageError> {
    let job_id_str: String = r.get("id");
    let seq_id_str: String = r.get("sequence_id");
    let tier_str: String = r.get("tier");
    let status_str: String = r.get("status");
    let created_at_str: String = r.get("created_at");
    let started_at_str: Option<String> = r.get("started_at");
    let completed_at_str: Option<String> = r.get("completed_at");

    let id = Uuid::from_str(&job_id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
    let sequence_id =
        Uuid::from_str(&seq_id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
    let tier = match tier_str.as_str() {
        "FastScreening" => PipelineTier::FastScreening,
        "HighFidelity" => PipelineTier::HighFidelity,
        _ => PipelineTier::FullValidation,
    };
    let status = match status_str.as_str() {
        "Pending" => JobStatus::Pending,
        "Queued" => JobStatus::Queued,
        "Running" => JobStatus::Running,
        "Completed" => JobStatus::Completed,
        "Failed" => JobStatus::Failed,
        "Cancelled" => JobStatus::Cancelled,
        _ => JobStatus::Pending,
    };
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map_err(|e| StorageError::NotFound(e.to_string()))?
        .with_timezone(&Utc);
    let parse = |s: Option<String>| {
        s.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|t| t.with_timezone(&Utc))
        })
    };
    Ok(PipelineJob {
        id,
        sequence_id,
        tier,
        status,
        priority: r.get("priority"),
        created_at,
        started_at: parse(started_at_str),
        completed_at: parse(completed_at_str),
        error_log: r.get("error_log"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::create_in_memory_pool;
    use proteus_core::models::{PlddtDistribution, Sequence};

    #[tokio::test]
    async fn test_storage_lifecycle() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);

        // 1. Insert & retrieve sequence
        let seq = Sequence {
            id: Uuid::new_v4(),
            header: "test_protein".into(),
            fasta: "ACDEFGHIKLMNPQRSTVWY".into(),
            length: 20,
            created_at: Utc::now(),
        };
        repo.insert_sequence(&seq).await.unwrap();

        let retrieved_seq = repo.get_sequence(seq.id).await.unwrap().unwrap();
        assert_eq!(retrieved_seq.header, "test_protein");
        assert_eq!(retrieved_seq.length, 20);

        // 2. Insert & retrieve job
        let job = PipelineJob {
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

        repo.update_job_status(job.id, JobStatus::Running, None)
            .await
            .unwrap();
        let running_job = repo.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(running_job.status, JobStatus::Running);
        assert!(running_job.started_at.is_some());

        repo.update_job_status(job.id, JobStatus::Completed, None)
            .await
            .unwrap();
        let completed_job = repo.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(completed_job.status, JobStatus::Completed);
        assert!(completed_job.completed_at.is_some());

        // 3. Insert & retrieve prediction
        let pred = Prediction {
            id: Uuid::new_v4(),
            job_id: job.id,
            pdb_path: "/tmp/test.pdb".into(),
            plddt: Some(85.5),
            confidence_category: Some("Confident".into()),
            metadata: Some(serde_json::json!({ "model": "esmfold" })),
        };
        repo.insert_prediction(&pred).await.unwrap();

        let retrieved_pred = repo.get_prediction_by_job(job.id).await.unwrap().unwrap();
        assert_eq!(retrieved_pred.pdb_path, "/tmp/test.pdb");
        assert_eq!(retrieved_pred.plddt, Some(85.5));

        // 4. Insert & retrieve metrics
        let metrics = BiophysicalMetrics {
            id: Uuid::new_v4(),
            prediction_id: pred.id,
            radius_of_gyration: 14.25,
            rmsd_to_reference: Some(1.12),
            contact_density: 0.18,
            plddt_distribution: PlddtDistribution {
                mean: 85.5,
                median: 86.0,
                high_confidence_fraction: 0.9,
                very_high_confidence_fraction: 0.4,
            },
            confidence_source: Default::default(),
            secondary_structure_summary: None,
            ramachandran_stats: None,
            steric_overlap: None,
            sasa_metrics: None,
            interaction_network: None,
            candidate_fitness_score: Some(88.5),
        };
        repo.insert_metrics(&metrics).await.unwrap();

        let retrieved_metrics = repo
            .get_metrics_by_prediction(pred.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved_metrics.radius_of_gyration, 14.25);
        assert_eq!(retrieved_metrics.rmsd_to_reference, Some(1.12));

        // 5. Audit logs
        repo.log_event(Some(job.id), "INFO", "Starting pipeline")
            .await
            .unwrap();
        repo.log_event(Some(job.id), "INFO", "Done").await.unwrap();

        let logs = repo.get_logs_for_job(job.id).await.unwrap();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].1, "Starting pipeline");

        // 6. CAS object indexing
        let hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        repo.record_cas_object(hash, 1024).await.unwrap();
        // Second record increments reference count
        repo.record_cas_object(hash, 1024).await.unwrap();

        let cas_rec = repo.get_cas_object(hash).await.unwrap().unwrap();
        assert_eq!(cas_rec.hash, hash);
        assert_eq!(cas_rec.size_bytes, 1024);
        assert_eq!(cas_rec.reference_count, 2);
        assert_eq!(repo.count_cas_objects().await.unwrap(), 1);
        assert_eq!(repo.total_cas_bytes().await.unwrap(), 1024);

        // 7. TES task lifecycle
        let task_id = "task-tes-12345";
        repo.insert_tes_task(
            task_id,
            "QUEUED",
            Some("screening_task"),
            Some("Screening test variant"),
            r#"{"id":"task-tes-12345","state":"QUEUED"}"#,
        )
        .await
        .unwrap();

        let fetched = repo.get_tes_task(task_id).await.unwrap().unwrap();
        assert_eq!(fetched.id, task_id);
        assert_eq!(fetched.state, "QUEUED");
        assert_eq!(fetched.name, Some("screening_task".into()));

        repo.update_tes_task_state(
            task_id,
            "COMPLETE",
            r#"{"id":"task-tes-12345","state":"COMPLETE"}"#,
        )
        .await
        .unwrap();

        let updated = repo.get_tes_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.state, "COMPLETE");

        let listed = repo.list_tes_tasks(Some("COMPLETE"), 10).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, task_id);
    }

    #[tokio::test]
    async fn task_listing_filters_and_pages_in_sql() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        for (i, (name, state)) in [
            ("pg-1", "COMPLETE"),
            ("pg-2", "COMPLETE"),
            ("pg-3", "RUNNING"),
            ("other", "COMPLETE"),
            ("pg%4", "COMPLETE"), // a literal % in the name must not act as a wildcard
        ]
        .iter()
        .enumerate()
        {
            repo.insert_tes_task(&format!("t{i}"), state, Some(name), None, "{}")
                .await
                .unwrap();
        }
        let page = repo
            .list_tes_tasks_page(None, Some("pg-"), 2, 0)
            .await
            .unwrap();
        assert_eq!(page.len(), 2);
        let rest = repo
            .list_tes_tasks_page(None, Some("pg-"), 2, 2)
            .await
            .unwrap();
        assert_eq!(rest.len(), 1);
        let complete = repo
            .list_tes_tasks_page(Some("COMPLETE"), Some("pg-"), 10, 0)
            .await
            .unwrap();
        assert_eq!(complete.len(), 2);
        let literal = repo
            .list_tes_tasks_page(None, Some("pg%"), 10, 0)
            .await
            .unwrap();
        assert_eq!(literal.len(), 1, "prefix with % must match literally");
        assert_eq!(literal[0].name.as_deref(), Some("pg%4"));
        assert_eq!(repo.count_tes_tasks(None, Some("pg-")).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn non_terminal_state_writes_do_not_overwrite_a_cancel() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        repo.insert_tes_task("t", "QUEUED", None, None, "{}")
            .await
            .unwrap();
        assert!(repo
            .update_tes_task_state_unless_canceled("t", "INITIALIZING", "{}")
            .await
            .unwrap());
        repo.update_tes_task_state("t", "CANCELED", "{}")
            .await
            .unwrap();
        // The worker, unaware of the cancel, tries to move on: the write must be refused.
        assert!(!repo
            .update_tes_task_state_unless_canceled("t", "RUNNING", "{}")
            .await
            .unwrap());
        assert_eq!(
            repo.get_tes_task("t").await.unwrap().unwrap().state,
            "CANCELED"
        );
    }

    /// Two jobs sharing a first byte, so the prefix lookup has something to be ambiguous about.
    #[tokio::test]
    async fn job_listing_is_newest_first_with_sequence_and_prediction() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let t0 = Utc::now() - chrono::Duration::hours(1);
        let mut ids = Vec::new();
        for (i, header) in ["first", "second", "third"].iter().enumerate() {
            let seq = Sequence {
                id: Uuid::new_v4(),
                header: header.to_string(),
                fasta: "ACDE".into(),
                length: 4 + i,
                created_at: t0,
            };
            repo.insert_sequence(&seq).await.unwrap();
            let job = PipelineJob {
                id: Uuid::new_v4(),
                sequence_id: seq.id,
                tier: PipelineTier::FastScreening,
                status: JobStatus::Queued,
                priority: 1,
                created_at: t0 + chrono::Duration::minutes(i as i64),
                started_at: None,
                completed_at: None,
                error_log: None,
            };
            repo.insert_job(&job).await.unwrap();
            ids.push(job.id);
        }
        // Two predictions for the second job: the listing shows the newer one, once.
        for (path, plddt) in [("old.pdb", 50.0), ("new.pdb", 80.0)] {
            repo.insert_prediction(&Prediction {
                id: Uuid::new_v4(),
                job_id: ids[1],
                pdb_path: path.into(),
                plddt: Some(plddt),
                confidence_category: None,
                metadata: Some(serde_json::json!({ "engine": "esmfold-api" })),
            })
            .await
            .unwrap();
        }

        let all = repo.list_jobs(10).await.unwrap();
        let order: Vec<_> = all.iter().map(|j| j.header.as_str()).collect();
        assert_eq!(order, ["third", "second", "first"]);
        assert_eq!(all[0].length, 6);
        assert_eq!(all[1].pdb_path.as_deref(), Some("new.pdb"));
        assert_eq!(all[1].plddt, Some(80.0));
        assert_eq!(all[1].metadata.as_ref().unwrap()["engine"], "esmfold-api");
        assert!(all[0].pdb_path.is_none() && all[2].plddt.is_none());
        assert_eq!(all[2].job.id, ids[0]);

        let two = repo.list_jobs(2).await.unwrap();
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].job.id, ids[2]);
    }

    #[tokio::test]
    async fn job_id_prefix_lookup_distinguishes_unique_from_ambiguous() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);

        let seq = Sequence {
            id: Uuid::new_v4(),
            header: "p".into(),
            fasta: "ACDEFGHIKL".into(),
            length: 10,
            created_at: Utc::now(),
        };
        repo.insert_sequence(&seq).await.unwrap();

        for id in [
            "ab12cd34-0000-4000-8000-000000000001",
            "ab12cd34-0000-4000-8000-000000000002",
            "ffffffff-0000-4000-8000-000000000003",
        ] {
            repo.insert_job(&PipelineJob {
                id: Uuid::parse_str(id).unwrap(),
                sequence_id: seq.id,
                tier: PipelineTier::FastScreening,
                status: JobStatus::Queued,
                priority: 0,
                created_at: Utc::now(),
                started_at: None,
                completed_at: None,
                error_log: None,
            })
            .await
            .unwrap();
        }

        // Unique prefix.
        assert_eq!(
            repo.find_job_ids_by_prefix("ffffffff", 4)
                .await
                .unwrap()
                .len(),
            1
        );
        // Shared prefix: both come back so the caller can report the ambiguity.
        assert_eq!(
            repo.find_job_ids_by_prefix("ab12cd34", 4)
                .await
                .unwrap()
                .len(),
            2
        );
        // No match.
        assert!(repo
            .find_job_ids_by_prefix("deadbeef", 4)
            .await
            .unwrap()
            .is_empty());
        // A LIKE wildcard must not match everything.
        assert!(repo
            .find_job_ids_by_prefix("%", 4)
            .await
            .unwrap()
            .is_empty());
        assert!(repo
            .find_job_ids_by_prefix("________", 4)
            .await
            .unwrap()
            .is_empty());
        // The limit is honoured.
        assert_eq!(
            repo.find_job_ids_by_prefix("ab12cd34", 1)
                .await
                .unwrap()
                .len(),
            1
        );
    }
    #[tokio::test]
    async fn only_pending_jobs_are_offered_to_the_daemon_and_each_is_claimed_once() {
        // Regression review: the daemon's queue poller took Queued jobs belonging to a running
        // `screen` in another process.
        let repo = ProteusRepository::new(crate::pool::create_in_memory_pool().await.unwrap());
        let seq = Sequence {
            id: Uuid::new_v4(),
            header: "x".into(),
            fasta: "ACDE".into(),
            length: 4,
            created_at: Utc::now(),
        };
        repo.insert_sequence(&seq).await.unwrap();
        let mut ids = Vec::new();
        for status in [JobStatus::Queued, JobStatus::Pending] {
            let job = PipelineJob {
                id: Uuid::new_v4(),
                sequence_id: seq.id,
                tier: PipelineTier::FastScreening,
                status,
                priority: 1,
                created_at: Utc::now(),
                started_at: None,
                completed_at: None,
                error_log: None,
            };
            repo.insert_job(&job).await.unwrap();
            ids.push(job.id);
        }
        assert_eq!(repo.pending_job_ids(10).await.unwrap(), vec![ids[1]]);
        assert!(repo.claim_job(ids[1]).await.unwrap());
        assert!(!repo.claim_job(ids[1]).await.unwrap(), "claimed twice");
        assert!(repo.pending_job_ids(10).await.unwrap().is_empty());
    }
}
