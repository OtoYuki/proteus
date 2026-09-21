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

        if let Some(r) = row {
            let job_id_str: String = r.get("id");
            let seq_id_str: String = r.get("sequence_id");
            let tier_str: String = r.get("tier");
            let status_str: String = r.get("status");
            let priority: i32 = r.get("priority");
            let created_at_str: String = r.get("created_at");
            let started_at_str: Option<String> = r.get("started_at");
            let completed_at_str: Option<String> = r.get("completed_at");
            let error_log: Option<String> = r.get("error_log");

            let job_id =
                Uuid::from_str(&job_id_str).map_err(|e| StorageError::NotFound(e.to_string()))?;
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
            let started_at = started_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|t| t.with_timezone(&Utc))
            });
            let completed_at = completed_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|t| t.with_timezone(&Utc))
            });

            Ok(Some(PipelineJob {
                id: job_id,
                sequence_id,
                tier,
                status,
                priority,
                created_at,
                started_at,
                completed_at,
                error_log,
            }))
        } else {
            Ok(None)
        }
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

    pub async fn get_prediction_by_job(
        &self,
        job_id: Uuid,
    ) -> Result<Option<Prediction>, StorageError> {
        let job_id_str = job_id.to_string();
        let row = sqlx::query(
            "SELECT id, job_id, pdb_path, plddt, confidence_category, metadata FROM predictions WHERE job_id = ?",
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
                clash_stats: None,
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
            clash_stats: None,
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
}
