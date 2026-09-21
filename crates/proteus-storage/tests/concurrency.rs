//! Storage behaviour under concurrent writers.
//!
//! SQLite in WAL mode allows many readers but **one writer at a time**. Proteus fans screening
//! jobs across a worker pool, so every worker's status update contends for that single writer.
//! These tests establish that the contention is handled (no `SQLITE_BUSY` reaching the caller,
//! no lost rows, readers not starved) and print the observed throughput, so any performance
//! claim about the storage layer comes from a measurement rather than an assumption.
//!
//! Run: `cargo test -p proteus-storage --release --test concurrency -- --nocapture`.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use proteus_core::models::{JobStatus, PipelineJob, PipelineTier, Sequence};
use proteus_storage::pool::create_sqlite_pool;
use proteus_storage::repository::ProteusRepository;
use uuid::Uuid;

fn sequence() -> Sequence {
    Sequence {
        id: Uuid::new_v4(),
        header: "concurrency_probe".into(),
        fasta: "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN".into(),
        length: 46,
        created_at: Utc::now(),
    }
}

fn job(sequence_id: Uuid) -> PipelineJob {
    PipelineJob {
        id: Uuid::new_v4(),
        sequence_id,
        tier: PipelineTier::FastScreening,
        status: JobStatus::Queued,
        priority: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error_log: None,
    }
}

/// Every concurrent insert must land: WAL serialises writers, it must not drop them.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_writers_do_not_lose_rows() {
    let dir = tempfile::tempdir().unwrap();
    let pool = create_sqlite_pool(dir.path().join("proteus.db"))
        .await
        .unwrap();
    let repo = Arc::new(ProteusRepository::new(pool));

    let seq = sequence();
    repo.insert_sequence(&seq).await.unwrap();

    const WORKERS: usize = 16;
    const PER_WORKER: usize = 25;
    let started = Instant::now();
    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let repo = repo.clone();
        let sequence_id = seq.id;
        handles.push(tokio::spawn(async move {
            let mut ids = Vec::with_capacity(PER_WORKER);
            for _ in 0..PER_WORKER {
                let j = job(sequence_id);
                repo.insert_job(&j).await.expect("insert under contention");
                ids.push(j.id);
            }
            ids
        }));
    }
    let mut all = Vec::new();
    for h in handles {
        all.extend(h.await.unwrap());
    }
    let elapsed = started.elapsed();

    let expected = WORKERS * PER_WORKER;
    assert_eq!(all.len(), expected);
    for id in &all {
        assert!(
            repo.get_job(*id).await.unwrap().is_some(),
            "job {id} was lost under concurrent insert"
        );
    }
    eprintln!(
        "{expected} inserts from {WORKERS} tasks in {elapsed:.2?} = {:.0} writes/s",
        expected as f64 / elapsed.as_secs_f64()
    );
}

/// Many tasks updating the same row: the write must converge and never surface a lock error.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_status_updates_converge() {
    let dir = tempfile::tempdir().unwrap();
    let pool = create_sqlite_pool(dir.path().join("proteus.db"))
        .await
        .unwrap();
    let repo = Arc::new(ProteusRepository::new(pool));

    let seq = sequence();
    repo.insert_sequence(&seq).await.unwrap();
    let j = job(seq.id);
    repo.insert_job(&j).await.unwrap();

    const WORKERS: usize = 12;
    const ROUNDS: usize = 20;
    let started = Instant::now();
    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let repo = repo.clone();
        let job_id = j.id;
        handles.push(tokio::spawn(async move {
            for _ in 0..ROUNDS {
                repo.update_job_status(job_id, JobStatus::Running, None)
                    .await
                    .expect("status update under contention");
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let elapsed = started.elapsed();

    let stored = repo.get_job(j.id).await.unwrap().unwrap();
    assert_eq!(stored.status, JobStatus::Running);
    let total = WORKERS * ROUNDS;
    eprintln!(
        "{total} updates to one row from {WORKERS} tasks in {elapsed:.2?} = {:.0} updates/s",
        total as f64 / elapsed.as_secs_f64()
    );
}

/// Readers must keep making progress while writes stream in — the whole reason for WAL mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn readers_are_not_starved_by_writers() {
    let dir = tempfile::tempdir().unwrap();
    let pool = create_sqlite_pool(dir.path().join("proteus.db"))
        .await
        .unwrap();
    let repo = Arc::new(ProteusRepository::new(pool));

    let seq = sequence();
    repo.insert_sequence(&seq).await.unwrap();
    let probe = job(seq.id);
    repo.insert_job(&probe).await.unwrap();

    let writer_repo = repo.clone();
    let sequence_id = seq.id;
    let writer = tokio::spawn(async move {
        for _ in 0..200 {
            let j = job(sequence_id);
            writer_repo.insert_job(&j).await.expect("writer");
        }
    });

    let started = Instant::now();
    let mut reads = 0usize;
    while !writer.is_finished() {
        assert!(repo.get_job(probe.id).await.unwrap().is_some());
        reads += 1;
    }
    writer.await.unwrap();
    eprintln!(
        "{reads} reads completed during 200 concurrent inserts ({:.2?})",
        started.elapsed()
    );
    assert!(
        reads > 10,
        "readers were starved: only {reads} reads got through"
    );
}
