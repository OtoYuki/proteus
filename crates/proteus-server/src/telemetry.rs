use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use proteus_engine::EngineEvent;

/// Buckets for task duration histogram (in seconds).
pub const TASK_DURATION_BUCKETS: [f64; 10] =
    [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];

/// Buckets for biophysical calculation histogram (in seconds).
pub const BIOPHYSICAL_DURATION_BUCKETS: [f64; 7] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1];

/// High-throughput, thread-safe, lock-free Prometheus OpenMetrics telemetry collector.
#[derive(Debug, Default)]
pub struct Telemetry {
    // Task counters
    pub tasks_queued: AtomicU64,
    pub tasks_running: AtomicU64,
    pub tasks_complete: AtomicU64,
    pub tasks_failed: AtomicU64,
    pub tasks_canceled: AtomicU64,

    // Concurrency gauge
    pub active_workers: AtomicI64,

    // CAS counters
    pub cas_hits: AtomicU64,
    pub cas_misses: AtomicU64,
    pub cas_stores: AtomicU64,
    pub cas_bytes_stored: AtomicU64,

    /// Start instants of TES tasks currently running, for the duration histogram.
    task_starts: Mutex<HashMap<String, Instant>>,

    // HTTP counters
    pub http_requests_total: AtomicU64,

    // Task duration histogram buckets
    task_duration_counts: [AtomicU64; 10],
    task_duration_sum_ms: AtomicU64,
    task_duration_total: AtomicU64,

    // Biophysical calculation histogram buckets
    bio_duration_counts: [AtomicU64; 7],
    bio_duration_sum_us: AtomicU64,
    bio_duration_total: AtomicU64,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one engine event into the counters. Every counter on `/metrics` is driven from
    /// here, so what the endpoint shows is exactly what the engine emitted.
    pub fn observe(&self, event: &EngineEvent) {
        match event {
            EngineEvent::TesTaskStarted { task_id } => {
                self.tasks_running.fetch_add(1, Ordering::Relaxed);
                self.active_workers.fetch_add(1, Ordering::Relaxed);
                self.task_starts
                    .lock()
                    .unwrap()
                    .insert(task_id.clone(), Instant::now());
            }
            EngineEvent::TesTaskCompleted { task_id }
            | EngineEvent::TesTaskFailed { task_id, .. } => {
                if matches!(event, EngineEvent::TesTaskCompleted { .. }) {
                    self.tasks_complete.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.tasks_failed.fetch_add(1, Ordering::Relaxed);
                }
                self.active_workers.fetch_sub(1, Ordering::Relaxed);
                if let Some(started) = self.task_starts.lock().unwrap().remove(task_id) {
                    self.record_task_duration(started.elapsed());
                }
            }
            // A cancel notice is not an end-of-work report; the worker sends that itself.
            EngineEvent::TesTaskCanceled { .. } => {}
            EngineEvent::JobStarted { .. } => {
                self.active_workers.fetch_add(1, Ordering::Relaxed);
            }
            EngineEvent::JobCompleted { .. } | EngineEvent::JobFailed { .. } => {
                self.active_workers.fetch_sub(1, Ordering::Relaxed);
            }
            EngineEvent::CasStored { duplicate, bytes } => {
                if *duplicate {
                    self.cas_hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.cas_misses.fetch_add(1, Ordering::Relaxed);
                    self.cas_stores.fetch_add(1, Ordering::Relaxed);
                    self.cas_bytes_stored.fetch_add(*bytes, Ordering::Relaxed);
                }
            }
            EngineEvent::BiophysicsAnalyzed { duration_ms, .. } => {
                self.record_biophysical_duration(Duration::from_millis(*duration_ms));
            }
            EngineEvent::JobQueued { .. } | EngineEvent::JobProgress { .. } => {}
        }
    }

    /// Records the execution duration of a task.
    pub fn record_task_duration(&self, duration: Duration) {
        let secs = duration.as_secs_f64();
        let ms = duration.as_millis() as u64;

        for (i, &bucket) in TASK_DURATION_BUCKETS.iter().enumerate() {
            if secs <= bucket {
                self.task_duration_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.task_duration_sum_ms.fetch_add(ms, Ordering::Relaxed);
        self.task_duration_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the calculation duration of a native biophysical metric.
    pub fn record_biophysical_duration(&self, duration: Duration) {
        let secs = duration.as_secs_f64();
        let us = duration.as_micros() as u64;

        for (i, &bucket) in BIOPHYSICAL_DURATION_BUCKETS.iter().enumerate() {
            if secs <= bucket {
                self.bio_duration_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.bio_duration_sum_us.fetch_add(us, Ordering::Relaxed);
        self.bio_duration_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Renders current metrics in standard Prometheus text format (v0.0.4).
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);

        // Task total counter
        out.push_str(
            "# HELP proteus_tasks_total Total count of pipeline and TES tasks scheduled\n",
        );
        out.push_str("# TYPE proteus_tasks_total counter\n");
        out.push_str(&format!(
            "proteus_tasks_total{{status=\"queued\"}} {}\n",
            self.tasks_queued.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_tasks_total{{status=\"running\"}} {}\n",
            self.tasks_running.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_tasks_total{{status=\"complete\"}} {}\n",
            self.tasks_complete.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_tasks_total{{status=\"failed\"}} {}\n",
            self.tasks_failed.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_tasks_total{{status=\"canceled\"}} {}\n",
            self.tasks_canceled.load(Ordering::Relaxed)
        ));

        // Active workers gauge
        let workers = self.active_workers.load(Ordering::Relaxed);
        out.push_str(
            "\n# HELP proteus_active_workers Number of concurrently running worker threads\n",
        );
        out.push_str("# TYPE proteus_active_workers gauge\n");
        out.push_str(&format!("proteus_active_workers {}\n", workers));

        // CAS operations counter
        out.push_str(
            "\n# HELP proteus_cas_operations_total Content-Addressable Storage operations\n",
        );
        out.push_str("# TYPE proteus_cas_operations_total counter\n");
        out.push_str(&format!(
            "proteus_cas_operations_total{{op=\"hit\"}} {}\n",
            self.cas_hits.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_cas_operations_total{{op=\"miss\"}} {}\n",
            self.cas_misses.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_cas_operations_total{{op=\"store\"}} {}\n",
            self.cas_stores.load(Ordering::Relaxed)
        ));

        // CAS bytes counter
        out.push_str("\n# HELP proteus_cas_bytes_total Total bytes processed through Content-Addressable Storage\n");
        out.push_str("# TYPE proteus_cas_bytes_total counter\n");
        out.push_str(&format!(
            "proteus_cas_bytes_total{{op=\"store\"}} {}\n",
            self.cas_bytes_stored.load(Ordering::Relaxed)
        ));

        // HTTP requests total
        out.push_str(
            "\n# HELP proteus_http_requests_total Lifetime HTTP requests handled by daemon\n",
        );
        out.push_str("# TYPE proteus_http_requests_total counter\n");
        out.push_str(&format!(
            "proteus_http_requests_total {}\n",
            self.http_requests_total.load(Ordering::Relaxed)
        ));

        // Task duration histogram
        out.push_str(
            "\n# HELP proteus_task_duration_seconds End-to-end task execution latency in seconds\n",
        );
        out.push_str("# TYPE proteus_task_duration_seconds histogram\n");
        let task_total = self.task_duration_total.load(Ordering::Relaxed);
        let task_sum_secs = self.task_duration_sum_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        for (i, &bucket) in TASK_DURATION_BUCKETS.iter().enumerate() {
            let count = self.task_duration_counts[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "proteus_task_duration_seconds_bucket{{le=\"{:.2}\"}} {}\n",
                bucket, count
            ));
        }
        out.push_str(&format!(
            "proteus_task_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
            task_total
        ));
        out.push_str(&format!(
            "proteus_task_duration_seconds_sum {:.4}\n",
            task_sum_secs
        ));
        out.push_str(&format!(
            "proteus_task_duration_seconds_count {}\n",
            task_total
        ));

        // Biophysical calculation duration histogram
        out.push_str("\n# HELP proteus_biophysical_duration_seconds Pure-Rust biophysical calculation latency in seconds\n");
        out.push_str("# TYPE proteus_biophysical_duration_seconds histogram\n");
        let bio_total = self.bio_duration_total.load(Ordering::Relaxed);
        let bio_sum_secs = self.bio_duration_sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        for (i, &bucket) in BIOPHYSICAL_DURATION_BUCKETS.iter().enumerate() {
            let count = self.bio_duration_counts[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "proteus_biophysical_duration_seconds_bucket{{le=\"{:.4}\"}} {}\n",
                bucket, count
            ));
        }
        out.push_str(&format!(
            "proteus_biophysical_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
            bio_total
        ));
        out.push_str(&format!(
            "proteus_biophysical_duration_seconds_sum {:.6}\n",
            bio_sum_secs
        ));
        out.push_str(&format!(
            "proteus_biophysical_duration_seconds_count {}\n",
            bio_total
        ));

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_engine::EngineEvent;

    fn started(id: &str) -> EngineEvent {
        EngineEvent::TesTaskStarted { task_id: id.into() }
    }

    #[test]
    fn task_lifecycle_events_drive_the_worker_gauge_and_duration_histogram() {
        let tel = Telemetry::new();
        tel.observe(&started("a"));
        tel.observe(&started("b"));
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 2);
        tel.observe(&EngineEvent::TesTaskCompleted {
            task_id: "a".into(),
        });
        tel.observe(&EngineEvent::TesTaskFailed {
            task_id: "b".into(),
            error: "x".into(),
        });
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 0);
        let out = tel.render_prometheus();
        assert!(
            out.contains("proteus_task_duration_seconds_count 2"),
            "{out}"
        );
    }

    #[test]
    fn cancel_notification_does_not_touch_the_worker_gauge() {
        let tel = Telemetry::new();
        // Cancelled before it ever started: no worker was ever busy.
        tel.observe(&EngineEvent::TesTaskCanceled {
            task_id: "q".into(),
        });
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 0);
        // Cancelled while running: the cancel notice plus the worker's own end report.
        tel.observe(&started("r"));
        tel.observe(&EngineEvent::TesTaskCanceled {
            task_id: "r".into(),
        });
        tel.observe(&EngineEvent::TesTaskFailed {
            task_id: "r".into(),
            error: "canceled".into(),
        });
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cas_and_biophysics_events_feed_their_counters() {
        let tel = Telemetry::new();
        tel.observe(&EngineEvent::CasStored {
            duplicate: false,
            bytes: 1000,
        });
        tel.observe(&EngineEvent::CasStored {
            duplicate: true,
            bytes: 1000,
        });
        tel.observe(&EngineEvent::BiophysicsAnalyzed {
            duration_ms: 3,
            residues: 46,
        });
        let out = tel.render_prometheus();
        assert!(
            out.contains("proteus_cas_operations_total{op=\"hit\"} 1"),
            "{out}"
        );
        assert!(
            out.contains("proteus_cas_operations_total{op=\"miss\"} 1"),
            "{out}"
        );
        assert!(
            out.contains("proteus_cas_operations_total{op=\"store\"} 1"),
            "{out}"
        );
        assert!(
            out.contains("proteus_cas_bytes_total{op=\"store\"} 1000"),
            "{out}"
        );
        assert!(
            out.contains("proteus_biophysical_duration_seconds_count 1"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn collector_keeps_counting_after_the_broadcast_channel_lags() {
        let tel = std::sync::Arc::new(Telemetry::new());
        let (tx, rx) = tokio::sync::broadcast::channel(4);
        crate::server::spawn_collector(tel.clone(), rx);
        // Ten sends before the collector is ever polled: it wakes to `Lagged(6)`.
        for i in 0..10 {
            tx.send(started(&i.to_string())).unwrap();
        }
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 4);
        // And it is still alive afterwards.
        tx.send(EngineEvent::TesTaskCompleted {
            task_id: "9".into(),
        })
        .unwrap();
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert_eq!(tel.active_workers.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_telemetry_rendering() {
        let tel = Telemetry::new();
        tel.tasks_queued.fetch_add(5, Ordering::Relaxed);
        tel.tasks_running.fetch_add(2, Ordering::Relaxed);
        tel.tasks_complete.fetch_add(3, Ordering::Relaxed);
        tel.active_workers.fetch_add(2, Ordering::Relaxed);
        tel.cas_hits.fetch_add(10, Ordering::Relaxed);
        tel.cas_stores.fetch_add(4, Ordering::Relaxed);

        tel.record_task_duration(Duration::from_millis(150));
        tel.record_biophysical_duration(Duration::from_micros(800));

        let rendered = tel.render_prometheus();
        assert!(rendered.contains("proteus_tasks_total{status=\"queued\"} 5"));
        assert!(rendered.contains("proteus_tasks_total{status=\"complete\"} 3"));
        assert!(rendered.contains("proteus_active_workers 2"));
        assert!(rendered.contains("proteus_cas_operations_total{op=\"hit\"} 10"));
        assert!(rendered.contains("proteus_task_duration_seconds_count 1"));
        assert!(rendered.contains("proteus_biophysical_duration_seconds_count 1"));
    }
}
