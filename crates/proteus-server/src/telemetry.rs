use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

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

    // Concurrency gauges
    pub active_workers: AtomicI64,
    pub task_queue_depth: AtomicI64,

    // CAS counters
    pub cas_hits: AtomicU64,
    pub cas_misses: AtomicU64,
    pub cas_stores: AtomicU64,
    pub cas_reads: AtomicU64,
    pub cas_bytes_stored: AtomicU64,
    pub cas_bytes_read: AtomicU64,

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
        let workers = self.active_workers.load(Ordering::Relaxed).max(0);
        out.push_str(
            "\n# HELP proteus_active_workers Number of concurrently running worker threads\n",
        );
        out.push_str("# TYPE proteus_active_workers gauge\n");
        out.push_str(&format!("proteus_active_workers {}\n", workers));

        // Task queue depth gauge
        let queue_depth = self.task_queue_depth.load(Ordering::Relaxed).max(0);
        out.push_str("\n# HELP proteus_task_queue_depth Pending tasks waiting in queue\n");
        out.push_str("# TYPE proteus_task_queue_depth gauge\n");
        out.push_str(&format!("proteus_task_queue_depth {}\n", queue_depth));

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
        out.push_str(&format!(
            "proteus_cas_operations_total{{op=\"read\"}} {}\n",
            self.cas_reads.load(Ordering::Relaxed)
        ));

        // CAS bytes counter
        out.push_str("\n# HELP proteus_cas_bytes_total Total bytes processed through Content-Addressable Storage\n");
        out.push_str("# TYPE proteus_cas_bytes_total counter\n");
        out.push_str(&format!(
            "proteus_cas_bytes_total{{op=\"store\"}} {}\n",
            self.cas_bytes_stored.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "proteus_cas_bytes_total{{op=\"read\"}} {}\n",
            self.cas_bytes_read.load(Ordering::Relaxed)
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
