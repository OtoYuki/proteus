-- Proteus-rs Initial SQLite Schema

CREATE TABLE IF NOT EXISTS sequences (
    id TEXT PRIMARY KEY NOT NULL,
    header TEXT NOT NULL,
    fasta TEXT NOT NULL,
    length INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY NOT NULL,
    sequence_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_log TEXT,
    FOREIGN KEY (sequence_id) REFERENCES sequences (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT NOT NULL,
    pdb_path TEXT NOT NULL,
    plddt REAL,
    confidence_category TEXT,
    metadata TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY NOT NULL,
    prediction_id TEXT NOT NULL,
    radius_of_gyration REAL NOT NULL,
    rmsd_to_reference REAL,
    contact_density REAL NOT NULL,
    mean_plddt REAL NOT NULL,
    median_plddt REAL NOT NULL,
    high_conf_fraction REAL NOT NULL,
    very_high_conf_fraction REAL NOT NULL,
    metrics_json TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_sequence_id ON jobs(sequence_id);
CREATE INDEX IF NOT EXISTS idx_predictions_job_id ON predictions(job_id);
CREATE INDEX IF NOT EXISTS idx_metrics_prediction_id ON metrics(prediction_id);
