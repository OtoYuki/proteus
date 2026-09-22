use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    MigrationError(#[from] sqlx::migrate::MigrateError),

    #[error("Entity not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Parquet error: {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    ArrowError(#[from] arrow_schema::ArrowError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CAS integrity violation: {0}")]
    IntegrityViolation(String),

    #[error("CAS error: {0}")]
    CasError(String),

    #[error("unsupported export format '{0}': use .parquet, .csv or .json")]
    UnsupportedExportFormat(String),
}
