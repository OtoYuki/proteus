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
}
