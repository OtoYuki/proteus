use crate::error::StorageError;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

const MIGRATION_SQL: &str = include_str!("../migrations/0001_initial.sql");

pub async fn create_sqlite_pool<P: AsRef<Path>>(db_path: P) -> Result<SqlitePool, StorageError> {
    let path = db_path.as_ref();
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| StorageError::DatabaseError(sqlx::Error::Io(e)))?;
    }

    // `filename` takes the path as it is; splicing it into a `sqlite://` URL let `?` and `%`
    // in a data directory be read as URL syntax ("unknown value 'ro/proteus.db' for mode").
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(Duration::from_secs(5));

    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(options)
        .await?;

    apply_migrations(&pool).await?;
    Ok(pool)
}

pub async fn create_in_memory_pool() -> Result<SqlitePool, StorageError> {
    let options = SqliteConnectOptions::from_str("sqlite::memory:")?
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    apply_migrations(&pool).await?;
    Ok(pool)
}

async fn apply_migrations(pool: &SqlitePool) -> Result<(), StorageError> {
    for statement in MIGRATION_SQL.split(';') {
        let trimmed = statement.trim();
        if !trimmed.is_empty() {
            sqlx::query(trimmed).execute(pool).await?;
        }
    }
    Ok(())
}
