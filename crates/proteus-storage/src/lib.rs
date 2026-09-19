//! Embedded SQLite storage for Proteus using SQLx.

pub mod error;
pub mod pool;
pub mod repository;

pub use error::StorageError;
pub use pool::create_sqlite_pool;
