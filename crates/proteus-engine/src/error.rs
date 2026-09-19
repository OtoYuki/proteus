use proteus_core::CoreError;
use proteus_storage::StorageError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Core error: {0}")]
    Core(#[from] CoreError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Container runtime error: {0}")]
    Container(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
