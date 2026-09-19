use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Invalid FASTA: {0}")]
    InvalidFasta(String),

    #[error("Structure parse error: {0}")]
    StructureParseError(String),

    #[error("Analysis error: {0}")]
    AnalysisError(String),
}
