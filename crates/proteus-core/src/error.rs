use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Invalid FASTA: {0}")]
    InvalidFasta(String),

    #[error("Invalid mutagenesis window: {0}")]
    InvalidWindow(String),

    #[error("Structure parse error: {0}")]
    StructureParseError(String),

    #[error("Analysis error: {0}")]
    AnalysisError(String),

    /// A predictor's side file (PAE matrix, scores) or a per-residue data file.
    #[error("{0}")]
    ParseError(String),
}
