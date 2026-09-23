use thiserror::Error;

#[derive(Error, Debug)]
pub enum RenderError {
    #[error("Geometry error: {0}")]
    Geometry(String),

    #[error("Rasterization error: {0}")]
    Rasterization(String),

    #[error("Terminal error: {0}")]
    Terminal(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDB parse error: {0}")]
    PdbParse(String),

    #[error("Invalid viewport: {0}")]
    InvalidViewport(String),
}
