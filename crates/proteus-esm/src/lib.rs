//! ESM-2 protein language model inference in pure Rust, on top of [`candle`](https://github.com/huggingface/candle).
//!
//! Loads ESM-2 masked-LM checkpoints from safetensors — `facebook/esm2_t{6,12,30,33}_*` directly
//! from the Hub, the 3B/15B models from local files after conversion — and
//! exposes per-position log-probabilities, zero-shot mutation scores (wild-type marginals and
//! masked marginals, Meier et al. 2021) and full 20×L deep-mutational-scan matrices.
//!
//! Numerical parity with `transformers.EsmForMaskedLM` is pinned by the parity tests in
//! `tests/parity.rs` against reference logits produced by `validate/esm_reference.py`.
#![forbid(unsafe_code)]

#[cfg(feature = "hub")]
pub mod hub;
mod model;
mod scoring;
mod tokenizer;

pub use candle_core::Device;
pub use model::{Esm2, EsmConfig};
pub use scoring::{
    parse_mutation, scan, score_masked_marginal, score_wt_marginal, MarginalScorer, Mutation,
    ScanRow,
};
pub use tokenizer::{Tokenizer, AMINO_ACIDS, MASK_ID, VOCAB};

/// Errors from loading or running a model.
#[derive(Debug, thiserror::Error)]
pub enum EsmError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("config: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("hub: {0}")]
    Hub(String),
    #[error("sequence: {0}")]
    Sequence(String),
    #[error("mutation: {0}")]
    Mutation(String),
}

pub type Result<T> = std::result::Result<T, EsmError>;
