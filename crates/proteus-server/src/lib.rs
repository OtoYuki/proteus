//! Headless Axum server and OpenAPI daemon for Proteus.

pub mod api;
pub mod dto;
pub mod server;

pub use server::{build_router, run_server, AppState};
