//! Headless Axum server and OpenAPI daemon for Proteus.

pub mod api;
pub mod dto;
pub mod server;
pub mod telemetry;
pub mod tes_api;

pub use server::{build_router, run_server, AppState};
pub use telemetry::Telemetry;
