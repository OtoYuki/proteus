//! Headless Axum server and OpenAPI daemon for Proteus.

pub mod api;
pub mod dto;
pub mod server;
pub mod telemetry;
pub mod tes_api;

pub use server::{
    build_router, build_router_with_options, run_server, run_server_with_options, AppState,
    ServerOptions,
};
pub use telemetry::Telemetry;
