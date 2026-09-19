use crate::api::{
    enqueue_job, get_job, get_metrics, get_prediction, get_sequence, health_check,
    stream_job_events, submit_sequence, ApiDoc,
};
use axum::routing::{get, post};
use axum::Router;
use proteus_engine::PipelineScheduler;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: PipelineScheduler,
}

pub fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/sequences", post(submit_sequence))
        .route("/api/v1/sequences/{id}", get(get_sequence))
        .route("/api/v1/jobs", post(enqueue_job))
        .route("/api/v1/jobs/{id}", get(get_job))
        .route("/api/v1/jobs/{id}/events", get(stream_job_events))
        .route("/api/v1/predictions/by-job/{job_id}", get(get_prediction))
        .route(
            "/api/v1/metrics/by-prediction/{prediction_id}",
            get(get_metrics),
        )
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(
    addr: SocketAddr,
    scheduler: PipelineScheduler,
) -> Result<(), std::io::Error> {
    let state = AppState { scheduler };
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Proteus server running on http://{}", addr);
    tracing::info!(
        "Swagger UI documentation available at http://{}/swagger-ui",
        addr
    );
    axum::serve(listener, app).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use proteus_engine::simulated::SimulatedRunner;
    use proteus_storage::pool::create_in_memory_pool;
    use proteus_storage::repository::ProteusRepository;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_and_api_endpoints() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let runner = Arc::new(SimulatedRunner::new());
        let tmp = tempdir().unwrap();
        let scheduler = PipelineScheduler::new(repo, runner, tmp.path().to_path_buf());
        let state = AppState { scheduler };
        let app = build_router(state);

        // Test GET /health
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Test POST /api/v1/sequences with valid FASTA
        let payload = serde_json::json!({
            "fasta": ">test_protein\nACDEFGHIKLMNPQRSTVWY"
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/sequences")
                    .header("Content-Type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        // Test POST /api/v1/sequences with invalid FASTA
        let bad_payload = serde_json::json!({
            "fasta": "invalid fasta without header"
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/sequences")
                    .header("Content-Type", "application/json")
                    .body(Body::from(bad_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
