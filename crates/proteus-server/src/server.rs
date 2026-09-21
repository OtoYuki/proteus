use crate::api::{
    enqueue_job, get_job, get_metrics, get_prediction, get_prediction_pdb, get_sequence,
    health_check, stream_job_events, submit_sequence, view_structure, ApiDoc,
};
use crate::telemetry::Telemetry;
use crate::tes_api::{cancel_task, create_task, get_service_info, get_task, list_tasks};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use proteus_engine::{EngineEvent, PipelineScheduler};
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: PipelineScheduler,
    pub telemetry: Arc<Telemetry>,
}

/// Network-facing options for [`run_server`].
#[derive(Clone, Default)]
pub struct ServerOptions {
    /// When set, every `/v1`, `/ga4gh` and `/api` request must carry `Authorization: Bearer <token>`.
    pub auth_token: Option<String>,
}

impl AppState {
    pub fn new(scheduler: PipelineScheduler) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        let tel = telemetry.clone();
        let mut rx = scheduler.subscribe();
        tokio::spawn(async move {
            while let Ok(event) = rx.recv().await {
                match event {
                    EngineEvent::TesTaskStarted { .. } => {
                        tel.tasks_running.fetch_add(1, Ordering::Relaxed);
                        tel.active_workers.fetch_add(1, Ordering::Relaxed);
                    }
                    EngineEvent::TesTaskCompleted { .. } => {
                        tel.tasks_complete.fetch_add(1, Ordering::Relaxed);
                        tel.active_workers.fetch_sub(1, Ordering::Relaxed);
                    }
                    EngineEvent::TesTaskFailed { .. } => {
                        tel.tasks_failed.fetch_add(1, Ordering::Relaxed);
                        tel.active_workers.fetch_sub(1, Ordering::Relaxed);
                    }
                    EngineEvent::JobStarted { .. } => {
                        tel.active_workers.fetch_add(1, Ordering::Relaxed);
                    }
                    EngineEvent::JobCompleted { .. } | EngineEvent::JobFailed { .. } => {
                        tel.active_workers.fetch_sub(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        });

        Self {
            scheduler,
            telemetry,
        }
    }
}

pub async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);
    let body = state.telemetry.render_prometheus();
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// Reject requests whose `Authorization: Bearer …` does not match the configured token
/// (constant-time comparison). Returns 401 with a `WWW-Authenticate` header.
async fn require_bearer(
    State(token): State<Arc<String>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use subtle::ConstantTimeEq;
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or("");
    let ok = presented.len() == token.len() && presented.as_bytes().ct_eq(token.as_bytes()).into();
    if ok {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            axum::Json(serde_json::json!({ "error": "missing or invalid bearer token" })),
        )
            .into_response()
    }
}

pub fn build_router(state: AppState) -> Router {
    build_router_with_options(state, ServerOptions::default())
}

pub fn build_router_with_options(state: AppState, options: ServerOptions) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Everything except /health, /metrics and the Swagger UI sits behind the optional bearer token.
    let protected = Router::new()
        // GA4GH TES v1.1 Standard Endpoints
        .route("/v1/tasks", post(create_task).get(list_tasks))
        .route("/v1/tasks/{id}", get(get_task).post(cancel_task))
        .route("/v1/tasks/{id}/cancel", post(cancel_task))
        .route("/v1/service-info", get(get_service_info))
        .route("/v1/tasks/service-info", get(get_service_info))
        // GA4GH TES v1.1 Aliases
        .route("/ga4gh/tes/v1/tasks", post(create_task).get(list_tasks))
        .route("/ga4gh/tes/v1/tasks/{id}", get(get_task).post(cancel_task))
        .route("/ga4gh/tes/v1/tasks/{id}/cancel", post(cancel_task))
        .route("/ga4gh/tes/v1/service-info", get(get_service_info))
        // Native Proteus API Endpoints
        .route("/api/v1/sequences", post(submit_sequence))
        .route("/api/v1/sequences/{id}", get(get_sequence))
        .route("/api/v1/jobs", post(enqueue_job))
        .route("/api/v1/jobs/{id}", get(get_job))
        .route("/api/v1/jobs/{id}/events", get(stream_job_events))
        .route("/api/v1/predictions/by-job/{job_id}", get(get_prediction))
        .route(
            "/api/v1/predictions/by-job/{job_id}/pdb",
            get(get_prediction_pdb),
        )
        .route(
            "/api/v1/metrics/by-prediction/{prediction_id}",
            get(get_metrics),
        )
        .route("/view/{job_id}", get(view_structure));
    let protected = match options.auth_token {
        Some(token) if !token.is_empty() => protected.layer(axum::middleware::from_fn_with_state(
            Arc::new(token),
            require_bearer,
        )),
        _ => protected,
    };

    Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
        .merge(protected)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(
    addr: SocketAddr,
    scheduler: PipelineScheduler,
) -> Result<(), std::io::Error> {
    run_server_with_options(addr, scheduler, ServerOptions::default()).await
}

pub async fn run_server_with_options(
    addr: SocketAddr,
    scheduler: PipelineScheduler,
    options: ServerOptions,
) -> Result<(), std::io::Error> {
    let executor_kind = scheduler.tes_config().executor.kind();
    let allowlist = scheduler.tes_config().allow_images.len();
    if !addr.ip().is_loopback() {
        if options.auth_token.is_none() {
            tracing::warn!(
                "binding to {} without --auth-token: anyone who can reach this port can submit tasks",
                addr
            );
        }
        if allowlist == 0 {
            tracing::warn!("no --allow-image patterns: any container image will be accepted");
        }
    }
    tracing::info!("TES executor backend: {executor_kind}");
    let state = AppState::new(scheduler);

    let app = build_router_with_options(state, options);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Proteus server running on http://{}", addr);
    tracing::info!(
        "Swagger UI documentation available at http://{}/swagger-ui",
        addr
    );
    tracing::info!("Prometheus metrics available at http://{}/metrics", addr);
    tracing::info!("GA4GH TES v1.1 endpoint available at http://{}/v1", addr);
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
        let state = AppState::new(scheduler);
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

        // Test GET /metrics (Prometheus OpenMetrics)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("proteus_tasks_total{status=\"queued\"}"));
        assert!(body_str.contains("proteus_active_workers"));
        assert!(body_str.contains("proteus_cas_operations_total"));

        // Test GET /v1/service-info
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/service-info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let service_info: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(service_info["id"], "org.ga4gh.proteus");
        assert_eq!(service_info["type"]["artifact"], "tes");

        // Test POST /v1/tasks (Submit TES task)
        let tes_payload = serde_json::json!({
            "name": "screening_variant_1",
            "description": "Nextflow TES automated pipeline test",
            "executors": [
                {
                    "image": "docker.io/library/alpine:3.20",
                    "command": ["echo", "TES task executed"],
                    "workdir": "/work"
                }
            ]
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tasks")
                    .header("Content-Type", "application/json")
                    .body(Body::from(tes_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let create_resp: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let task_id = create_resp["id"].as_str().unwrap();
        assert!(!task_id.is_empty());

        // Test GET /v1/tasks/{id} with view=FULL
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/tasks/{task_id}?view=FULL"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Test GET /v1/tasks (List tasks)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/tasks")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Test GET /ga4gh/tes/v1/service-info (Alias)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/ga4gh/tes/v1/service-info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Test POST /v1/tasks/{id}:cancel
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/tasks/{task_id}:cancel"))
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
