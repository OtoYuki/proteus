use crate::api::{
    enqueue_job, get_job, get_metrics, get_prediction, get_prediction_pdb, get_sequence,
    health_check, stream_job_events, submit_sequence, view_structure, ApiDoc,
};
use crate::telemetry::Telemetry;
use crate::tes_api::{
    cancel_task, cancel_task_colon, create_task, get_service_info, get_task, list_tasks,
};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use proteus_engine::{EngineEvent, PipelineScheduler};
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
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
        spawn_collector(telemetry.clone(), scheduler.subscribe());

        Self {
            scheduler,
            telemetry,
        }
    }
}

/// Feed engine events into the telemetry counters for the life of the process. A slow
/// consumer sees `RecvError::Lagged` when the broadcast buffer wraps; that drops events but
/// must not end the collector.
pub fn spawn_collector(
    telemetry: Arc<Telemetry>,
    mut rx: tokio::sync::broadcast::Receiver<EngineEvent>,
) {
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => telemetry.observe(&event),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("telemetry collector lagged; {n} engine events not counted");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
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
        // RFC 7235: the scheme is case-insensitive.
        .and_then(|v| {
            v.get(..7)
                .filter(|scheme| scheme.eq_ignore_ascii_case("bearer "))
                .map(|_| &v[7..])
        })
        .map(str::trim)
        .unwrap_or("");
    // Compare fixed-length digests so neither the token's length nor a matching prefix shows
    // up in the response time.
    let ok: bool = blake3::hash(presented.as_bytes())
        .as_bytes()
        .ct_eq(blake3::hash(token.as_bytes()).as_bytes())
        .into();
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
    // No CORS layer, deliberately. `Access-Control-Allow-Origin: *` let any web page a user
    // visited drive a loopback daemon (submit tasks, run commands under the host executor).
    // TES clients are not browsers, and the Swagger UI and `/view` pages are same-origin.

    // Everything except /health, /metrics and the Swagger UI sits behind the optional bearer token.
    let protected = Router::new()
        // GA4GH TES v1.1 Standard Endpoints
        .route("/v1/tasks", post(create_task).get(list_tasks))
        .route("/v1/tasks/{id}", get(get_task).post(cancel_task_colon))
        .route("/v1/tasks/{id}/cancel", post(cancel_task))
        .route("/v1/service-info", get(get_service_info))
        .route("/v1/tasks/service-info", get(get_service_info))
        // GA4GH TES v1.1 Aliases
        .route("/ga4gh/tes/v1/tasks", post(create_task).get(list_tasks))
        .route(
            "/ga4gh/tes/v1/tasks/{id}",
            get(get_task).post(cancel_task_colon),
        )
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

#[cfg(test)]
mod sse_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use proteus_engine::simulated::SimulatedRunner;
    use proteus_storage::pool::create_in_memory_pool;
    use proteus_storage::repository::ProteusRepository;
    use tempfile::tempdir;
    use tower::ServiceExt;

    /// A client that subscribes after the job finished must still learn its state, and the
    /// stream must end instead of hanging forever.
    #[tokio::test]
    async fn job_event_stream_starts_with_the_current_status_and_ends_when_terminal() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let tmp = tempdir().unwrap();
        let scheduler = PipelineScheduler::new(
            repo.clone(),
            Arc::new(SimulatedRunner::new()),
            tmp.path().to_path_buf(),
        );
        let seq = proteus_core::models::Sequence {
            id: uuid::Uuid::new_v4(),
            header: "x".into(),
            fasta: "ACDEFGHIKLMNPQRSTVWY".into(),
            length: 20,
            created_at: chrono::Utc::now(),
        };
        repo.insert_sequence(&seq).await.unwrap();
        let job = proteus_core::models::PipelineJob {
            id: uuid::Uuid::new_v4(),
            sequence_id: seq.id,
            tier: proteus_core::models::PipelineTier::FastScreening,
            status: proteus_core::models::JobStatus::Queued,
            priority: 1,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error_log: None,
        };
        repo.insert_job(&job).await.unwrap();
        scheduler.process_job(job.id).await.unwrap();

        let app = build_router(AppState::new(scheduler));
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/jobs/{}/events", job.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Would never resolve if the stream stayed open.
        let body = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            axum::body::to_bytes(response.into_body(), 1 << 20),
        )
        .await
        .expect("event stream did not end")
        .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"Completed\""), "{text}");
    }

    #[tokio::test]
    async fn job_event_stream_for_an_unknown_job_is_404() {
        let pool = create_in_memory_pool().await.unwrap();
        let repo = ProteusRepository::new(pool);
        let tmp = tempdir().unwrap();
        let scheduler = PipelineScheduler::new(
            repo,
            Arc::new(SimulatedRunner::new()),
            tmp.path().to_path_buf(),
        );
        let app = build_router(AppState::new(scheduler));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/jobs/00000000-0000-0000-0000-000000000000/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    // ---- Regressions from the 2026-09-23 bug hunt -------------------------------------

    async fn test_app(dir: &std::path::Path) -> Router {
        let repo = ProteusRepository::new(create_in_memory_pool().await.unwrap());
        let scheduler =
            PipelineScheduler::new(repo, Arc::new(SimulatedRunner::new()), dir.to_path_buf());
        build_router(AppState::new(scheduler))
    }

    async fn call(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, String) {
        let r = app.clone().oneshot(req).await.unwrap();
        let (status, headers) = (r.status(), r.headers().clone());
        let body = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        (status, headers, String::from_utf8_lossy(&body).into_owned())
    }

    async fn submit(app: &Router, body: serde_json::Value) -> String {
        let (status, _, text) = call(
            app,
            Request::post("/v1/tasks")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        serde_json::from_str::<serde_json::Value>(&text).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn sleeper() -> serde_json::Value {
        serde_json::json!({
            "executors": [{ "image": "alpine", "command": ["sleep", "30"] }],
            "inputs": [{ "path": "/data/in.txt", "content": "SECRET-INPUT" }],
        })
    }

    #[tokio::test]
    async fn cross_origin_requests_get_no_cors_grant() {
        // Reported: `Access-Control-Allow-Origin: *` let any web page drive a loopback daemon.
        let tmp = tempdir().unwrap();
        let app = test_app(tmp.path()).await;
        let (_, headers, _) = call(
            &app,
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/tasks")
                .header("origin", "https://evil.example")
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(headers.get("access-control-allow-origin").is_none());
    }

    #[tokio::test]
    async fn a_plain_post_to_a_task_does_not_cancel_it() {
        // Reported: `POST /v1/tasks/{id}` without `:cancel` cancelled the task.
        let tmp = tempdir().unwrap();
        let app = test_app(tmp.path()).await;
        let id = submit(&app, sleeper()).await;
        let (status, _, _) = call(
            &app,
            Request::post(format!("/v1/tasks/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        let (_, _, text) = call(
            &app,
            Request::get(format!("/v1/tasks/{id}?view=MINIMAL"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(!text.contains("CANCELED"), "{text}");
        let (status, _, _) = call(
            &app,
            Request::post(format!("/v1/tasks/{id}:cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn views_project_what_tes_says_they_project() {
        // Reported: MINIMAL carried `resources` and `executors`; BASIC carried input content;
        // `view=minimal` was accepted on the list endpoint but a plain-text 400 on a task.
        let tmp = tempdir().unwrap();
        let app = test_app(tmp.path()).await;
        let id = submit(&app, sleeper()).await;
        let (status, _, text) = call(
            &app,
            Request::get(format!("/v1/tasks/{id}?view=minimal"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["id", "state"], "{text}");

        let (_, _, text) = call(
            &app,
            Request::get(format!("/v1/tasks/{id}?view=BASIC"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(!text.contains("SECRET-INPUT"), "{text}");
        let (_, _, text) = call(
            &app,
            Request::get(format!("/v1/tasks/{id}?view=FULL"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(text.contains("SECRET-INPUT"), "{text}");

        let (status, _, text) = call(
            &app,
            Request::get(format!("/v1/tasks/{id}?view=bogus"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            serde_json::from_str::<serde_json::Value>(&text).is_ok(),
            "{text}"
        );
    }

    #[tokio::test]
    async fn list_filters_reject_what_they_cannot_mean() {
        // Reported: `state=BOGUS` gave an empty 200, and a page_token for u64::MAX gave the
        // first page (and would overflow in a debug build).
        use base64::Engine;
        let tmp = tempdir().unwrap();
        let app = test_app(tmp.path()).await;
        let (status, _, _) = call(
            &app,
            Request::get("/v1/tasks?state=BOGUS")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _, _) = call(
            &app,
            Request::get("/v1/tasks?state=RUNNING")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(u64::MAX.to_string());
        let (status, _, _) = call(
            &app,
            Request::get(format!("/v1/tasks?page_token={token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}
