use crate::dto::{
    EnqueueJobRequest, EnqueueJobResponse, SubmitSequenceRequest, SubmitSequenceResponse,
};
use crate::server::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use futures_util::Stream;
use proteus_core::models::{BiophysicalMetrics, JobStatus, PipelineJob, Prediction, Sequence};
use proteus_core::sequence::validate_and_parse_fasta;
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use utoipa::OpenApi;
use uuid::Uuid;

#[derive(OpenApi)]
#[openapi(
    paths(
        submit_sequence,
        get_sequence,
        enqueue_job,
        get_job,
        get_prediction,
        get_metrics,
        health_check
    ),
    components(
        schemas(
            SubmitSequenceRequest,
            SubmitSequenceResponse,
            EnqueueJobRequest,
            EnqueueJobResponse,
            proteus_core::models::Sequence,
            proteus_core::models::PipelineJob,
            proteus_core::models::PipelineTier,
            proteus_core::models::JobStatus,
            proteus_core::models::Prediction,
            proteus_core::models::BiophysicalMetrics,
            proteus_core::models::PlddtDistribution,
            proteus_core::structure::SecondaryStructureSummary,
            proteus_core::structure::SecondaryStructure,
            proteus_core::structure::RamachandranStats,
            proteus_core::clash::ClashStats,
            proteus_core::clash::StericClash,
            proteus_core::sasa::SasaMetrics,
            proteus_core::interactions::InteractionNetwork,
            proteus_core::interactions::InteractionSummary,
            proteus_core::interactions::HydrogenBond,
            proteus_core::interactions::HBondCategory,
            proteus_core::interactions::SaltBridge,
            proteus_core::interactions::PiStacking,
            proteus_core::interactions::PiStackingCategory,
            proteus_core::interactions::CationPiInteraction,
            proteus_core::ranking::CandidateFitness
        )
    ),
    tags(
        (name = "Proteus", description = "High-throughput Bio-Compute API")
    )
)]
pub struct ApiDoc;

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "Service healthy"))
)]
pub async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "proteus-server",
        "timestamp": Utc::now()
    }))
}

#[utoipa::path(
    post,
    path = "/api/v1/sequences",
    request_body = SubmitSequenceRequest,
    responses(
        (status = 201, description = "Sequence parsed & stored", body = SubmitSequenceResponse),
        (status = 400, description = "Invalid FASTA")
    )
)]
pub async fn submit_sequence(
    State(state): State<AppState>,
    Json(payload): Json<SubmitSequenceRequest>,
) -> Response {
    let sequence = match validate_and_parse_fasta(&payload.fasta) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    if let Err(e) = state.scheduler.repo().insert_sequence(&sequence).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response();
    }

    (
        StatusCode::CREATED,
        Json(SubmitSequenceResponse {
            sequence_id: sequence.id,
            header: sequence.header,
            length: sequence.length,
        }),
    )
        .into_response()
}

#[utoipa::path(
    get,
    path = "/api/v1/sequences/{id}",
    params(("id" = Uuid, Path, description = "Sequence UUID")),
    responses(
        (status = 200, description = "Sequence details", body = Sequence),
        (status = 404, description = "Sequence not found")
    )
)]
pub async fn get_sequence(State(state): State<AppState>, Path(id): Path<Uuid>) -> Response {
    match state.scheduler.repo().get_sequence(id).await {
        Ok(Some(seq)) => Json(seq).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "Sequence not found").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/api/v1/jobs",
    request_body = EnqueueJobRequest,
    responses(
        (status = 202, description = "Job accepted & queued", body = EnqueueJobResponse),
        (status = 404, description = "Sequence not found")
    )
)]
pub async fn enqueue_job(
    State(state): State<AppState>,
    Json(payload): Json<EnqueueJobRequest>,
) -> Response {
    let seq = match state
        .scheduler
        .repo()
        .get_sequence(payload.sequence_id)
        .await
    {
        Ok(Some(s)) => s,
        Ok(None) => return (StatusCode::NOT_FOUND, "Sequence not found").into_response(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let job_id = Uuid::new_v4();
    let job = PipelineJob {
        id: job_id,
        sequence_id: seq.id,
        tier: payload.tier,
        status: JobStatus::Queued,
        priority: payload.priority,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error_log: None,
    };

    if let Err(e) = state.scheduler.repo().insert_job(&job).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response();
    }

    // Spawn pipeline processing in background
    let sched = state.scheduler.clone();
    tokio::spawn(async move {
        if let Err(e) = sched.process_job(job_id).await {
            tracing::error!("Background job {} error: {}", job_id, e);
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(EnqueueJobResponse {
            job_id,
            sequence_id: seq.id,
            status: "Queued".to_string(),
        }),
    )
        .into_response()
}

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{id}",
    params(("id" = Uuid, Path, description = "Job UUID")),
    responses(
        (status = 200, description = "Job details", body = PipelineJob),
        (status = 404, description = "Job not found")
    )
)]
pub async fn get_job(State(state): State<AppState>, Path(id): Path<Uuid>) -> Response {
    match state.scheduler.repo().get_job(id).await {
        Ok(Some(job)) => Json(job).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "Job not found").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn stream_job_events(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.scheduler.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |item| match item {
        Ok(event) => {
            let matches = match &event {
                proteus_engine::EngineEvent::JobQueued { job_id, .. } => *job_id == id,
                proteus_engine::EngineEvent::JobStarted { job_id } => *job_id == id,
                proteus_engine::EngineEvent::JobProgress { job_id, .. } => *job_id == id,
                proteus_engine::EngineEvent::JobCompleted { job_id, .. } => *job_id == id,
                proteus_engine::EngineEvent::JobFailed { job_id, .. } => *job_id == id,
            };
            if matches {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Some(Ok(Event::default().data(data)))
            } else {
                None
            }
        }
        Err(_) => None,
    });

    Sse::new(stream)
}

#[utoipa::path(
    get,
    path = "/api/v1/predictions/by-job/{job_id}",
    params(("job_id" = Uuid, Path, description = "Job UUID")),
    responses(
        (status = 200, description = "Prediction result", body = Prediction),
        (status = 404, description = "Prediction not found")
    )
)]
pub async fn get_prediction(State(state): State<AppState>, Path(job_id): Path<Uuid>) -> Response {
    match state.scheduler.repo().get_prediction_by_job(job_id).await {
        Ok(Some(pred)) => Json(pred).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "Prediction not found").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/v1/metrics/by-prediction/{prediction_id}",
    params(("prediction_id" = Uuid, Path, description = "Prediction UUID")),
    responses(
        (status = 200, description = "Biophysical metrics", body = BiophysicalMetrics),
        (status = 404, description = "Metrics not found")
    )
)]
pub async fn get_metrics(
    State(state): State<AppState>,
    Path(prediction_id): Path<Uuid>,
) -> Response {
    match state
        .scheduler
        .repo()
        .get_metrics_by_prediction(prediction_id)
        .await
    {
        Ok(Some(metrics)) => Json(metrics).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "Metrics not found").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn get_prediction_pdb(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
) -> Response {
    match state.scheduler.repo().get_prediction_by_job(job_id).await {
        Ok(Some(pred)) => match tokio::fs::read_to_string(&pred.pdb_path).await {
            Ok(content) => (
                [(axum::http::header::CONTENT_TYPE, "chemical/x-pdb")],
                content,
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read PDB artifact: {e}"),
            )
                .into_response(),
        },
        Ok(None) => (StatusCode::NOT_FOUND, "Prediction not found").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {e}"),
        )
            .into_response(),
    }
}

pub async fn view_structure(Path(job_id): Path<Uuid>) -> axum::response::Html<String> {
    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Proteus 3D Structure Viewer</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/molstar@3.30.0/build/viewer/molstar.css" />
    <script type="text/javascript" src="https://unpkg.com/molstar@3.30.0/build/viewer/molstar.js"></script>
    <style>
        body, html {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #111; color: #fff; }}
        #app {{ width: 100%; height: 100%; position: absolute; }}
        #header {{ position: absolute; top: 12px; left: 16px; z-index: 1000; background: rgba(0,0,0,0.75); padding: 8px 16px; border-radius: 8px; backdrop-filter: blur(8px); border: 1px solid #333; }}
        #header h1 {{ margin: 0; font-size: 14px; font-weight: 600; color: #4ade80; }}
        #header p {{ margin: 2px 0 0 0; font-size: 11px; color: #aaa; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Proteus Bio-Compute 3D Viewer</h1>
        <p>Job ID: {job_id}</p>
    </div>
    <div id="app"></div>
    <script>
        document.addEventListener('DOMContentLoaded', async () => {{
            const viewer = await molstar.Viewer.create('app', {{
                layoutIsExpanded: false,
                layoutShowControls: true,
                layoutShowRemoteState: false,
                layoutShowSequence: true,
                layoutShowLog: false,
                viewportShowExpand: false,
            }});
            await viewer.loadStructureFromUrl('/api/v1/predictions/by-job/{job_id}/pdb', 'pdb', false, {{
                representationStyle: {{
                    type: 'cartoon',
                    color: 'plddt',
                }}
            }});
        }});
    </script>
</body>
</html>"#
    );
    axum::response::Html(html)
}
