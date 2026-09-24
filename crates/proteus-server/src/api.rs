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
            proteus_core::confidence::ConfidenceSource,
            proteus_core::rama8000::RamaClass,
            proteus_core::structure::SecondaryStructureSummary,
            proteus_core::structure::SecondaryStructure,
            proteus_core::structure::RamachandranStats,
            proteus_core::clash::StericOverlapStats,
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
            proteus_core::ranking::CandidateFitness,
            proteus_core::tes::TesTask,
            proteus_core::tes::TesState,
            proteus_core::tes::TesExecutor,
            proteus_core::tes::TesInput,
            proteus_core::tes::TesOutput,
            proteus_core::tes::TesFileType,
            proteus_core::tes::TesResources,
            proteus_core::tes::TesTaskLog,
            proteus_core::tes::TesExecutorLog,
            proteus_core::tes::TesOutputFileLog,
            proteus_core::tes::TesCreateTaskResponse,
            proteus_core::tes::TesListTasksResponse,
            proteus_core::tes::TesCancelTaskResponse,
            proteus_core::tes::TesServiceInfo,
            proteus_core::tes::TesServiceType,
            proteus_core::tes::TesServiceOrganization
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

/// Server-sent events for one job. The first event is the job's current record, so a client
/// that subscribes after the job finished still learns its state; the stream ends after the
/// terminal event instead of staying open forever. Unknown jobs are 404.
pub async fn stream_job_events(State(state): State<AppState>, Path(id): Path<Uuid>) -> Response {
    // Subscribe before reading the record, so a completion between the two is not missed.
    let rx = state.scheduler.subscribe();
    let job = match state.scheduler.repo().get_job(id).await {
        Ok(Some(j)) => j,
        Ok(None) => return (StatusCode::NOT_FOUND, "Job not found").into_response(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let terminal = matches!(
        job.status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
    );
    let first = Event::default()
        .event("status")
        .data(serde_json::to_string(&job).unwrap_or_default());
    let first = tokio_stream::once(Ok::<Event, Infallible>(first));

    let live = BroadcastStream::new(rx)
        .filter_map(move |item| {
            let event = item.ok()?;
            let (matches, ends) = match &event {
                proteus_engine::EngineEvent::JobQueued { job_id, .. }
                | proteus_engine::EngineEvent::JobStarted { job_id }
                | proteus_engine::EngineEvent::JobProgress { job_id, .. } => (*job_id == id, false),
                proteus_engine::EngineEvent::JobCompleted { job_id, .. }
                | proteus_engine::EngineEvent::JobFailed { job_id, .. } => (*job_id == id, true),
                _ => (false, false),
            };
            matches.then(|| {
                (
                    Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&event).unwrap_or_default()),
                    ),
                    ends,
                )
            })
        })
        // Emit up to and including the terminal event, then stop.
        .map_while({
            let mut done = false;
            move |(ev, ends)| {
                if done {
                    return None;
                }
                done = ends;
                Some(ev)
            }
        });
    let live = live.take_while(move |_| !terminal);

    Sse::new(first.chain(live)).into_response()
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

/// Browser viewer page for a job's prediction, drawn by `proteus_render::web` from the same
/// ribbon, DSSP and measurements as the terminal viewer, and coloured by pLDDT only when the file
/// really is a prediction. A job without a prediction is a 404 rather than an empty viewer.
pub async fn view_structure(State(state): State<AppState>, Path(job_id): Path<Uuid>) -> Response {
    let pred = match state.scheduler.repo().get_prediction_by_job(job_id).await {
        Ok(Some(p)) => p,
        Ok(None) => return (StatusCode::NOT_FOUND, "Prediction not found").into_response(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    let engine = proteus_engine::engine_name(pred.metadata.as_ref());
    let caption = if engine == proteus_engine::ENGINE_SIMULATED {
        format!("job {job_id} — SIMULATED: synthetic helix, not a prediction")
    } else if let Some(d) = proteus_engine::tier_downgrade(pred.metadata.as_ref()) {
        format!(
            "job {job_id} — {engine}; tier '{}' not honoured: {}",
            d.requested, d.reason
        )
    } else {
        format!("job {job_id} — engine: {engine}")
    };

    // Everything is embedded rather than fetched by the page: a second request would carry no
    // Authorization header and be refused behind --auth-token, and the page must also open from
    // disk with no server at all.
    let text = match proteus_core::io::read_structure_text(std::path::Path::new(&pred.pdb_path)) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read prediction artifact: {e}"),
            )
                .into_response()
        }
    };
    // Building the mesh is CPU work (0.7 s for a 2 900-residue spike); keep it off the runtime.
    let model_path = std::path::PathBuf::from(&pred.pdb_path);
    let rendered = tokio::task::spawn_blocking(move || {
        proteus_render::parse_pdb_structure(&text).map(|mut structure| {
            // PAE and pTM the predictor wrote beside the model (Boltz), when it did.
            if let Ok(c) = proteus_core::pae::read_confidence(&model_path, None) {
                let _ = structure.attach_confidence(c);
            }
            let scheme = structure.default_color_scheme();
            proteus_render::web::WebPage {
                title: "Proteus structure viewer",
                caption: &caption,
                structure: &structure,
                scheme,
            }
            .render()
        })
    })
    .await;
    let html = match rendered {
        Ok(Ok(html)) => html,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build the viewer for this prediction: {e}"),
            )
                .into_response()
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Viewer task failed: {e}"),
            )
                .into_response()
        }
    };
    axum::response::Html(html).into_response()
}
