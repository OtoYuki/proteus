use crate::server::AppState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use proteus_core::tes::{
    TesCancelTaskResponse, TesCreateTaskResponse, TesListTasksResponse, TesServiceInfo, TesTask,
    TesTaskView,
};
use proteus_engine::CancelOutcome;
use serde::Deserialize;
use std::sync::atomic::Ordering;

#[derive(Debug, Deserialize, Default)]
pub struct GetTaskQuery {
    pub view: Option<TesTaskView>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ListTasksQuery {
    pub name_prefix: Option<String>,
    pub page_size: Option<usize>,
    pub page_token: Option<String>,
    pub state: Option<String>,
    pub view: Option<TesTaskView>,
}

/// POST /v1/tasks and /ga4gh/tes/v1/tasks
pub async fn create_task(
    State(state): State<AppState>,
    Json(task): Json<TesTask>,
) -> Result<(StatusCode, Json<TesCreateTaskResponse>), (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);
    state.telemetry.tasks_queued.fetch_add(1, Ordering::Relaxed);

    match state.scheduler.submit_tes_task(task).await {
        Ok(id) => Ok((StatusCode::OK, Json(TesCreateTaskResponse { id }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

/// GET /v1/tasks/{id} and /ga4gh/tes/v1/tasks/{id}
pub async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<GetTaskQuery>,
) -> Result<Json<TesTask>, (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let view = query.view.unwrap_or(TesTaskView::Basic);

    match state.scheduler.repo().get_tes_task(&id).await {
        Ok(Some(record)) => match serde_json::from_str::<TesTask>(&record.task_json) {
            Ok(task) => Ok(Json(task.project_view(view))),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to parse task JSON: {e}") })),
            )),
        },
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Task '{id}' not found") })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

/// GET /v1/tasks and /ga4gh/tes/v1/tasks
pub async fn list_tasks(
    State(state): State<AppState>,
    Query(query): Query<ListTasksQuery>,
) -> Result<Json<TesListTasksResponse>, (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let view = query.view.unwrap_or(TesTaskView::Minimal);
    let limit = query.page_size.unwrap_or(256) as i64;

    match state
        .scheduler
        .repo()
        .list_tes_tasks(query.state.as_deref(), limit)
        .await
    {
        Ok(records) => {
            let mut tasks = Vec::with_capacity(records.len());
            for rec in records {
                if let Ok(task) = serde_json::from_str::<TesTask>(&rec.task_json) {
                    if let Some(ref prefix) = query.name_prefix {
                        if !task.name.as_deref().unwrap_or("").starts_with(prefix) {
                            continue;
                        }
                    }
                    tasks.push(task.project_view(view));
                }
            }
            Ok(Json(TesListTasksResponse {
                tasks,
                next_page_token: None,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

/// POST /v1/tasks/{id}:cancel, /v1/tasks/{id}/cancel, and /ga4gh/tes/v1/tasks/{id}:cancel
pub async fn cancel_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<TesCancelTaskResponse>), (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let clean_id = id.strip_suffix(":cancel").unwrap_or(&id);

    match state.scheduler.cancel_tes_task(clean_id).await {
        Ok(CancelOutcome::Canceled) => {
            state
                .telemetry
                .tasks_canceled
                .fetch_add(1, Ordering::Relaxed);
            Ok((StatusCode::OK, Json(TesCancelTaskResponse {})))
        }
        // Idempotent: a cancel after completion is not an error (TES clients retry cancels).
        Ok(CancelOutcome::AlreadyTerminal) => Ok((StatusCode::OK, Json(TesCancelTaskResponse {}))),
        Ok(CancelOutcome::NotFound) => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Task '{clean_id}' not found") })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

/// GET /v1/service-info, /v1/tasks/service-info, /ga4gh/tes/v1/service-info
pub async fn get_service_info(State(state): State<AppState>) -> impl IntoResponse {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);
    Json(TesServiceInfo::default())
}
