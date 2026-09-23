use crate::server::AppState;
use axum::extract::{Path, RawQuery, State};
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

/// `view` as TES clients send it: `MINIMAL`, `BASIC` or `FULL`, in any case.
fn parse_view(v: &str) -> Result<TesTaskView, String> {
    match v.to_ascii_uppercase().as_str() {
        "MINIMAL" => Ok(TesTaskView::Minimal),
        "BASIC" => Ok(TesTaskView::Basic),
        "FULL" => Ok(TesTaskView::Full),
        other => Err(format!("unknown view '{other}' (MINIMAL, BASIC or FULL)")),
    }
}

/// The TES 1.1 task states a `state` filter may name.
const TES_STATES: &[&str] = &[
    "UNKNOWN",
    "QUEUED",
    "INITIALIZING",
    "RUNNING",
    "PAUSED",
    "COMPLETE",
    "EXECUTOR_ERROR",
    "SYSTEM_ERROR",
    "CANCELED",
    "PREEMPTED",
    "CANCELING",
];

fn bad_request(msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg.into() })),
    )
}

#[derive(Debug, Deserialize, Default)]
pub struct ListTasksQuery {
    pub name_prefix: Option<String>,
    pub page_size: Option<usize>,
    pub page_token: Option<String>,
    pub state: Option<String>,
    pub view: Option<TesTaskView>,
    /// TES 1.1 `tag_key` (repeatable); paired by position with `tag_value`.
    #[serde(default)]
    pub tag_key: Vec<String>,
    #[serde(default)]
    pub tag_value: Vec<String>,
}

impl ListTasksQuery {
    /// `axum::extract::Query` cannot decode repeated keys into a `Vec`, so the query string is
    /// parsed by hand (`tag_key=a&tag_key=b&tag_value=x`).
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        let mut q = ListTasksQuery::default();
        for (k, v) in form_urlencoded::parse(raw.unwrap_or("").as_bytes()) {
            match k.as_ref() {
                "name_prefix" => q.name_prefix = Some(v.into_owned()),
                "page_size" => {
                    q.page_size = Some(v.parse().map_err(|_| "page_size must be an integer")?)
                }
                "page_token" => q.page_token = Some(v.into_owned()),
                "state" => {
                    if !TES_STATES.contains(&v.as_ref()) {
                        return Err(format!("unknown state '{v}'"));
                    }
                    q.state = Some(v.into_owned())
                }
                "view" => q.view = Some(parse_view(&v)?),
                "tag_key" => q.tag_key.push(v.into_owned()),
                "tag_value" => q.tag_value.push(v.into_owned()),
                _ => {}
            }
        }
        Ok(q)
    }

    /// TES 1.1 tag semantics: every `tag_key[i]` must exist; when `tag_value[i]` is given it
    /// must match exactly. A key without a value matches any value.
    fn tags_match(&self, tags: &std::collections::HashMap<String, String>) -> bool {
        self.tag_key
            .iter()
            .enumerate()
            .all(|(i, key)| match tags.get(key) {
                None => false,
                Some(actual) => self.tag_value.get(i).is_none_or(|want| want == actual),
            })
    }
}

fn encode_page_token(offset: usize) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(offset.to_string())
}

fn decode_page_token(token: &str) -> Result<usize, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .map_err(|_| "invalid page_token")?;
    // Offsets go to SQL as i64; anything past that is not a token this server issued.
    std::str::from_utf8(&bytes)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| i64::try_from(n).is_ok())
        .ok_or_else(|| "invalid page_token".to_string())
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
        // Policy/validation rejections (image allow-list, relative paths) are client errors.
        Err(proteus_engine::error::EngineError::Tes(msg)) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )),
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
    RawQuery(raw): RawQuery,
) -> Result<Json<TesTask>, (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let mut view = TesTaskView::Basic;
    for (k, v) in form_urlencoded::parse(raw.as_deref().unwrap_or("").as_bytes()) {
        if k == "view" {
            view = parse_view(&v).map_err(bad_request)?;
        }
    }

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
    RawQuery(raw): RawQuery,
) -> Result<Json<TesListTasksResponse>, (StatusCode, Json<serde_json::Value>)> {
    state
        .telemetry
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let query = ListTasksQuery::parse(raw.as_deref()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        )
    })?;
    let view = query.view.unwrap_or(TesTaskView::Minimal);
    // TES: page_size default 256, max 2048.
    let page_size = query.page_size.unwrap_or(256).clamp(1, 2048);
    let offset = match query.page_token.as_deref() {
        Some(t) => decode_page_token(t).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            )
        })?,
        None => 0,
    };

    let repo = state.scheduler.repo();
    let db_err = |e: proteus_storage::StorageError| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    };

    // `state` and `name_prefix` are filtered and paged in SQL. Tags live inside the task JSON,
    // so a tag filter still has to scan the (state/prefix-filtered) rows.
    let (tasks, next_page_token) = if query.tag_key.is_empty() {
        let rows = repo
            .list_tes_tasks_page(
                query.state.as_deref(),
                query.name_prefix.as_deref(),
                page_size as i64,
                offset as i64,
            )
            .await
            .map_err(db_err)?;
        let total = repo
            .count_tes_tasks(query.state.as_deref(), query.name_prefix.as_deref())
            .await
            .map_err(db_err)? as usize;
        let tasks: Vec<TesTask> = rows
            .iter()
            .filter_map(|rec| serde_json::from_str::<TesTask>(&rec.task_json).ok())
            .map(|t| t.project_view(view))
            .collect();
        let end = offset + rows.len();
        (tasks, (end < total).then(|| encode_page_token(end)))
    } else {
        let rows = repo
            .list_tes_tasks_page(
                query.state.as_deref(),
                query.name_prefix.as_deref(),
                i64::MAX,
                0,
            )
            .await
            .map_err(db_err)?;
        let matching: Vec<TesTask> = rows
            .iter()
            .filter_map(|rec| serde_json::from_str::<TesTask>(&rec.task_json).ok())
            .filter(|t| query.tags_match(&t.tags))
            .collect();
        let end = (offset + page_size).min(matching.len());
        let next = (end < matching.len()).then(|| encode_page_token(end));
        let tasks = matching
            .into_iter()
            .skip(offset)
            .take(page_size)
            .map(|t| t.project_view(view))
            .collect();
        (tasks, next)
    };
    Ok(Json(TesListTasksResponse {
        tasks,
        next_page_token,
    }))
}

/// POST /v1/tasks/{id}:cancel and /ga4gh/tes/v1/tasks/{id}:cancel. The router matches
/// `{id}:cancel` as a single path segment; a POST to `/v1/tasks/{id}` without the suffix is
/// not a TES operation and must not cancel anything.
pub async fn cancel_task_colon(
    state: State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<TesCancelTaskResponse>), (StatusCode, Json<serde_json::Value>)> {
    if !id.ends_with(":cancel") {
        return Err((
            StatusCode::METHOD_NOT_ALLOWED,
            Json(serde_json::json!({
                "error": "POST is only defined on /tasks/{id}:cancel"
            })),
        ));
    }
    cancel_task(state, Path(id)).await
}

/// POST /v1/tasks/{id}/cancel (and the `:cancel` form, via [`cancel_task_colon`]).
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
    let mut info = TesServiceInfo::default();
    let tes = state.scheduler.tes_config();
    info.tags
        .insert("proteus.executor".into(), tes.executor.kind().into());
    info.tags.insert(
        "proteus.image_allowlist".into(),
        if tes.allow_images.is_empty() {
            "*".to_string()
        } else {
            tes.allow_images
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(",")
        },
    );
    info.tags.insert(
        "proteus.file_allowlist".into(),
        tes.allow_dirs
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    info.tags.insert(
        "proteus.executor_timeout_seconds".into(),
        tes.executor_timeout.as_secs().to_string(),
    );
    Json(info)
}
