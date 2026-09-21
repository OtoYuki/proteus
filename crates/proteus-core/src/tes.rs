use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GA4GH TES Task Execution lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TesState {
    #[default]
    Unknown,
    Queued,
    Initializing,
    Running,
    Paused,
    Complete,
    ExecutorError,
    SystemError,
    Canceled,
}

impl std::fmt::Display for TesState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "UNKNOWN"),
            Self::Queued => write!(f, "QUEUED"),
            Self::Initializing => write!(f, "INITIALIZING"),
            Self::Running => write!(f, "RUNNING"),
            Self::Paused => write!(f, "PAUSED"),
            Self::Complete => write!(f, "COMPLETE"),
            Self::ExecutorError => write!(f, "EXECUTOR_ERROR"),
            Self::SystemError => write!(f, "SYSTEM_ERROR"),
            Self::Canceled => write!(f, "CANCELED"),
        }
    }
}

/// Target view representation when querying TES tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TesTaskView {
    Minimal,
    #[default]
    Basic,
    Full,
}

/// File type for TES inputs and outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TesFileType {
    #[default]
    File,
    Directory,
}

fn default_file_type() -> TesFileType {
    TesFileType::File
}

/// Input file or directory specification for a TES task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub path: String,
    #[serde(rename = "type", default = "default_file_type")]
    pub type_: TesFileType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Output file or directory specification for a TES task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub path: String,
    #[serde(rename = "type", default = "default_file_type")]
    pub type_: TesFileType,
}

/// Specification for a container or command executor within a TES task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesExecutor {
    pub image: String,
    pub command: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdin: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub ignore_error: bool,
}

/// Compute resource requests for a TES task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesResources {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preemptible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_gb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_gb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zones: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_parameters: Option<HashMap<String, String>>,
    /// TES 1.1: when true, the server must reject unknown `backend_parameters`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_parameters_strict: Option<bool>,
}

/// Log record for a single executor within a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesExecutorLog {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

/// Log record for an output file produced by a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesOutputFileLog {
    pub url: String,
    pub path: String,
    /// int64 in the TES schema, which the protobuf JSON mapping encodes as a **string**;
    /// strict clients (Sprocket/Crankshaft's `tes` crate) reject a bare integer.
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "int64_as_string",
        default
    )]
    pub size_bytes: Option<u64>,
}

/// Serde helpers for TES int64 fields (`size_bytes`): serialize as decimal string, accept both
/// string and integer on input.
pub mod int64_as_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<u64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(n) => n.to_string().serialize(s),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(u64),
            Str(String),
            None,
        }
        match Raw::deserialize(d)? {
            Raw::Num(n) => Ok(Some(n)),
            Raw::Str(s) => s.parse().map(Some).map_err(serde::de::Error::custom),
            Raw::None => Ok(None),
        }
    }
}

/// Complete execution log record for a task attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesTaskLog {
    #[serde(default)]
    pub logs: Vec<TesExecutorLog>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(default)]
    pub outputs: Vec<TesOutputFileLog>,
    #[serde(default)]
    pub system_logs: Vec<String>,
}

/// Standard GA4GH TES Task definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesTask {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub state: TesState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<TesInput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<TesOutput>,
    #[serde(default)]
    pub resources: TesResources,
    #[serde(default)]
    pub executors: Vec<TesExecutor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub volumes: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<TesTaskLog>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creation_time: Option<String>,
}

impl TesTask {
    /// Projects this task according to the requested view level.
    pub fn project_view(&self, view: TesTaskView) -> Self {
        match view {
            TesTaskView::Minimal => Self {
                id: self.id.clone(),
                state: self.state,
                ..Default::default()
            },
            TesTaskView::Basic => {
                // Basic view includes all metadata, inputs, outputs, executors, but excludes executor logs stdout/stderr
                let mut basic = self.clone();
                for log in &mut basic.logs {
                    for exec_log in &mut log.logs {
                        exec_log.stdout = None;
                        exec_log.stderr = None;
                    }
                    log.system_logs.clear();
                }
                basic
            }
            TesTaskView::Full => self.clone(),
        }
    }
}

/// Response returned upon task creation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesCreateTaskResponse {
    pub id: String,
}

/// Paginated list of tasks response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesListTasksResponse {
    pub tasks: Vec<TesTask>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Response returned when a task is canceled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesCancelTaskResponse {}

/// GA4GH Service Type definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesServiceType {
    pub group: String,
    pub artifact: String,
    pub version: String,
}

/// GA4GH Service Organization definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesServiceOrganization {
    pub name: String,
    pub url: String,
}

/// GA4GH Service Info response (v1.0 standard).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TesServiceInfo {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub type_: TesServiceType,
    pub description: String,
    pub organization: TesServiceOrganization,
    #[serde(rename = "contactUrl", skip_serializing_if = "Option::is_none")]
    pub contact_url: Option<String>,
    #[serde(rename = "documentationUrl", skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    pub environment: String,
    pub version: String,
    pub storage: Vec<String>,
    /// Free-form server properties (`proteus.executor`, `proteus.image_allowlist`, …).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
}

impl Default for TesServiceInfo {
    fn default() -> Self {
        Self {
            id: "org.ga4gh.proteus".into(),
            name: "Proteus GA4GH Task Execution Service".into(),
            type_: TesServiceType {
                group: "org.ga4gh".into(),
                artifact: "tes".into(),
                version: "1.1.0".into(),
            },
            description: "High-throughput pure-Rust bio-compute task execution daemon with all-atom biophysics and CAS deduplication".into(),
            organization: TesServiceOrganization {
                name: "Proteus Contributors".into(),
                url: "https://github.com/OtoYuki/proteus".into(),
            },
            contact_url: Some("https://github.com/OtoYuki/proteus/issues".into()),
            documentation_url: Some("https://github.com/OtoYuki/proteus".into()),
            created_at: "2026-09-21T00:00:00Z".into(),
            updated_at: "2026-09-21T00:00:00Z".into(),
            environment: "production".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            storage: vec!["file".into(), "http".into()],
            tags: HashMap::new(),
        }
    }
}
