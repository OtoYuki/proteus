use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum PipelineTier {
    FastScreening,  // ESMFold
    HighFidelity,   // Boltz-1 / ColabFold
    FullValidation, // GROMACS Molecular Dynamics
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum JobStatus {
    Pending,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Sequence {
    pub id: Uuid,
    pub header: String,
    pub fasta: String,
    pub length: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PipelineJob {
    pub id: Uuid,
    pub sequence_id: Uuid,
    pub tier: PipelineTier,
    pub status: JobStatus,
    pub priority: i32,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_log: Option<String>,
}

impl PipelineJob {
    pub fn tier_slug(&self) -> &'static str {
        match self.tier {
            PipelineTier::FastScreening => "fast",
            PipelineTier::HighFidelity => "sota",
            PipelineTier::FullValidation => "md",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Prediction {
    pub id: Uuid,
    pub job_id: Uuid,
    pub pdb_path: String,
    pub plddt: Option<f64>,
    pub confidence_category: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct BiophysicalMetrics {
    pub id: Uuid,
    pub prediction_id: Uuid,
    pub radius_of_gyration: f64,
    pub rmsd_to_reference: Option<f64>,
    pub contact_density: f64,
    pub plddt_distribution: PlddtDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PlddtDistribution {
    pub mean: f64,
    pub median: f64,
    pub high_confidence_fraction: f64,      // pLDDT > 70
    pub very_high_confidence_fraction: f64, // pLDDT > 90
}
