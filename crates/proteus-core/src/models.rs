use crate::clash::StericOverlapStats;
use crate::sasa::SasaMetrics;
use crate::structure::{RamachandranStats, SecondaryStructureSummary};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum PipelineTier {
    /// Fast single-sequence transformer screening (ESMFold / ESM-2)
    FastScreening,
    /// All-atom diffusion structure prediction (Boltz) in a user-supplied container image
    HighFidelity,
    /// Molecular-mechanics relaxation (OpenMM). Not implemented: refused by every runner
    FullValidation,
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
            PipelineTier::HighFidelity => "boltz",
            PipelineTier::FullValidation => "relax",
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
    /// Whether `plddt_distribution` is a real confidence (predicted model) or just the
    /// B-factor column of an experimental structure.
    #[serde(default)]
    pub confidence_source: crate::confidence::ConfidenceSource,
    pub secondary_structure_summary: Option<SecondaryStructureSummary>,
    pub ramachandran_stats: Option<RamachandranStats>,
    /// Heavy-atom steric overlaps (not a MolProbity clashscore; see `clash::StericOverlapStats`).
    #[serde(alias = "clash_stats")]
    pub steric_overlap: Option<StericOverlapStats>,
    pub sasa_metrics: Option<SasaMetrics>,
    pub interaction_network: Option<crate::interactions::InteractionNetwork>,
    pub candidate_fitness_score: Option<f64>,
}

impl BiophysicalMetrics {
    /// pLDDT statistics, only meaningful for predicted structures. `None` when the
    /// B-factor column is known to be experimental.
    pub fn plddt(&self) -> Option<&PlddtDistribution> {
        match self.confidence_source {
            crate::confidence::ConfidenceSource::ExperimentalBFactor => None,
            _ => Some(&self.plddt_distribution),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PlddtDistribution {
    pub mean: f64,
    pub median: f64,
    pub high_confidence_fraction: f64,      // pLDDT > 70
    pub very_high_confidence_fraction: f64, // pLDDT > 90
}
