use proteus_core::models::PipelineTier;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Deserialize, ToSchema)]
pub struct SubmitSequenceRequest {
    pub fasta: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SubmitSequenceResponse {
    pub sequence_id: Uuid,
    pub header: String,
    pub length: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct EnqueueJobRequest {
    pub sequence_id: Uuid,
    pub tier: PipelineTier,
    #[serde(default)]
    pub priority: i32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EnqueueJobResponse {
    pub job_id: Uuid,
    pub sequence_id: Uuid,
    pub status: String,
}
