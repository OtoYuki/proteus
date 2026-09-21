use crate::models::BiophysicalMetrics;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CandidateFitness {
    pub total_score: f64,
    pub plddt_component: f64,
    pub compactness_component: f64,
    pub ramachandran_component: f64,
    pub hydrophobic_burial_component: f64,
    pub interaction_network_component: f64,
    pub tier_label: String,
}

/// Compute a multi-objective candidate fitness score (0.0 - 100.0)
/// combining folding confidence, tertiary compactness, stereochemical validity,
/// hydrophobic core stability, and non-covalent interaction network density.
pub fn evaluate_candidate_fitness(
    metrics: &BiophysicalMetrics,
    residue_count: usize,
) -> CandidateFitness {
    let n_res = residue_count.max(1) as f64;

    // 1. pLDDT Component (weight: 0.30)
    // Mean pLDDT is natively [0.0, 100.0]
    let plddt_component = metrics.plddt_distribution.mean.clamp(0.0, 100.0);

    // 2. Compactness Component (weight: 0.20)
    // Expected globular protein Rg follows Flory scaling: Rg_expected ≈ 2.82 * N^0.392
    let expected_rg = 2.82 * n_res.powf(0.392);
    let actual_rg = metrics.radius_of_gyration;

    let rg_ratio = actual_rg / expected_rg.max(1.0);
    let compactness_component = if rg_ratio <= 1.05 {
        100.0
    } else if rg_ratio >= 2.0 {
        0.0
    } else {
        (1.0 - (rg_ratio - 1.05) / 0.95) * 100.0
    };

    // 3. Ramachandran Stereochemistry Component (weight: 0.15)
    let ramachandran_component = if let Some(ref rama) = metrics.ramachandran_stats {
        (rama.favored_fraction * 100.0 + rama.allowed_fraction * 50.0).clamp(0.0, 100.0)
    } else {
        85.0 // Default baseline when only CA is present
    };

    // 4. Hydrophobic Core Burial Component (weight: 0.15)
    let hydrophobic_burial_component = if let Some(ref sasa) = metrics.sasa_metrics {
        (sasa.hydrophobic_burial_ratio * 100.0).clamp(0.0, 100.0)
    } else {
        // Fallback approximation via contact density (contact density > 0.08 indicates solid core)
        (metrics.contact_density * 800.0).clamp(0.0, 100.0)
    };

    // 5. Non-Covalent Interaction Network Component (weight: 0.20)
    let interaction_network_component = if let Some(ref net) = metrics.interaction_network {
        let tertiary_hbonds = (net.summary.bb_sc_hbonds + net.summary.sc_sc_hbonds) as f64;
        let bb_hbonds = net.summary.bb_bb_hbonds as f64;
        let salt_bridges = net.summary.total_salt_bridges as f64;
        let pi_stacks = net.summary.total_pi_pi_stacks as f64;
        let cation_pi = net.summary.total_cation_pi as f64;

        let raw_stabilization = 0.5 * bb_hbonds
            + 1.0 * tertiary_hbonds
            + 2.5 * salt_bridges
            + 2.0 * pi_stacks
            + 2.0 * cation_pi;

        let normalized_density = (raw_stabilization / n_res) * 100.0;
        // Typical well-folded globular protein has normalized density ~40.0 - 70.0
        (normalized_density / 60.0 * 100.0).clamp(0.0, 100.0)
    } else {
        (metrics.contact_density * 750.0).clamp(50.0, 85.0)
    };

    // 6. MolProbity Steric Clash Penalty
    // Normal protein crystal structures have clashscore < 5. Clashes > 15 incur penalty.
    let clash_penalty = if let Some(ref clash) = metrics.clash_stats {
        (clash.clashscore * 0.5).min(20.0)
    } else {
        0.0
    };

    // Weighted composite score
    let total_score = (0.30 * plddt_component
        + 0.20 * compactness_component
        + 0.15 * ramachandran_component
        + 0.15 * hydrophobic_burial_component
        + 0.20 * interaction_network_component
        - clash_penalty)
        .clamp(0.0, 100.0);

    let tier_label = if total_score >= 82.0 {
        "Lead Candidate (Synthesis Priority)".to_string()
    } else if total_score >= 68.0 {
        "Viable Scaffold (Optimization Candidate)".to_string()
    } else if total_score >= 50.0 {
        "Metastable / High Flexibility".to_string()
    } else {
        "Unviable (Aggregated / Disordered)".to_string()
    };

    CandidateFitness {
        total_score,
        plddt_component,
        compactness_component,
        ramachandran_component,
        hydrophobic_burial_component,
        interaction_network_component,
        tier_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::PlddtDistribution;
    use uuid::Uuid;

    #[test]
    fn test_high_quality_candidate_fitness() {
        let metrics = BiophysicalMetrics {
            id: Uuid::new_v4(),
            prediction_id: Uuid::new_v4(),
            radius_of_gyration: 10.0,
            rmsd_to_reference: None,
            contact_density: 0.12,
            plddt_distribution: PlddtDistribution {
                mean: 92.5,
                median: 93.0,
                high_confidence_fraction: 0.95,
                very_high_confidence_fraction: 0.85,
            },
            secondary_structure_summary: None,
            ramachandran_stats: None,
            clash_stats: None,
            sasa_metrics: None,
            interaction_network: None,
            candidate_fitness_score: None,
        };

        let fitness = evaluate_candidate_fitness(&metrics, 46);
        assert!(
            fitness.total_score > 85.0,
            "Expected fitness > 85, got {}",
            fitness.total_score
        );
        assert_eq!(fitness.tier_label, "Lead Candidate (Synthesis Priority)");
    }
}
