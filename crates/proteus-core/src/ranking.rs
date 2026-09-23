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

    // 1. pLDDT Component (weight: 0.30). Absent for experimental structures, whose B-factor
    // column is not a confidence; its weight is then redistributed over the other terms.
    let has_plddt = metrics.plddt().is_some();
    let plddt_component = metrics
        .plddt()
        .map(|p| p.mean.clamp(0.0, 100.0))
        .unwrap_or(0.0);

    // 2. Compactness Component (weight: 0.20)
    // Empirical folded-protein scaling Rg ≈ 2.2·N^0.38 Å (Skolnick & Kolinski; Hong & Lei
    // 2009 give the same form). Measured on the validation corpus the single-chain entries sit
    // at 0.93–1.06 of this line. Full credit up to 1.10×, none from 2.0× (a denatured chain
    // follows ≈1.93·N^0.598 Å, i.e. 2.5–3× the folded value at typical lengths).
    let expected_rg = 2.2 * n_res.powf(0.38);
    let actual_rg = metrics.radius_of_gyration;

    let rg_ratio = actual_rg / expected_rg.max(1.0);
    let compactness_component = if rg_ratio <= 1.10 {
        100.0
    } else if rg_ratio >= 2.0 {
        0.0
    } else {
        (1.0 - (rg_ratio - 1.10) / 0.90) * 100.0
    };

    // 3. Ramachandran Stereochemistry Component (weight: 0.15)
    // `None` means a C-alpha-only trace, which has no φ/ψ to judge (see
    // `analyze_pdb_detailed_with_source`). A full-atom model whose peptide bonds are all
    // broken still has stats, with nothing favoured, and scores 0 here as it should.
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
    // Normal protein crystal structures have heavy_atom_overlap_score < 5. Clashes > 15 incur penalty.
    let clash_penalty = if let Some(ref clash) = metrics.steric_overlap {
        (clash.heavy_atom_overlap_score * 0.5).min(20.0)
    } else {
        0.0
    };

    // Weighted composite score: pLDDT 0.30, compactness 0.20, Ramachandran 0.15,
    // burial 0.15, network 0.20. Without pLDDT the remaining weights are scaled by 1/0.70.
    let (w_p, w_c, w_r, w_b, w_n) = if has_plddt {
        (0.30, 0.20, 0.15, 0.15, 0.20)
    } else {
        (0.0, 0.20 / 0.70, 0.15 / 0.70, 0.15 / 0.70, 0.20 / 0.70)
    };
    let total_score = (w_p * plddt_component
        + w_c * compactness_component
        + w_r * ramachandran_component
        + w_b * hydrophobic_burial_component
        + w_n * interaction_network_component
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
            confidence_source: Default::default(),
            secondary_structure_summary: None,
            ramachandran_stats: None,
            steric_overlap: None,
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

    fn with_rg(rg: f64) -> BiophysicalMetrics {
        BiophysicalMetrics {
            id: Uuid::new_v4(),
            prediction_id: Uuid::new_v4(),
            radius_of_gyration: rg,
            rmsd_to_reference: None,
            contact_density: 0.1,
            plddt_distribution: PlddtDistribution {
                mean: 90.0,
                median: 90.0,
                high_confidence_fraction: 1.0,
                very_high_confidence_fraction: 0.5,
            },
            confidence_source: Default::default(),
            secondary_structure_summary: None,
            ramachandran_stats: None,
            steric_overlap: None,
            sasa_metrics: None,
            interaction_network: None,
            candidate_fitness_score: None,
        }
    }

    /// Folded-protein scaling from the literature (Rg ≈ 2.2·N^0.38 Å): a structure on that
    /// line is fully compact, one 1.6× wider is clearly penalised, one 2.5× wider (an extended
    /// chain) scores nothing. The old constant (2.82·N^0.392) put the 1.6× case at 85/100.
    #[test]
    fn compactness_is_calibrated_to_folded_protein_scaling() {
        let n = 150usize;
        let folded = 2.2 * (n as f64).powf(0.38);
        assert_eq!(
            evaluate_candidate_fitness(&with_rg(folded), n).compactness_component,
            100.0
        );
        assert_eq!(
            evaluate_candidate_fitness(&with_rg(folded * 0.95), n).compactness_component,
            100.0
        );
        let loose = evaluate_candidate_fitness(&with_rg(folded * 1.6), n).compactness_component;
        assert!(loose < 60.0, "1.6x folded Rg scored {loose}");
        assert!(loose > 0.0);
        assert_eq!(
            evaluate_candidate_fitness(&with_rg(folded * 2.5), n).compactness_component,
            0.0
        );
    }

    #[test]
    fn experimental_structure_redistributes_plddt_weight() {
        let base = BiophysicalMetrics {
            id: Uuid::new_v4(),
            prediction_id: Uuid::new_v4(),
            radius_of_gyration: 10.0,
            rmsd_to_reference: None,
            contact_density: 0.12,
            plddt_distribution: PlddtDistribution {
                mean: 0.0,
                median: 0.0,
                high_confidence_fraction: 0.0,
                very_high_confidence_fraction: 0.0,
            },
            confidence_source: crate::confidence::ConfidenceSource::Predicted,
            secondary_structure_summary: None,
            ramachandran_stats: None,
            steric_overlap: None,
            sasa_metrics: None,
            interaction_network: None,
            candidate_fitness_score: None,
        };
        let predicted_zero = evaluate_candidate_fitness(&base, 46);
        let experimental = BiophysicalMetrics {
            confidence_source: crate::confidence::ConfidenceSource::ExperimentalBFactor,
            ..base
        };
        let exp = evaluate_candidate_fitness(&experimental, 46);
        assert!(exp.total_score > predicted_zero.total_score);
        // Four-term weighted mean with weights scaled by 1/0.70.
        let expected = (0.20 * exp.compactness_component
            + 0.15 * exp.ramachandran_component
            + 0.15 * exp.hydrophobic_burial_component
            + 0.20 * exp.interaction_network_component)
            / 0.70;
        assert!((exp.total_score - expected).abs() < 1e-9);
        assert_eq!(exp.plddt_component, 0.0);
    }
}
