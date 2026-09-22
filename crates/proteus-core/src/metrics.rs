use crate::error::CoreError;
use crate::models::{BiophysicalMetrics, PlddtDistribution};
use nalgebra::{Matrix3, Vector3, SVD};
use std::path::Path;
use uuid::Uuid;

/// Compute center of mass for a set of atom coordinates.
pub fn compute_center_of_mass(coords: &[Vector3<f64>]) -> Result<Vector3<f64>, CoreError> {
    if coords.is_empty() {
        return Err(CoreError::AnalysisError(
            "Cannot compute center of mass for empty coordinates".into(),
        ));
    }
    let sum: Vector3<f64> = coords.iter().sum();
    Ok(sum / (coords.len() as f64))
}

/// Compute Radius of Gyration ($R_g$) over C-alpha coordinates.
pub fn compute_radius_of_gyration(coords: &[Vector3<f64>]) -> Result<f64, CoreError> {
    if coords.is_empty() {
        return Err(CoreError::AnalysisError(
            "Cannot compute Rg for empty coordinates".into(),
        ));
    }
    let com = compute_center_of_mass(coords)?;
    let sq_dist_sum: f64 = coords.iter().map(|p| (p - com).norm_squared()).sum();
    Ok((sq_dist_sum / (coords.len() as f64)).sqrt())
}

#[derive(Debug, Clone)]
pub struct KabschSuperpositionResult {
    pub rmsd: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub aligned_coords: Vec<Vector3<f64>>,
}

/// Compute Kabsch optimal superposition and transformation between two aligned coordinate sets.
pub fn compute_kabsch_superposition(
    p_coords: &[Vector3<f64>],
    q_coords: &[Vector3<f64>],
) -> Result<KabschSuperpositionResult, CoreError> {
    if p_coords.len() != q_coords.len() {
        return Err(CoreError::AnalysisError(format!(
            "Coordinate length mismatch for RMSD: {} vs {}",
            p_coords.len(),
            q_coords.len()
        )));
    }
    let n = p_coords.len();
    if n == 0 {
        return Err(CoreError::AnalysisError(
            "Cannot compute RMSD for zero atoms".into(),
        ));
    }

    let p_com = compute_center_of_mass(p_coords)?;
    let q_com = compute_center_of_mass(q_coords)?;

    let p_centered: Vec<Vector3<f64>> = p_coords.iter().map(|p| p - p_com).collect();
    let q_centered: Vec<Vector3<f64>> = q_coords.iter().map(|q| q - q_com).collect();

    // Covariance matrix H = P^T * Q
    let mut h = Matrix3::zeros();
    for (p, q) in p_centered.iter().zip(q_centered.iter()) {
        h += p * q.transpose();
    }

    // SVD of H: H = U * S * V^T
    let svd = SVD::new(h, true, true);
    let u = svd.u.ok_or_else(|| {
        CoreError::AnalysisError("SVD decomposition failed to produce U matrix".into())
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        CoreError::AnalysisError("SVD decomposition failed to produce V^T matrix".into())
    })?;

    let v = v_t.transpose();
    let mut d = Matrix3::identity();
    let det = (v * u.transpose()).determinant();
    if det < 0.0 {
        d[(2, 2)] = -1.0;
    }

    let rotation = v * d * u.transpose();
    let translation = q_com - rotation * p_com;

    // Compute RMSD and transformed coordinates P' = rotation * (P - COM_p) + COM_q
    let mut sum_sq_diff = 0.0;
    let mut aligned_coords = Vec::with_capacity(n);
    for (p, q) in p_centered.iter().zip(q_centered.iter()) {
        let p_rotated = rotation * p;
        sum_sq_diff += (p_rotated - q).norm_squared();
        aligned_coords.push(p_rotated + q_com);
    }

    let rmsd = (sum_sq_diff / (n as f64)).sqrt();

    Ok(KabschSuperpositionResult {
        rmsd,
        rotation,
        translation,
        aligned_coords,
    })
}

/// Compute Kabsch optimal superposition C-alpha RMSD between two aligned sets of coordinates.
pub fn compute_kabsch_rmsd(
    p_coords: &[Vector3<f64>],
    q_coords: &[Vector3<f64>],
) -> Result<f64, CoreError> {
    Ok(compute_kabsch_superposition(p_coords, q_coords)?.rmsd)
}

/// Compute residue contact density (non-consecutive residues |i-j| >= 4 with C-alpha distance <= 8.0 A).
pub fn compute_contact_density(coords: &[Vector3<f64>], threshold_angstrom: f64) -> f64 {
    let n = coords.len();
    if n < 5 {
        return 0.0;
    }

    let threshold_sq = threshold_angstrom * threshold_angstrom;
    let mut contacts = 0usize;
    let mut total_pairs = 0usize;

    for i in 0..n {
        for j in (i + 4)..n {
            total_pairs += 1;
            let dist_sq = (coords[i] - coords[j]).norm_squared();
            if dist_sq <= threshold_sq {
                contacts += 1;
            }
        }
    }

    if total_pairs == 0 {
        0.0
    } else {
        (contacts as f64) / (total_pairs as f64)
    }
}

#[derive(Debug, Clone)]
pub struct DetailedBiophysicalAnalysis {
    pub metrics: BiophysicalMetrics,
    pub plddts: Vec<f64>,
    pub ramachandran_points: Vec<(
        Option<f64>,
        Option<f64>,
        crate::structure::RamachandranRegion,
    )>,
}

/// Analyse a structure file (PDB / mmCIF, optionally gzip-compressed) from disk.
pub fn analyze_pdb_file(
    path: &Path,
    reference_path: Option<&Path>,
) -> Result<BiophysicalMetrics, CoreError> {
    let loaded = crate::io::load_structure(path)?;
    let ref_pdb = match reference_path {
        Some(p) => Some(crate::io::open_structure(p)?),
        None => None,
    };
    let detailed = analyze_pdb_detailed_with_header(
        &loaded.pdb,
        ref_pdb.as_ref(),
        Some(&loaded.header_preview),
    )?;
    Ok(detailed.metrics)
}

/// Detailed biophysical analysis on an open PDB structure with per-residue profiles.
pub fn analyze_pdb_detailed(
    pdb: &pdbtbx::PDB,
    reference_pdb: Option<&pdbtbx::PDB>,
) -> Result<DetailedBiophysicalAnalysis, CoreError> {
    analyze_pdb_detailed_with_header(pdb, reference_pdb, None)
}

/// As [`analyze_pdb_detailed`], with raw header text (EXPDTA/TITLE/mmCIF categories that
/// pdbtbx does not retain) to improve pLDDT-vs-B-factor provenance detection.
pub fn analyze_pdb_detailed_with_header(
    pdb: &pdbtbx::PDB,
    reference_pdb: Option<&pdbtbx::PDB>,
    extra_header: Option<&str>,
) -> Result<DetailedBiophysicalAnalysis, CoreError> {
    // All metrics are defined on protein heavy atoms: drop solvent, ions, ligands, hydrogens.
    let protein = crate::io::protein_heavy_atoms(pdb);
    let pdb = &protein;
    let mut ca_coords: Vec<Vector3<f64>> = Vec::new();
    let mut plddts: Vec<f64> = Vec::new();
    let mut all_atoms: Vec<crate::sasa::AtomDescriptor> = Vec::new();

    let backbones = crate::backbone::extract_backbone(pdb);
    for r in &backbones {
        if let Some(ca) = r.ca {
            ca_coords.push(ca);
            plddts.push(r.b_factor);
        }
    }
    for atom in pdb.atoms() {
        let coord = Vector3::new(atom.x(), atom.y(), atom.z());
        all_atoms.push(crate::sasa::AtomDescriptor::new(
            coord,
            crate::io::element_symbol(atom),
        ));
    }

    if ca_coords.is_empty() {
        return Err(CoreError::StructureParseError(
            "No C-alpha atoms found in PDB file".into(),
        ));
    }

    let rg = compute_radius_of_gyration(&ca_coords)?;
    let contact_density = compute_contact_density(&ca_coords, 8.0);

    // RMSD to reference if provided. The reference is normalised exactly like the query
    // (protein residues, heavy atoms, first altloc, one CA per residue) so that alternate
    // conformations or a calcium ion named `CA` cannot change the atom count.
    let rmsd = if let Some(ref_pdb) = reference_pdb {
        let ref_protein = crate::io::protein_heavy_atoms(ref_pdb);
        let ref_ca: Vec<Vector3<f64>> = crate::backbone::extract_backbone(&ref_protein)
            .iter()
            .filter_map(|r| r.ca)
            .collect();
        Some(compute_kabsch_rmsd(&ca_coords, &ref_ca)?)
    } else {
        None
    };

    let header = format!(
        "{}\n{}",
        crate::confidence::pdb_header_text(pdb),
        extra_header.unwrap_or("")
    );
    let confidence_source = crate::confidence::detect_confidence_source(&header, &plddts);

    // Normalize pLDDT if model wrote it in [0.0, 1.0] range (e.g. ESMFold)
    let max_plddt = plddts.iter().copied().fold(f64::MIN, f64::max);
    if confidence_source.is_predicted() && max_plddt <= 1.0 && max_plddt > 0.0 {
        for v in &mut plddts {
            *v *= 100.0;
        }
    }

    // pLDDT statistics
    let n_plddt = plddts.len() as f64;
    let mean_plddt = plddts.iter().sum::<f64>() / n_plddt;
    let mut sorted_plddt = plddts.clone();
    sorted_plddt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_plddt = sorted_plddt[sorted_plddt.len() / 2];

    let high_conf = plddts.iter().filter(|&&v| v >= 70.0).count() as f64 / n_plddt;
    let very_high_conf = plddts.iter().filter(|&&v| v >= 90.0).count() as f64 / n_plddt;

    // Ramachandran backbone dihedral angles with MolProbity residue-specific context
    let mut rama_points: Vec<(Option<f64>, Option<f64>, crate::rama8000::RamaClass)> = Vec::new();
    let mut ramachandran_points = Vec::new();
    let n_res = backbones.len();
    for i in 0..n_res {
        let phi = if i > 0 {
            crate::backbone::phi(&backbones[i - 1], &backbones[i])
        } else {
            None
        };
        let psi = if i + 1 < n_res {
            crate::backbone::psi(&backbones[i], &backbones[i + 1])
        } else {
            None
        };
        let omega = if i > 0 {
            crate::backbone::omega(&backbones[i - 1], &backbones[i])
        } else {
            None
        };
        let next_name = if i + 1 < n_res && !backbones[i + 1].chain_break_before {
            Some(backbones[i + 1].name.as_str())
        } else {
            None
        };
        let class = crate::rama8000::RamaClass::classify(&backbones[i].name, next_name, omega);
        let region = match (phi, psi) {
            (Some(p), Some(s)) => crate::rama8000::evaluate(class, p, s),
            _ => crate::structure::RamachandranRegion::Outlier,
        };
        ramachandran_points.push((phi, psi, region));
        rama_points.push((phi, psi, class));
    }

    let ss_summary = crate::structure::assign_secondary_structure(&backbones);
    let rama_stats = crate::structure::evaluate_ramachandran(&rama_points);
    let sasa_metrics = crate::sasa::compute_sasa(&all_atoms);
    let steric_overlap = crate::clash::compute_steric_overlap(pdb);
    let interaction_network = crate::interactions::compute_interaction_network(pdb);

    let mut metrics = BiophysicalMetrics {
        id: Uuid::new_v4(),
        prediction_id: Uuid::new_v4(),
        radius_of_gyration: rg,
        rmsd_to_reference: rmsd,
        contact_density,
        plddt_distribution: PlddtDistribution {
            mean: mean_plddt,
            median: median_plddt,
            high_confidence_fraction: high_conf,
            very_high_confidence_fraction: very_high_conf,
        },
        confidence_source,
        secondary_structure_summary: Some(ss_summary),
        ramachandran_stats: Some(rama_stats),
        steric_overlap: Some(steric_overlap),
        sasa_metrics: Some(sasa_metrics),
        interaction_network: Some(interaction_network),
        candidate_fitness_score: None,
    };

    let fitness = crate::ranking::evaluate_candidate_fitness(&metrics, ca_coords.len());
    metrics.candidate_fitness_score = Some(fitness.total_score);

    Ok(DetailedBiophysicalAnalysis {
        metrics,
        plddts,
        ramachandran_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_center_of_mass() {
        let pts = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let com = compute_center_of_mass(&pts).unwrap();
        assert!((com.x - 1.0 / 3.0).abs() < 1e-9);
        assert!((com.y - 1.0 / 3.0).abs() < 1e-9);
        assert!((com.z - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_radius_of_gyration() {
        // 4 points on unit circle in XY plane: (1,0), (-1,0), (0,1), (0,-1)
        // COM is (0,0,0). Each distance squared is 1.0. Mean is 1.0. Sqrt is 1.0.
        let pts = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ];
        let rg = compute_radius_of_gyration(&pts).unwrap();
        assert!((rg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_kabsch_rmsd_identical_and_transformed() {
        let p = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 1.0, -1.0),
            Vector3::new(-2.0, 3.0, 5.0),
        ];

        // Identical set
        let rmsd_self = compute_kabsch_rmsd(&p, &p).unwrap();
        assert!(rmsd_self < 1e-9, "Self RMSD must be ~0, got {}", rmsd_self);

        // Rotated by 90 deg around Z, plus translation (10, -5, 2)
        let q: Vec<Vector3<f64>> = p
            .iter()
            .map(|pt| {
                // (x, y, z) -> (-y + 10, x - 5, z + 2)
                Vector3::new(-pt.y + 10.0, pt.x - 5.0, pt.z + 2.0)
            })
            .collect();

        let rmsd_rot = compute_kabsch_rmsd(&p, &q).unwrap();
        assert!(
            rmsd_rot < 1e-7,
            "Rotated/translated RMSD must be ~0 after Kabsch alignment, got {}",
            rmsd_rot
        );

        let sup = compute_kabsch_superposition(&p, &q).unwrap();
        assert!(sup.rmsd < 1e-7);
        for (aligned, target) in sup.aligned_coords.iter().zip(q.iter()) {
            assert!((aligned - target).norm() < 1e-6);
        }
    }

    #[test]
    fn test_contact_density() {
        // 6 points in a line: dist between i and i+4 is 4.0 (<= 8.0)
        let pts: Vec<Vector3<f64>> = (0..6).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect();
        let density = compute_contact_density(&pts, 8.0);
        assert!(density > 0.0);
    }

    #[test]
    fn reference_with_altlocs_and_ions_is_normalised_like_the_query() {
        let base =
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 10.00           O\n\
ATOM      5  N   GLY A   2       3.300   1.500   0.000  1.00 10.00           N\n\
ATOM      6  CA  GLY A   2       4.000   2.700   0.000  1.00 10.00           C\n\
ATOM      7  C   GLY A   2       5.500   2.600   0.000  1.00 10.00           C\n\
ATOM      8  O   GLY A   2       6.100   1.500   0.000  1.00 10.00           O\n\
ATOM      9  N   SER A   3       6.100   3.800   0.000  1.00 10.00           N\n\
ATOM     10  CA  SER A   3       7.500   4.000   0.000  1.00 10.00           C\n\
ATOM     11  C   SER A   3       8.000   5.400   0.000  1.00 10.00           C\n\
ATOM     12  O   SER A   3       7.200   6.300   0.000  1.00 10.00           O\n";
        // Same protein, but the reference carries an alternate conformation for one CA and a
        // calcium ion (residue `CA`, atom `CA`): neither may change the CA count.
        let reference = base.replace(
            "ATOM     10  CA  SER A   3       7.500   4.000   0.000  1.00 10.00           C\n",
            "ATOM     10  CA ASER A   3       7.500   4.000   0.000  0.50 10.00           C\n\
ATOM     11  CA BSER A   3       7.600   4.100   0.000  0.50 10.00           C\n",
        ) + "HETATM   13 CA    CA A 101      20.000  20.000  20.000  1.00 10.00          CA\nEND\n";
        let dir = tempfile::tempdir().unwrap();
        let q = dir.path().join("q.pdb");
        let r = dir.path().join("r.pdb");
        std::fs::write(&q, format!("{base}END\n")).unwrap();
        std::fs::write(&r, reference).unwrap();
        let m = analyze_pdb_file(&q, Some(&r)).unwrap();
        let rmsd = m.rmsd_to_reference.expect("RMSD against a valid reference");
        assert!(rmsd < 1e-6, "identical backbone, got RMSD {rmsd}");
    }

    #[test]
    fn test_analyze_synthetic_pdb() {
        let mut tmp = tempfile::Builder::new().suffix(".pdb").tempfile().unwrap();
        let pdb_content = r#"HEADER    TEST PDB
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  ALA A   2       1.500   2.000   0.500  1.00 92.00           C
ATOM      3  CA  ALA A   3       3.000   0.500   1.500  1.00 78.00           C
ATOM      4  CA  ALA A   4       4.500   2.500   2.000  1.00 95.00           C
ATOM      5  CA  ALA A   5       6.000   1.000   3.500  1.00 88.00           C
END
"#;
        tmp.write_all(pdb_content.as_bytes()).unwrap();

        let metrics = analyze_pdb_file(tmp.path(), None).unwrap();
        assert!(metrics.radius_of_gyration > 0.0);
        assert!(metrics.plddt_distribution.mean > 80.0);
        assert_eq!(metrics.plddt_distribution.high_confidence_fraction, 1.0); // all >= 70
        assert_eq!(
            metrics.plddt_distribution.very_high_confidence_fraction,
            0.4
        ); // 2 out of 5 >= 90
    }
}
