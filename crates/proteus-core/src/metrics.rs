use crate::error::CoreError;
use crate::models::{BiophysicalMetrics, PlddtDistribution};
use nalgebra::{Matrix3, Vector3, SVD};
use pdbtbx::{open, Atom, StrictnessLevel};
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

/// Compute Kabsch optimal superposition and C-alpha RMSD between two aligned sets of coordinates.
pub fn compute_kabsch_rmsd(
    p_coords: &[Vector3<f64>],
    q_coords: &[Vector3<f64>],
) -> Result<f64, CoreError> {
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

    // Compute RMSD with rotated P
    let mut sum_sq_diff = 0.0;
    for (p, q) in p_centered.iter().zip(q_centered.iter()) {
        let p_rotated = rotation * p;
        sum_sq_diff += (p_rotated - q).norm_squared();
    }

    Ok((sum_sq_diff / (n as f64)).sqrt())
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

/// Extract C-alpha coordinates and b-factors (pLDDT) from a PDB file.
pub fn analyze_pdb_file(
    path: &Path,
    reference_path: Option<&Path>,
) -> Result<BiophysicalMetrics, CoreError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| CoreError::StructureParseError("Invalid UTF-8 in PDB path".into()))?;
    let (pdb, _errors) = open(path_str, StrictnessLevel::Loose)
        .map_err(|e| CoreError::StructureParseError(format!("Failed to open PDB file: {e:?}")))?;

    let mut ca_coords: Vec<Vector3<f64>> = Vec::new();
    let mut plddts: Vec<f64> = Vec::new();

    for residue in pdb.residues() {
        for atom in residue.atoms() {
            if atom.name() == "CA" {
                ca_coords.push(Vector3::new(atom.x(), atom.y(), atom.z()));
                plddts.push(atom.b_factor());
            }
        }
    }

    if ca_coords.is_empty() {
        return Err(CoreError::StructureParseError(
            "No C-alpha atoms found in PDB file".into(),
        ));
    }

    let rg = compute_radius_of_gyration(&ca_coords)?;
    let contact_density = compute_contact_density(&ca_coords, 8.0);

    // RMSD to reference if provided
    let rmsd = if let Some(ref_p) = reference_path {
        let ref_str = ref_p.to_str().ok_or_else(|| {
            CoreError::StructureParseError("Invalid UTF-8 in reference path".into())
        })?;
        let (ref_pdb, _) = open(ref_str, StrictnessLevel::Loose).map_err(|e| {
            CoreError::StructureParseError(format!("Failed to open reference PDB file: {e:?}"))
        })?;
        let ref_ca: Vec<Vector3<f64>> = ref_pdb
            .residues()
            .flat_map(|r| r.atoms())
            .filter(|a: &&Atom| a.name() == "CA")
            .map(|a| Vector3::new(a.x(), a.y(), a.z()))
            .collect();
        Some(compute_kabsch_rmsd(&ca_coords, &ref_ca)?)
    } else {
        None
    };

    // pLDDT statistics
    let n_plddt = plddts.len() as f64;
    let mean_plddt = plddts.iter().sum::<f64>() / n_plddt;
    let mut sorted_plddt = plddts.clone();
    sorted_plddt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_plddt = sorted_plddt[sorted_plddt.len() / 2];

    let high_conf = plddts.iter().filter(|&&v| v >= 70.0).count() as f64 / n_plddt;
    let very_high_conf = plddts.iter().filter(|&&v| v >= 90.0).count() as f64 / n_plddt;

    Ok(BiophysicalMetrics {
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
    }

    #[test]
    fn test_contact_density() {
        // 6 points in a line: dist between i and i+4 is 4.0 (<= 8.0)
        let pts: Vec<Vector3<f64>> = (0..6).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect();
        let density = compute_contact_density(&pts, 8.0);
        assert!(density > 0.0);
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
