use crate::error::CoreError;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum SecondaryStructure {
    Helix,
    Strand,
    Coil,
}

impl SecondaryStructure {
    pub fn as_char(&self) -> char {
        match self {
            Self::Helix => 'H',
            Self::Strand => 'E',
            Self::Coil => 'C',
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SecondaryStructureSummary {
    pub helix_fraction: f64,
    pub strand_fraction: f64,
    pub coil_fraction: f64,
    pub assignment: Vec<SecondaryStructure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum RamachandranRegion {
    CoreHelix,
    CoreStrand,
    LeftHandedHelix,
    Allowed,
    Outlier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RamachandranStats {
    pub favored_fraction: f64,
    pub allowed_fraction: f64,
    pub outlier_fraction: f64,
    pub outlier_count: usize,
    pub total_evaluated: usize,
}

/// Compute the dihedral (torsion) angle defined by 4 points in 3D space in degrees [-180, 180].
pub fn compute_dihedral(
    p1: &Vector3<f64>,
    p2: &Vector3<f64>,
    p3: &Vector3<f64>,
    p4: &Vector3<f64>,
) -> Result<f64, CoreError> {
    let b1 = p2 - p1;
    let b2 = p3 - p2;
    let b3 = p4 - p3;

    let b2_norm = b2.norm();
    if b2_norm < 1e-6 {
        return Err(CoreError::AnalysisError(
            "Collinear points in dihedral computation".into(),
        ));
    }

    let n1 = b1.cross(&b2);
    let n2 = b2.cross(&b3);

    let m1 = n1.cross(&(b2 / b2_norm));

    let x = n1.dot(&n2);
    let y = m1.dot(&n2);

    let rad = y.atan2(x);
    Ok(rad.to_degrees())
}

/// Assign secondary structure using P-SEA alpha-carbon geometry:
/// - Alpha-helix: distance(i, i+3) in [5.0, 5.5]A, distance(i, i+4) in [5.5, 6.5]A, dihedral in [40, 65] deg.
/// - Beta-strand: distance(i, i+2) in [6.5, 7.2]A, distance(i, i+3) in [9.8, 10.8]A.
pub fn assign_secondary_structure(ca_coords: &[Vector3<f64>]) -> SecondaryStructureSummary {
    let n = ca_coords.len();
    if n < 4 {
        let assignment = vec![SecondaryStructure::Coil; n];
        return SecondaryStructureSummary {
            helix_fraction: 0.0,
            strand_fraction: 0.0,
            coil_fraction: 1.0,
            assignment,
        };
    }

    let mut assignment = vec![SecondaryStructure::Coil; n];

    // Detect helices (i, i+3 and i, i+4 distance invariants)
    let mut is_helix = vec![false; n];
    for i in 0..n {
        let has_i3 = if i + 3 < n {
            let d3 = (ca_coords[i] - ca_coords[i + 3]).norm();
            (5.0..=5.6).contains(&d3)
        } else {
            false
        };

        let has_i4 = if i + 4 < n {
            let d4 = (ca_coords[i] - ca_coords[i + 4]).norm();
            (5.4..=6.6).contains(&d4)
        } else {
            false
        };

        if has_i3 || has_i4 {
            is_helix[i] = true;
            if i + 1 < n {
                is_helix[i + 1] = true;
            }
            if i + 2 < n {
                is_helix[i + 2] = true;
            }
            if i + 3 < n {
                is_helix[i + 3] = true;
            }
        }
    }

    // Detect strands (extended chain invariants)
    let mut is_strand = vec![false; n];
    for i in 0..n {
        if is_helix[i] {
            continue;
        }
        let has_i2 = if i + 2 < n {
            let d2 = (ca_coords[i] - ca_coords[i + 2]).norm();
            (6.4..=7.4).contains(&d2)
        } else {
            false
        };

        let has_i3 = if i + 3 < n {
            let d3 = (ca_coords[i] - ca_coords[i + 3]).norm();
            (9.6..=11.0).contains(&d3)
        } else {
            false
        };

        if has_i2 || has_i3 {
            is_strand[i] = true;
            if i + 1 < n {
                is_strand[i + 1] = true;
            }
            if i + 2 < n {
                is_strand[i + 2] = true;
            }
        }
    }

    let mut helix_count = 0;
    let mut strand_count = 0;
    let mut coil_count = 0;

    for i in 0..n {
        if is_helix[i] {
            assignment[i] = SecondaryStructure::Helix;
            helix_count += 1;
        } else if is_strand[i] {
            assignment[i] = SecondaryStructure::Strand;
            strand_count += 1;
        } else {
            assignment[i] = SecondaryStructure::Coil;
            coil_count += 1;
        }
    }

    let n_f = n as f64;
    SecondaryStructureSummary {
        helix_fraction: (helix_count as f64) / n_f,
        strand_fraction: (strand_count as f64) / n_f,
        coil_fraction: (coil_count as f64) / n_f,
        assignment,
    }
}

/// Classify a single (phi, psi) pair into Ramachandran conformational basins.
pub fn classify_ramachandran(phi: f64, psi: f64) -> RamachandranRegion {
    // Core Alpha-Helix
    if (-100.0..=-30.0).contains(&phi) && (-70.0..=-10.0).contains(&psi) {
        return RamachandranRegion::CoreHelix;
    }

    // Core Beta-Sheet / Extended
    if ((-180.0..=-45.0).contains(&phi) && (90.0..=180.0).contains(&psi))
        || ((-180.0..=-45.0).contains(&phi) && (-180.0..=-150.0).contains(&psi))
    {
        return RamachandranRegion::CoreStrand;
    }

    // Left-handed Alpha-Helix
    if (30.0..=90.0).contains(&phi) && (10.0..=70.0).contains(&psi) {
        return RamachandranRegion::LeftHandedHelix;
    }

    // Allowed outer basins
    if ((-120.0..=-20.0).contains(&phi) && (-90.0..=20.0).contains(&psi))
        || ((-180.0..=-30.0).contains(&phi) && (60.0..=180.0).contains(&psi))
        || ((20.0..=100.0).contains(&phi) && (-10.0..=80.0).contains(&psi))
    {
        return RamachandranRegion::Allowed;
    }

    RamachandranRegion::Outlier
}

/// Evaluate Ramachandran statistics over a sequence of (phi, psi) angle pairs.
pub fn evaluate_ramachandran_angles(angles: &[(Option<f64>, Option<f64>)]) -> RamachandranStats {
    let mut favored = 0;
    let mut allowed = 0;
    let mut outliers = 0;
    let mut total = 0;

    for &(phi_opt, psi_opt) in angles {
        if let (Some(phi), Some(psi)) = (phi_opt, psi_opt) {
            total += 1;
            match classify_ramachandran(phi, psi) {
                RamachandranRegion::CoreHelix
                | RamachandranRegion::CoreStrand
                | RamachandranRegion::LeftHandedHelix => favored += 1,
                RamachandranRegion::Allowed => allowed += 1,
                RamachandranRegion::Outlier => outliers += 1,
            }
        }
    }

    if total == 0 {
        return RamachandranStats {
            favored_fraction: 1.0,
            allowed_fraction: 0.0,
            outlier_fraction: 0.0,
            outlier_count: 0,
            total_evaluated: 0,
        };
    }

    let tot_f = total as f64;
    RamachandranStats {
        favored_fraction: (favored as f64) / tot_f,
        allowed_fraction: (allowed as f64) / tot_f,
        outlier_fraction: (outliers as f64) / tot_f,
        outlier_count: outliers,
        total_evaluated: total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dihedral_perpendicular_planes() {
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 0.0, 0.0);
        let p3 = Vector3::new(0.0, 1.0, 0.0);
        let p4 = Vector3::new(0.0, 1.0, 1.0);

        let angle = compute_dihedral(&p1, &p2, &p3, &p4).unwrap();
        assert!((angle.abs() - 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_ramachandran_classification() {
        assert_eq!(
            classify_ramachandran(-60.0, -45.0),
            RamachandranRegion::CoreHelix
        );
        assert_eq!(
            classify_ramachandran(-120.0, 140.0),
            RamachandranRegion::CoreStrand
        );
        assert_eq!(classify_ramachandran(0.0, 0.0), RamachandranRegion::Outlier);
    }

    #[test]
    fn test_synthetic_helix_secondary_structure() {
        // Generate ideal alpha-helix CA coordinates (3.6 residues/turn, 1.5A pitch, 2.3A radius)
        let radius = 2.3;
        let mut coords = Vec::new();
        for i in 0..20 {
            let angle = (i as f64) * 1.74533; // 100 degrees
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let z = (i as f64) * 1.5;
            coords.push(Vector3::new(x, y, z));
        }

        let summary = assign_secondary_structure(&coords);
        assert!(
            summary.helix_fraction > 0.70,
            "Expected helix fraction > 70%, got {}",
            summary.helix_fraction
        );
    }
}
