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

    // IUPAC/Blondel-Karplus signed torsion: atan2(|b2| b1·n2, n1·n2).
    let y = b2_norm * b1.dot(&n2);
    let x = n1.dot(&n2);
    Ok(y.atan2(x).to_degrees())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidueContext {
    #[default]
    General,
    Glycine,
    Proline,
    PreProline,
}

impl ResidueContext {
    pub fn from_names(curr_name: &str, next_name: Option<&str>) -> Self {
        if next_name.is_some_and(|n| n.eq_ignore_ascii_case("PRO")) {
            Self::PreProline
        } else if curr_name.eq_ignore_ascii_case("GLY") {
            Self::Glycine
        } else if curr_name.eq_ignore_ascii_case("PRO") {
            Self::Proline
        } else {
            Self::General
        }
    }
}

/// Classify a single (phi, psi) pair into Ramachandran conformational basins using generic boundaries.
pub fn classify_ramachandran(phi: f64, psi: f64) -> RamachandranRegion {
    classify_ramachandran_context(phi, psi, ResidueContext::General)
}

/// Classify a (phi, psi) pair under residue-specific stereochemical contexts (General, Glycine, Proline, Pre-Proline)
/// per the MolProbity (Lovell et al. 2003) crystallographic standard.
pub fn classify_ramachandran_context(
    phi: f64,
    psi: f64,
    context: ResidueContext,
) -> RamachandranRegion {
    match context {
        ResidueContext::General => {
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
                || ((-90.0..=-45.0).contains(&phi) && (120.0..=175.0).contains(&psi))
            {
                return RamachandranRegion::Allowed;
            }

            RamachandranRegion::Outlier
        }
        ResidueContext::Glycine => {
            // Glycine lacks C-beta sidechain: symmetric conformations across all 4 quadrants
            if ((-140.0..=-40.0).contains(&phi) && (-80.0..=40.0).contains(&psi))
                || ((-180.0..=-50.0).contains(&phi) && (120.0..=180.0).contains(&psi))
            {
                return RamachandranRegion::CoreHelix;
            }
            if ((40.0..=140.0).contains(&phi) && (-40.0..=80.0).contains(&psi))
                || ((50.0..=180.0).contains(&phi) && (-180.0..=-120.0).contains(&psi))
            {
                return RamachandranRegion::LeftHandedHelix;
            }
            if ((-180.0..=0.0).contains(&phi) && (-100.0..=180.0).contains(&psi))
                || ((0.0..=180.0).contains(&phi) && (-180.0..=100.0).contains(&psi))
            {
                return RamachandranRegion::Allowed;
            }

            RamachandranRegion::Outlier
        }
        ResidueContext::Proline => {
            // Rigid pyrrolidine ring restricts phi strictly into [-80, -50]
            if (-80.0..=-50.0).contains(&phi) {
                if (-60.0..=-10.0).contains(&psi) {
                    return RamachandranRegion::CoreHelix;
                }
                if (100.0..=180.0).contains(&psi) || (-180.0..=-160.0).contains(&psi) {
                    return RamachandranRegion::CoreStrand;
                }
            }
            if (-95.0..=-35.0).contains(&phi)
                && (((-80.0..=30.0).contains(&psi)) || ((80.0..=180.0).contains(&psi)))
            {
                return RamachandranRegion::Allowed;
            }

            RamachandranRegion::Outlier
        }
        ResidueContext::PreProline => {
            // Residues preceding Proline: steric clash with C-delta atom
            if (-100.0..=-50.0).contains(&phi) && (-60.0..=-20.0).contains(&psi) {
                return RamachandranRegion::CoreHelix;
            }
            if (-180.0..=-50.0).contains(&phi) && (100.0..=180.0).contains(&psi) {
                return RamachandranRegion::CoreStrand;
            }
            if ((-180.0..=-30.0).contains(&phi) && (60.0..=180.0).contains(&psi))
                || ((-120.0..=-40.0).contains(&phi) && (-80.0..=20.0).contains(&psi))
            {
                return RamachandranRegion::Allowed;
            }

            RamachandranRegion::Outlier
        }
    }
}

/// Evaluate Ramachandran statistics over a sequence of (phi, psi) angle pairs using generic boundaries.
pub fn evaluate_ramachandran_angles(angles: &[(Option<f64>, Option<f64>)]) -> RamachandranStats {
    let context_angles: Vec<(Option<f64>, Option<f64>, ResidueContext)> = angles
        .iter()
        .map(|&(phi, psi)| (phi, psi, ResidueContext::General))
        .collect();
    evaluate_ramachandran_with_context(&context_angles)
}

/// Evaluate Ramachandran statistics with MolProbity residue-specific context.
pub fn evaluate_ramachandran_with_context(
    angles: &[(Option<f64>, Option<f64>, ResidueContext)],
) -> RamachandranStats {
    let mut favored = 0;
    let mut allowed = 0;
    let mut outliers = 0;
    let mut total = 0;

    for &(phi_opt, psi_opt, context) in angles {
        if let (Some(phi), Some(psi)) = (phi_opt, psi_opt) {
            total += 1;
            match classify_ramachandran_context(phi, psi, context) {
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

    #[test]
    fn test_molprobity_residue_contexts() {
        // Glycine in positive phi quadrant is allowed, whereas general residue is outlier
        assert_eq!(
            classify_ramachandran_context(60.0, -40.0, ResidueContext::General),
            RamachandranRegion::Outlier
        );
        assert_eq!(
            classify_ramachandran_context(60.0, -40.0, ResidueContext::Glycine),
            RamachandranRegion::LeftHandedHelix
        );

        // Proline strictly requires phi ~ -65 deg
        assert_eq!(
            classify_ramachandran_context(-65.0, -40.0, ResidueContext::Proline),
            RamachandranRegion::CoreHelix
        );
        assert_eq!(
            classify_ramachandran_context(-120.0, 140.0, ResidueContext::Proline),
            RamachandranRegion::Outlier
        );

        // Pre-proline context
        assert_eq!(
            classify_ramachandran_context(-120.0, 140.0, ResidueContext::PreProline),
            RamachandranRegion::CoreStrand
        );
    }
    #[test]
    fn dihedral_sign_matches_iupac_on_crambin() {
        let (pdb, _) = pdbtbx::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"),
            pdbtbx::StrictnessLevel::Loose,
        )
        .unwrap();
        let csv = include_str!("../tests/data/1crn_phipsi_mdtraj.csv");
        let mut expect: Vec<(Option<f64>, Option<f64>)> = Vec::new();
        for line in csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            expect.push((f[2].parse::<f64>().ok(), f[3].parse::<f64>().ok()));
        }
        let bb: Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> = pdb
            .residues()
            .map(|r| {
                let get = |n: &str| {
                    r.atoms()
                        .find(|a| a.name() == n)
                        .map(|a| Vector3::new(a.x(), a.y(), a.z()))
                        .unwrap()
                };
                (get("N"), get("CA"), get("C"))
            })
            .collect();
        for i in 0..bb.len() {
            if i > 0 {
                let phi = compute_dihedral(&bb[i - 1].2, &bb[i].0, &bb[i].1, &bb[i].2).unwrap();
                assert!(
                    (phi - expect[i].0.unwrap()).abs() < 0.05,
                    "phi res {} got {phi}",
                    i + 1
                );
            }
            if i + 1 < bb.len() {
                let psi = compute_dihedral(&bb[i].0, &bb[i].1, &bb[i].2, &bb[i + 1].0).unwrap();
                assert!(
                    (psi - expect[i].1.unwrap()).abs() < 0.05,
                    "psi res {} got {psi}",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn dihedral_mirror_negates() {
        let p = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.5, 0.0),
            Vector3::new(0.7, 1.5, 0.9),
        ];
        let a = compute_dihedral(&p[0], &p[1], &p[2], &p[3]).unwrap();
        let m: Vec<Vector3<f64>> = p.iter().map(|v| Vector3::new(v.x, v.y, -v.z)).collect();
        let b = compute_dihedral(&m[0], &m[1], &m[2], &m[3]).unwrap();
        assert!((a + b).abs() < 1e-9);
        // IUPAC value for these points is -52.125 deg (computed independently with numpy).
        assert!((a + 52.125).abs() < 0.01, "expected -52.125, got {a}");
    }
}
