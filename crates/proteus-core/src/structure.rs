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
    /// Three-state assignment, one entry per residue with a C-alpha.
    pub assignment: Vec<SecondaryStructure>,
    /// Eight-state DSSP string (`H B E G I T S -`), same length as `assignment`.
    #[serde(default)]
    pub dssp: String,
}

/// MolProbity Ramachandran classification of one residue (Top8000 contours, see `rama8000`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum RamachandranRegion {
    Favored,
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

/// Assign secondary structure with Kabsch–Sander DSSP (via `proteus-dssp`).
///
/// Residues missing any of N, CA, C, O are skipped in the DSSP run; their slot in the
/// returned `assignment` is `Coil` so that indices line up with the backbone list.
pub fn assign_secondary_structure(
    backbone: &[crate::backbone::BackboneResidue],
) -> SecondaryStructureSummary {
    let mut index_map: Vec<Option<usize>> = Vec::with_capacity(backbone.len());
    let mut residues: Vec<proteus_dssp::Residue> = Vec::with_capacity(backbone.len());
    let mut pending_break = false;
    for r in backbone {
        match (r.n, r.ca, r.c, r.o) {
            (Some(n), Some(ca), Some(c), Some(o)) => {
                index_map.push(Some(residues.len()));
                residues.push(proteus_dssp::Residue {
                    n: [n.x, n.y, n.z],
                    ca: [ca.x, ca.y, ca.z],
                    c: [c.x, c.y, c.z],
                    o: [o.x, o.y, o.z],
                    is_proline: r.is_proline(),
                    chain_break_before: r.chain_break_before || pending_break,
                });
                pending_break = false;
            }
            _ => {
                // A residue with missing atoms breaks the peptide chain for DSSP purposes.
                index_map.push(None);
                pending_break = true;
            }
        }
    }
    let dssp8 = proteus_dssp::assign(&residues);
    let mut assignment = Vec::with_capacity(backbone.len());
    let mut dssp = String::with_capacity(backbone.len());
    for slot in &index_map {
        match slot {
            Some(k) => {
                let s = dssp8[*k];
                dssp.push(s.as_char());
                assignment.push(match s.simplify() {
                    proteus_dssp::Simple::Helix => SecondaryStructure::Helix,
                    proteus_dssp::Simple::Strand => SecondaryStructure::Strand,
                    proteus_dssp::Simple::Coil => SecondaryStructure::Coil,
                });
            }
            None => {
                dssp.push('-');
                assignment.push(SecondaryStructure::Coil);
            }
        }
    }
    let n = assignment.len().max(1) as f64;
    let count = |k: SecondaryStructure| assignment.iter().filter(|s| **s == k).count() as f64 / n;
    SecondaryStructureSummary {
        helix_fraction: count(SecondaryStructure::Helix),
        strand_fraction: count(SecondaryStructure::Strand),
        coil_fraction: count(SecondaryStructure::Coil),
        assignment,
        dssp,
    }
}

/// Aggregate MolProbity-style Ramachandran statistics over `(phi, psi, class)` points.
/// Residues without both angles are not evaluated.
pub fn evaluate_ramachandran(
    points: &[(Option<f64>, Option<f64>, crate::rama8000::RamaClass)],
) -> RamachandranStats {
    let (mut fav, mut allow, mut out, mut total) = (0usize, 0usize, 0usize, 0usize);
    for (phi, psi, class) in points {
        let (Some(p), Some(s)) = (phi, psi) else {
            continue;
        };
        total += 1;
        match crate::rama8000::evaluate(*class, *p, *s) {
            RamachandranRegion::Favored => fav += 1,
            RamachandranRegion::Allowed => allow += 1,
            RamachandranRegion::Outlier => out += 1,
        }
    }
    let n = total.max(1) as f64;
    RamachandranStats {
        favored_fraction: fav as f64 / n,
        allowed_fraction: allow as f64 / n,
        outlier_fraction: out as f64 / n,
        outlier_count: out,
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
    fn crambin_three_state_fractions_match_dssp() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
            .unwrap();
        let bb = crate::backbone::extract_backbone(&pdb);
        let s = assign_secondary_structure(&bb);
        assert_eq!(s.dssp, "-EE-SSHHHHHHHHHHHTTT--HHHHHHHHS-EE-SSS---GGG--");
        assert!(
            (s.helix_fraction - 0.478).abs() < 0.05,
            "{}",
            s.helix_fraction
        );
        assert!(
            (s.strand_fraction - 0.087).abs() < 0.05,
            "{}",
            s.strand_fraction
        );
        assert!(
            (s.coil_fraction - 0.435).abs() < 0.05,
            "{}",
            s.coil_fraction
        );
    }

    #[test]
    fn dihedral_sign_matches_iupac_on_crambin() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
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
