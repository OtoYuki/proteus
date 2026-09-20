use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SasaMetrics {
    pub total_sasa: f64,
    pub polar_sasa: f64,
    pub apolar_sasa: f64,
    pub hydrophobic_burial_ratio: f64,
}

pub struct AtomDescriptor {
    pub coord: Vector3<f64>,
    pub element: String,
}

impl AtomDescriptor {
    pub fn new(coord: Vector3<f64>, element: impl Into<String>) -> Self {
        Self {
            coord,
            element: element.into(),
        }
    }

    pub fn vdw_radius(&self) -> f64 {
        match self.element.to_ascii_uppercase().as_str() {
            "C" => 1.70,
            "N" => 1.55,
            "O" => 1.52,
            "S" => 1.80,
            "P" => 1.80,
            "H" => 1.20,
            _ => 1.60, // Standard default for organic bio-atoms
        }
    }

    pub fn is_polar(&self) -> bool {
        matches!(self.element.to_ascii_uppercase().as_str(), "N" | "O" | "P")
    }
}

/// Shrake-Rupley Solvent Accessible Surface Area (SASA) numerical calculation.
/// Uses a 96-point Fibonacci sphere and 1.4A water probe radius.
pub fn compute_sasa(atoms: &[AtomDescriptor]) -> SasaMetrics {
    if atoms.is_empty() {
        return SasaMetrics {
            total_sasa: 0.0,
            polar_sasa: 0.0,
            apolar_sasa: 0.0,
            hydrophobic_burial_ratio: 0.0,
        };
    }

    let probe_radius = 1.40;
    let n_points = 96usize;
    let sphere_points = generate_fibonacci_sphere(n_points);

    let mut total_sasa = 0.0;
    let mut polar_sasa = 0.0;
    let mut apolar_sasa = 0.0;
    let mut total_apolar_isolated = 0.0;

    let n = atoms.len();
    let expanded_radii: Vec<f64> = atoms
        .iter()
        .map(|a| a.vdw_radius() + probe_radius)
        .collect();

    for i in 0..n {
        let r_i = expanded_radii[i];
        let c_i = atoms[i].coord;
        let is_polar = atoms[i].is_polar();

        // Theoretical isolated surface area
        let isolated_area = 4.0 * std::f64::consts::PI * r_i * r_i;
        if !is_polar {
            total_apolar_isolated += isolated_area;
        }

        // Find candidate neighbor atoms within (r_i + max_neighbor_radius)
        let mut neighbors = Vec::new();
        for j in 0..n {
            if i != j {
                let d_sq = (c_i - atoms[j].coord).norm_squared();
                let max_d = r_i + expanded_radii[j];
                if d_sq < max_d * max_d {
                    neighbors.push((atoms[j].coord, expanded_radii[j]));
                }
            }
        }

        let mut accessible_points = 0usize;
        for pt in &sphere_points {
            let test_point = c_i + pt * r_i;
            let mut occluded = false;

            for (n_coord, n_r) in &neighbors {
                if (test_point - n_coord).norm_squared() < n_r * n_r {
                    occluded = true;
                    break;
                }
            }

            if !occluded {
                accessible_points += 1;
            }
        }

        let atom_sasa = isolated_area * (accessible_points as f64) / (n_points as f64);
        total_sasa += atom_sasa;
        if is_polar {
            polar_sasa += atom_sasa;
        } else {
            apolar_sasa += atom_sasa;
        }
    }

    let hydrophobic_burial_ratio = if total_apolar_isolated > 0.0 {
        ((total_apolar_isolated - apolar_sasa) / total_apolar_isolated).clamp(0.0, 1.0)
    } else {
        0.0
    };

    SasaMetrics {
        total_sasa,
        polar_sasa,
        apolar_sasa,
        hydrophobic_burial_ratio,
    }
}

/// Generate equidistant points on a unit sphere using the Fibonacci spiral.
fn generate_fibonacci_sphere(samples: usize) -> Vec<Vector3<f64>> {
    let mut points = Vec::with_capacity(samples);
    let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // Golden angle in radians

    for i in 0..samples {
        let y = 1.0 - (i as f64 / (samples - 1) as f64) * 2.0;
        let radius = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi * (i as f64);

        let x = theta.cos() * radius;
        let z = theta.sin() * radius;
        points.push(Vector3::new(x, y, z));
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_carbon_atom_sasa() {
        let atoms = vec![AtomDescriptor::new(Vector3::new(0.0, 0.0, 0.0), "C")];
        let sasa = compute_sasa(&atoms);

        // Theoretical area = 4 * pi * (1.7 + 1.4)^2 = 4 * pi * 3.1^2 = ~120.76 A^2
        assert!((sasa.total_sasa - 120.76).abs() < 5.0);
        assert_eq!(sasa.polar_sasa, 0.0);
        assert!(sasa.apolar_sasa > 100.0);
        assert_eq!(sasa.hydrophobic_burial_ratio, 0.0); // Completely exposed
    }

    #[test]
    fn test_two_touching_atoms_occlusion() {
        let atoms = vec![
            AtomDescriptor::new(Vector3::new(0.0, 0.0, 0.0), "C"),
            AtomDescriptor::new(Vector3::new(2.5, 0.0, 0.0), "C"),
        ];
        let sasa = compute_sasa(&atoms);

        // Two touching atoms should have less total SASA than 2 * single isolated atom
        assert!(sasa.total_sasa < 240.0);
        assert!(sasa.hydrophobic_burial_ratio > 0.0);
    }
}
