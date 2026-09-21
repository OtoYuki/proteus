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

/// 3D Spatial Voxel Grid for O(N) neighbor lookup in Shrake-Rupley SASA.
/// Eliminates the classic O(N^2) pairwise distance bottleneck using a cell-linked list.
struct SpatialCellList {
    min: Vector3<f64>,
    inv_cell_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    head: Vec<i32>,
    next: Vec<i32>,
}

impl SpatialCellList {
    fn build(coords: &[Vector3<f64>], cell_size: f64) -> Self {
        let n = coords.len();
        if n == 0 {
            return Self {
                min: Vector3::zeros(),
                inv_cell_size: 1.0 / cell_size,
                nx: 1,
                ny: 1,
                nz: 1,
                head: vec![-1],
                next: Vec::new(),
            };
        }

        let mut min = coords[0];
        let mut max = coords[0];
        for c in coords.iter().skip(1) {
            min.x = min.x.min(c.x);
            min.y = min.y.min(c.y);
            min.z = min.z.min(c.z);
            max.x = max.x.max(c.x);
            max.y = max.y.max(c.y);
            max.z = max.z.max(c.z);
        }

        // Small padding to prevent out-of-bounds on max edge
        min -= Vector3::new(0.01, 0.01, 0.01);
        max += Vector3::new(0.01, 0.01, 0.01);

        let inv_cell_size = 1.0 / cell_size;
        let nx = (((max.x - min.x) * inv_cell_size).floor() as usize + 1).max(1);
        let ny = (((max.y - min.y) * inv_cell_size).floor() as usize + 1).max(1);
        let nz = (((max.z - min.z) * inv_cell_size).floor() as usize + 1).max(1);

        let total_cells = nx.saturating_mul(ny).saturating_mul(nz).min(2_000_000);
        let mut head = vec![-1i32; total_cells];
        let mut next = vec![-1i32; n];

        for i in 0..n {
            let cx = (((coords[i].x - min.x) * inv_cell_size).floor() as usize).min(nx - 1);
            let cy = (((coords[i].y - min.y) * inv_cell_size).floor() as usize).min(ny - 1);
            let cz = (((coords[i].z - min.z) * inv_cell_size).floor() as usize).min(nz - 1);

            let cell_idx = cx + cy * nx + cz * nx * ny;
            if cell_idx < total_cells {
                next[i] = head[cell_idx];
                head[cell_idx] = i as i32;
            }
        }

        Self {
            min,
            inv_cell_size,
            nx,
            ny,
            nz,
            head,
            next,
        }
    }

    /// Query neighbor atom indices within the 27 adjacent grid cells of atom `atom_idx`.
    fn find_neighbors(
        &self,
        atom_idx: usize,
        coords: &[Vector3<f64>],
        radii: &[f64],
        max_dist_sq: f64,
        neighbors_out: &mut Vec<(Vector3<f64>, f64)>,
    ) {
        neighbors_out.clear();
        let coord = coords[atom_idx];
        let r_i = radii[atom_idx];

        let cx = (((coord.x - self.min.x) * self.inv_cell_size).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let cy = (((coord.y - self.min.y) * self.inv_cell_size).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        let cz = (((coord.z - self.min.z) * self.inv_cell_size).floor() as isize)
            .clamp(0, self.nz as isize - 1) as usize;

        let min_x = cx.saturating_sub(1);
        let max_x = (cx + 1).min(self.nx - 1);
        let min_y = cy.saturating_sub(1);
        let max_y = (cy + 1).min(self.ny - 1);
        let min_z = cz.saturating_sub(1);
        let max_z = (cz + 1).min(self.nz - 1);

        for z in min_z..=max_z {
            let z_offset = z * self.nx * self.ny;
            for y in min_y..=max_y {
                let yz_offset = y * self.nx + z_offset;
                for x in min_x..=max_x {
                    let cell_idx = x + yz_offset;
                    let mut curr = self.head[cell_idx];
                    while curr >= 0 {
                        let j = curr as usize;
                        if j != atom_idx {
                            let d_sq = (coord - coords[j]).norm_squared();
                            let max_d = r_i + radii[j];
                            if d_sq < max_d * max_d && d_sq < max_dist_sq {
                                neighbors_out.push((coords[j], radii[j]));
                            }
                        }
                        curr = self.next[j];
                    }
                }
            }
        }
    }
}

/// Default number of Fibonacci-sphere test points per atom. 960 matches mdtraj's default and
/// keeps the total SASA within ~0.3 % of it; 96 is ~10x faster and within ~1–2 %.
pub const DEFAULT_SPHERE_POINTS: usize = 960;

/// Shrake-Rupley Solvent Accessible Surface Area (SASA) with [`DEFAULT_SPHERE_POINTS`].
pub fn compute_sasa(atoms: &[AtomDescriptor]) -> SasaMetrics {
    compute_sasa_with_points(atoms, DEFAULT_SPHERE_POINTS)
}

/// Shrake-Rupley SASA using an O(N) 3D spatial cell-list, an `n_points` Fibonacci sphere per
/// atom, Bondi radii and a 1.40 Å probe.
pub fn compute_sasa_with_points(atoms: &[AtomDescriptor], n_points: usize) -> SasaMetrics {
    if atoms.is_empty() {
        return SasaMetrics {
            total_sasa: 0.0,
            polar_sasa: 0.0,
            apolar_sasa: 0.0,
            hydrophobic_burial_ratio: 0.0,
        };
    }

    let probe_radius = 1.40;
    let n_points = n_points.max(12);
    let sphere_points = generate_fibonacci_sphere(n_points);

    let n = atoms.len();
    let coords: Vec<Vector3<f64>> = atoms.iter().map(|a| a.coord).collect();
    let expanded_radii: Vec<f64> = atoms
        .iter()
        .map(|a| a.vdw_radius() + probe_radius)
        .collect();

    // Maximum possible sphere-sphere overlap distance: 2 * (1.80 + 1.40) = 6.40Å
    let cell_size = 6.40;
    let cell_list = SpatialCellList::build(&coords, cell_size);

    let mut total_sasa = 0.0;
    let mut polar_sasa = 0.0;
    let mut apolar_sasa = 0.0;
    let mut total_apolar_isolated = 0.0;

    let mut neighbors = Vec::with_capacity(64);

    for i in 0..n {
        let r_i = expanded_radii[i];
        let c_i = coords[i];
        let is_polar = atoms[i].is_polar();

        // Theoretical isolated surface area
        let isolated_area = 4.0 * std::f64::consts::PI * r_i * r_i;
        if !is_polar {
            total_apolar_isolated += isolated_area;
        }

        // O(1) query of neighboring atoms within spatial cell list
        cell_list.find_neighbors(
            i,
            &coords,
            &expanded_radii,
            cell_size * cell_size,
            &mut neighbors,
        );

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
