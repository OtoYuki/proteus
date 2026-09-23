use super::frame::compute_bishop_frames;
use super::spline::interpolate_catmull_rom;
use nalgebra::Vector3;
use proteus_core::structure::SecondaryStructure;

#[derive(Debug, Clone, Copy)]
pub struct Vertex3D {
    pub position: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub plddt: f32,
    pub secondary_structure: SecondaryStructure,
    pub residue_index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<[u32; 3]>,
}

impl TriangleMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Merge another triangle mesh into this mesh, offsetting indices.
    pub fn merge(&mut self, other: TriangleMesh) {
        let base_idx = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        for tri in other.indices {
            self.indices
                .push([tri[0] + base_idx, tri[1] + base_idx, tri[2] + base_idx]);
        }
    }
}

/// Extrude a Richardson cartoon ribbon mesh from protein C-alpha coordinates,
/// secondary structure assignments, and pLDDT values.
/// Build the cartoon ribbon.
///
/// `guides` is the ribbon's wide axis per residue, taken from the backbone carbonyl and
/// flip-corrected (Carson & Bugg 1986) — the same construction PyMOL, Mol* and Chimera use. It
/// is what makes a β-strand's flat face lie in the sheet and show the strand's real twist.
/// Between residues the axis is interpolated from one guide to the next, so the face twists
/// smoothly along the spline instead of turning all at once at each residue boundary.
/// `None` for a residue whose C or O is absent (a C-alpha-only trace, which the offline
/// simulator and coarse-grained models produce); those fall back to parallel transport, which
/// is smooth and twist-free but carries no physical meaning.
pub fn generate_cartoon_mesh(
    ca_coords: &[Vector3<f64>],
    ss_assignments: &[SecondaryStructure],
    plddts: &[f64],
    guides: &[Option<Vector3<f64>>],
    subdivisions_per_residue: usize,
) -> TriangleMesh {
    let n_res = ca_coords.len();
    if n_res < 2 {
        return TriangleMesh::new();
    }

    // 1. Spline interpolation
    let spline_points = interpolate_catmull_rom(ca_coords, subdivisions_per_residue);
    if spline_points.is_empty() {
        return TriangleMesh::new();
    }

    // 2. Bishop frame orientation
    let tangents: Vec<Vector3<f64>> = spline_points.iter().map(|p| p.tangent).collect();
    let frames = compute_bishop_frames(&tangents);

    let mut mesh = TriangleMesh::new();
    let n_pts = spline_points.len();

    // Profile ring definition: 8-sided circle/ellipse for smooth shading
    let ring_sides = 8usize;
    let mut ring_offsets = Vec::with_capacity(ring_sides);
    for i in 0..ring_sides {
        let angle = (i as f64) * std::f64::consts::TAU / (ring_sides as f64);
        ring_offsets.push((angle.cos(), angle.sin()));
    }

    // Identify beta-strand terminations to generate Richardson arrowheads pointing towards C-terminus
    let is_strand_terminus = |idx: usize| -> bool {
        if ss_assignments.get(idx) == Some(&SecondaryStructure::Strand) {
            idx + 1 >= n_res || ss_assignments.get(idx + 1) != Some(&SecondaryStructure::Strand)
        } else {
            false
        }
    };

    for (k, (s_pt, frame)) in spline_points.iter().zip(frames.iter()).enumerate() {
        let res_idx = s_pt.residue_index.min(n_res - 1);
        // Prefer the carbonyl-derived axis, made perpendicular to the tangent. Falls back to
        // the parallel-transported frame when this residue has no backbone O.
        let (axis_wide, axis_thin) = match interpolated_guide(guides, res_idx, s_pt.parameter) {
            Some(g) => {
                let t = s_pt.tangent.normalize();
                let perp = g - t * g.dot(&t);
                if perp.norm() > 1e-6 {
                    let n1 = perp.normalize();
                    (n1, t.cross(&n1).normalize())
                } else {
                    (frame.normal1, frame.normal2)
                }
            }
            None => (frame.normal1, frame.normal2),
        };
        let ss = ss_assignments
            .get(res_idx)
            .copied()
            .unwrap_or(SecondaryStructure::Coil);
        let plddt = plddts.get(res_idx).copied().unwrap_or(75.0) as f32;

        // Radii depending on secondary structure
        let (rx, ry) = match ss {
            SecondaryStructure::Helix => (1.5, 0.45), // Wide helical ribbon
            SecondaryStructure::Strand => {
                if is_strand_terminus(res_idx) {
                    // Richardson beta-sheet arrowhead:
                    // Flares to 2.8Å barb at u=0.25, then tapers to 0.2Å tip at u=1.0
                    let u = s_pt.parameter.clamp(0.0, 1.0);
                    let rx = if u <= 0.25 {
                        1.8 + (2.8 - 1.8) * (u / 0.25)
                    } else {
                        2.8 - (2.8 - 0.2) * ((u - 0.25) / 0.75)
                    };
                    let ry = 0.25 * (1.0 - 0.4 * u);
                    (rx, ry)
                } else {
                    (1.8, 0.25) // Flat beta sheet
                }
            }
            SecondaryStructure::Coil => (0.35, 0.35), // Thin flexible loop tube
        };

        for &(cos_a, sin_a) in &ring_offsets {
            let offset = axis_wide * (rx * cos_a) + axis_thin * (ry * sin_a);
            let normal = (axis_wide * cos_a + axis_thin * sin_a).normalize();
            let pos = s_pt.position + offset;

            mesh.vertices.push(Vertex3D {
                position: Vector3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                normal: Vector3::new(normal.x as f32, normal.y as f32, normal.z as f32),
                plddt,
                secondary_structure: ss,
                residue_index: res_idx,
            });
        }

        // Connect adjacent rings with quad faces
        if k > 0 {
            let cur_ring_start = (k * ring_sides) as u32;
            let prev_ring_start = ((k - 1) * ring_sides) as u32;

            for j in 0..ring_sides {
                let next_j = (j + 1) % ring_sides;

                let p0 = prev_ring_start + j as u32;
                let p1 = prev_ring_start + next_j as u32;
                let c0 = cur_ring_start + j as u32;
                let c1 = cur_ring_start + next_j as u32;

                mesh.indices.push([p0, c0, c1]);
                mesh.indices.push([p0, c1, p1]);
            }
        }
    }

    // Add end caps to N-terminal (first ring) and C-terminal (last ring)
    if n_pts > 0 {
        // N-terminal cap
        let center_n = spline_points[0].position;
        let norm_n = -frames[0].tangent;
        let v_center_n = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex3D {
            position: Vector3::new(center_n.x as f32, center_n.y as f32, center_n.z as f32),
            normal: Vector3::new(norm_n.x as f32, norm_n.y as f32, norm_n.z as f32),
            plddt: plddts.first().copied().unwrap_or(75.0) as f32,
            secondary_structure: ss_assignments
                .first()
                .copied()
                .unwrap_or(SecondaryStructure::Coil),
            residue_index: 0,
        });

        for j in 0..ring_sides {
            let next_j = (j + 1) % ring_sides;
            mesh.indices.push([v_center_n, next_j as u32, j as u32]);
        }

        // C-terminal cap
        let last_k = n_pts - 1;
        let center_c = spline_points[last_k].position;
        let norm_c = frames[last_k].tangent;
        let v_center_c = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex3D {
            position: Vector3::new(center_c.x as f32, center_c.y as f32, center_c.z as f32),
            normal: Vector3::new(norm_c.x as f32, norm_c.y as f32, norm_c.z as f32),
            plddt: plddts.last().copied().unwrap_or(75.0) as f32,
            secondary_structure: ss_assignments
                .last()
                .copied()
                .unwrap_or(SecondaryStructure::Coil),
            residue_index: n_res - 1,
        });

        let last_ring_start = (last_k * ring_sides) as u32;
        for j in 0..ring_sides {
            let next_j = (j + 1) % ring_sides;
            mesh.indices.push([
                v_center_c,
                last_ring_start + j as u32,
                last_ring_start + next_j as u32,
            ]);
        }
    }

    mesh
}

/// The ribbon's wide axis at fraction `u` of the way from C-alpha `i` to C-alpha `i + 1`.
///
/// Using residue `i`'s guide unchanged over the whole span held the face still inside a
/// residue and then turned it by the full guide-to-guide angle at the boundary: on 1UBQ a
/// median 29–64° turn between consecutive rings across a boundary against 5–7° within one.
/// Residue `i`'s carbonyl belongs to the peptide between C-alphas `i` and `i + 1`, so its guide
/// is placed mid-span and blended (normalised linear interpolation) towards the neighbouring
/// peptide's on either side; that brings the boundary turns down to 13–29°, the rest being the
/// spline's own curvature at the C-alpha. The guides are flip-corrected upstream, so neighbours
/// agree in sign; should they not (an uncorrected caller), blending would pass near zero, and
/// residue `i`'s guide is used as before.
fn interpolated_guide(guides: &[Option<Vector3<f64>>], i: usize, u: f64) -> Option<Vector3<f64>> {
    let own = guides.get(i).copied().flatten()?;
    let at = |j: Option<usize>| j.and_then(|j| guides.get(j).copied().flatten());
    // Residue i's carbonyl belongs to the peptide between C-alpha i and i+1, so its guide is
    // anchored mid-span (u = 0.5) and blended towards the neighbouring peptide's on each side.
    let u = u.clamp(0.0, 1.0);
    let (other, t) = if u < 0.5 {
        (at(i.checked_sub(1)), 0.5 - u)
    } else {
        (at(Some(i + 1)), u - 0.5)
    };
    let Some(other) = other else {
        return Some(own);
    };
    if own.dot(&other) <= 0.0 {
        return Some(own);
    }
    let g = own * (1.0 - t) + other * t;
    if g.norm() > 1e-9 {
        Some(g.normalize())
    } else {
        Some(own)
    }
}

/// Generate a 3D cylindrical stick mesh connecting p1 to p2.
/// Used for covalent disulfide bridges (S-S bonds) and sidechain sticks.
pub fn generate_cylinder_mesh(
    p1: Vector3<f32>,
    p2: Vector3<f32>,
    radius: f32,
    sides: usize,
    residue_index: usize,
    secondary_structure: SecondaryStructure,
    plddt: f32,
) -> TriangleMesh {
    let axis = p2 - p1;
    let len = axis.norm();
    if len < 1e-4 {
        return TriangleMesh::new();
    }
    let dir = axis / len;

    // Perpendicular coordinate frame
    let up = if dir.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = dir.cross(&up).normalize();
    let v = dir.cross(&u).normalize();

    let mut mesh = TriangleMesh::new();
    let ring_sides = sides.max(4);

    // Ring 0 at p1, Ring 1 at p2
    for &center in &[p1, p2] {
        for i in 0..ring_sides {
            let angle = (i as f32) * std::f32::consts::TAU / (ring_sides as f32);
            let normal = u * angle.cos() + v * angle.sin();
            let pos = center + normal * radius;
            mesh.vertices.push(Vertex3D {
                position: pos,
                normal,
                plddt,
                secondary_structure,
                residue_index,
            });
        }
    }

    // Side faces (quads connecting ring 0 to ring 1)
    for i in 0..ring_sides {
        let next_i = (i + 1) % ring_sides;
        let p0 = i as u32;
        let p1_idx = next_i as u32;
        let c0 = (i + ring_sides) as u32;
        let c1 = (next_i + ring_sides) as u32;

        mesh.indices.push([p0, c0, c1]);
        mesh.indices.push([p0, c1, p1_idx]);
    }

    mesh
}

/// Generate a UV sphere mesh at specified center and radius (for atomic beads and sulfur caps).
pub fn generate_sphere_mesh(
    center: Vector3<f32>,
    radius: f32,
    lat_bands: usize,
    lon_bands: usize,
    residue_index: usize,
    secondary_structure: SecondaryStructure,
    plddt: f32,
) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();
    let lats = lat_bands.max(3);
    let lons = lon_bands.max(3);

    for lat in 0..=lats {
        let theta = (lat as f32) * std::f32::consts::PI / (lats as f32);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lons {
            let phi = (lon as f32) * std::f32::consts::TAU / (lons as f32);
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let normal = Vector3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
            let pos = center + normal * radius;

            mesh.vertices.push(Vertex3D {
                position: pos,
                normal,
                plddt,
                secondary_structure,
                residue_index,
            });
        }
    }

    for lat in 0..lats {
        for lon in 0..lons {
            let first = (lat * (lons + 1) + lon) as u32;
            let second = first + lons as u32 + 1;

            mesh.indices.push([first, second, first + 1]);
            mesh.indices.push([second, second + 1, first + 1]);
        }
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartoon_mesh_generation() {
        let ca_coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 1.0, 0.0),
            Vector3::new(11.4, 2.0, 0.0),
        ];
        let ss = vec![SecondaryStructure::Helix; 4];
        let plddts = vec![90.0; 4];

        let mesh = generate_cartoon_mesh(&ca_coords, &ss, &plddts, &vec![None; ca_coords.len()], 4);
        assert!(mesh.vertex_count() > 50);
        assert!(mesh.triangle_count() > 50);
    }

    #[test]
    fn test_cylinder_and_sphere_generation() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 2.0, 0.0);
        let cylinder = generate_cylinder_mesh(p1, p2, 0.2, 8, 0, SecondaryStructure::Coil, 85.0);
        assert_eq!(cylinder.vertex_count(), 16);
        assert_eq!(cylinder.triangle_count(), 16);

        let sphere = generate_sphere_mesh(p1, 0.5, 6, 8, 0, SecondaryStructure::Coil, 85.0);
        assert!(sphere.vertex_count() > 0);
        assert!(sphere.triangle_count() > 0);

        let mut combined = cylinder;
        combined.merge(sphere);
        assert!(combined.vertex_count() > 16);
    }
}
