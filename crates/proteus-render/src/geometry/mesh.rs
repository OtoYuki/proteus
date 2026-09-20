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
}

/// Extrude a Richardson cartoon ribbon mesh from protein C-alpha coordinates,
/// secondary structure assignments, and pLDDT values.
pub fn generate_cartoon_mesh(
    ca_coords: &[Vector3<f64>],
    ss_assignments: &[SecondaryStructure],
    plddts: &[f64],
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
            let offset = frame.normal1 * (rx * cos_a) + frame.normal2 * (ry * sin_a);
            let normal = (frame.normal1 * cos_a + frame.normal2 * sin_a).normalize();
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

        let mesh = generate_cartoon_mesh(&ca_coords, &ss, &plddts, 4);
        assert!(mesh.vertex_count() > 50);
        assert!(mesh.triangle_count() > 50);
    }
}
