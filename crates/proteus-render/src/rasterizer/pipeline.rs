use super::buffer::{ColorRGB, Framebuffer};
use super::camera::OrbitCamera;
use super::shader::{
    plddt_to_color, rainbow_color, secondary_structure_to_color, shade_blinn_phong, ColorScheme,
};
use crate::geometry::mesh::TriangleMesh;
use nalgebra::Vector3;

pub struct Rasterizer {
    pub color_scheme: ColorScheme,
    pub enable_ssao: bool,
    pub enable_outlines: bool,
    projected: Vec<Vector3<f32>>,
    shaded_colors: Vec<ColorRGB>,
}

impl Rasterizer {
    pub fn new(color_scheme: ColorScheme) -> Self {
        Self {
            color_scheme,
            enable_ssao: true,
            enable_outlines: true,
            projected: Vec::new(),
            shaded_colors: Vec::new(),
        }
    }

    /// Render a triangle mesh to the framebuffer with post-processing (SSAO & outlines).
    pub fn render(&mut self, mesh: &TriangleMesh, camera: &OrbitCamera, fb: &mut Framebuffer) {
        self.render_with_scheme(mesh, camera, fb, self.color_scheme);
    }

    /// Render a triangle mesh using an explicit color scheme and apply post-processing.
    pub fn render_with_scheme(
        &mut self,
        mesh: &TriangleMesh,
        camera: &OrbitCamera,
        fb: &mut Framebuffer,
        scheme: ColorScheme,
    ) {
        self.rasterize_mesh(mesh, camera, fb, scheme);
        self.apply_post_processing(fb);
    }

    /// Pure geometry and rasterization pass into z-buffer without applying post-processing.
    /// Used when compositing multiple meshes (e.g. ribbon + disulfides or superposition) into a shared buffer.
    pub fn rasterize_mesh(
        &mut self,
        mesh: &TriangleMesh,
        camera: &OrbitCamera,
        fb: &mut Framebuffer,
        scheme: ColorScheme,
    ) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }

        let n_verts = mesh.vertices.len();
        self.projected.resize(n_verts, Vector3::zeros());
        self.shaded_colors.resize(n_verts, ColorRGB::BLACK);

        let rot_mat = camera.rotation_matrix();
        let total_residues = mesh
            .vertices
            .iter()
            .map(|v| v.residue_index)
            .max()
            .unwrap_or(1)
            + 1;

        // 1. Vertex transform pass and depth range calculation
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for (i, v) in mesh.vertices.iter().enumerate() {
            let (sx, sy, sz) = camera.project(v.position, &rot_mat, fb.width, fb.height);
            self.projected[i] = Vector3::new(sx, sy, sz);
            min_z = min_z.min(sz);
            max_z = max_z.max(sz);
        }

        let z_range = (max_z - min_z).max(1e-3);

        // Gouraud lighting with atmospheric depth cueing (fog)
        for (i, v) in mesh.vertices.iter().enumerate() {
            let sz = self.projected[i].z;
            let depth_fraction = ((sz - min_z) / z_range).clamp(0.0, 1.0);
            // Linear depth cueing: foreground is 100% brightness, background dims gracefully to 55%
            let fog_factor = 1.0 - 0.45 * depth_fraction;

            let view_normal = rot_mat * v.normal;
            let base_color = match scheme {
                ColorScheme::Plddt => plddt_to_color(v.plddt),
                ColorScheme::SecondaryStructure => {
                    secondary_structure_to_color(v.secondary_structure)
                }
                ColorScheme::Rainbow => rainbow_color(v.residue_index, total_residues),
                ColorScheme::Solid(c) => c,
            };

            let lit = shade_blinn_phong(base_color, view_normal);
            self.shaded_colors[i] = lit.scale(fog_factor);
        }

        // 2. Triangle rasterization pass
        let width_f = fb.width as f32;
        let height_f = fb.height as f32;

        for tri in &mesh.indices {
            let idx0 = tri[0] as usize;
            let idx1 = tri[1] as usize;
            let idx2 = tri[2] as usize;

            if idx0 >= n_verts || idx1 >= n_verts || idx2 >= n_verts {
                continue;
            }

            let p0 = self.projected[idx0];
            let p1 = self.projected[idx1];
            let p2 = self.projected[idx2];

            // Determinant (twice the signed area of triangle in screen space)
            let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if det.abs() < 1e-5 {
                continue;
            }
            let inv_det = 1.0 / det;

            // Bounding box clamped to framebuffer bounds
            let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as usize;
            let max_x = p0.x.max(p1.x).max(p2.x).ceil().min(width_f - 1.0).max(0.0) as usize;
            let min_y = p0.y.min(p1.y).min(p2.y).floor().max(0.0) as usize;
            let max_y = p0.y.max(p1.y).max(p2.y).ceil().min(height_f - 1.0).max(0.0) as usize;

            if min_x > max_x || min_y > max_y {
                continue;
            }

            let c0 = self.shaded_colors[idx0];
            let c1 = self.shaded_colors[idx1];
            let c2 = self.shaded_colors[idx2];

            let c0_r = c0.r as f32;
            let c0_g = c0.g as f32;
            let c0_b = c0.b as f32;
            let c1_r = c1.r as f32;
            let c1_g = c1.g as f32;
            let c1_b = c1.b as f32;
            let c2_r = c2.r as f32;
            let c2_g = c2.g as f32;
            let c2_b = c2.b as f32;

            for y in min_y..=max_y {
                let py = y as f32 + 0.5;
                for x in min_x..=max_x {
                    let px = x as f32 + 0.5;

                    // Barycentric coordinates
                    let w0 = ((p1.x - px) * (p2.y - py) - (p2.x - px) * (p1.y - py)) * inv_det;
                    let w1 = ((p2.x - px) * (p0.y - py) - (p0.x - px) * (p2.y - py)) * inv_det;
                    let w2 = 1.0 - w0 - w1;

                    // Small epsilon tolerance to eliminate cracks between adjacent rasterized triangles
                    const EPS: f32 = -1e-4;
                    if w0 >= EPS && w1 >= EPS && w2 >= EPS {
                        let depth = w0 * p0.z + w1 * p1.z + w2 * p2.z;
                        let r = (w0 * c0_r + w1 * c1_r + w2 * c2_r).clamp(0.0, 255.0) as u8;
                        let g = (w0 * c0_g + w1 * c1_g + w2 * c2_g).clamp(0.0, 255.0) as u8;
                        let b = (w0 * c0_b + w1 * c1_b + w2 * c2_b).clamp(0.0, 255.0) as u8;

                        fb.set_pixel(x, y, ColorRGB::new(r, g, b), depth);
                    }
                }
            }
        }
    }

    /// Apply Screen-Space Ambient Occlusion (SSAO) and silhouette edge darkening.
    pub fn apply_post_processing(&self, fb: &mut Framebuffer) {
        if !self.enable_ssao && !self.enable_outlines {
            return;
        }

        let width = fb.width;
        let height = fb.height;
        if width < 3 || height < 3 {
            return;
        }

        // 8 sample offsets on two concentric rings for SSAO
        let ssao_offsets: [(isize, isize); 8] = [
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-3, -3),
            (3, 3),
            (-3, 3),
            (3, -3),
        ];

        // 4 cardinal offsets for silhouette edge detection
        let edge_offsets: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        let orig_colors = fb.colors.clone();

        for y in 0..height {
            let y_i = y as isize;
            for x in 0..width {
                let x_i = x as isize;
                let idx = y * width + x;
                let depth = fb.depths[idx];

                if depth >= f32::INFINITY {
                    continue;
                }

                let mut is_edge = false;
                if self.enable_outlines {
                    for &(dx, dy) in &edge_offsets {
                        let nx = x_i + dx;
                        let ny = y_i + dy;
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let n_idx = (ny as usize) * width + (nx as usize);
                            let n_depth = fb.depths[n_idx];
                            if n_depth >= f32::INFINITY || (n_depth - depth).abs() > 4.0 {
                                is_edge = true;
                                break;
                            }
                        } else {
                            is_edge = true;
                            break;
                        }
                    }
                }

                if is_edge {
                    fb.colors[idx] = orig_colors[idx].scale(0.35);
                } else if self.enable_ssao {
                    let mut occlusion = 0.0f32;
                    let mut valid_samples = 0usize;

                    for &(dx, dy) in &ssao_offsets {
                        let nx = x_i + dx;
                        let ny = y_i + dy;
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let n_idx = (ny as usize) * width + (nx as usize);
                            let n_depth = fb.depths[n_idx];
                            if n_depth < f32::INFINITY {
                                valid_samples += 1;
                                let diff = depth - n_depth;
                                if diff > 0.05 && diff < 8.0 {
                                    occlusion += (diff / 4.0).min(1.0);
                                }
                            }
                        }
                    }

                    if valid_samples > 0 {
                        let occ_factor = (occlusion / valid_samples as f32) * 0.45;
                        let ao = (1.0 - occ_factor).clamp(0.50, 1.0);
                        fb.colors[idx] = orig_colors[idx].scale(ao);
                    }
                }
            }
        }
    }
}
