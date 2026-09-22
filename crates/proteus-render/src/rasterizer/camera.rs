use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub center: Vector3<f32>,
    pub yaw: f32,   // Rotation around Y axis in radians
    pub pitch: f32, // Rotation around X axis in radians
    pub roll: f32,  // Rotation around Z axis in radians
    pub zoom: f32,  // Multiplier, default 1.0
    pub bounding_radius: f32,
    pub pan: Vector3<f32>,
    /// Orientation the interactive rotation is applied on top of (identity, or the
    /// principal-axis frame from [`OrbitCamera::oriented`]); `reset()` returns to it.
    pub base: Matrix3<f32>,
    /// Half-extents of the model along the screen x and y axes of the base frame. When set,
    /// the initial view fits these to the viewport instead of the bounding sphere, so a rod
    /// fills the wide side of a terminal rather than being scaled to the short side.
    pub half_extents: Option<(f32, f32)>,
}

impl OrbitCamera {
    pub fn new(center: Vector3<f32>, bounding_radius: f32) -> Self {
        Self {
            center,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            zoom: 1.0,
            bounding_radius: bounding_radius.max(5.0),
            pan: Vector3::zeros(),
            base: Matrix3::identity(),
            half_extents: None,
        }
    }

    /// A camera whose initial view is the model's principal-axis frame: the longest axis of
    /// `points` runs along screen x, the second along screen y, and the viewer looks down the
    /// shortest (PyMOL's `orient`). Without this a long helix or coiled coil whose axis
    /// happens to lie along z is seen end-on as a dot. Degenerate clouds fall back to identity.
    pub fn oriented(center: Vector3<f32>, bounding_radius: f32, points: &[Vector3<f32>]) -> Self {
        let mut cam = Self::new(center, bounding_radius);
        if points.len() < 2 {
            return cam;
        }
        let n = points.len() as f32;
        let mean = points.iter().sum::<Vector3<f32>>() / n;
        let mut cov = Matrix3::zeros();
        for p in points {
            let d = p - mean;
            cov += d * d.transpose();
        }
        cov /= n;
        if cov.iter().any(|v| !v.is_finite()) || cov.norm() < 1e-6 {
            return cam;
        }
        let eig = cov.symmetric_eigen();
        // Sort axes by decreasing variance.
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| {
            eig.eigenvalues[b]
                .partial_cmp(&eig.eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let e1 = eig.eigenvectors.column(order[0]).into_owned();
        let e2 = eig.eigenvectors.column(order[1]).into_owned();
        // Right-handed frame: the view axis is e1 × e2 rather than the raw third eigenvector,
        // whose sign is arbitrary.
        let e3 = e1.cross(&e2);
        let base = Matrix3::from_rows(&[e1.transpose(), e2.transpose(), e3.transpose()]);
        if base.iter().all(|v| v.is_finite()) && (base.determinant() - 1.0).abs() < 1e-3 {
            cam.base = base;
            let (mut hx, mut hy) = (0.0f32, 0.0f32);
            for p in points {
                let v = base * (p - center);
                hx = hx.max(v.x.abs());
                hy = hy.max(v.y.abs());
            }
            // Ribbons and side chains extend a few Å past the C-alpha trace.
            const MARGIN: f32 = 4.0;
            cam.half_extents = Some((hx + MARGIN, hy + MARGIN));
        }
        cam
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.5, 1.5);
    }

    pub fn adjust_zoom(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(0.2, 5.0);
    }

    pub fn reset(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.roll = 0.0;
        self.zoom = 1.0;
        self.pan = Vector3::zeros();
    }

    /// Construct 3x3 rotation matrix for current camera orientation
    pub fn rotation_matrix(&self) -> Matrix3<f32> {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cr = self.roll.cos();
        let sr = self.roll.sin();

        let r_yaw = Matrix3::new(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
        let r_pitch = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
        let r_roll = Matrix3::new(cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0);

        r_yaw * r_pitch * r_roll * self.base
    }

    /// Project a world-space point to 2D screen coordinates with depth: `(x, y, depth_z)`.
    ///
    /// The scale is **isotropic** — one world unit is the same number of cells horizontally and
    /// vertically. Terminal cells are about twice as tall as they are wide, and the backends
    /// (Braille 2×4, half-block 1×2) already supply that compensation by packing 2 or 4
    /// sub-pixels per cell vertically; applying it here as well squashed every structure by 2×
    /// (fixed in 914e854, pinned by `projection_is_isotropic`).
    ///
    /// Depth increases away from the viewer, who sits at +Z.
    pub fn project(
        &self,
        point: Vector3<f32>,
        rot_mat: &Matrix3<f32>,
        width: usize,
        height: usize,
    ) -> (f32, f32, f32) {
        let centered = point - self.center;
        let view = rot_mat * centered + self.pan;

        let sphere_fit = (width.min(height) as f32) * 0.90 / (self.bounding_radius * 2.0);
        let scale = self.zoom
            * match self.half_extents {
                // Fit the oriented extents to the viewport, but never smaller than the sphere
                // fit (a globular model gains, a rod gains a lot, nothing loses).
                Some((hx, hy)) if hx > 0.0 && hy > 0.0 => {
                    let fit =
                        (width as f32 * 0.90 / (2.0 * hx)).min(height as f32 * 0.90 / (2.0 * hy));
                    fit.max(sphere_fit)
                }
                _ => sphere_fit,
            };

        let screen_x = (width as f32 * 0.5) + view.x * scale;
        let screen_y = (height as f32 * 0.5) - view.y * scale;
        // Smaller depth is closer to camera; viewer is at +Z
        let depth_z = -view.z;

        (screen_x, screen_y, depth_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> OrbitCamera {
        OrbitCamera::new(Vector3::new(1.0, -2.0, 3.0), 10.0)
    }

    #[test]
    fn rotation_matrix_is_a_proper_rotation() {
        let mut c = cam();
        for (yaw, pitch) in [(0.0, 0.0), (0.7, -0.4), (2.9, 1.2), (-1.1, 0.9)] {
            c.reset();
            c.rotate(yaw, pitch);
            let r = c.rotation_matrix();
            // Orthonormal: R Rᵀ = I, and det = +1 (a rotation, not a reflection).
            let id = r * r.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (id[(i, j)] - want).abs() < 1e-5,
                        "R Rᵀ[{i},{j}] = {} for yaw {yaw} pitch {pitch}",
                        id[(i, j)]
                    );
                }
            }
            assert!(
                (r.determinant() - 1.0).abs() < 1e-5,
                "det = {} for yaw {yaw} pitch {pitch}",
                r.determinant()
            );
            // Rotation preserves length.
            let v = Vector3::new(3.0, -4.0, 12.0);
            assert!(((r * v).norm() - v.norm()).abs() < 1e-4);
        }
    }

    #[test]
    fn centre_projects_to_the_middle_of_the_viewport() {
        let c = cam();
        let r = c.rotation_matrix();
        let (x, y, _) = c.project(c.center, &r, 80, 40);
        assert!((x - 40.0).abs() < 1e-4, "x = {x}");
        assert!((y - 20.0).abs() < 1e-4, "y = {y}");
    }

    /// Regression for 914e854: equal world displacements must move the projection by the same
    /// number of cells in x and y. A vertical "font aspect" factor here squashed every model.
    #[test]
    fn projection_is_isotropic() {
        let c = cam();
        let r = Matrix3::identity();
        let (w, h) = (100, 100);
        let (cx, cy, _) = c.project(c.center, &r, w, h);
        let d = 2.5f32;
        let (x1, _, _) = c.project(c.center + Vector3::new(d, 0.0, 0.0), &r, w, h);
        let (_, y1, _) = c.project(c.center + Vector3::new(0.0, d, 0.0), &r, w, h);
        let dx = (x1 - cx).abs();
        let dy = (y1 - cy).abs();
        assert!(dx > 1.0, "displacement too small to be meaningful: {dx}");
        assert!(
            (dx - dy).abs() < 1e-4,
            "anisotropic projection: {dx} cells horizontally vs {dy} vertically"
        );
    }

    #[test]
    fn screen_y_grows_downward_and_depth_grows_away_from_the_viewer() {
        let c = cam();
        let r = Matrix3::identity();
        let (_, y_up, _) = c.project(c.center + Vector3::new(0.0, 1.0, 0.0), &r, 80, 40);
        let (_, y_down, _) = c.project(c.center - Vector3::new(0.0, 1.0, 0.0), &r, 80, 40);
        assert!(y_up < y_down, "+Y must map to a smaller row index");

        // The viewer is at +Z, so a point with larger z is nearer and must have smaller depth.
        let (_, _, near) = c.project(c.center + Vector3::new(0.0, 0.0, 4.0), &r, 80, 40);
        let (_, _, far) = c.project(c.center - Vector3::new(0.0, 0.0, 4.0), &r, 80, 40);
        assert!(
            near < far,
            "depth {near} (near) should be below {far} (far)"
        );
    }

    /// A rod-shaped model (a long helix, a coiled coil) must not be viewed end-on by default:
    /// the initial orientation puts the longest principal axis across the screen and looks
    /// along the shortest.
    #[test]
    fn default_orientation_looks_along_the_shortest_principal_axis() {
        let pts: Vec<Vector3<f32>> = (0..50)
            .map(|i| Vector3::new(0.3 * (i % 3) as f32, 0.1 * (i % 2) as f32, i as f32 * 1.5))
            .collect();
        let c = OrbitCamera::oriented(Vector3::new(0.3, 0.05, 36.75), 40.0, &pts);
        let r = c.rotation_matrix();
        let (x0, y0, _) = c.project(pts[0], &r, 100, 100);
        let (x1, y1, _) = c.project(pts[49], &r, 100, 100);
        let on_screen = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        assert!(
            on_screen > 60.0,
            "rod spans only {on_screen:.1} cells of a 100-cell viewport"
        );
        // In a wide viewport the fit follows the oriented extents, not the bounding sphere:
        // the rod uses most of the width instead of being scaled to the short side.
        let (wx0, _, _) = c.project(pts[0], &r, 100, 48);
        let (wx1, _, _) = c.project(pts[49], &r, 100, 48);
        assert!(
            (wx1 - wx0).abs() > 75.0,
            "rod spans only {:.1} of 100 columns in a 100×48 viewport",
            (wx1 - wx0).abs()
        );
        // …and still stays inside the viewport.
        for p in &pts {
            let (x, y, _) = c.project(*p, &r, 100, 48);
            assert!(
                (0.0..=100.0).contains(&x) && (0.0..=48.0).contains(&y),
                "({x}, {y})"
            );
        }
        // The long axis lies along screen x (the wider terminal dimension), not y.
        assert!((x1 - x0).abs() > (y1 - y0).abs());
        // The base orientation survives reset() and interactive rotation stays a rotation.
        let mut c2 = c.clone();
        c2.rotate(0.4, -0.3);
        c2.reset();
        let r2 = c2.rotation_matrix();
        assert!((r2 - r).norm() < 1e-6);
        assert!((r.determinant() - 1.0).abs() < 1e-5);
    }

    /// Fewer than two points, or a degenerate cloud, must not panic or produce NaN.
    #[test]
    fn orientation_is_robust_to_degenerate_input() {
        for pts in [
            vec![],
            vec![Vector3::new(1.0, 2.0, 3.0)],
            vec![Vector3::new(1.0, 2.0, 3.0); 5],
        ] {
            let c = OrbitCamera::oriented(Vector3::zeros(), 10.0, &pts);
            let r = c.rotation_matrix();
            assert!(r.iter().all(|v| v.is_finite()));
            assert!((r.determinant() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn zoom_scales_the_projection_linearly() {
        let mut c = cam();
        let r = Matrix3::identity();
        let probe = c.center + Vector3::new(3.0, 0.0, 0.0);
        let (x1, _, _) = c.project(probe, &r, 80, 80);
        let base = x1 - 40.0;
        c.adjust_zoom(2.0);
        let (x2, _, _) = c.project(probe, &r, 80, 80);
        assert!(
            ((x2 - 40.0) - 2.0 * base).abs() < 1e-3,
            "2x zoom gave {}, expected {}",
            x2 - 40.0,
            2.0 * base
        );
    }

    #[test]
    fn a_structure_at_the_bounding_radius_stays_inside_the_viewport() {
        let c = cam();
        let r = c.rotation_matrix();
        let (w, h) = (120usize, 60usize);
        // Worst case: a point on the bounding sphere along the screen x axis.
        let edge = c.center + Vector3::new(c.bounding_radius, 0.0, 0.0);
        let (x, y, _) = c.project(edge, &r, w, h);
        assert!(x >= 0.0 && x <= w as f32, "x = {x} outside 0..{w}");
        assert!(y >= 0.0 && y <= h as f32, "y = {y} outside 0..{h}");
    }
}
