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
        }
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

        r_yaw * r_pitch * r_roll
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

        let scale = self.zoom * (width.min(height) as f32) * 0.90 / (self.bounding_radius * 2.0);

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
