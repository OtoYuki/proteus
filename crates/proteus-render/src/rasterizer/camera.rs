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

    /// Project a world-space point to 2D screen coordinates with depth.
    /// Screen output: (x, y, depth_z).
    /// Applies 0.5 vertical font aspect ratio compensation.
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
