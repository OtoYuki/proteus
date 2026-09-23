#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorRGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl ColorRGB {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const BLACK: Self = Self::new(0, 0, 0);
    pub const WHITE: Self = Self::new(255, 255, 255);

    pub fn scale(&self, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        Self {
            r: (self.r as f32 * f) as u8,
            g: (self.g as f32 * f) as u8,
            b: (self.b as f32 * f) as u8,
        }
    }

    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: (a.r as f32 + (b.r as f32 - a.r as f32) * t) as u8,
            g: (a.g as f32 + (b.g as f32 - a.g as f32) * t) as u8,
            b: (a.b as f32 + (b.b as f32 - a.b as f32) * t) as u8,
        }
    }
}

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub colors: Vec<ColorRGB>,
    pub depths: Vec<f32>,
}

impl Framebuffer {
    /// # Panics
    /// If `width * height` overflows `usize`; callers taking sizes from users validate them
    /// first (see `proteus_render::viewport_pixels`).
    pub fn new(width: usize, height: usize) -> Self {
        let size = width
            .checked_mul(height)
            .expect("framebuffer dimensions overflow usize");
        Self {
            width,
            height,
            colors: vec![ColorRGB::BLACK; size],
            depths: vec![f32::INFINITY; size],
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            let size = width
                .checked_mul(height)
                .expect("framebuffer dimensions overflow usize");
            self.colors.resize(size, ColorRGB::BLACK);
            self.depths.resize(size, f32::INFINITY);
        }
    }

    pub fn clear(&mut self, bg: ColorRGB) {
        self.colors.fill(bg);
        self.depths.fill(f32::INFINITY);
    }

    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<ColorRGB> {
        if x < self.width && y < self.height {
            Some(self.colors[y * self.width + x])
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: ColorRGB, depth: f32) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            if depth < self.depths[idx] {
                self.depths[idx] = depth;
                self.colors[idx] = color;
            }
        }
    }
}
