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

    /// Box-filter this `k`-times supersampled buffer into `out` (`out` is `width / k` by
    /// `height / k`). A pixel is drawn when enough of its `k × k` samples hit geometry (depth
    /// finite): at least 2 of 9, so a ribbon thinner than a pixel still shows instead of falling
    /// between samples. It takes the mean colour of the samples that hit, so it stays exactly
    /// black (the "empty" colour the terminal compositors leave transparent) only where nothing
    /// was drawn, and edges never blend towards a background the terminal may not share.
    pub fn downsample_into(&self, out: &mut Framebuffer, k: usize) {
        let k = k.max(1);
        let (w, h) = (self.width / k, self.height / k);
        out.resize(w, h);
        let need = ((k * k * 2) as f32 / 9.0).ceil().max(1.0) as usize;
        for y in 0..h {
            for x in 0..w {
                let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0usize);
                let mut depth = f32::INFINITY;
                for sy in 0..k {
                    let row = (y * k + sy) * self.width + x * k;
                    for sx in 0..k {
                        let i = row + sx;
                        if self.depths[i].is_finite() {
                            let c = self.colors[i];
                            r += c.r as u32;
                            g += c.g as u32;
                            b += c.b as u32;
                            n += 1;
                            depth = depth.min(self.depths[i]);
                        }
                    }
                }
                let o = y * w + x;
                if n >= need {
                    let n32 = n as u32;
                    let mut c = ColorRGB::new((r / n32) as u8, (g / n32) as u8, (b / n32) as u8);
                    // Pure black means "empty" downstream; a drawn pixel never is.
                    if c == ColorRGB::BLACK {
                        c = ColorRGB::new(1, 1, 1);
                    }
                    out.colors[o] = c;
                    out.depths[o] = depth;
                } else {
                    out.colors[o] = ColorRGB::BLACK;
                    out.depths[o] = f32::INFINITY;
                }
            }
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

#[cfg(test)]
mod downsample_tests {
    use super::*;

    /// A 3×3-supersampled pixel is drawn when at least 2 of its 9 samples hit, in the mean colour
    /// of those samples; otherwise it stays empty (pure black, transparent downstream).
    #[test]
    fn coverage_decides_and_hits_are_averaged() {
        let mut hi = Framebuffer::new(6, 3);
        // Left pixel: one hit only. Right pixel: two hits, 100 and 200.
        hi.set_pixel(0, 0, ColorRGB::new(50, 50, 50), 1.0);
        hi.set_pixel(3, 0, ColorRGB::new(100, 0, 0), 1.0);
        hi.set_pixel(5, 2, ColorRGB::new(200, 0, 0), 2.0);
        let mut out = Framebuffer::new(1, 1);
        hi.downsample_into(&mut out, 3);
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.colors[0], ColorRGB::BLACK);
        assert!(out.depths[0].is_infinite());
        assert_eq!(out.colors[1], ColorRGB::new(150, 0, 0));
        assert_eq!(out.depths[1], 1.0);
    }
}
