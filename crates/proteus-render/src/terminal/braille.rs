use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use std::fmt::Write;

/// 2x4 Subpixel Braille wireframe / high-density compositor.
/// Maps a 2x4 pixel grid into a single Unicode Braille pattern character (U+2800..U+28FF),
/// giving a 2x horizontal and 4x vertical resolution boost.
pub struct BrailleRenderer;

impl BrailleRenderer {
    // Dot position bitmask lookups:
    // (0,0)->1, (0,1)->2, (0,2)->4, (1,0)->8, (1,1)->16, (1,2)->32, (0,3)->64, (1,3)->128
    const DOT_MASKS: [[u8; 2]; 4] = [
        [0x01, 0x08], // dy = 0: dx=0, dx=1
        [0x02, 0x10], // dy = 1: dx=0, dx=1
        [0x04, 0x20], // dy = 2: dx=0, dx=1
        [0x40, 0x80], // dy = 3: dx=0, dx=1
    ];

    /// Render a static snapshot using Braille characters with 24-bit Truecolor foregrounds.
    pub fn render_snapshot(fb: &Framebuffer) -> String {
        let cols = fb.width.div_ceil(2);
        let rows = fb.height.div_ceil(4);
        let mut out = String::with_capacity(cols * rows * 20);

        let mut last_fg: Option<ColorRGB> = None;

        for r in 0..rows {
            for c in 0..cols {
                let mut mask = 0u8;
                let mut sum_r = 0u32;
                let mut sum_g = 0u32;
                let mut sum_b = 0u32;
                let mut dot_count = 0u32;

                for dy in 0..4 {
                    let y = r * 4 + dy;
                    if y >= fb.height {
                        continue;
                    }
                    for dx in 0..2 {
                        let x = c * 2 + dx;
                        if x >= fb.width {
                            continue;
                        }

                        let idx = y * fb.width + x;
                        if fb.depths[idx] < f32::INFINITY {
                            mask |= Self::DOT_MASKS[dy][dx];
                            let color = fb.colors[idx];
                            sum_r += color.r as u32;
                            sum_g += color.g as u32;
                            sum_b += color.b as u32;
                            dot_count += 1;
                        }
                    }
                }

                if mask == 0 || dot_count == 0 {
                    out.push(' ');
                    continue;
                }

                let avg_color = ColorRGB::new(
                    (sum_r / dot_count) as u8,
                    (sum_g / dot_count) as u8,
                    (sum_b / dot_count) as u8,
                );

                if last_fg != Some(avg_color) {
                    let _ = write!(
                        out,
                        "\x1b[38;2;{};{};{}m",
                        avg_color.r, avg_color.g, avg_color.b
                    );
                    last_fg = Some(avg_color);
                }

                let ch = char::from_u32(0x2800 + mask as u32).unwrap_or(' ');
                out.push(ch);
            }

            out.push_str("\x1b[0m\n");
            last_fg = None;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braille_renderer() {
        let mut fb = Framebuffer::new(4, 8);
        fb.set_pixel(0, 0, ColorRGB::new(255, 255, 255), 1.0);
        let s = BrailleRenderer::render_snapshot(&fb);
        assert!(s.contains('\u{2801}')); // Dot 1 set
    }
}
