//! Caption text drawn into an RGB frame: Geist Mono (embedded, OFL) rasterised by `fontdue`,
//! alpha-blended. Pure CPU, so the captions are the same pixels whichever renderer drew the
//! structure underneath.

use crate::rasterizer::buffer::ColorRGB;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

const REGULAR_TTF: &[u8] = include_bytes!("../../assets/film/fonts/GeistMono-Regular.ttf");
const SEMIBOLD_TTF: &[u8] = include_bytes!("../../assets/film/fonts/GeistMono-SemiBold.ttf");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weight {
    Regular,
    SemiBold,
}

fn fonts() -> &'static (Arc<fontdue::Font>, Arc<fontdue::Font>) {
    static FONTS: OnceLock<(Arc<fontdue::Font>, Arc<fontdue::Font>)> = OnceLock::new();
    FONTS.get_or_init(|| {
        let load = |bytes: &[u8]| {
            Arc::new(
                fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default())
                    .expect("embedded Geist Mono parses"),
            )
        };
        (load(REGULAR_TTF), load(SEMIBOLD_TTF))
    })
}

type Glyph = (fontdue::Metrics, Vec<u8>);

/// Draws text into frames. Holds a glyph cache, so each rendering thread keeps its own.
#[derive(Default)]
pub struct TextPainter {
    cache: RefCell<HashMap<(char, u32, Weight), Glyph>>,
}

/// An RGB frame, `width × height`, three bytes per pixel, rows top to bottom.
pub struct Canvas<'a> {
    pub rgb: &'a mut [u8],
    pub width: usize,
    pub height: usize,
}

impl Canvas<'_> {
    /// Blend `color` over the pixel at `(x, y)` with opacity `a` (0–1).
    #[inline]
    pub fn blend(&mut self, x: usize, y: usize, color: ColorRGB, a: f32) {
        if x >= self.width || y >= self.height || a <= 0.0 {
            return;
        }
        let a = a.min(1.0);
        let i = (y * self.width + x) * 3;
        for (k, c) in [color.r, color.g, color.b].into_iter().enumerate() {
            let under = self.rgb[i + k] as f32;
            self.rgb[i + k] = (under + (c as f32 - under) * a).round() as u8;
        }
    }

    /// A filled rectangle blended at opacity `a`.
    pub fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: ColorRGB, a: f32) {
        for yy in y..(y + h).min(self.height) {
            for xx in x..(x + w).min(self.width) {
                self.blend(xx, yy, color, a);
            }
        }
    }
}

impl TextPainter {
    pub fn new() -> Self {
        Self::default()
    }

    fn with_glyph<R>(&self, ch: char, px: f32, weight: Weight, f: impl FnOnce(&Glyph) -> R) -> R {
        let key = (ch, (px * 4.0).round() as u32, weight);
        let mut cache = self.cache.borrow_mut();
        let glyph = cache.entry(key).or_insert_with(|| {
            let (regular, semibold) = fonts();
            let font = match weight {
                Weight::Regular => regular,
                Weight::SemiBold => semibold,
            };
            font.rasterize(ch, px)
        });
        f(glyph)
    }

    /// Width of `text` in pixels at size `px`.
    pub fn measure(&self, text: &str, px: f32, weight: Weight) -> f32 {
        text.chars()
            .map(|c| self.with_glyph(c, px, weight, |(m, _)| m.advance_width))
            .sum()
    }

    /// Ascent above the baseline at size `px`, for laying out lines.
    pub fn ascent(&self, px: f32) -> f32 {
        let (regular, _) = fonts();
        regular
            .horizontal_line_metrics(px)
            .map(|m| m.ascent)
            .unwrap_or(px * 0.8)
    }

    /// Draw `text` with its baseline at `y` and its left edge at `x`; returns the pen position
    /// after the last glyph.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        canvas: &mut Canvas<'_>,
        x: f32,
        y: f32,
        text: &str,
        px: f32,
        weight: Weight,
        color: ColorRGB,
        alpha: f32,
    ) -> f32 {
        let mut pen = x;
        for ch in text.chars() {
            self.with_glyph(ch, px, weight, |(m, coverage)| {
                let left = (pen + m.xmin as f32).round() as isize;
                let top = (y - m.ymin as f32 - m.height as f32).round() as isize;
                for gy in 0..m.height {
                    for gx in 0..m.width {
                        let cov = coverage[gy * m.width + gx] as f32 / 255.0;
                        let (px_x, px_y) = (left + gx as isize, top + gy as isize);
                        if cov > 0.0 && px_x >= 0 && px_y >= 0 {
                            canvas.blend(px_x as usize, px_y as usize, color, cov * alpha);
                        }
                    }
                }
                pen += m.advance_width;
            });
        }
        pen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_glyphs_exist_in_the_embedded_font() {
        let (regular, semibold) = fonts();
        for ch in "H102A · −4 H-bonds +1 π-stack → pLDDT 91 Å C-alpha RMSD 0.20 … not shown: lost gained parent ±".chars() {
            assert!(regular.has_glyph(ch), "Geist Mono Regular lacks {ch:?}");
            assert!(semibold.has_glyph(ch), "Geist Mono SemiBold lacks {ch:?}");
        }
    }

    #[test]
    fn text_is_drawn_and_measured_consistently() {
        let painter = TextPainter::new();
        let (w, h) = (200, 40);
        let mut rgb = vec![0u8; w * h * 3];
        let mut c = Canvas {
            rgb: &mut rgb,
            width: w,
            height: h,
        };
        let end = painter.draw(
            &mut c,
            4.0,
            30.0,
            "K15A",
            20.0,
            Weight::SemiBold,
            ColorRGB::WHITE,
            1.0,
        );
        let measured = painter.measure("K15A", 20.0, Weight::SemiBold);
        assert!((end - 4.0 - measured).abs() < 1e-3);
        assert!(rgb.iter().any(|&v| v > 200), "some pixels are fully inked");
    }
}
