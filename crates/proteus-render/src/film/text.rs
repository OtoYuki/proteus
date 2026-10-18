//! Caption text drawn into an RGB frame: Geist Mono (embedded, OFL), outlines read by `skrifa`
//! and turned into coverage by `ab_glyph_rasterizer`, alpha-blended. Pure CPU, so the captions
//! are the same pixels whichever renderer drew the structure underneath.

use crate::rasterizer::buffer::ColorRGB;
use ab_glyph_rasterizer::{point, Point, Rasterizer};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::prelude::{FontRef, LocationRef, MetadataProvider, Size};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

const REGULAR_TTF: &[u8] = include_bytes!("../../assets/film/fonts/GeistMono-Regular.ttf");
const SEMIBOLD_TTF: &[u8] = include_bytes!("../../assets/film/fonts/GeistMono-SemiBold.ttf");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weight {
    Regular,
    SemiBold,
}

fn font(weight: Weight) -> FontRef<'static> {
    static FONTS: OnceLock<(FontRef<'static>, FontRef<'static>)> = OnceLock::new();
    let (regular, semibold) = FONTS.get_or_init(|| {
        (
            FontRef::new(REGULAR_TTF).expect("embedded Geist Mono Regular parses"),
            FontRef::new(SEMIBOLD_TTF).expect("embedded Geist Mono SemiBold parses"),
        )
    });
    match weight {
        Weight::Regular => regular.clone(),
        Weight::SemiBold => semibold.clone(),
    }
}

/// A rasterised glyph: coverage (0–255) over a `width × height` box whose top-left corner is
/// `left` right of the pen and `top` above the baseline.
struct Glyph {
    left: i32,
    top: i32,
    width: usize,
    height: usize,
    advance: f32,
    coverage: Vec<u8>,
}

enum Segment {
    Line(Point, Point),
    Quad(Point, Point, Point),
    Cubic(Point, Point, Point, Point),
}

/// Collects an outline (pixels, y up) as segments, closing every contour.
#[derive(Default)]
struct Path {
    segments: Vec<Segment>,
    start: Option<Point>,
    current: Point,
}

impl Path {
    fn close_contour(&mut self) {
        if let Some(start) = self.start.take() {
            if start != self.current {
                self.segments.push(Segment::Line(self.current, start));
            }
            self.current = start;
        }
    }
}

impl OutlinePen for Path {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close_contour();
        self.start = Some(point(x, y));
        self.current = point(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.segments.push(Segment::Line(self.current, point(x, y)));
        self.current = point(x, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.segments
            .push(Segment::Quad(self.current, point(cx, cy), point(x, y)));
        self.current = point(x, y);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.segments.push(Segment::Cubic(
            self.current,
            point(cx0, cy0),
            point(cx1, cy1),
            point(x, y),
        ));
        self.current = point(x, y);
    }
    fn close(&mut self) {
        self.close_contour();
    }
}

fn rasterize(ch: char, px: f32, weight: Weight) -> Glyph {
    let font = font(weight);
    let size = Size::new(px);
    let gid = font.charmap().map(ch).unwrap_or_default();
    let advance = font
        .glyph_metrics(size, LocationRef::default())
        .advance_width(gid)
        .unwrap_or(0.0);
    let mut path = Path::default();
    if let Some(outline) = font.outline_glyphs().get(gid) {
        let _ = outline.draw(DrawSettings::unhinted(size, LocationRef::default()), &mut path);
    }
    path.close_contour();
    let pts = path.segments.iter().flat_map(|s| match s {
        Segment::Line(a, b) => vec![*a, *b],
        Segment::Quad(a, b, c) => vec![*a, *b, *c],
        Segment::Cubic(a, b, c, d) => vec![*a, *b, *c, *d],
    });
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    if path.segments.is_empty() || x1 <= x0 || y1 <= y0 {
        return Glyph {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
            advance,
            coverage: Vec::new(),
        };
    }
    let (left, bottom, right, top) = (x0.floor(), y0.floor(), x1.ceil(), y1.ceil());
    let (width, height) = ((right - left) as usize, (top - bottom) as usize);
    // Outline space is y up; the bitmap is y down from the box's top.
    let map = |p: Point| point(p.x - left, top - p.y);
    let mut r = Rasterizer::new(width, height);
    for s in &path.segments {
        match *s {
            Segment::Line(a, b) => r.draw_line(map(a), map(b)),
            Segment::Quad(a, b, c) => r.draw_quad(map(a), map(b), map(c)),
            Segment::Cubic(a, b, c, d) => r.draw_cubic(map(a), map(b), map(c), map(d)),
        }
    }
    let mut coverage = vec![0u8; width * height];
    r.for_each_pixel(|i, a| coverage[i] = (a.clamp(0.0, 1.0) * 255.0).round() as u8);
    Glyph {
        left: left as i32,
        top: top as i32,
        width,
        height,
        advance,
        coverage,
    }
}

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
        let glyph = cache
            .entry(key)
            .or_insert_with(|| rasterize(ch, px, weight));
        f(glyph)
    }

    /// Width of `text` in pixels at size `px`.
    pub fn measure(&self, text: &str, px: f32, weight: Weight) -> f32 {
        text.chars()
            .map(|c| self.with_glyph(c, px, weight, |g| g.advance))
            .sum()
    }

    /// Ascent above the baseline at size `px`, for laying out lines.
    pub fn ascent(&self, px: f32) -> f32 {
        font(Weight::Regular)
            .metrics(Size::new(px), LocationRef::default())
            .ascent
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
            self.with_glyph(ch, px, weight, |g| {
                let left = (pen + g.left as f32).round() as isize;
                let top = (y - g.top as f32).round() as isize;
                for gy in 0..g.height {
                    for gx in 0..g.width {
                        let cov = g.coverage[gy * g.width + gx] as f32 / 255.0;
                        let (px_x, px_y) = (left + gx as isize, top + gy as isize);
                        if cov > 0.0 && px_x >= 0 && px_y >= 0 {
                            canvas.blend(px_x as usize, px_y as usize, color, cov * alpha);
                        }
                    }
                }
                pen += g.advance;
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
        for weight in [Weight::Regular, Weight::SemiBold] {
            let charmap = font(weight).charmap();
            for ch in "H102A · −4 H-bonds +1 π-stack → pLDDT 91 Å C-alpha RMSD 0.20 … not shown: lost gained parent ±".chars() {
                assert!(charmap.map(ch).is_some(), "Geist Mono {weight:?} lacks {ch:?}");
            }
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
