use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use std::fmt::Write;

/// Universal 24-bit Truecolor Half-Block compositor.
/// Each character cell combines two vertical pixels into one cell using `▀` (U+2580):
/// - Top pixel = Foreground color
/// - Bottom pixel = Background color
///
/// Black is the framebuffer's "nothing drawn here", and it is left to the terminal's own
/// background rather than painted black: a cell with one empty half is drawn as `▀` or `▄` in
/// the other half's colour over the default background, so a ribbon's edge does not carry a
/// black fringe on a terminal whose background is not black.
pub struct HalfBlockRenderer {
    prev_top: Vec<ColorRGB>,
    prev_bottom: Vec<ColorRGB>,
    last_fg: Option<ColorRGB>,
    last_bg: Option<ColorRGB>,
}

impl HalfBlockRenderer {
    pub fn new() -> Self {
        Self {
            prev_top: Vec::new(),
            prev_bottom: Vec::new(),
            last_fg: None,
            last_bg: None,
        }
    }

    /// Render a single static snapshot of the framebuffer into an ANSI string.
    /// Framebuffer height must be even; if odd, the last row is padded with black.
    pub fn render_snapshot(fb: &Framebuffer) -> String {
        let cols = fb.width;
        let char_rows = fb.height.div_ceil(2);
        let mut out = String::with_capacity(cols * char_rows * 25);

        let mut cur_fg: Option<ColorRGB> = None;
        let mut cur_bg: Option<ColorRGB> = None;

        for r in 0..char_rows {
            let y_top = r * 2;
            let y_bottom = y_top + 1;

            for x in 0..cols {
                let top = fb.get_pixel(x, y_top).unwrap_or(ColorRGB::BLACK);
                let bottom = if y_bottom < fb.height {
                    fb.get_pixel(x, y_bottom).unwrap_or(ColorRGB::BLACK)
                } else {
                    ColorRGB::BLACK
                };

                emit_cell(&mut out, top, bottom, &mut cur_fg, &mut cur_bg);
            }

            // Reset at line boundary
            out.push_str("\x1b[0m\n");
            cur_fg = None;
            cur_bg = None;
        }

        out
    }

    /// Forget what is on screen, so the next frame repaints every cell. Needed after the
    /// screen is cleared: a resize can keep the cell count (80×20 → 40×40), and a dashboard
    /// toggle on a narrow terminal keeps the view's size, and neither is otherwise detected.
    pub fn invalidate(&mut self) {
        self.prev_top.clear();
        self.prev_bottom.clear();
        self.last_fg = None;
        self.last_bg = None;
    }

    /// Differential update: only emits ANSI codes for cells that changed from the previous frame.
    /// Clears and repaints if dimensions changed.
    pub fn render_differential(
        &mut self,
        fb: &Framebuffer,
        out: &mut String,
        screen_offset_row: u16,
        screen_offset_col: u16,
    ) {
        let cols = fb.width;
        let char_rows = fb.height.div_ceil(2);
        let total_cells = cols * char_rows;

        // Whatever was written since the last frame (dashboard, HUD) may have changed the
        // colours, so start from a known state: `None` is the terminal default.
        out.push_str("\x1b[0m");
        self.last_fg = None;
        self.last_bg = None;

        let resized = self.prev_top.len() != total_cells;
        if resized {
            self.prev_top.resize(total_cells, ColorRGB::BLACK);
            self.prev_bottom.resize(total_cells, ColorRGB::BLACK);
            self.last_fg = None;
            self.last_bg = None;
        }

        for r in 0..char_rows {
            let y_top = r * 2;
            let y_bottom = y_top + 1;

            let mut cursor_moved = false;

            for x in 0..cols {
                let cell_idx = r * cols + x;
                let top = fb.get_pixel(x, y_top).unwrap_or(ColorRGB::BLACK);
                let bottom = if y_bottom < fb.height {
                    fb.get_pixel(x, y_bottom).unwrap_or(ColorRGB::BLACK)
                } else {
                    ColorRGB::BLACK
                };

                // Skip if cell has not changed and screen wasn't resized
                if !resized
                    && self.prev_top[cell_idx] == top
                    && self.prev_bottom[cell_idx] == bottom
                {
                    cursor_moved = false;
                    continue;
                }

                self.prev_top[cell_idx] = top;
                self.prev_bottom[cell_idx] = bottom;

                // Move cursor to specific cell position if not sequential
                if !cursor_moved {
                    let _ = write!(
                        out,
                        "\x1b[{};{}H",
                        screen_offset_row + r as u16 + 1,
                        screen_offset_col + x as u16 + 1
                    );
                    cursor_moved = true;
                }

                emit_cell(out, top, bottom, &mut self.last_fg, &mut self.last_bg);
            }
        }
    }
}

/// Append one cell, emitting only the colour changes it needs. `fg`/`bg` track the terminal's
/// current colours, `None` meaning its default; [`ColorRGB::BLACK`] is an empty pixel.
fn emit_cell(
    out: &mut String,
    top: ColorRGB,
    bottom: ColorRGB,
    fg: &mut Option<ColorRGB>,
    bg: &mut Option<ColorRGB>,
) {
    let (glyph, want_fg, want_bg) = match (top == ColorRGB::BLACK, bottom == ColorRGB::BLACK) {
        (true, true) => (' ', *fg, None),
        (false, true) => ('▀', Some(top), None),
        (true, false) => ('▄', Some(bottom), None),
        (false, false) => ('▀', Some(top), Some(bottom)),
    };
    if *fg != want_fg {
        if let Some(c) = want_fg {
            let _ = write!(out, "\x1b[38;2;{};{};{}m", c.r, c.g, c.b);
        }
        *fg = want_fg;
    }
    if *bg != want_bg {
        match want_bg {
            Some(c) => {
                let _ = write!(out, "\x1b[48;2;{};{};{}m", c.r, c.g, c.b);
            }
            None => out.push_str("\x1b[49m"),
        }
        *bg = want_bg;
    }
    out.push(glyph);
}

impl Default for HalfBlockRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halfblock_snapshot() {
        let mut fb = Framebuffer::new(4, 4);
        fb.set_pixel(0, 0, ColorRGB::new(255, 0, 0), 1.0);
        fb.set_pixel(0, 1, ColorRGB::new(0, 255, 0), 1.0);

        let s = HalfBlockRenderer::render_snapshot(&fb);
        assert!(s.contains('▀'));
        assert!(s.contains("\x1b[38;2;255;0;0m"));
        assert!(s.contains("\x1b[48;2;0;255;0m"));
    }

    /// An empty pixel is the terminal's background, not black paint. A cell with only one
    /// half drawn used to set `48;2;0;0;0`, which shows as a black fringe along every edge of
    /// the ribbon on a terminal with any other background.
    #[test]
    fn empty_pixels_use_the_default_background() {
        let red = ColorRGB::new(255, 0, 0);
        let blue = ColorRGB::new(0, 0, 255);
        let mut fb = Framebuffer::new(3, 2);
        fb.set_pixel(0, 0, red, 1.0); // top half only
        fb.set_pixel(1, 1, blue, 1.0); // bottom half only
        fb.set_pixel(2, 0, red, 1.0); // both halves
        fb.set_pixel(2, 1, blue, 1.0);

        let snapshot = HalfBlockRenderer::render_snapshot(&fb);
        let mut differential = String::new();
        HalfBlockRenderer::new().render_differential(&fb, &mut differential, 0, 0);

        for out in [&snapshot, &differential] {
            assert!(
                !out.contains("48;2;0;0;0") && !out.contains("38;2;0;0;0"),
                "an empty half was painted black: {out:?}"
            );
            // Top-only cell: upper half block in red over the default background.
            assert!(out.contains("\x1b[38;2;255;0;0m▀"), "{out:?}");
            // Bottom-only cell: lower half block in blue over the default background.
            assert!(out.contains("\x1b[38;2;0;0;255m▄"), "{out:?}");
            // Both halves drawn: foreground over an explicit background, as before.
            assert!(
                out.contains("\x1b[38;2;255;0;0m\x1b[48;2;0;0;255m▀"),
                "{out:?}"
            );
        }
    }
}
