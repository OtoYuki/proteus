use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use std::fmt::Write;

/// Universal 24-bit Truecolor Half-Block compositor.
/// Each character cell combines two vertical pixels into one cell using `▀` (U+2580):
/// - Top pixel = Foreground color
/// - Bottom pixel = Background color
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

                if top == ColorRGB::BLACK && bottom == ColorRGB::BLACK {
                    if cur_fg.is_some() || cur_bg.is_some() {
                        out.push_str("\x1b[0m");
                        cur_fg = None;
                        cur_bg = None;
                    }
                    out.push(' ');
                    continue;
                }

                // Optimize ANSI escape sequences
                if cur_fg != Some(top) {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", top.r, top.g, top.b);
                    cur_fg = Some(top);
                }
                if cur_bg != Some(bottom) {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", bottom.r, bottom.g, bottom.b);
                    cur_bg = Some(bottom);
                }

                out.push('▀');
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

                if top == ColorRGB::BLACK && bottom == ColorRGB::BLACK {
                    if self.last_fg.is_some() || self.last_bg.is_some() {
                        out.push_str("\x1b[0m");
                        self.last_fg = None;
                        self.last_bg = None;
                    }
                    out.push(' ');
                    continue;
                }

                if self.last_fg != Some(top) {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", top.r, top.g, top.b);
                    self.last_fg = Some(top);
                }
                if self.last_bg != Some(bottom) {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", bottom.r, bottom.g, bottom.b);
                    self.last_bg = Some(bottom);
                }

                out.push('▀');
            }
        }
    }
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
}
