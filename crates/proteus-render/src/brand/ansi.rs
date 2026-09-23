//! Brand colours as raw ANSI escape strings, for the hand-drawn terminal surfaces (the 3-D
//! viewer's status bar and dashboard). Same rules as the home screen: roles at the terminal's
//! colour depth, and nothing but bold and dim under `NO_COLOR`.

use super::{to_ansi256, ColorDepth, Role, Theme, DARK};
use crate::rasterizer::buffer::ColorRGB;

pub const RESET: &str = "\x1b[0m";

#[derive(Debug, Clone, Copy)]
pub struct Ansi {
    pub depth: ColorDepth,
    pub theme: Theme,
}

impl Ansi {
    pub fn detect() -> Self {
        Self::with_depth(ColorDepth::detect())
    }

    pub fn with_depth(depth: ColorDepth) -> Self {
        Self { depth, theme: DARK }
    }

    /// Foreground escape for a role.
    pub fn fg(&self, role: Role) -> String {
        match self.depth {
            ColorDepth::None => match role {
                Role::Dim | Role::Line => "\x1b[2m".into(),
                Role::Accent | Role::Bad => "\x1b[1m".into(),
                _ => String::new(),
            },
            ColorDepth::Ansi16 => match role.ansi16() {
                Some(i) if i < 8 => format!("\x1b[3{i}m"),
                Some(i) => format!("\x1b[9{}m", i - 8),
                None => String::new(),
            },
            _ => self.rgb(role.rgb(&self.theme)),
        }
    }

    /// Foreground escape for a data colour (pLDDT, structure). In 16 colours the nearest
    /// standard colour; under `NO_COLOR`, nothing, so callers must not rely on it alone.
    pub fn rgb(&self, c: ColorRGB) -> String {
        match self.depth {
            ColorDepth::TrueColor => format!("\x1b[38;2;{};{};{}m", c.r, c.g, c.b),
            ColorDepth::Ansi256 => format!("\x1b[38;5;{}m", to_ansi256(c)),
            ColorDepth::Ansi16 => {
                let i = nearest_ansi16(c);
                if i < 8 {
                    format!("\x1b[3{i}m")
                } else {
                    format!("\x1b[9{}m", i - 8)
                }
            }
            ColorDepth::None => String::new(),
        }
    }

    /// `text` in a role's colour, then reset (a bare string under `NO_COLOR` with no style).
    pub fn paint(&self, role: Role, text: &str) -> String {
        let on = self.fg(role);
        if on.is_empty() {
            text.to_string()
        } else {
            format!("{on}{text}{RESET}")
        }
    }

    pub fn paint_rgb(&self, c: ColorRGB, text: &str) -> String {
        let on = self.rgb(c);
        if on.is_empty() {
            text.to_string()
        } else {
            format!("{on}{text}{RESET}")
        }
    }

    pub fn bold(&self, text: &str) -> String {
        format!("\x1b[1m{text}{RESET}")
    }

    pub fn colours(&self) -> bool {
        self.depth != ColorDepth::None
    }
}

/// The nearest of the 16 standard colours, by the xterm default palette.
fn nearest_ansi16(c: ColorRGB) -> u8 {
    const XTERM16: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (205, 0, 0),
        (0, 205, 0),
        (205, 205, 0),
        (0, 0, 238),
        (205, 0, 205),
        (0, 205, 205),
        (229, 229, 229),
        (127, 127, 127),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (92, 92, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    let d = |(r, g, b): (u8, u8, u8)| {
        (r as i32 - c.r as i32).pow(2)
            + (g as i32 - c.g as i32).pow(2)
            + (b as i32 - c.b as i32).pow(2)
    };
    (0..16).min_by_key(|&i| d(XTERM16[i])).unwrap_or(7) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_follow_the_depth() {
        let t = Ansi::with_depth(ColorDepth::TrueColor);
        assert_eq!(t.fg(Role::Accent), "\x1b[38;2;153;146;11m");
        assert!(Ansi::with_depth(ColorDepth::Ansi256)
            .fg(Role::Accent)
            .starts_with("\x1b[38;5;"));
        assert_eq!(
            Ansi::with_depth(ColorDepth::Ansi16).fg(Role::Accent),
            "\x1b[33m"
        );
        assert_eq!(
            Ansi::with_depth(ColorDepth::Ansi16).fg(Role::Dim),
            "\x1b[90m"
        );
        let none = Ansi::with_depth(ColorDepth::None);
        assert_eq!(none.paint(Role::Sea, "x"), "x");
        assert_eq!(none.paint_rgb(ColorRGB::new(0, 83, 214), "█"), "█");
        assert_eq!(nearest_ansi16(ColorRGB::new(0, 83, 214)), 4);
    }
}
