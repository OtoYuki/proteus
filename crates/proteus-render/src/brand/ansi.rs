//! Brand colours as raw ANSI escape strings, for the hand-drawn terminal surfaces (the 3-D
//! viewer's status bar and dashboard). Same rules as the home screen: roles at the terminal's
//! colour depth, nothing but bold and dim under `NO_COLOR`, and no escapes at all for a dumb
//! terminal or a pipe (`for_stdout`). The 3-D picture itself is drawn in 24-bit colour by the
//! terminal back-ends and is not governed by this.

use super::{to_ansi256, ColorDepth, Role, Theme, DARK};
use crate::rasterizer::buffer::ColorRGB;

pub const RESET: &str = "\x1b[0m";

#[derive(Debug, Clone, Copy)]
pub struct Ansi {
    pub depth: ColorDepth,
    pub theme: Theme,
    /// No escape sequences at all, not even bold: the output is going to a pipe or a file.
    pub plain: bool,
}

impl Ansi {
    pub fn detect() -> Self {
        // A dumb terminal takes no escape sequences at all, not even bold.
        let dumb = std::env::var("TERM").is_ok_and(|t| t == "dumb");
        Self {
            plain: dumb,
            ..Self::with_depth(ColorDepth::detect())
        }
    }

    /// For lines printed to standard output: no colour when it is not a terminal (a pipe, a
    /// file), so what a script captures is plain text.
    pub fn for_stdout() -> Self {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            Self::detect()
        } else {
            Self {
                plain: true,
                ..Self::with_depth(ColorDepth::None)
            }
        }
    }

    pub fn with_depth(depth: ColorDepth) -> Self {
        Self {
            depth,
            theme: DARK,
            plain: false,
        }
    }

    /// Foreground escape for a role.
    pub fn fg(&self, role: Role) -> String {
        if self.plain {
            return String::new();
        }
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
                let i = ansi16_for_data(c);
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
        if self.plain {
            text.to_string()
        } else {
            format!("\x1b[1m{text}{RESET}")
        }
    }

    pub fn colours(&self) -> bool {
        self.depth != ColorDepth::None
    }
}

/// A data colour in 16 colours, by hue: nearest-RGB sends the brand's muted colours to grey
/// (helix, strand, target and reference all became colour 8). Pale or grey colours map to
/// white or grey; everything else to the chromatic colour of the nearest hue.
fn ansi16_for_data(c: ColorRGB) -> u8 {
    let (r, g, b) = (c.r as f32, c.g as f32, c.b as f32);
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    if max - min < 40.0 {
        return if max > 160.0 { 7 } else { 8 };
    }
    let d = max - min;
    let hue = if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    // Bins, not nearest hue: orange (under 40°) counts as red, so the AlphaFold <50 band
    // (orange) and 50–70 band (yellow) stay apart.
    match hue {
        h if h < 40.0 => 1,
        h if h < 90.0 => 3,
        h if h < 150.0 => 2,
        h if h < 210.0 => 6,
        h if h < 270.0 => 4,
        h if h < 330.0 => 5,
        _ => 1,
    }
}

/// The nearest of the 16 standard colours, by the xterm default palette.
#[cfg(test)]
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
        // Data colours stay apart in 16 colours (by nearest RGB they were all grey).
        use crate::brand::structure::{COIL, HELIX, REFERENCE, STRAND, TARGET};
        assert_eq!(nearest_ansi16(HELIX), 8, "the old mapping, for the record");
        assert_eq!(
            ansi16_for_data(HELIX),
            1,
            "Clay is an orange-tan: red in 16 colours"
        );
        assert_eq!(ansi16_for_data(STRAND), 6);
        assert_eq!(ansi16_for_data(COIL), 7);
        assert_ne!(ansi16_for_data(TARGET), ansi16_for_data(REFERENCE));
        use crate::rasterizer::shader::plddt_to_color as p;
        let bands: Vec<u8> = [95.0, 80.0, 60.0, 25.0]
            .map(|v| ansi16_for_data(p(v)))
            .to_vec();
        assert_eq!(bands, [4, 6, 3, 1], "pLDDT bands blue, cyan, yellow, red");
        let plain = Ansi {
            plain: true,
            ..Ansi::with_depth(ColorDepth::TrueColor)
        };
        let out = plain.paint(Role::Accent, "a") + &plain.bold("b") + &plain.paint(Role::Dim, "c");
        assert_eq!(out, "abc", "a pipe gets no escape sequences at all");
    }
}
