//! The Proteus identity, defined once: palette roles, terminal colour depth, the mark and the
//! dot-matrix wordmark. Every surface (the home screen, the terminal viewer, the browser page,
//! the brand assets) reads from here. See `docs/design/2026-09-23-proteus-identity-design.md`,
//! whose measured numbers the tests below recompute.

pub mod ansi;
pub mod assets;
pub mod mark;
pub mod matrix;

use crate::rasterizer::buffer::ColorRGB;

const fn hex(v: u32) -> ColorRGB {
    ColorRGB::new((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

/// The s1re.sh identity v0.1 palette, as defined there.
pub mod palette {
    use super::{hex, ColorRGB};
    pub const ROOT: ColorRGB = hex(0x141C10);
    pub const CREAM: ColorRGB = hex(0xFBFFE1);
    pub const CHARTREUSE: ColorRGB = hex(0x99920B);
    pub const CLAY: ColorRGB = hex(0xD8A664);
    pub const MOSS: ColorRGB = hex(0x5A6042);
    pub const KHAKI: ColorRGB = hex(0xD1CF8B);
    /// Proteus's one addition: a cool data colour, so that structure colours stay apart for
    /// colour-blind readers (the palette alone gives ΔE 12.5 under deuteranopia).
    pub const TIDE: ColorRGB = hex(0x4F9A94);
}

/// Secondary-structure colours: Clay, Tide and a pale cream. The smallest pairwise ΔE after
/// simulating deutan, protan and tritan vision is at least 30 (asserted below).
pub mod structure {
    use super::{hex, palette, ColorRGB};
    pub const HELIX: ColorRGB = palette::CLAY;
    pub const STRAND: ColorRGB = palette::TIDE;
    pub const COIL: ColorRGB = hex(0xE7E9C8);
    /// Disulfide bonds.
    pub const DISULFIDE: ColorRGB = palette::CHARTREUSE;
    /// `view --compare`: the superposed target and its reference (ΔE 51 apart under
    /// deuteranopia).
    pub const TARGET: ColorRGB = palette::TIDE;
    pub const REFERENCE: ColorRGB = palette::CLAY;
}

/// Colours by role, for a dark or a light ground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub ground: ColorRGB,
    /// Panels, one step off the ground.
    pub surface: ColorRGB,
    /// Hairlines and borders; never text.
    pub line: ColorRGB,
    pub text: ColorRGB,
    pub muted: ColorRGB,
    pub dim: ColorRGB,
    /// The one loud colour: selection, the mark, emphasis.
    pub accent: ColorRGB,
    /// Cursor, running, highlights.
    pub warm: ColorRGB,
    pub sea: ColorRGB,
    /// Failure. Never on its own: always with a glyph and a word.
    pub bad: ColorRGB,
}

pub const DARK: Theme = Theme {
    ground: palette::ROOT,
    surface: hex(0x1B2516),
    line: palette::MOSS,
    text: palette::CREAM,
    muted: palette::KHAKI,
    dim: hex(0x9A9F80),
    accent: palette::CHARTREUSE,
    warm: palette::CLAY,
    sea: palette::TIDE,
    bad: hex(0xE0694A),
};

pub const LIGHT: Theme = Theme {
    ground: palette::CREAM,
    surface: hex(0xF3F6D6),
    line: palette::KHAKI,
    text: palette::ROOT,
    muted: palette::MOSS,
    dim: hex(0x646948),
    accent: hex(0x6B660A),
    warm: palette::CLAY,
    sea: hex(0x2F7771),
    bad: hex(0xB4472B),
};

/// The one-line signature under the lockup.
pub const SIGNATURE: &str = "a s1re.sh project";

/// How many colours the terminal can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    /// `NO_COLOR` is set: no colour escapes at all.
    None,
    /// The 16 ANSI colours, chosen by role.
    Ansi16,
    /// The xterm 256-colour palette.
    Ansi256,
    /// 24-bit colour.
    TrueColor,
}

impl ColorDepth {
    /// From the environment, by the conventions terminals use: `NO_COLOR` (no-color.org) wins;
    /// `COLORTERM=truecolor|24bit` means 24-bit; a `TERM` naming `256color` means 256.
    pub fn detect() -> Self {
        Self::from_env(|k| std::env::var(k).ok())
    }

    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Self {
        let term = get("TERM").unwrap_or_default();
        if get("NO_COLOR").is_some_and(|v| !v.is_empty()) || term == "dumb" {
            return Self::None;
        }
        let colorterm = get("COLORTERM").unwrap_or_default().to_ascii_lowercase();
        if colorterm == "truecolor" || colorterm == "24bit" {
            return Self::TrueColor;
        }
        // OpenSSH does not pass COLORTERM on, so over SSH a truecolor terminal is known only by
        // its TERM: the terminfo `-direct` and `-truecolor` variants, and the terminals that
        // name themselves and draw 24-bit colour.
        const TRUECOLOR_TERMS: [&str; 5] = ["kitty", "ghostty", "alacritty", "foot", "wezterm"];
        if term.ends_with("-direct")
            || term.contains("truecolor")
            || TRUECOLOR_TERMS.iter().any(|t| term.contains(t))
        {
            return Self::TrueColor;
        }
        if term.contains("256color") {
            return Self::Ansi256;
        }
        Self::Ansi16
    }

    /// Whether the home screen paints its own ground (it keeps the terminal's in 16 colours).
    pub fn paints_ground(self) -> bool {
        matches!(self, Self::TrueColor | Self::Ansi256)
    }
}

/// The nearest entry of the xterm 256-colour palette: the 6×6×6 cube (levels 0, 95, 135, 175,
/// 215, 255) or the 24-step grey ramp (8 + 10·i), whichever is closer in RGB.
pub fn to_ansi256(c: ColorRGB) -> u8 {
    const LEVELS: [i32; 6] = [0, 95, 135, 175, 215, 255];
    let nearest = |v: u8| {
        (0..6)
            .min_by_key(|&i| (LEVELS[i] - v as i32).abs())
            .unwrap_or(0)
    };
    let (ri, gi, bi) = (nearest(c.r), nearest(c.g), nearest(c.b));
    let cube = (LEVELS[ri], LEVELS[gi], LEVELS[bi]);
    let avg = (c.r as i32 + c.g as i32 + c.b as i32) / 3;
    let gi_ramp = ((avg - 8 + 5) / 10).clamp(0, 23);
    let grey = 8 + 10 * gi_ramp;
    let dist = |(r, g, b): (i32, i32, i32)| {
        (r - c.r as i32).pow(2) + (g - c.g as i32).pow(2) + (b - c.b as i32).pow(2)
    };
    if dist((grey, grey, grey)) < dist(cube) {
        232 + gi_ramp as u8
    } else {
        16 + 36 * ri as u8 + 6 * gi as u8 + bi as u8
    }
}

/// A role's colour in the 16-colour palette (standard ANSI indices 0–15). `None` keeps the
/// terminal's default foreground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Text,
    Muted,
    Dim,
    Line,
    Accent,
    Warm,
    Sea,
    Bad,
}

impl Role {
    pub fn rgb(self, t: &Theme) -> ColorRGB {
        match self {
            Role::Text => t.text,
            Role::Muted => t.muted,
            Role::Dim => t.dim,
            Role::Line => t.line,
            Role::Accent => t.accent,
            Role::Warm => t.warm,
            Role::Sea => t.sea,
            Role::Bad => t.bad,
        }
    }

    pub fn ansi16(self) -> Option<u8> {
        match self {
            Role::Text => None,
            Role::Muted => None,
            Role::Dim | Role::Line => Some(8),
            Role::Accent => Some(3),
            Role::Warm => Some(11),
            Role::Sea => Some(6),
            Role::Bad => Some(1),
        }
    }
}

/// The 16 ANSI colours of a terminal dressed in the identity, for a theme's ground: each slot a
/// role uses in 16 colours ([`Role::ansi16`]) holds that role's colour, and the rest stay within
/// the palette. The README recordings are rendered with it, and `docs/brand/` ships it as a
/// kitty theme.
pub fn terminal_palette(t: &Theme) -> [ColorRGB; 16] {
    let cream = structure::COIL;
    [
        t.surface, // black: one step off the ground, so "black" text never vanishes
        t.bad, t.muted, // green: Khaki, the palette's green-leaning neutral
        t.accent, t.sea,  // blue
        t.warm, // magenta
        t.sea, cream, // white
        t.dim, t.bad, t.muted, t.warm, t.sea, t.warm, t.sea, t.text,
    ]
}

/// WCAG 2.x relative luminance and contrast ratio.
pub fn contrast(a: ColorRGB, b: ColorRGB) -> f64 {
    let lum = |c: ColorRGB| {
        let [r, g, b] = [c.r, c.g, c.b].map(srgb_to_linear);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    };
    let (x, y) = (lum(a), lum(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

fn srgb_to_linear(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// `#rrggbb`.
pub fn css(c: ColorRGB) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
}

/// CSS custom properties for a theme, as the browser page declares them.
pub fn css_vars(t: &Theme) -> String {
    let pairs = [
        ("ground", t.ground),
        ("surface", t.surface),
        ("line", t.line),
        ("text", t.text),
        ("muted", t.muted),
        ("dim", t.dim),
        ("accent", t.accent),
        ("warm", t.warm),
        ("sea", t.sea),
        ("bad", t.bad),
    ];
    pairs
        .iter()
        .map(|(k, c)| format!("--{k}:{};", css(*c)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
    const DEUTAN: [[f64; 3]; 3] = [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ];
    const PROTAN: [[f64; 3]; 3] = [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ];
    const TRITAN: [[f64; 3]; 3] = [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ];

    fn lab(c: ColorRGB, m: Option<&[[f64; 3]; 3]>) -> [f64; 3] {
        let mut v = [c.r, c.g, c.b].map(srgb_to_linear);
        if let Some(m) = m {
            v = [0, 1, 2]
                .map(|i| (m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]).clamp(0.0, 1.0));
        }
        let [r, g, b] = v;
        let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
        let f = |t: f64| {
            if t > 0.008856 {
                t.cbrt()
            } else {
                7.787 * t + 16.0 / 116.0
            }
        };
        [
            116.0 * f(y) - 16.0,
            500.0 * (f(x) - f(y)),
            200.0 * (f(y) - f(z)),
        ]
    }

    fn delta_e(a: ColorRGB, b: ColorRGB, m: Option<&[[f64; 3]; 3]>) -> f64 {
        let (p, q) = (lab(a, m), lab(b, m));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    #[test]
    fn contrast_matches_the_published_wcag_examples() {
        assert!((contrast(ColorRGB::BLACK, ColorRGB::WHITE) - 21.0).abs() < 1e-9);
        assert!((contrast(palette::CREAM, palette::ROOT) - 17.03).abs() < 0.01);
        assert!((contrast(palette::MOSS, palette::ROOT) - 2.65).abs() < 0.01);
    }

    #[test]
    fn the_terminal_palette_gives_each_role_its_own_colour() {
        let p = terminal_palette(&DARK);
        for role in [
            Role::Dim,
            Role::Line,
            Role::Accent,
            Role::Warm,
            Role::Sea,
            Role::Bad,
        ] {
            let i = role.ansi16().unwrap() as usize;
            if role != Role::Line {
                assert_eq!(p[i], role.rgb(&DARK), "{role:?} at index {i}");
            }
        }
        for (i, c) in p.iter().enumerate().skip(1) {
            assert!(
                contrast(*c, DARK.ground) >= 3.0,
                "ANSI {i} {} is readable on the ground",
                css(*c)
            );
        }
    }

    #[test]
    fn every_text_role_is_readable_on_ground_and_surface() {
        for (name, t) in [("dark", DARK), ("light", LIGHT)] {
            for bg in [t.ground, t.surface] {
                for (role, c) in [("text", t.text), ("muted", t.muted), ("dim", t.dim)] {
                    let r = contrast(c, bg);
                    assert!(r >= 4.5, "{name} {role} on {}: {r:.2}", css(bg));
                }
                for (role, c) in [("accent", t.accent), ("sea", t.sea), ("bad", t.bad)] {
                    let r = contrast(c, bg);
                    assert!(r >= 4.5, "{name} {role} text on {}: {r:.2}", css(bg));
                }
            }
            assert!(
                contrast(t.warm, t.ground) >= 2.0,
                "{name} warm is at least visible"
            );
        }
    }

    #[test]
    fn structure_colours_stay_apart_for_colour_blind_readers() {
        let set = [structure::HELIX, structure::STRAND, structure::COIL];
        for (vision, m) in [
            ("normal", None),
            ("deutan", Some(&DEUTAN)),
            ("protan", Some(&PROTAN)),
            ("tritan", Some(&TRITAN)),
        ] {
            let mut worst = f64::MAX;
            for i in 0..3 {
                for j in i + 1..3 {
                    worst = worst.min(delta_e(set[i], set[j], m));
                }
            }
            assert!(worst >= 30.0, "{vision}: smallest ΔE {worst:.1}");
        }
        // And the palette alone would not have been enough (the reason Tide exists).
        let palette_only = [palette::CLAY, palette::CHARTREUSE, palette::KHAKI];
        let worst = (0..3)
            .flat_map(|i| (i + 1..3).map(move |j| (i, j)))
            .map(|(i, j)| delta_e(palette_only[i], palette_only[j], Some(&DEUTAN)))
            .fold(f64::MAX, f64::min);
        assert!(worst < 15.0, "{worst}");
    }

    #[test]
    fn colour_depth_follows_the_terminal_conventions() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(
            ColorDepth::from_env(env(&[("NO_COLOR", "1"), ("COLORTERM", "truecolor")])),
            ColorDepth::None
        );
        assert_eq!(
            ColorDepth::from_env(env(&[("NO_COLOR", ""), ("COLORTERM", "truecolor")])),
            ColorDepth::TrueColor
        );
        assert_eq!(
            ColorDepth::from_env(env(&[("COLORTERM", "24bit")])),
            ColorDepth::TrueColor
        );
        assert_eq!(
            ColorDepth::from_env(env(&[("TERM", "xterm-256color")])),
            ColorDepth::Ansi256
        );
        assert_eq!(
            ColorDepth::from_env(env(&[("TERM", "xterm")])),
            ColorDepth::Ansi16
        );
        assert_eq!(
            ColorDepth::from_env(env(&[("TERM", "dumb")])),
            ColorDepth::None
        );
        assert_eq!(ColorDepth::from_env(env(&[])), ColorDepth::Ansi16);
        assert_eq!(
            ColorDepth::from_env(env(&[("TERM", "dumb"), ("COLORTERM", "truecolor")])),
            ColorDepth::None,
            "a dumb terminal takes no escapes, whatever COLORTERM says"
        );
        for t in [
            "xterm-kitty",
            "xterm-ghostty",
            "alacritty",
            "foot",
            "wezterm",
            "xterm-direct",
            "st-truecolor",
        ] {
            let depth = ColorDepth::from_env(|k: &str| (k == "TERM").then(|| t.to_string()));
            assert_eq!(depth, ColorDepth::TrueColor, "{t} over SSH (no COLORTERM)");
        }
    }

    #[test]
    fn the_256_colour_mapping_picks_exact_entries_and_greys() {
        assert_eq!(to_ansi256(ColorRGB::new(0, 0, 0)), 16);
        assert_eq!(to_ansi256(ColorRGB::new(255, 255, 255)), 231);
        assert_eq!(to_ansi256(ColorRGB::new(95, 135, 175)), 16 + 36 + 2 * 6 + 3);
        assert_eq!(to_ansi256(ColorRGB::new(128, 128, 128)), 244);
        // Every brand colour lands within a visibly close entry (no hue jumps).
        for c in [
            palette::ROOT,
            palette::CREAM,
            palette::CHARTREUSE,
            palette::CLAY,
            palette::MOSS,
            palette::KHAKI,
            palette::TIDE,
        ] {
            let i = to_ansi256(c);
            assert!(i >= 16, "{}", css(c));
        }
    }

    #[test]
    fn css_vars_carry_every_role() {
        let v = css_vars(&DARK);
        assert!(v.contains("--ground:#141c10;"));
        assert!(v.contains("--accent:#99920b;"));
        assert_eq!(v.matches("--").count(), 10);
    }
}
