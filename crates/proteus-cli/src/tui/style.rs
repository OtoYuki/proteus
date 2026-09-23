//! The Proteus identity as ratatui styles, at the terminal's colour depth.
//!
//! Every colour on the home screen comes from a `brand` role through [`Look`]. In truecolor
//! and 256 colours the screen paints its own ground; in 16 colours it keeps the terminal's
//! and uses the ANSI colours by role; with `NO_COLOR` it uses bold, dim and reverse only.

use proteus_render::brand::{self, ColorDepth, Role, Theme};
use proteus_render::rasterizer::ColorRGB;
use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy)]
pub struct Look {
    pub depth: ColorDepth,
    pub theme: Theme,
    /// Motion at meaningful moments (the launch, a running job). `NO_MOTION` turns it off.
    pub motion: bool,
}

impl Look {
    pub fn detect() -> Self {
        Self {
            depth: ColorDepth::detect(),
            theme: brand::DARK,
            motion: std::env::var_os("NO_MOTION").is_none_or(|v| v.is_empty()),
        }
    }

    pub fn with_depth(depth: ColorDepth) -> Self {
        Self {
            depth,
            theme: brand::DARK,
            motion: true,
        }
    }

    fn rgb(&self, c: ColorRGB) -> Color {
        match self.depth {
            ColorDepth::TrueColor => Color::Rgb(c.r, c.g, c.b),
            ColorDepth::Ansi256 => Color::Indexed(brand::to_ansi256(c)),
            ColorDepth::Ansi16 | ColorDepth::None => Color::Reset,
        }
    }

    /// The foreground of a role.
    pub fn fg(&self, role: Role) -> Style {
        match self.depth {
            ColorDepth::None => match role {
                Role::Dim | Role::Line => Style::new().add_modifier(Modifier::DIM),
                Role::Accent | Role::Bad => Style::new().add_modifier(Modifier::BOLD),
                _ => Style::new(),
            },
            ColorDepth::Ansi16 => match role.ansi16() {
                Some(i) => Style::new().fg(Color::Indexed(i)),
                None => Style::new(),
            },
            _ => Style::new().fg(self.rgb(role.rgb(&self.theme))),
        }
    }

    /// The whole screen's base: the ground, where the depth allows painting it.
    pub fn base(&self) -> Style {
        if self.depth.paints_ground() {
            Style::new()
                .bg(self.rgb(self.theme.ground))
                .fg(self.rgb(self.theme.text))
        } else {
            Style::new()
        }
    }

    /// A raised panel (help overlay, the selected row).
    pub fn surface(&self) -> Style {
        if self.depth.paints_ground() {
            Style::new().bg(self.rgb(self.theme.surface))
        } else {
            Style::new().add_modifier(Modifier::REVERSED)
        }
    }

    /// The selected row: the surface plus bold, and an accent marker drawn by the caller.
    pub fn selected(&self) -> Style {
        if self.depth.paints_ground() {
            self.surface().add_modifier(Modifier::BOLD)
        } else {
            Style::new().add_modifier(Modifier::REVERSED | Modifier::BOLD)
        }
    }

    /// The Clay block cursor of the s1re.sh prompt.
    pub fn cursor(&self) -> Style {
        self.fg(Role::Warm)
    }

    pub fn text(&self) -> Style {
        self.fg(Role::Text)
    }
    pub fn muted(&self) -> Style {
        self.fg(Role::Muted)
    }
    pub fn dim(&self) -> Style {
        self.fg(Role::Dim)
    }
    pub fn line(&self) -> Style {
        self.fg(Role::Line)
    }
    pub fn accent(&self) -> Style {
        self.fg(Role::Accent)
    }
    pub fn warm(&self) -> Style {
        self.fg(Role::Warm)
    }
    pub fn sea(&self) -> Style {
        self.fg(Role::Sea)
    }
    pub fn bad(&self) -> Style {
        self.fg(Role::Bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_depth_maps_roles_the_documented_way() {
        let t = Look::with_depth(ColorDepth::TrueColor);
        assert_eq!(t.accent().fg, Some(Color::Rgb(0x99, 0x92, 0x0B)));
        assert!(t.base().bg.is_some());

        let c256 = Look::with_depth(ColorDepth::Ansi256);
        assert!(matches!(c256.accent().fg, Some(Color::Indexed(i)) if i >= 16));

        let c16 = Look::with_depth(ColorDepth::Ansi16);
        assert_eq!(c16.accent().fg, Some(Color::Indexed(3)));
        assert_eq!(
            c16.base(),
            Style::new(),
            "the terminal's own ground is kept"
        );

        let none = Look::with_depth(ColorDepth::None);
        for s in [
            none.accent(),
            none.bad(),
            none.dim(),
            none.base(),
            none.selected(),
        ] {
            assert_eq!(s.fg, None);
            assert_eq!(s.bg, None);
        }
    }
}
