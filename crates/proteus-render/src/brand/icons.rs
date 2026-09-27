//! Icons: Material Design glyphs from the Nerd Fonts set, which kitty, WezTerm and Ghostty
//! draw from fonts they ship, whatever the user's font. Elsewhere there is no telling whether
//! the font has them, so the default there is no icon at all rather than a box.

/// An icon of the home screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    /// Jobs: a flask, an experiment run.
    Jobs,
    /// Structures: a molecule.
    Structures,
    /// Run: a play mark.
    Run,
}

impl Icon {
    pub const ALL: [Icon; 3] = [Icon::Jobs, Icon::Structures, Icon::Run];

    /// The glyph (`nf-md-flask_outline`, `nf-md-molecule`, `nf-md-play`).
    pub fn glyph(self) -> char {
        match self {
            Icon::Jobs => '\u{F0096}',
            Icon::Structures => '\u{F0BAC}',
            Icon::Run => '\u{F040A}',
        }
    }
}

/// Whether to draw icons: `PROTEUS_ICONS=nerd` or `none` decides; otherwise yes in a terminal
/// that ships the glyphs (kitty, WezTerm, Ghostty).
pub fn available() -> bool {
    decide(
        std::env::var("PROTEUS_ICONS").ok().as_deref(),
        std::env::var("TERM").ok().as_deref(),
        std::env::var("TERM_PROGRAM").ok().as_deref(),
        std::env::var_os("KITTY_WINDOW_ID").is_some(),
    )
}

fn decide(setting: Option<&str>, term: Option<&str>, program: Option<&str>, kitty: bool) -> bool {
    match setting.map(str::trim) {
        Some("nerd" | "on" | "1") => return true,
        Some("none" | "off" | "0") => return false,
        _ => {}
    }
    kitty
        || matches!(term, Some("xterm-kitty" | "xterm-ghostty" | "wezterm"))
        || matches!(program, Some("WezTerm" | "ghostty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icons_only_where_the_glyphs_ship_or_when_asked() {
        assert!(decide(None, Some("xterm-kitty"), None, false));
        assert!(
            decide(None, Some("xterm-256color"), None, true),
            "kitty inside tmux"
        );
        assert!(decide(None, None, Some("WezTerm"), false));
        assert!(!decide(None, Some("xterm-256color"), None, false));
        assert!(decide(Some("nerd"), Some("linux"), None, false));
        assert!(!decide(Some("none"), Some("xterm-kitty"), None, true));
        let all: Vec<char> = Icon::ALL.iter().map(|i| i.glyph()).collect();
        assert_eq!(all.len(), 3);
    }
}
