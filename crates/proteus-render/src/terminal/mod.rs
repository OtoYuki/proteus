pub mod braille;
pub mod halfblock;
pub mod kitty;
pub mod sixel;

pub use braille::BrailleRenderer;
pub use halfblock::HalfBlockRenderer;
pub use kitty::KittyRenderer;
pub use sixel::SixelRenderer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalBackend {
    #[default]
    HalfBlock,
    Kitty,
    Braille,
    /// DEC Sixel: true pixels on xterm, mlterm, foot, contour, WezTerm and Windows Terminal,
    /// which the kitty protocol does not reach.
    Sixel,
}

impl std::str::FromStr for TerminalBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "halfblock" | "half-block" | "block" => Ok(Self::HalfBlock),
            "kitty" => Ok(Self::Kitty),
            "braille" => Ok(Self::Braille),
            "sixel" => Ok(Self::Sixel),
            other => Err(format!(
                "Unknown terminal backend: '{other}'. Expected halfblock, braille, sixel or kitty."
            )),
        }
    }
}
