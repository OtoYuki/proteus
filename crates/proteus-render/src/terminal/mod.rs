pub mod braille;
pub mod halfblock;
pub mod kitty;

pub use braille::BrailleRenderer;
pub use halfblock::HalfBlockRenderer;
pub use kitty::KittyRenderer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalBackend {
    #[default]
    HalfBlock,
    Kitty,
    Braille,
}

impl std::str::FromStr for TerminalBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "halfblock" | "half-block" | "block" => Ok(Self::HalfBlock),
            "kitty" => Ok(Self::Kitty),
            "braille" => Ok(Self::Braille),
            other => Err(format!(
                "Unknown terminal backend: '{other}'. Expected halfblock, kitty, or braille."
            )),
        }
    }
}
