//! The command-line surface: the parser, the shared value enums and the small helpers
//! every subcommand needs. One module per subcommand lives in [`crate::cmd`].

use crate::cmd;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use proteus_core::models::PipelineTier;
use proteus_engine::oci::OciRunner;
use proteus_engine::simulated::SimulatedRunner;
use proteus_engine::{AutoRunner, ComputeRunner, EsmApiRunner};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "proteus")]
#[command(
    about = "The protein-engineering design loop in one binary: mutate, fold, analyse, view, and serve GA4GH TES",
    long_about = None
)]
#[command(version)]
pub struct Cli {
    /// None only for bare `proteus`, which opens the home screen in a terminal.
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ConfidenceSourceArg {
    Auto,
    Predicted,
    Experimental,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Fold one sequence as a tracked job (prints its id)
    Submit(cmd::submit::Args),
    /// State, tier and timings of a job
    Status(cmd::status::Args),
    /// Full biophysical report of a finished job
    Inspect(cmd::inspect::Args),
    /// Quality metrics of structure files: one full report, or a table over many (Parquet/CSV/JSON)
    Analyze(cmd::analyze::Args),
    /// Cartoon ribbon in the terminal (half-block, Braille, Sixel or kitty graphics)
    View(cmd::view::Args),
    /// Write a variant library: alanine scan or site-saturation mutagenesis
    Mutate(cmd::mutate::Args),
    /// Fold a variant library, analyse every model, rank, and export
    Screen(cmd::screen::Args),
    /// ESM-2 zero-shot mutation scores and in-silico deep mutational scans
    Esm(cmd::esm::Args),
    /// Run the daemon: GA4GH TES 1.1 server and job API
    Serve(cmd::serve::Args),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum ExecutorMode {
    Container,
    Host,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum CliTier {
    Fast,
    Sota,
    Full,
}

impl From<CliTier> for PipelineTier {
    fn from(t: CliTier) -> Self {
        match t {
            CliTier::Fast => PipelineTier::FastScreening,
            CliTier::Sota => PipelineTier::HighFidelity,
            CliTier::Full => PipelineTier::FullValidation,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum CliMutagenesisMode {
    #[value(name = "alanine")]
    Alanine,
    #[value(name = "saturation")]
    Saturation,
}

impl From<CliMutagenesisMode> for proteus_core::mutagenesis::MutagenesisMode {
    fn from(m: CliMutagenesisMode) -> Self {
        match m {
            CliMutagenesisMode::Alanine => {
                proteus_core::mutagenesis::MutagenesisMode::AlanineScanning
            }
            CliMutagenesisMode::Saturation => {
                proteus_core::mutagenesis::MutagenesisMode::Saturation
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum CliBackend {
    #[value(name = "halfblock", alias = "half-block")]
    HalfBlock,
    #[value(name = "braille")]
    Braille,
    #[value(name = "sixel")]
    Sixel,
    #[value(name = "kitty")]
    Kitty,
}

impl From<CliBackend> for proteus_render::terminal::TerminalBackend {
    fn from(b: CliBackend) -> Self {
        match b {
            CliBackend::HalfBlock => proteus_render::terminal::TerminalBackend::HalfBlock,
            CliBackend::Braille => proteus_render::terminal::TerminalBackend::Braille,
            CliBackend::Sixel => proteus_render::terminal::TerminalBackend::Sixel,
            CliBackend::Kitty => proteus_render::terminal::TerminalBackend::Kitty,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum CliColorScheme {
    #[value(name = "plddt")]
    Plddt,
    #[value(name = "ss", alias = "secondary-structure")]
    SecondaryStructure,
    #[value(name = "rainbow")]
    Rainbow,
}

impl From<CliColorScheme> for proteus_render::rasterizer::ColorScheme {
    fn from(c: CliColorScheme) -> Self {
        match c {
            CliColorScheme::Plddt => proteus_render::rasterizer::ColorScheme::Plddt,
            CliColorScheme::SecondaryStructure => {
                proteus_render::rasterizer::ColorScheme::SecondaryStructure
            }
            CliColorScheme::Rainbow => proteus_render::rasterizer::ColorScheme::Rainbow,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum RunnerMode {
    Auto,
    Oci,
    Simulated,
    EsmApi,
}

pub fn resolve_runner(mode: RunnerMode) -> Result<Arc<dyn ComputeRunner>> {
    match mode {
        RunnerMode::Auto => Ok(Arc::new(AutoRunner::new())),
        RunnerMode::Simulated => Ok(Arc::new(SimulatedRunner::new())),
        RunnerMode::EsmApi => Ok(Arc::new(EsmApiRunner::new())),
        RunnerMode::Oci => {
            let oci = OciRunner::new().context("Failed to initialize OCI container runner")?;
            Ok(Arc::new(oci))
        }
    }
}

/// Whether a structure from `engine` may enter the screening leaderboard. The offline
/// simulator writes an ideal helix that does not depend on the sequence, so its output is only
/// ranked when the user asked for the simulator explicitly; a silent fallback is dropped.
pub fn rankable(engine: &str, runner: RunnerMode) -> bool {
    engine != proteus_engine::ENGINE_SIMULATED || runner == RunnerMode::Simulated
}

/// Fit a table to where it is going. On a terminal, cells wrap inside the table so it is never
/// wider than the window (a table wider than the terminal is hard-wrapped by the terminal and
/// its borders fall apart). Into a pipe or a file nothing wraps, so rows stay whole for grep.
pub fn fit_table(table: &mut comfy_table::Table) {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        table.set_content_arrangement(comfy_table::ContentArrangement::Dynamic);
        if let Ok((cols, _)) = crossterm::terminal::size() {
            if cols >= 40 {
                table.set_width(cols);
            }
        }
    } else {
        table.set_content_arrangement(comfy_table::ContentArrangement::Disabled);
    }
}

/// A brand role as an indicatif/console style token (`{bar:40.<token>}`), at the terminal's
/// colour depth: `#rrggbb`, a 256-colour index, or an ANSI name. `console` turns colour off
/// itself under `NO_COLOR` and in a pipe.
pub fn tint(role: proteus_render::brand::Role) -> String {
    use proteus_render::brand::{self, ColorDepth, Role};
    let c = role.rgb(&brand::DARK);
    match ColorDepth::detect() {
        ColorDepth::TrueColor => brand::css(c),
        ColorDepth::Ansi256 => brand::to_ansi256(c).to_string(),
        _ => match role {
            Role::Accent => "yellow",
            Role::Warm => "yellow.bright",
            Role::Sea => "cyan",
            Role::Bad => "red",
            // Bright black (grey), as brand::ansi uses: plain black vanishes on a dark terminal.
            Role::Dim | Role::Line => "black.bright",
            Role::Text | Role::Muted => "white",
        }
        .to_string(),
    }
}

/// `$PROTEUS_DATA_DIR` if set (containers, CI), else `~/.local/share/proteus`.
pub fn get_default_data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("PROTEUS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    dirs_next_or_home().join(".local/share/proteus")
}

pub fn dirs_next_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulated_structures_rank_only_when_the_simulator_was_requested() {
        assert!(!rankable(
            proteus_engine::ENGINE_SIMULATED,
            RunnerMode::Auto
        ));
        assert!(!rankable(
            proteus_engine::ENGINE_SIMULATED,
            RunnerMode::EsmApi
        ));
        assert!(rankable(
            proteus_engine::ENGINE_SIMULATED,
            RunnerMode::Simulated
        ));
        assert!(rankable(
            proteus_engine::ENGINE_ESMFOLD_API,
            RunnerMode::Auto
        ));
        assert!(rankable(proteus_engine::ENGINE_OCI, RunnerMode::Auto));
    }
}
