pub mod dashboard;
pub mod viewer;

pub use dashboard::{DashboardData, DashboardPage, DashboardRenderer, DashboardView};
pub use viewer::{
    run_interactive_viewer, InterfaceColors, ScoreColors, TerminalFinding, ViewerConfig,
};
