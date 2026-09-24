pub mod dashboard;
pub mod viewer;

pub use dashboard::{DashboardData, DashboardRenderer};
pub use viewer::{run_interactive_viewer, ScoreColors, TerminalFinding, ViewerConfig};
