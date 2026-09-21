use proteus_core::models::BiophysicalMetrics;
use proteus_core::structure::RamachandranRegion;
use std::fmt::Write;

/// Biophysical data bundle used to render the live side-panel dashboard.
#[derive(Debug, Clone)]
pub struct DashboardData {
    pub title: String,
    pub num_residues: usize,
    pub num_disulfides: usize,
    pub metrics: Option<BiophysicalMetrics>,
    pub plddts: Vec<f64>,
    pub ramachandran_points: Vec<(Option<f64>, Option<f64>, RamachandranRegion)>,
}

/// Computes visible character width of a string, ignoring ANSI escape sequences.
pub fn visible_width(s: &str) -> usize {
    let mut in_escape = false;
    let mut count = 0;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' || c == 'H' || c == 'J' || c == 'K' {
                in_escape = false;
            }
        } else {
            count += 1;
        }
    }
    count
}

/// Pads a formatted string with spaces until its visible width matches target_width.
pub fn pad_to_width(s: &str, target_width: usize) -> String {
    let vis = visible_width(s);
    if vis >= target_width {
        s.to_string()
    } else {
        let mut padded = String::with_capacity(s.len() + (target_width - vis));
        padded.push_str(s);
        for _ in 0..(target_width - vis) {
            padded.push(' ');
        }
        padded
    }
}

#[derive(Default)]
pub struct DashboardRenderer;

impl DashboardRenderer {
    pub fn new() -> Self {
        Self
    }

    /// Render all dashboard sections and append positioned ANSI output into `out_buf`.
    pub fn render_to_buffer(
        &self,
        data: &DashboardData,
        out: &mut String,
        screen_offset_col: u16,
        screen_offset_row: u16,
        width: usize,
        height: usize,
    ) {
        if width < 20 || height < 10 {
            return;
        }

        let lines = self.generate_lines(data, width, height);
        for (i, line) in lines.iter().enumerate().take(height) {
            let row = screen_offset_row + i as u16 + 1;
            let col = screen_offset_col + 1;
            let padded = pad_to_width(line, width);
            let _ = write!(out, "\x1b[{row};{col}H{padded}\x1b[0m");
        }
    }

    /// Generate the formatted list of lines for the dashboard.
    pub fn generate_lines(&self, data: &DashboardData, width: usize, height: usize) -> Vec<String> {
        let mut lines = Vec::with_capacity(height);

        // Header Title
        let header_title = format!(" {} ({} res) ", data.title, data.num_residues);
        let header_bar_len = width.saturating_sub(header_title.len() + 2);
        lines.push(format!(
            "\x1b[1;36m┌─{}\x1b[0m\x1b[38;5;240m{:─<w$}┐\x1b[0m",
            header_title,
            "",
            w = header_bar_len
        ));

        // Adaptive vertical layout
        let (plot_h, has_telemetry) = if height >= 30 {
            (11, true)
        } else if height >= 24 {
            (8, true)
        } else if height >= 18 {
            (6, true)
        } else {
            (5, false)
        };

        // 1. Ramachandran Section
        self.render_ramachandran_section(data, width, plot_h, &mut lines);

        // 2. pLDDT Confidence Profile
        let remaining = height.saturating_sub(lines.len());
        if remaining >= 5 {
            self.render_plddt_section(data, width, &mut lines);
        }

        // 3. Biophysical Telemetry Card
        let remaining = height.saturating_sub(lines.len());
        if has_telemetry && remaining >= 4 {
            self.render_telemetry_section(data, width, remaining, &mut lines);
        }

        // Pad with empty rows to fill height
        while lines.len() < height {
            lines.push(String::new());
        }

        lines
    }

    fn render_ramachandran_section(
        &self,
        data: &DashboardData,
        width: usize,
        plot_h: usize,
        lines: &mut Vec<String>,
    ) {
        let plot_w = width.saturating_sub(8).clamp(16, 36);

        // Section header
        let title = " Ramachandran (φ, ψ) ";
        let bar_len = width.saturating_sub(title.len() + 2);
        lines.push(format!(
            "\x1b[1;35m├─{}\x1b[0m\x1b[38;5;240m{:─<w$}┤\x1b[0m",
            title,
            "",
            w = bar_len
        ));

        // Precompute residue points on the grid
        let mut grid_markers = vec![None; plot_w * plot_h];
        for (phi_opt, psi_opt, region) in &data.ramachandran_points {
            if let (Some(phi), Some(psi)) = (*phi_opt, *psi_opt) {
                let gx = (((phi - (-180.0)) / 360.0) * (plot_w as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_w - 1) as f64) as usize;
                let gy = (((180.0 - psi) / 360.0) * (plot_h as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_h - 1) as f64) as usize;

                let idx = gy * plot_w + gx;
                let (sym, col) = match region {
                    RamachandranRegion::Outlier => ('▲', "\x1b[1;31m"),
                    RamachandranRegion::Allowed => ('●', "\x1b[1;33m"),
                    _ => ('●', "\x1b[1;32m"),
                };

                // Prioritize outliers over allowed over favored in cell overlap
                if let Some((prev_sym, _)) = grid_markers[idx] {
                    if prev_sym == '▲' {
                        continue;
                    }
                    if prev_sym == '●' && sym != '▲' {
                        continue;
                    }
                }
                grid_markers[idx] = Some((sym, col));
            }
        }

        let mid_x = (plot_w - 1) / 2;
        let mid_y = (plot_h - 1) / 2;

        // Render grid rows
        for r in 0..plot_h {
            let left_label = if r == 0 {
                " +180°│"
            } else if r == mid_y / 2 {
                "  +90°│"
            } else if r == mid_y {
                "    0°├"
            } else if r == mid_y + (plot_h - 1 - mid_y) / 2 {
                "  -90°│"
            } else if r == plot_h - 1 {
                " -180°│"
            } else {
                "      │"
            };

            let right_border = if r == mid_y { "┤" } else { "│" };

            let mut row_str = String::with_capacity(plot_w * 4 + 10);
            row_str.push_str("\x1b[38;5;244m");
            row_str.push_str(left_label);
            row_str.push_str("\x1b[0m");

            let psi_c = 180.0 - (r as f64 / (plot_h - 1) as f64) * 360.0;

            for c in 0..plot_w {
                let idx = r * plot_w + c;
                if let Some((sym, col)) = grid_markers[idx] {
                    let _ = write!(row_str, "{col}{sym}\x1b[0m");
                } else {
                    let phi_c = -180.0 + (c as f64 / (plot_w - 1) as f64) * 360.0;

                    if c == mid_x && r == mid_y {
                        row_str.push_str("\x1b[38;5;240m┼\x1b[0m");
                    } else if c == mid_x {
                        row_str.push_str("\x1b[38;5;240m│\x1b[0m");
                    } else if r == mid_y {
                        row_str.push_str("\x1b[38;5;240m─\x1b[0m");
                    } else if (-180.0..=-45.0).contains(&phi_c)
                        && (psi_c >= 90.0 || psi_c <= -150.0)
                    {
                        // Core beta sheet basin
                        row_str.push_str("\x1b[38;5;238m·\x1b[0m");
                    } else if (-100.0..=-30.0).contains(&phi_c) && (-70.0..=-10.0).contains(&psi_c)
                    {
                        // Core alpha helix basin
                        row_str.push_str("\x1b[38;5;238m·\x1b[0m");
                    } else if (30.0..=90.0).contains(&phi_c) && (10.0..=70.0).contains(&psi_c) {
                        // Left-handed helix basin
                        row_str.push_str("\x1b[38;5;238m·\x1b[0m");
                    } else {
                        row_str.push(' ');
                    }
                }
            }

            row_str.push_str("\x1b[38;5;244m");
            row_str.push_str(right_border);
            row_str.push_str("\x1b[0m");

            lines.push(row_str);
        }

        // Horizontal axis footer
        let mut axis_bar = String::from("      └───");
        for c in 4..plot_w {
            if c == mid_x {
                axis_bar.push('┴');
            } else {
                axis_bar.push('─');
            }
        }
        axis_bar.push('┘');
        lines.push(format!("\x1b[38;5;244m{axis_bar}\x1b[0m"));

        // Axis scale markers
        lines.push(format!(
            "       \x1b[38;5;242m-180°{: <pad$}0°{: >pad2$}+180°\x1b[0m",
            "",
            "",
            pad = (plot_w / 2).saturating_sub(6),
            pad2 = (plot_w / 2).saturating_sub(5)
        ));

        // Conformation summary stats
        if let Some(rama) = data
            .metrics
            .as_ref()
            .and_then(|m| m.ramachandran_stats.as_ref())
        {
            lines.push(format!(
                " \x1b[32mFav: {:.1}%\x1b[0m │ \x1b[33mAll: {:.1}%\x1b[0m │ \x1b[31mOutliers: {}\x1b[0m",
                rama.favored_fraction * 100.0,
                rama.allowed_fraction * 100.0,
                rama.outlier_count
            ));
        }
    }

    fn render_plddt_section(&self, data: &DashboardData, width: usize, lines: &mut Vec<String>) {
        let title = " pLDDT Confidence Profile ";
        let bar_len = width.saturating_sub(title.len() + 2);
        lines.push(format!(
            "\x1b[1;34m├─{}\x1b[0m\x1b[38;5;240m{:─<w$}┤\x1b[0m",
            title,
            "",
            w = bar_len
        ));

        if let Some(plddt_dist) = data.metrics.as_ref().map(|m| &m.plddt_distribution) {
            lines.push(format!(
                " Mean: \x1b[1m{:.1}\x1b[0m │ Med: \x1b[1m{:.1}\x1b[0m │ ≥70: \x1b[32m{:.1}%\x1b[0m",
                plddt_dist.mean,
                plddt_dist.median,
                plddt_dist.high_confidence_fraction * 100.0
            ));
        }

        // Resampled per-residue pLDDT bar
        let n_res = data.plddts.len();
        if n_res > 0 {
            let bar_w = width.saturating_sub(4).clamp(10, 44);
            let mut bar_str = String::with_capacity(bar_w * 20);
            bar_str.push(' ');

            for col in 0..bar_w {
                let start_idx = col * n_res / bar_w;
                let end_idx = ((col + 1) * n_res / bar_w).max(start_idx + 1).min(n_res);

                let slice_plddts = &data.plddts[start_idx..end_idx];
                let avg = if !slice_plddts.is_empty() {
                    slice_plddts.iter().sum::<f64>() / slice_plddts.len() as f64
                } else {
                    70.0
                };

                // Color code: Very High (Blue), High (Cyan), Low (Yellow), Very Low (Orange/Red)
                let block_color = if avg >= 90.0 {
                    "\x1b[38;2;30;64;175m" // Deep Blue
                } else if avg >= 70.0 {
                    "\x1b[38;2;56;189;248m" // Cyan
                } else if avg >= 50.0 {
                    "\x1b[38;2;250;204;21m" // Yellow
                } else {
                    "\x1b[38;2;239;68;68m" // Red
                };

                let _ = write!(bar_str, "{block_color}█\x1b[0m");
            }

            lines.push(bar_str);

            // Sequence range indices
            lines.push(format!(
                " \x1b[38;5;242m1{: <pad$}{}\x1b[0m",
                "",
                n_res,
                pad = bar_w.saturating_sub(format!("{n_res}").len() + 1)
            ));
        }
    }

    fn render_telemetry_section(
        &self,
        data: &DashboardData,
        width: usize,
        max_rows: usize,
        lines: &mut Vec<String>,
    ) {
        let title = " Biophysical Telemetry ";
        let bar_len = width.saturating_sub(title.len() + 2);
        lines.push(format!(
            "\x1b[1;33m├─{}\x1b[0m\x1b[38;5;240m{:─<w$}┤\x1b[0m",
            title,
            "",
            w = bar_len
        ));

        let mut row_count = 1;

        if let Some(ref m) = data.metrics {
            if row_count < max_rows {
                lines.push(format!(
                    " Radius of Gyration (Rg) : \x1b[1m{:.3} Å\x1b[0m",
                    m.radius_of_gyration
                ));
                row_count += 1;
            }

            if row_count < max_rows {
                lines.push(format!(
                    " Contact Density (≤8Å)   : \x1b[1m{:.1}%\x1b[0m (Cα)",
                    m.contact_density * 100.0
                ));
                row_count += 1;
            }

            if let Some(ref sasa) = m.sasa_metrics {
                if row_count < max_rows {
                    lines.push(format!(
                        " SASA Surface Area       : \x1b[1m{:.1} Å²\x1b[0m",
                        sasa.total_sasa
                    ));
                    row_count += 1;
                }
                if row_count < max_rows {
                    lines.push(format!(
                        " Hydrophobic Core Burial : \x1b[1m{:.1}%\x1b[0m",
                        sasa.hydrophobic_burial_ratio * 100.0
                    ));
                    row_count += 1;
                }
            }

            if let Some(ref ss) = m.secondary_structure_summary {
                if row_count < max_rows {
                    lines.push(format!(
                        " 2° Structure : \x1b[35mα {:.0}%\x1b[0m │ \x1b[33mβ {:.0}%\x1b[0m │ \x1b[37mCoil {:.0}%\x1b[0m",
                        ss.helix_fraction * 100.0,
                        ss.strand_fraction * 100.0,
                        ss.coil_fraction * 100.0
                    ));
                    row_count += 1;
                }
            }

            if data.num_disulfides > 0 && row_count < max_rows {
                lines.push(format!(
                    " Disulfide Bridges (S-S) : \x1b[1;33m{} covalent pairs\x1b[0m",
                    data.num_disulfides
                ));
                row_count += 1;
            }

            if let Some(ref clash) = m.clash_stats {
                if row_count < max_rows {
                    let score_color = if clash.clashscore < 5.0 {
                        "\x1b[1;32m"
                    } else if clash.clashscore < 15.0 {
                        "\x1b[1;33m"
                    } else {
                        "\x1b[1;31m"
                    };
                    lines.push(format!(
                        " Clashscore (>0.4Å)     : {score_color}{:.1}\x1b[0m ({} overlaps)",
                        clash.clashscore, clash.clash_count
                    ));
                    row_count += 1;
                }
            }

            if let Some(fitness) = m.candidate_fitness_score {
                if row_count < max_rows {
                    lines.push(format!(
                        " Candidate Fitness Score : \x1b[1;32m{:.1} / 100\x1b[0m",
                        fitness
                    ));
                }
            }
        } else {
            lines.push(" [Biophysical analysis calculating...]".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::models::PlddtDistribution;
    use proteus_core::structure::RamachandranStats;

    #[test]
    fn test_visible_width_and_padding() {
        let plain = "Hello, world!";
        assert_eq!(visible_width(plain), 13);

        let styled = "\x1b[1;32mHello\x1b[0m, \x1b[38;2;255;0;0mworld\x1b[0m!";
        assert_eq!(visible_width(styled), 13);

        let padded = pad_to_width(styled, 20);
        assert_eq!(visible_width(&padded), 20);
        assert!(padded.ends_with("       "));
    }

    #[test]
    fn test_dashboard_renderer_generation() {
        let data = DashboardData {
            title: "1CRN".to_string(),
            num_residues: 46,
            num_disulfides: 3,
            metrics: Some(BiophysicalMetrics {
                id: uuid::Uuid::new_v4(),
                prediction_id: uuid::Uuid::new_v4(),
                radius_of_gyration: 9.762,
                rmsd_to_reference: None,
                contact_density: 0.148,
                plddt_distribution: PlddtDistribution {
                    mean: 91.2,
                    median: 93.4,
                    high_confidence_fraction: 0.957,
                    very_high_confidence_fraction: 0.782,
                },
                confidence_source: Default::default(),
                secondary_structure_summary: None,
                ramachandran_stats: Some(RamachandranStats {
                    favored_fraction: 0.955,
                    allowed_fraction: 0.045,
                    outlier_fraction: 0.0,
                    outlier_count: 0,
                    total_evaluated: 44,
                }),
                clash_stats: None,
                sasa_metrics: None,
                interaction_network: None,
                candidate_fitness_score: Some(87.4),
            }),
            plddts: vec![92.0; 46],
            ramachandran_points: vec![
                (Some(-60.0), Some(-45.0), RamachandranRegion::Favored),
                (Some(-120.0), Some(135.0), RamachandranRegion::Favored),
            ],
        };

        let renderer = DashboardRenderer::new();
        let lines = renderer.generate_lines(&data, 45, 30);

        assert_eq!(lines.len(), 30);
        assert!(lines[0].contains("1CRN"));
        assert!(lines[0].contains("46 res"));

        let full_text = lines.join("\n");
        assert!(full_text.contains("Ramachandran"));
        assert!(full_text.contains("pLDDT"));
        assert!(full_text.contains("Biophysical Telemetry"));
        assert!(full_text.contains("9.762"));
    }
}
