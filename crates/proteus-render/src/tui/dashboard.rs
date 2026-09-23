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

/// Split `s` into ANSI escape sequences (`true`) and runs of visible text (`false`).
///
/// A CSI sequence (`ESC [ … final`) ends at its final byte (`@` to `~`); any other escape is
/// taken as `ESC` plus one character.
fn ansi_segments(s: &str) -> impl Iterator<Item = (bool, &str)> {
    let mut rest = s;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        if let Some(after) = rest.strip_prefix('\x1b') {
            let len = if let Some(params) = after.strip_prefix('[') {
                match params.find(|c: char| ('@'..='~').contains(&c)) {
                    Some(i) => 2 + i + 1,
                    None => rest.len(),
                }
            } else {
                1 + after.chars().next().map_or(0, char::len_utf8)
            };
            let (esc, tail) = rest.split_at(len);
            rest = tail;
            Some((true, esc))
        } else {
            let len = rest.find('\x1b').unwrap_or(rest.len());
            let (text, tail) = rest.split_at(len);
            rest = tail;
            Some((false, text))
        }
    })
}

/// Terminal columns `s` occupies: display width (a CJK character is two columns, a combining
/// mark none), ignoring ANSI escape sequences.
pub fn visible_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    ansi_segments(s)
        .filter(|(esc, _)| !esc)
        .map(|(_, text)| text.width())
        .sum()
}

/// Cut `s` to at most `max_width` terminal columns, keeping its escape sequences intact. A
/// line wider than the terminal wraps, and on the bottom row that scrolls the whole screen, so
/// every positioned line has to go through this.
pub fn truncate_to_width(s: &str, max_width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if visible_width(s) <= max_width {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut used = 0usize;
    let mut styled = false;
    'segments: for (esc, part) in ansi_segments(s) {
        if esc {
            out.push_str(part);
            styled = true;
            continue;
        }
        for c in part.chars() {
            let w = c.width().unwrap_or(0);
            if used + w > max_width {
                break 'segments;
            }
            used += w;
            out.push(c);
        }
    }
    if styled {
        out.push_str("\x1b[0m");
    }
    out
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

/// Exactly `width` terminal columns: truncated if longer, space-padded if shorter.
pub fn fit_to_width(s: &str, width: usize) -> String {
    pad_to_width(&truncate_to_width(s, width), width)
}

/// A section rule `{left}─{title}───…{right}` exactly `width` columns wide (the title is cut
/// if it does not fit).
fn section_rule(style: &str, left: char, title: &str, right: char, width: usize) -> String {
    let title = truncate_to_width(title, width.saturating_sub(3));
    let bar_len = width.saturating_sub(visible_width(&title) + 3);
    format!(
        "{style}{left}─{title}\x1b[0m\x1b[38;5;240m{:─<bar_len$}{right}\x1b[0m",
        ""
    )
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
            let fitted = fit_to_width(line, width);
            let _ = write!(out, "\x1b[{row};{col}H{fitted}\x1b[0m");
        }
    }

    /// Generate the formatted list of lines for the dashboard.
    pub fn generate_lines(&self, data: &DashboardData, width: usize, height: usize) -> Vec<String> {
        let mut lines = Vec::with_capacity(height);

        // Header Title
        let header_title = format!(" {} ({} res) ", data.title, data.num_residues);
        lines.push(section_rule("\x1b[1;36m", '┌', &header_title, '┐', width));

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

        // Pad with empty rows to fill height, and never let a line run past the panel: a
        // plot row at the minimum plot width is wider than a narrow panel.
        lines.truncate(height);
        while lines.len() < height {
            lines.push(String::new());
        }
        for line in &mut lines {
            *line = truncate_to_width(line, width);
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
        lines.push(section_rule(
            "\x1b[1;35m",
            '├',
            " Ramachandran (φ, ψ) ",
            '┤',
            width,
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
        let predicted = data
            .metrics
            .as_ref()
            .map(|m| m.plddt().is_some())
            .unwrap_or(true);
        let title = if predicted {
            " pLDDT Confidence Profile "
        } else {
            " B-factor Profile (experimental; no pLDDT) "
        };
        lines.push(section_rule("\x1b[1;34m", '├', title, '┤', width));

        if let Some(m) = data.metrics.as_ref() {
            match m.plddt() {
                Some(plddt_dist) => lines.push(format!(
                    " Mean: \x1b[1m{:.1}\x1b[0m │ Med: \x1b[1m{:.1}\x1b[0m │ ≥70: \x1b[32m{:.1}%\x1b[0m",
                    plddt_dist.mean,
                    plddt_dist.median,
                    plddt_dist.high_confidence_fraction * 100.0
                )),
                None => lines.push(format!(
                    " Mean B: \x1b[1m{:.1} Å²\x1b[0m │ Med: \x1b[1m{:.1}\x1b[0m │ \x1b[38;5;242mnot a confidence\x1b[0m",
                    m.plddt_distribution.mean, m.plddt_distribution.median
                )),
            }
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

                // Color code: Very High (Blue), High (Cyan), Low (Yellow), Very Low (Orange/Red).
                // Experimental B-factors: a neutral grey ramp scaled to the structure's own range.
                let block_color = if !predicted {
                    let (lo, hi) = data
                        .plddts
                        .iter()
                        .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                    let t = if hi > lo { (avg - lo) / (hi - lo) } else { 0.5 };
                    let g = 90 + (t * 140.0) as u8;
                    let _ = write!(bar_str, "\x1b[38;2;{g};{g};{g}m█\x1b[0m");
                    continue;
                } else if avg >= 90.0 {
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
        lines.push(section_rule(
            "\x1b[1;33m",
            '├',
            " Biophysical Telemetry ",
            '┤',
            width,
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

            if let Some(ref clash) = m.steric_overlap {
                if row_count < max_rows {
                    let score_color = if clash.heavy_atom_overlap_score < 5.0 {
                        "\x1b[1;32m"
                    } else if clash.heavy_atom_overlap_score < 15.0 {
                        "\x1b[1;33m"
                    } else {
                        "\x1b[1;31m"
                    };
                    lines.push(format!(
                        " Clash/1k (no H)        : {score_color}{:.1}\x1b[0m ({} overlaps)",
                        clash.heavy_atom_overlap_score, clash.clash_count
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

    /// Display width, not bytes or chars: `Å`/`φ` are two bytes and one column, a CJK
    /// character is one char and two columns.
    #[test]
    fn widths_are_display_columns_and_truncation_keeps_escapes() {
        assert_eq!(visible_width("φ, ψ Å"), 6);
        assert_eq!(visible_width("蛋白"), 4);
        assert_eq!(visible_width("\x1b[38;2;1;2;3mab\x1b[0m\x1b[5Gc"), 3);

        let styled = "\x1b[1;36mΩmega蛋白質\x1b[0m tail";
        for w in 0..16 {
            let cut = truncate_to_width(styled, w);
            assert!(visible_width(&cut) <= w, "{w}: {cut:?}");
            assert!(cut.starts_with("\x1b[1;36m"), "escape dropped at {w}");
            assert_eq!(visible_width(&fit_to_width(styled, w)), w);
        }
        // A wide character that does not fit is dropped whole, not split.
        assert_eq!(visible_width(&truncate_to_width("a蛋", 2)), 1);
        assert_eq!(truncate_to_width("short", 10), "short");
    }

    fn sample_data(title: &str) -> DashboardData {
        DashboardData {
            title: title.to_string(),
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
                steric_overlap: None,
                sasa_metrics: None,
                interaction_network: None,
                candidate_fitness_score: Some(87.4),
            }),
            plddts: vec![92.0; 46],
            ramachandran_points: vec![
                (Some(-60.0), Some(-45.0), RamachandranRegion::Favored),
                (Some(-120.0), Some(135.0), RamachandranRegion::Favored),
            ],
        }
    }

    /// Every dashboard line must fit its panel: a line one column too wide wraps into the next
    /// row (the header's closing `┐` used to land on the Ramachandran rule), and section rules
    /// must reach exactly to the panel edge. Measured in display columns — the titles carry
    /// `φ`, `ψ` and whatever the file name is.
    #[test]
    fn every_dashboard_line_fits_the_panel_width() {
        let renderer = DashboardRenderer::new();
        for title in [
            "1crn.pdb",
            "a_very_long_structure_file_name_蛋白質_model_0001.cif",
        ] {
            let data = sample_data(title);
            for width in 20..=90 {
                for height in [10, 17, 18, 24, 30, 45] {
                    let lines = renderer.generate_lines(&data, width, height);
                    assert_eq!(lines.len(), height);
                    for (i, line) in lines.iter().enumerate() {
                        assert!(
                            visible_width(line) <= width,
                            "{title} {width}x{height} line {i} is {} columns: {line:?}",
                            visible_width(line)
                        );
                        let plain: String = ansi_segments(line)
                            .filter(|(esc, _)| !esc)
                            .map(|(_, t)| t)
                            .collect();
                        if plain.starts_with("┌─") || plain.starts_with("├─") {
                            assert_eq!(
                                visible_width(line),
                                width,
                                "{title} {width}x{height}: rule {i} does not reach the edge"
                            );
                            assert!(
                                plain.ends_with('┐') || plain.ends_with('┤'),
                                "{title} {width}x{height}: rule {i} lost its corner: {plain}"
                            );
                        }
                    }
                }
            }
        }
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
                steric_overlap: None,
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
