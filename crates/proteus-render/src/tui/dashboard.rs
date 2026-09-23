use crate::brand::ansi::{Ansi, RESET};
use crate::brand::Role;
use crate::rasterizer::buffer::ColorRGB;
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
    use unicode_width::UnicodeWidthStr;
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
        // Measure prefixes of the run with the same function as `visible_width`, so that a
        // sequence such as ❤ + VS16 (two columns together) is never cut to a wider result.
        let mut taken = 0;
        for (i, c) in part.char_indices() {
            let end = i + c.len_utf8();
            if used + part[..end].width() > max_width {
                out.push_str(&part[..taken]);
                break 'segments;
            }
            taken = end;
        }
        out.push_str(part);
        used += part.width();
    }
    if styled {
        out.push_str(RESET);
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
/// if it does not fit): the title in `title_style`, the rule in the brand's line colour.
fn section_rule(
    a: &Ansi,
    title_style: &str,
    left: char,
    title: &str,
    right: char,
    width: usize,
) -> String {
    let title = truncate_to_width(title, width.saturating_sub(3));
    let bar_len = width.saturating_sub(visible_width(&title) + 3);
    let line = a.fg(Role::Line);
    format!(
        "{line}{left}─{RESET}{title_style}{title}{RESET}{line}{:─<bar_len$}{right}{RESET}",
        ""
    )
}

/// Ramachandran markers: outliers over allowed over favoured where they share a cell. Each
/// region has its own glyph, so the plot reads without colour.
/// Items joined by two spaces, as many whole items as fit in `width` columns: a row never ends
/// in half a number ("coil 4" for "coil 43%").
fn fit_items(items: &[String], width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    for item in items {
        let w = visible_width(item) + if out.is_empty() { 0 } else { 2 };
        if used + w > width {
            break;
        }
        if !out.is_empty() {
            out.push_str("  ");
        }
        out.push_str(item);
        used += w;
    }
    out
}

fn rama_marker(region: RamachandranRegion) -> (u8, char, Role) {
    match region {
        RamachandranRegion::Outlier => (2, '▲', Role::Bad),
        RamachandranRegion::Allowed => (1, '○', Role::Warm),
        RamachandranRegion::Favored => (0, '●', Role::Accent),
    }
}

pub struct DashboardRenderer {
    ansi: Ansi,
}

impl Default for DashboardRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardRenderer {
    /// Colours at the terminal's detected depth.
    pub fn new() -> Self {
        Self {
            ansi: Ansi::detect(),
        }
    }

    pub fn with_ansi(ansi: Ansi) -> Self {
        Self { ansi }
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
            let _ = write!(out, "\x1b[{row};{col}H{fitted}{RESET}");
        }
    }

    /// Generate the formatted list of lines for the dashboard.
    pub fn generate_lines(&self, data: &DashboardData, width: usize, height: usize) -> Vec<String> {
        let a = &self.ansi;
        let mut lines = Vec::with_capacity(height);

        // Header: the structure, in the text colour, bold.
        let header_title = format!(" {} · {} residues ", data.title, data.num_residues);
        lines.push(section_rule(
            a,
            &format!("{}\x1b[1m", a.fg(Role::Text)),
            '┌',
            &header_title,
            '┐',
            width,
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

        self.render_ramachandran_section(data, width, plot_h, &mut lines);

        let remaining = height.saturating_sub(lines.len());
        if remaining >= 5 {
            self.render_plddt_section(data, width, &mut lines);
        }

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
        let a = &self.ansi;
        let plot_w = width.saturating_sub(8).clamp(16, 36);
        lines.push(section_rule(
            a,
            &a.fg(Role::Muted),
            '├',
            " (ramachandran φ, ψ) ",
            '┤',
            width,
        ));

        // Precompute residue points on the grid
        let mut grid_markers: Vec<Option<(u8, char, Role)>> = vec![None; plot_w * plot_h];
        for (phi_opt, psi_opt, region) in &data.ramachandran_points {
            if let (Some(phi), Some(psi)) = (*phi_opt, *psi_opt) {
                let gx = (((phi - (-180.0)) / 360.0) * (plot_w as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_w - 1) as f64) as usize;
                let gy = (((180.0 - psi) / 360.0) * (plot_h as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_h - 1) as f64) as usize;
                let idx = gy * plot_w + gx;
                let marker = rama_marker(*region);
                if grid_markers[idx].is_none_or(|(rank, _, _)| marker.0 > rank) {
                    grid_markers[idx] = Some(marker);
                }
            }
        }

        let mid_x = (plot_w - 1) / 2;
        let mid_y = (plot_h - 1) / 2;
        let line = a.fg(Role::Line);
        let dim = a.fg(Role::Dim);

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
            let _ = write!(row_str, "{dim}{left_label}{RESET}");
            let psi_c = 180.0 - (r as f64 / (plot_h - 1) as f64) * 360.0;

            for c in 0..plot_w {
                let idx = r * plot_w + c;
                if let Some((_, sym, role)) = grid_markers[idx] {
                    row_str.push_str(&a.paint(role, &sym.to_string()));
                } else {
                    let phi_c = -180.0 + (c as f64 / (plot_w - 1) as f64) * 360.0;
                    let basin = ((-180.0..=-45.0).contains(&phi_c)
                        && (psi_c >= 90.0 || psi_c <= -150.0))
                        || ((-100.0..=-30.0).contains(&phi_c) && (-70.0..=-10.0).contains(&psi_c))
                        || ((30.0..=90.0).contains(&phi_c) && (10.0..=70.0).contains(&psi_c));
                    let ch = if c == mid_x && r == mid_y {
                        "┼"
                    } else if c == mid_x {
                        "│"
                    } else if r == mid_y {
                        "─"
                    } else if basin {
                        // The favoured basins (β, right-handed α, left-handed α).
                        "·"
                    } else {
                        " "
                    };
                    if ch == " " {
                        row_str.push(' ');
                    } else {
                        let _ = write!(row_str, "{line}{ch}{RESET}");
                    }
                }
            }
            let _ = write!(row_str, "{dim}{right_border}{RESET}");
            lines.push(row_str);
        }

        // "      └" sits under the left label; the plot's own columns start after it.
        let mut axis_bar = String::from("      └");
        for c in 0..plot_w {
            axis_bar.push(if c == mid_x { '┴' } else { '─' });
        }
        axis_bar.push('┘');
        lines.push(format!("{dim}{axis_bar}{RESET}"));
        lines.push(format!(
            "       {dim}-180°{: <pad$}φ 0°{: >pad2$}+180°{RESET}",
            "",
            "",
            pad = (plot_w / 2).saturating_sub(7),
            pad2 = (plot_w / 2).saturating_sub(6)
        ));

        if let Some(rama) = data
            .metrics
            .as_ref()
            .and_then(|m| m.ramachandran_stats.as_ref())
        {
            let pct = |x: f64| format!("{:.1}%", x * 100.0);
            let items = [
                format!(
                    "{} {}",
                    a.paint(Role::Accent, "●"),
                    a.paint(
                        Role::Text,
                        &format!("favoured {}", pct(rama.favored_fraction))
                    )
                ),
                format!(
                    "{} {}",
                    a.paint(Role::Bad, "▲"),
                    a.paint(Role::Text, &format!("outliers {}", rama.outlier_count))
                ),
                format!(
                    "{} {}",
                    a.paint(Role::Warm, "○"),
                    a.paint(
                        Role::Text,
                        &format!("allowed {}", pct(rama.allowed_fraction))
                    )
                ),
            ];
            lines.push(format!(" {}", fit_items(&items, width.saturating_sub(1))));
        }
    }

    fn render_plddt_section(&self, data: &DashboardData, width: usize, lines: &mut Vec<String>) {
        let a = &self.ansi;
        let predicted = data
            .metrics
            .as_ref()
            .map(|m| m.plddt().is_some())
            .unwrap_or(true);
        let title = if predicted {
            " (plddt, AlphaFold colours) "
        } else {
            " (b-factor · not a confidence) "
        };
        lines.push(section_rule(a, &a.fg(Role::Muted), '├', title, '┤', width));

        if let Some(m) = data.metrics.as_ref() {
            let dim = |s: &str| a.paint(Role::Dim, s);
            match m.plddt() {
                Some(p) => lines.push(format!(
                    " {} {}  {} {}  {} {}",
                    dim("mean"),
                    a.bold(&format!("{:.1}", p.mean)),
                    dim("median"),
                    a.bold(&format!("{:.1}", p.median)),
                    dim("≥70"),
                    a.bold(&format!("{:.1}%", p.high_confidence_fraction * 100.0)),
                )),
                None => lines.push(format!(
                    " {} {}  {} {}",
                    dim("mean B"),
                    a.bold(&format!("{:.1} Å²", m.plddt_distribution.mean)),
                    dim("median"),
                    a.bold(&format!("{:.1}", m.plddt_distribution.median)),
                )),
            }
        }

        // Resampled per-residue strip.
        let n_res = data.plddts.len();
        if n_res > 0 {
            let bar_w = width.saturating_sub(4).clamp(10, 44);
            let mut bar_str = String::with_capacity(bar_w * 20);
            bar_str.push(' ');
            let (lo, hi) = data
                .plddts
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            for col in 0..bar_w {
                let start_idx = col * n_res / bar_w;
                let end_idx = ((col + 1) * n_res / bar_w).max(start_idx + 1).min(n_res);
                let slice = &data.plddts[start_idx..end_idx];
                let avg = slice.iter().sum::<f64>() / slice.len().max(1) as f64;
                if predicted {
                    // The AlphaFold bands, the same colours as the ribbon; without colour, the
                    // band shows as shade density instead.
                    let glyph = match avg {
                        _ if a.colours() => "█",
                        v if v >= 90.0 => "█",
                        v if v >= 70.0 => "▓",
                        v if v >= 50.0 => "▒",
                        _ => "░",
                    };
                    bar_str.push_str(
                        &a.paint_rgb(crate::rasterizer::shader::plddt_to_color(avg as f32), glyph),
                    );
                } else {
                    // B-factors: a neutral ramp over the structure's own range, Moss to Khaki.
                    let t = if hi > lo {
                        ((avg - lo) / (hi - lo)) as f32
                    } else {
                        0.5
                    };
                    let c = ColorRGB::lerp(
                        crate::brand::palette::MOSS,
                        crate::brand::palette::KHAKI,
                        t,
                    );
                    bar_str.push_str(&a.paint_rgb(c, "█"));
                }
            }
            lines.push(bar_str);
            lines.push(format!(
                " {}1{: <pad$}{}{RESET}",
                a.fg(Role::Dim),
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
        let a = &self.ansi;
        lines.push(section_rule(
            a,
            &a.fg(Role::Muted),
            '├',
            " (measurements) ",
            '┤',
            width,
        ));
        // The label column shrinks on a narrow panel, so the values are never what is cut.
        let label_w = if width >= 56 {
            23
        } else {
            (width / 2).saturating_sub(2).max(8)
        };
        let row = |label: &str, value: String| {
            let label = truncate_to_width(label, label_w);
            format!(
                " {} {value}",
                a.paint(Role::Dim, &format!("{label:<label_w$}"))
            )
        };
        let value_w = width.saturating_sub(label_w + 2);

        let mut row_count = 1;
        let Some(ref m) = data.metrics else {
            lines.push(format!(" {}", a.paint(Role::Dim, "measuring…")));
            return;
        };
        let mut push = |line: String, lines: &mut Vec<String>| {
            if row_count < max_rows {
                lines.push(line);
                row_count += 1;
            }
        };

        push(
            row(
                "radius of gyration",
                a.bold(&format!("{:.3} Å", m.radius_of_gyration)),
            ),
            lines,
        );
        push(
            row(
                "contact density ≤8 Å",
                format!(
                    "{} {}",
                    a.bold(&format!("{:.1}%", m.contact_density * 100.0)),
                    a.paint(Role::Dim, "Cα")
                ),
            ),
            lines,
        );
        if let Some(ref sasa) = m.sasa_metrics {
            push(
                row("SASA", a.bold(&format!("{:.1} Å²", sasa.total_sasa))),
                lines,
            );
            push(
                row(
                    "hydrophobic burial",
                    a.bold(&format!("{:.1}%", sasa.hydrophobic_burial_ratio * 100.0)),
                ),
                lines,
            );
        }
        if let Some(ref ss) = m.secondary_structure_summary {
            use crate::brand::structure::{COIL, HELIX, STRAND};
            let items = [
                format!(
                    "{} {}",
                    a.paint_rgb(HELIX, "■"),
                    a.paint(Role::Text, &format!("α {:.0}%", ss.helix_fraction * 100.0))
                ),
                format!(
                    "{} {}",
                    a.paint_rgb(STRAND, "■"),
                    a.paint(Role::Text, &format!("β {:.0}%", ss.strand_fraction * 100.0))
                ),
                format!(
                    "{} {}",
                    a.paint_rgb(COIL, "■"),
                    a.paint(
                        Role::Text,
                        &format!("coil {:.0}%", ss.coil_fraction * 100.0)
                    )
                ),
            ];
            push(
                row("secondary structure", fit_items(&items, value_w)),
                lines,
            );
        }
        if data.num_disulfides > 0 {
            push(
                row(
                    "disulfide bonds",
                    a.paint_rgb(
                        crate::brand::structure::DISULFIDE,
                        &format!("{} pairs", data.num_disulfides),
                    ),
                ),
                lines,
            );
        }
        if let Some(ref clash) = m.steric_overlap {
            let (glyph, role) = if clash.heavy_atom_overlap_score < 5.0 {
                ("✓", Role::Accent)
            } else if clash.heavy_atom_overlap_score < 15.0 {
                ("!", Role::Warm)
            } else {
                ("✗", Role::Bad)
            };
            push(
                row(
                    "overlaps /1000 atoms",
                    format!(
                        "{} {}",
                        a.paint(
                            role,
                            &format!("{glyph} {:.1}", clash.heavy_atom_overlap_score)
                        ),
                        a.paint(Role::Dim, &format!("({}, no H)", clash.clash_count))
                    ),
                ),
                lines,
            );
        }
        if let Some(fitness) = m.candidate_fitness_score {
            push(
                row("triage score", a.bold(&format!("{fitness:.1} / 100"))),
                lines,
            );
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

        use crate::brand::{ansi::Ansi, ColorDepth};
        let renderer = DashboardRenderer::with_ansi(Ansi::with_depth(ColorDepth::TrueColor));
        let lines = renderer.generate_lines(&data, 45, 30);

        assert_eq!(lines.len(), 30);
        assert!(lines[0].contains("1CRN"));
        assert!(lines[0].contains("46 residues"));

        let full_text = lines.join("\n");
        assert!(full_text.contains("(ramachandran"));
        assert!(full_text.contains("(plddt") || full_text.contains("(b-factor"));
        assert!(full_text.contains("(measurements)"));
        assert!(full_text.contains("9.762"));
        if full_text.contains("(plddt") {
            // pLDDT 92 everywhere: the strip is the AlphaFold ≥90 blue, #0053D6, as in the ribbon.
            assert!(full_text.contains("\x1b[38;2;0;83;214m█"), "{full_text:?}");
        }

        // Without colour, not one colour escape, and every Ramachandran region still has its
        // own glyph.
        let plain = DashboardRenderer::with_ansi(Ansi::with_depth(ColorDepth::None))
            .generate_lines(&data, 45, 30)
            .join("\n");
        assert!(
            !plain.contains("\x1b[38;") && !plain.contains("\x1b[3"),
            "{plain:?}"
        );
        assert!(plain.contains('●'));
    }

    #[test]
    fn truncation_never_returns_a_wider_line_and_rows_keep_whole_numbers() {
        // ❤ + VS16 is two columns together; cutting to 8 used to leave 9.
        let s = "abcdefg\u{2764}\u{fe0f}";
        assert_eq!(visible_width(s), 9);
        assert!(visible_width(&truncate_to_width(s, 8)) <= 8);
        let items = [
            "α 25%".to_string(),
            "β 43%".to_string(),
            "coil 32%".to_string(),
        ];
        assert_eq!(fit_items(&items, 14), "α 25%  β 43%");
        assert_eq!(fit_items(&items, 100), "α 25%  β 43%  coil 32%");
    }

    #[test]
    fn the_ramachandran_axis_ticks_line_up_with_the_plot() {
        use crate::brand::{ansi::Ansi, ColorDepth};
        let data = DashboardData {
            title: "t".into(),
            num_residues: 3,
            num_disulfides: 0,
            metrics: None,
            plddts: vec![],
            ramachandran_points: vec![],
        };
        let lines = DashboardRenderer::with_ansi(Ansi::with_depth(ColorDepth::None))
            .generate_lines(&data, 40, 30);
        let plain: Vec<String> = lines
            .iter()
            .map(|l| {
                ansi_segments(l)
                    .filter(|(e, _)| !e)
                    .map(|(_, t)| t)
                    .collect()
            })
            .collect();
        let cross = plain
            .iter()
            .find_map(|l| l.find('┼').map(|b| l[..b].chars().count()))
            .unwrap();
        let axis = plain.iter().find(|l| l.contains('└')).unwrap();
        let tick = axis[..axis.find('┴').unwrap()].chars().count();
        assert_eq!(tick, cross, "{axis}");
    }
}
