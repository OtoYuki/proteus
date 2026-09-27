//! The side panel of the interactive viewer: four pages (overview, geometry, confidence,
//! measurements) under a segmented selector, each laid out in the house style: section titles
//! in `(parentheses)` followed by a hairline, a blank line between sections, labels in a dim
//! fixed column, values in one column with the number bold and the unit dim.

use crate::brand::ansi::{Ansi, RESET};
use crate::brand::Role;
use crate::rasterizer::buffer::ColorRGB;
use proteus_core::models::BiophysicalMetrics;
use proteus_core::structure::RamachandranRegion;
use std::fmt::Write;

/// Biophysical data bundle used to render the live side-panel dashboard.
#[derive(Debug, Clone, Default)]
pub struct DashboardData {
    pub title: String,
    pub num_residues: usize,
    pub num_disulfides: usize,
    pub metrics: Option<BiophysicalMetrics>,
    pub plddts: Vec<f64>,
    pub ramachandran_points: Vec<(Option<f64>, Option<f64>, RamachandranRegion)>,
    /// PAE and pTM, when the predictor's files were found.
    pub confidence: Option<proteus_core::pae::PredictionConfidence>,
    /// A complex's interfaces (each chain against the rest); the smallest binder is shown.
    pub interfaces: Vec<crate::InterfaceView>,
    /// The ribbon's chains in order, as (chain ID, residues): consecutive runs of residues.
    pub chains: Vec<(String, usize)>,
    /// Eight-state DSSP, one character per ribbon residue; empty when unknown.
    pub dssp: String,
    /// Chain ID → the name the input gave it, for the interface labels.
    pub chain_names: Vec<(String, String)>,
}

impl DashboardData {
    /// Chain IDs (`A` or `A,C`) as `PD-L1 (A)`, bare where there is no name.
    pub fn name_chains(&self, ids: &str) -> String {
        ids.split(',')
            .map(|id| {
                let id = id.trim();
                self.chain_names
                    .iter()
                    .find(|(c, _)| c == id)
                    .map_or(id.to_string(), |(_, n)| format!("{n} ({id})"))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The dashboard's pages, in selector order; keys `1`–`4` pick them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashboardPage {
    #[default]
    Overview,
    Geometry,
    Confidence,
    Measurements,
}

impl DashboardPage {
    pub const ALL: [Self; 4] = [
        Self::Overview,
        Self::Geometry,
        Self::Confidence,
        Self::Measurements,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Overview => "overview",
            Self::Geometry => "geometry",
            Self::Confidence => "confidence",
            Self::Measurements => "measurements",
        }
    }

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|p| *p == self).unwrap_or(0)
    }

    /// The page a digit key selects: `1` is the first.
    pub fn from_key(c: char) -> Option<Self> {
        let i = c.to_digit(10)? as usize;
        (1..=Self::ALL.len()).contains(&i).then(|| Self::ALL[i - 1])
    }
}

/// What the dashboard shows: the page, and how far a page taller than the panel is scrolled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DashboardView {
    pub page: DashboardPage,
    pub scroll: usize,
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

/// `left` and `right` on one line of `width` columns, `right` flush with the right edge; the
/// right part is left out when both do not fit with at least `gap` columns between them.
pub fn split_line(left: &str, right: &str, width: usize, gap: usize) -> String {
    let (lw, rw) = (visible_width(left), visible_width(right));
    if right.is_empty() || lw + gap + rw > width {
        return truncate_to_width(left, width);
    }
    format!("{left}{}{right}", " ".repeat(width - lw - rw))
}

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

/// Ramachandran markers: outliers over allowed over favoured where they share a cell. Each
/// region has its own glyph, so the plot reads without colour.
fn rama_marker(region: RamachandranRegion) -> (u8, char, Role) {
    match region {
        RamachandranRegion::Outlier => (2, '▲', Role::Bad),
        RamachandranRegion::Allowed => (1, '○', Role::Warm),
        RamachandranRegion::Favored => (0, '●', Role::Accent),
    }
}

/// How a check came out, always shown as glyph and word together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Good,
    Warn,
    Bad,
}

impl Level {
    fn mark(self) -> (&'static str, Role) {
        match self {
            Level::Good => ("✓", Role::Accent),
            Level::Warn => ("!", Role::Warm),
            Level::Bad => ("✗", Role::Bad),
        }
    }
}

/// A group of labelled rows under one section title. A row with an empty label is a line of
/// its own (a legend, a verdict).
struct Group {
    title: String,
    note: String,
    rows: Vec<(String, String)>,
}

impl Group {
    fn new(title: &str, note: &str) -> Self {
        Self {
            title: title.to_string(),
            note: note.to_string(),
            rows: Vec::new(),
        }
    }
    fn row(&mut self, label: &str, value: String) {
        self.rows.push((label.to_string(), value));
    }
}

/// Left and right margins inside the panel, so that nothing touches the separator.
const MARGIN_L: usize = 2;
const MARGIN_R: usize = 1;
/// Narrowest column of a two-column page.
const MIN_COLUMN: usize = 32;

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

    /// Render the dashboard's current page as positioned ANSI output into `out`. Clamps
    /// `view.scroll` to what the page has.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_buffer(
        &self,
        data: &DashboardData,
        view: &mut DashboardView,
        out: &mut String,
        screen_offset_col: u16,
        screen_offset_row: u16,
        width: usize,
        height: usize,
    ) {
        if width < 20 || height < 10 {
            return;
        }
        let lines = self.generate_lines(data, view, width, height);
        for (i, line) in lines.iter().enumerate().take(height) {
            let row = screen_offset_row + i as u16 + 1;
            let col = screen_offset_col + 1;
            let fitted = fit_to_width(line, width);
            let _ = write!(out, "\x1b[{row};{col}H{fitted}{RESET}");
        }
    }

    /// The panel as `height` lines of at most `width` columns: the page selector, a hairline,
    /// then the page (scrolled by `view.scroll`, which is clamped here).
    pub fn generate_lines(
        &self,
        data: &DashboardData,
        view: &mut DashboardView,
        width: usize,
        height: usize,
    ) -> Vec<String> {
        let a = &self.ansi;
        let w = width.saturating_sub(MARGIN_L + MARGIN_R).max(1);
        let mut head = vec![
            self.tabs_line(data, view.page, w),
            a.paint(Role::Line, &"─".repeat(w)),
        ];
        if height >= 24 {
            head.push(String::new());
        }
        let avail = height.saturating_sub(head.len());
        let body = self.page_lines(data, view.page, w, avail);

        let mut lines = head;
        if body.len() > avail && avail >= 2 {
            // Taller than the panel: a window onto it, and the last row says what is hidden.
            let visible = avail - 1;
            let max_scroll = body.len() - visible;
            view.scroll = view.scroll.min(max_scroll);
            lines.extend_from_slice(&body[view.scroll..view.scroll + visible]);
            let below = max_scroll - view.scroll;
            let mut parts = Vec::new();
            if view.scroll > 0 {
                parts.push(format!("↑ {} above", view.scroll));
            }
            if below > 0 {
                parts.push(format!("↓ {below} more"));
            }
            parts.push("pgup pgdn".to_string());
            let note = parts.join("  ");
            lines.push(split_line("", &a.paint(Role::Dim, &note), w, 0));
        } else {
            view.scroll = 0;
            lines.extend(body);
        }

        lines.truncate(height);
        let margin = " ".repeat(MARGIN_L);
        let mut lines: Vec<String> = lines
            .into_iter()
            .map(|l| {
                if l.is_empty() {
                    l
                } else {
                    truncate_to_width(&format!("{margin}{l}"), width)
                }
            })
            .collect();
        while lines.len() < height {
            lines.push(String::new());
        }
        lines
    }

    /// The page selector: the showing page filled in the accent, the others quiet with their
    /// number key; on a narrow panel the numbers go, then the other pages' names.
    fn tabs_line(&self, data: &DashboardData, page: DashboardPage, w: usize) -> String {
        let a = &self.ansi;
        let tab = |p: DashboardPage, number: bool, name: bool| -> String {
            let n = p.index() + 1;
            let text = match (name, number) {
                (true, true) => format!(" {} {n} ", p.name()),
                (true, false) => format!(" {} ", p.name()),
                (false, _) => format!(" {n} "),
            };
            if p == page {
                a.tab_on(&text)
            } else if name {
                let body = format!(" {} ", p.name());
                if number {
                    format!(
                        "{}{} ",
                        a.paint(Role::Muted, &body),
                        a.paint(Role::Dim, &n.to_string())
                    )
                } else {
                    a.paint(Role::Muted, &body)
                }
            } else {
                a.paint(Role::Dim, &text)
            }
        };
        let variants = [
            DashboardPage::ALL.map(|p| tab(p, true, true)),
            DashboardPage::ALL.map(|p| tab(p, false, true)),
            DashboardPage::ALL.map(|p| tab(p, true, p == page)),
        ];
        let line = variants
            .iter()
            .map(|v| v.join(" "))
            .find(|l| visible_width(l) <= w)
            .unwrap_or_else(|| tab(page, false, true));
        let chains = data.chains.len();
        let size = if chains > 1 {
            format!("{} residues · {chains} chains", data.num_residues)
        } else {
            format!("{} residues", data.num_residues)
        };
        split_line(&line, &a.paint(Role::Dim, &size), w, 3)
    }

    fn page_lines(
        &self,
        data: &DashboardData,
        page: DashboardPage,
        w: usize,
        avail: usize,
    ) -> Vec<String> {
        match page {
            DashboardPage::Overview => self.overview_page(data, w),
            DashboardPage::Geometry => self.geometry_page(data, w, avail),
            DashboardPage::Confidence => self.confidence_page(data, w, avail),
            DashboardPage::Measurements => self.measurements_page(data, w),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Building blocks

    /// `(title) note ─────` to the width `w`.
    fn rule(&self, title: &str, note: &str, w: usize) -> String {
        let a = &self.ansi;
        let head = format!("({title})");
        let mut s = a.paint(Role::Muted, &head);
        let mut used = visible_width(&head);
        if !note.is_empty() {
            s.push(' ');
            s.push_str(&a.paint(Role::Dim, note));
            used += 1 + visible_width(note);
        }
        if used + 2 < w {
            s.push(' ');
            s.push_str(&a.paint(Role::Line, &"─".repeat(w - used - 1)));
        }
        truncate_to_width(&s, w)
    }

    /// A bold number and its dim unit.
    fn num(&self, n: &str, unit: &str) -> String {
        let a = &self.ansi;
        if unit.is_empty() {
            a.bold(n)
        } else {
            format!("{} {}", a.bold(n), a.paint(Role::Dim, unit))
        }
    }

    /// A small bar for a fraction 0–1: heavy in the accent for the filled part, a hairline for
    /// the rest (heavy and light strokes read apart without colour).
    fn gauge(&self, frac: f64, cells: usize) -> String {
        let a = &self.ansi;
        let frac = if frac.is_finite() {
            frac.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let on = (frac * cells as f64).round() as usize;
        format!(
            "{}{}",
            a.paint(Role::Accent, &"━".repeat(on)),
            a.paint(Role::Line, &"─".repeat(cells - on))
        )
    }

    /// A label in the dim label column, then the value.
    fn kv(&self, label_w: usize, label: &str, value: &str) -> String {
        let label = truncate_to_width(label, label_w);
        format!(
            "{}  {value}",
            self.ansi.paint(Role::Dim, &pad_to_width(&label, label_w))
        )
    }

    /// A value with a gauge after it, the gauges of consecutive rows lined up.
    fn gauge_value(&self, value: String, frac: f64, room: usize) -> String {
        const VALUE_W: usize = 11;
        let cells = room.saturating_sub(VALUE_W + 1).min(18);
        if cells < 5 {
            return value;
        }
        format!(
            "{} {}",
            pad_to_width(&value, VALUE_W),
            self.gauge(frac, cells)
        )
    }

    /// A check: its glyph and label in the label column, then what was found.
    fn check(&self, label_w: usize, level: Level, label: &str, value: String) -> String {
        let a = &self.ansi;
        let (glyph, role) = level.mark();
        let label = truncate_to_width(label, label_w.saturating_sub(2));
        format!(
            "{} {}  {value}",
            a.paint(role, glyph),
            a.paint(Role::Text, &pad_to_width(&label, label_w.saturating_sub(2)))
        )
    }

    /// The label column for groups stacked in one column: their widest label.
    fn label_width(groups: &[Group], w: usize) -> usize {
        groups
            .iter()
            .flat_map(|g| g.rows.iter())
            .map(|(l, _)| visible_width(l))
            .max()
            .unwrap_or(0)
            .min(w * 3 / 5)
    }

    /// A group as lines `w` wide: its rule, then its rows, labels `label_w` wide.
    fn group_lines(&self, g: &Group, w: usize, label_w: usize) -> Vec<String> {
        let mut out = vec![self.rule(&g.title, &g.note, w)];
        for (label, value) in &g.rows {
            let line = if label.is_empty() {
                value.clone()
            } else {
                self.kv(label_w, label, value)
            };
            out.push(truncate_to_width(&line, w));
        }
        out
    }

    /// Groups one under the other, a blank line between; in two columns when the panel is
    /// wide enough, split where the two come out most even.
    fn columns(&self, groups: &[Group], w: usize) -> Vec<String> {
        // One label column for the whole stack, so its values line up.
        let stack = |gs: &[Group], cw: usize| -> Vec<String> {
            let label_w = Self::label_width(gs, cw);
            let mut out = Vec::new();
            for g in gs {
                if !out.is_empty() {
                    out.push(String::new());
                }
                out.extend(self.group_lines(g, cw, label_w));
            }
            out
        };
        if groups.len() < 2 || w < 2 * MIN_COLUMN + 3 {
            return stack(groups, w);
        }
        let cw = (w - 3) / 2;
        let heights: Vec<usize> = groups.iter().map(|g| g.rows.len() + 2).collect();
        let total: usize = heights.iter().sum();
        let mut best = (usize::MAX, 1);
        let mut left = 0;
        for (k, h) in heights.iter().enumerate().take(groups.len() - 1) {
            left += h;
            let tallest = left.max(total - left);
            if tallest < best.0 {
                best = (tallest, k + 1);
            }
        }
        let (l, r) = (stack(&groups[..best.1], cw), stack(&groups[best.1..], cw));
        (0..l.len().max(r.len()))
            .map(|i| {
                let left = l.get(i).map_or("", String::as_str);
                let right = r.get(i).map_or("", String::as_str);
                if right.is_empty() {
                    left.to_string()
                } else {
                    format!("{}   {right}", pad_to_width(left, cw))
                }
            })
            .collect()
    }

    /// Mean of `values` over the residues each of `cols` columns covers.
    fn resample(values: &[f64], cols: usize) -> Vec<f64> {
        let n = values.len();
        (0..cols)
            .map(|c| {
                let s = c * n / cols;
                let e = ((c + 1) * n / cols).max(s + 1).min(n);
                let slice = &values[s.min(n - 1)..e];
                slice.iter().sum::<f64>() / slice.len().max(1) as f64
            })
            .collect()
    }

    fn predicted(data: &DashboardData) -> bool {
        // Unknown (no analysis) counts as predicted, as in the viewer's legend.
        data.metrics.as_ref().is_none_or(|m| m.plddt().is_some())
    }

    /// Per-residue pLDDT as `cols` cells in the AlphaFold bands (shade density without colour);
    /// for an experimental structure the B-factors on a neutral Moss-to-Khaki ramp.
    fn plddt_cells(&self, data: &DashboardData, cols: usize) -> String {
        let a = &self.ansi;
        if data.plddts.is_empty() || cols == 0 {
            return String::new();
        }
        let predicted = Self::predicted(data);
        let (lo, hi) = data
            .plddts
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        let mut s = String::with_capacity(cols * 24);
        for avg in Self::resample(&data.plddts, cols) {
            if predicted {
                let glyph = match avg {
                    _ if a.colours() => "█",
                    v if v >= 90.0 => "█",
                    v if v >= 70.0 => "▓",
                    v if v >= 50.0 => "▒",
                    _ => "░",
                };
                s.push_str(
                    &a.paint_rgb(crate::rasterizer::shader::plddt_to_color(avg as f32), glyph),
                );
            } else {
                let t = if hi > lo {
                    ((avg - lo) / (hi - lo)) as f32
                } else {
                    0.5
                };
                let c =
                    ColorRGB::lerp(crate::brand::palette::MOSS, crate::brand::palette::KHAKI, t);
                s.push_str(&a.paint_rgb(c, "█"));
            }
        }
        s
    }

    /// Secondary structure from the DSSP string as `cols` cells: helix, strand or coil,
    /// whichever covers most of the residues in a cell. Glyphs differ too (█ ▒ ─).
    fn ss_cells(&self, data: &DashboardData, cols: usize) -> Option<String> {
        use crate::brand::structure::{COIL, HELIX, STRAND};
        let a = &self.ansi;
        let ss: Vec<u8> = data
            .dssp
            .chars()
            .map(|c| match c {
                'H' | 'G' | 'I' => 0,
                'E' | 'B' => 1,
                _ => 2,
            })
            .collect();
        let n = ss.len();
        if n == 0 || cols == 0 {
            return None;
        }
        let mut s = String::with_capacity(cols * 24);
        for c in 0..cols {
            let st = c * n / cols;
            let e = ((c + 1) * n / cols).max(st + 1).min(n);
            let mut count = [0usize; 3];
            for k in &ss[st.min(n - 1)..e] {
                count[*k as usize] += 1;
            }
            let kind = (0..3).max_by_key(|&k| (count[k], 2 - k)).unwrap_or(2);
            let (col, glyph) = match kind {
                0 => (HELIX, "█"),
                1 => (STRAND, if a.colours() { "█" } else { "▒" }),
                _ => (COIL, "─"),
            };
            s.push_str(&a.paint_rgb(col, glyph));
        }
        Some(s)
    }

    /// Chain IDs over a `cols`-wide residue track, each at the column its chain starts; `None`
    /// for one chain.
    fn chain_ruler(&self, data: &DashboardData, cols: usize) -> Option<String> {
        let n: usize = data.chains.iter().map(|(_, k)| k).sum();
        if data.chains.len() < 2 || n == 0 || cols == 0 {
            return None;
        }
        let mut row = vec![' '; cols];
        let mut start = 0;
        for (id, k) in &data.chains {
            let c = (start * cols / n).min(cols - 1);
            // A chain too short to label keeps its boundary tick only.
            row[c] = '▏';
            if c + 1 < cols && *k * cols / n >= 2 {
                row[c + 1] = id.chars().next().unwrap_or('?');
            }
            start += k;
        }
        Some(
            self.ansi
                .paint(Role::Dim, &row.into_iter().collect::<String>()),
        )
    }

    /// `1 … n` under a `cols`-wide residue track.
    fn residue_axis(&self, cols: usize, n: usize) -> String {
        let end = n.to_string();
        let pad = cols.saturating_sub(1 + end.len());
        self.ansi
            .paint(Role::Dim, &format!("1{}{end}", " ".repeat(pad)))
    }

    /// The pLDDT bands as swatch and range, in the strip's own glyphs without colour.
    fn plddt_legend(&self) -> Vec<String> {
        let a = &self.ansi;
        let p = crate::rasterizer::shader::plddt_to_color;
        [
            (95.0, "█", ">90"),
            (80.0, "▓", "70–90"),
            (60.0, "▒", "50–70"),
            (25.0, "░", "<50"),
        ]
        .iter()
        .map(|(v, g, label)| {
            let glyph = if a.colours() { "■" } else { g };
            format!(
                "{} {}",
                a.paint_rgb(p(*v), glyph),
                a.paint(Role::Muted, label)
            )
        })
        .collect()
    }

    fn ss_legend(&self, data: &DashboardData) -> Vec<String> {
        use crate::brand::structure::{COIL, HELIX, STRAND};
        let a = &self.ansi;
        let fractions = data
            .metrics
            .as_ref()
            .and_then(|m| m.secondary_structure_summary.as_ref())
            .map(|s| (s.helix_fraction, s.strand_fraction, s.coil_fraction));
        [
            (HELIX, "■", "█", "helix", fractions.map(|f| f.0)),
            (STRAND, "■", "▒", "strand", fractions.map(|f| f.1)),
            (COIL, "■", "─", "coil", fractions.map(|f| f.2)),
        ]
        .iter()
        .map(|(c, sw, g, word, f)| {
            let glyph = if a.colours() { sw } else { g };
            let pct = f.map_or(String::new(), |f| format!(" {:.0}%", f * 100.0));
            format!(
                "{} {}{}",
                a.paint_rgb(*c, glyph),
                a.paint(Role::Muted, word),
                a.paint(Role::Text, &pct)
            )
        })
        .collect()
    }

    fn interface(data: &DashboardData) -> Option<&crate::InterfaceView> {
        data.interfaces.iter().min_by_key(|v| v.binder_size)
    }

    fn chain_size(data: &DashboardData, ids: &str) -> usize {
        let ids: Vec<&str> = ids.split(',').collect();
        data.chains
            .iter()
            .filter(|(c, _)| ids.contains(&c.as_str()))
            .map(|(_, k)| k)
            .sum()
    }

    fn rama_level(&self, m: &BiophysicalMetrics) -> Option<(Level, String)> {
        let r = m.ramachandran_stats.as_ref()?;
        // MolProbity's goals: over 98 % favoured and under 0.2 % outliers.
        let level = if r.outlier_fraction <= 0.002 && r.favored_fraction >= 0.98 {
            Level::Good
        } else if r.outlier_fraction <= 0.02 {
            Level::Warn
        } else {
            Level::Bad
        };
        Some((
            level,
            format!(
                "{} {}",
                self.num(&format!("{:.1}", r.favored_fraction * 100.0), "% favoured"),
                self.ansi.paint(
                    Role::Dim,
                    &format!(
                        "· {} outlier{}",
                        r.outlier_count,
                        if r.outlier_count == 1 { "" } else { "s" }
                    )
                )
            ),
        ))
    }

    // -----------------------------------------------------------------------------------------
    // Pages

    /// The verdict, the key numbers with gauges, the checks, and the chain at a glance.
    fn overview_page(&self, data: &DashboardData, w: usize) -> Vec<String> {
        let a = &self.ansi;
        let lw = 14.min(w / 2);
        let room = w.saturating_sub(lw + 2);
        let mut out: Vec<String> = Vec::new();
        let section = |out: &mut Vec<String>, title: &str, note: &str| {
            if !out.is_empty() {
                out.push(String::new());
            }
            out.push(self.rule(title, note, w));
        };
        let m = data.metrics.as_ref();
        let predicted = Self::predicted(data);
        let conf = data.confidence.as_ref();

        // The headline: the interface for a complex, the model's confidence otherwise.
        if let Some(v) = Self::interface(data) {
            let im = &v.metrics;
            section(
                &mut out,
                "interface",
                &format!(
                    "{} → {}",
                    data.name_chains(&im.binder_chains),
                    data.name_chains(&im.target_chains)
                ),
            );
            match im.ipsae_min {
                Some(x) if x > 0.61 => {
                    out.push(a.bold(&a.paint(Role::Accent, "● confident interface")));
                    out.push(a.paint(Role::Dim, &format!("ipSAE_min {x:.3} is above 0.61")));
                }
                Some(x) => {
                    out.push(a.bold(&a.paint(Role::Warm, "○ not a confident interface")));
                    out.push(a.paint(Role::Dim, &format!("ipSAE_min {x:.3} is 0.61 or below")));
                }
                None => {
                    out.push(a.paint(Role::Muted, "– no verdict"));
                    out.push(a.paint(Role::Dim, "no PAE beside this model"));
                }
            }
            out.push(String::new());
            let target = Self::chain_size(data, &im.target_chains);
            out.push(self.kv(
                lw,
                "binder",
                &format!(
                    "{}  {}",
                    a.bold(&data.name_chains(&im.binder_chains)),
                    a.paint(
                        Role::Dim,
                        &format!(
                            "{} residues · {} in contact",
                            v.binder_size, im.binder_interface_residues
                        )
                    )
                ),
            ));
            out.push(self.kv(
                lw,
                "target",
                &format!(
                    "{}  {}",
                    a.bold(&data.name_chains(&im.target_chains)),
                    a.paint(
                        Role::Dim,
                        &format!(
                            "{target} residues · {} in contact",
                            im.target_interface_residues
                        )
                    )
                ),
            ));
        } else if let Some(m) = m {
            section(
                &mut out,
                "model",
                if predicted {
                    "predicted"
                } else {
                    "experimental"
                },
            );
            match m.plddt() {
                Some(p) => {
                    let (glyph, word, role) = match p.mean {
                        x if x >= 90.0 => ("●", "very confident model", Role::Accent),
                        x if x >= 70.0 => ("●", "confident model", Role::Accent),
                        x if x >= 50.0 => ("○", "low confidence", Role::Warm),
                        _ => ("▲", "very low confidence", Role::Bad),
                    };
                    out.push(a.bold(&a.paint(role, &format!("{glyph} {word}"))));
                    out.push(a.paint(
                        Role::Dim,
                        &format!(
                            "mean pLDDT {:.1} · {:.0}% of residues at 70 or more",
                            p.mean,
                            p.high_confidence_fraction * 100.0
                        ),
                    ));
                }
                None => {
                    out.push(a.bold(&a.paint(Role::Sea, "◆ experimental structure")));
                    out.push(a.paint(Role::Dim, "its B-factors are not confidences"));
                }
            }
        } else {
            section(&mut out, "model", "");
            out.push(a.paint(Role::Dim, "– not measured"));
        }

        // Key numbers.
        section(&mut out, "at a glance", "");
        let chains = data.chains.len();
        out.push(self.kv(
            lw,
            "residues",
            &format!(
                "{}{}",
                a.bold(&data.num_residues.to_string()),
                if chains > 1 {
                    a.paint(Role::Dim, &format!("  in {chains} chains"))
                } else {
                    String::new()
                }
            ),
        ));
        if let Some(v) = Self::interface(data) {
            if let Some(x) = v.metrics.ipsae_min {
                out.push(self.kv(
                    lw,
                    "ipSAE min",
                    &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
                ));
            }
        }
        if let Some(x) = conf.and_then(|c| c.iptm).filter(|_| chains > 1) {
            out.push(self.kv(
                lw,
                "ipTM",
                &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
            ));
        }
        if let Some(x) = conf.and_then(|c| c.ptm) {
            out.push(self.kv(
                lw,
                "pTM",
                &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
            ));
        }
        if let Some(p) = m.and_then(|m| m.plddt()) {
            out.push(self.kv(
                lw,
                "pLDDT mean",
                &self.gauge_value(
                    self.num(&format!("{:.1}", p.mean), ""),
                    p.mean / 100.0,
                    room,
                ),
            ));
        }
        if let Some(x) = m.and_then(|m| m.candidate_fitness_score) {
            out.push(self.kv(
                lw,
                "triage score",
                &self.gauge_value(self.num(&format!("{x:.1}"), "/ 100"), x / 100.0, room),
            ));
        }

        // Checks, each glyph and words.
        if let Some(m) = m {
            let mut checks = Vec::new();
            if let Some((level, text)) = self.rama_level(m) {
                checks.push(self.check(lw, level, "Ramachandran", text));
            }
            if let Some(c) = &m.steric_overlap {
                let level = match c.heavy_atom_overlap_score {
                    x if x < 5.0 => Level::Good,
                    x if x < 15.0 => Level::Warn,
                    _ => Level::Bad,
                };
                checks.push(self.check(
                    lw,
                    level,
                    "overlaps",
                    self.num(
                        &format!("{:.1}", c.heavy_atom_overlap_score),
                        "per 1000 atoms",
                    ),
                ));
            }
            if let Some(g) = &m.covalent_geometry {
                let level = if g.bonds.outliers == 0 {
                    Level::Good
                } else if g.bonds.outliers * 100 <= g.bonds.n {
                    Level::Warn
                } else {
                    Level::Bad
                };
                checks.push(self.check(
                    lw,
                    level,
                    "bond lengths",
                    format!(
                        "{} {}",
                        self.num(
                            &g.bonds.rmsz.map_or("–".into(), |z| format!("{z:.2}")),
                            "RMSZ"
                        ),
                        a.paint(Role::Dim, &format!("· {} beyond 4σ", g.bonds.outliers))
                    ),
                ));
                if let Some(pct) = g.rotamer_outlier_pct() {
                    let level = if pct < 1.0 { Level::Good } else { Level::Warn };
                    checks.push(self.check(
                        lw,
                        level,
                        "rotamers",
                        self.num(&format!("{pct:.1}"), "% outliers"),
                    ));
                }
            }
            if !checks.is_empty() {
                section(&mut out, "checks", "");
                out.extend(checks);
            }
        }

        // The chain from N to C: chains, fold, confidence.
        let tw = w.saturating_sub(lw + 2);
        if tw >= 8 && (!data.plddts.is_empty() || !data.dssp.is_empty()) {
            section(&mut out, "along the chain", "N → C");
            let pad = " ".repeat(lw + 2);
            if let Some(r) = self.chain_ruler(data, tw) {
                out.push(format!("{pad}{r}"));
            }
            if let Some(s) = self.ss_cells(data, tw) {
                out.push(self.kv(lw, "fold", &s));
            }
            if !data.plddts.is_empty() {
                out.push(self.kv(
                    lw,
                    if predicted { "pLDDT" } else { "B-factor" },
                    &self.plddt_cells(data, tw),
                ));
            }
            out.push(format!(
                "{pad}{}",
                self.residue_axis(tw, data.num_residues.max(data.plddts.len()))
            ));
            // The legends go under the track, or both from the left edge when either is wider.
            let mut legends = Vec::new();
            if !data.dssp.is_empty() {
                legends.push(self.ss_legend(data));
            }
            if predicted && !data.plddts.is_empty() {
                legends.push(self.plddt_legend());
            }
            let widest = legends
                .iter()
                .map(|l| l.iter().map(|i| visible_width(i) + 2).sum::<usize>())
                .max()
                .unwrap_or(0);
            let (indent, room) = if widest <= tw + 2 {
                (pad.as_str(), tw)
            } else {
                ("", w)
            };
            for l in legends {
                out.push(format!("{indent}{}", fit_items(&l, room)));
            }
        }
        out.into_iter().map(|l| truncate_to_width(&l, w)).collect()
    }

    /// A large Ramachandran plot and the backbone, side-chain and covalent numbers.
    fn geometry_page(&self, data: &DashboardData, w: usize, avail: usize) -> Vec<String> {
        let a = &self.ansi;
        let m = data.metrics.as_ref();
        let mut groups = Vec::new();
        if let Some(r) = m.and_then(|m| m.ramachandran_stats.as_ref()) {
            let mut g = Group::new("backbone", "");
            g.row(
                "favoured",
                self.num(&format!("{:.1}", r.favored_fraction * 100.0), "%"),
            );
            g.row(
                "allowed",
                self.num(&format!("{:.1}", r.allowed_fraction * 100.0), "%"),
            );
            g.row(
                "outliers",
                format!(
                    "{} {}",
                    self.num(&r.outlier_count.to_string(), ""),
                    a.paint(Role::Dim, &format!("of {}", r.total_evaluated))
                ),
            );
            if let Some(cg) = m.and_then(|m| m.covalent_geometry.as_ref()) {
                g.row(
                    "cis peptides",
                    format!(
                        "{} {}",
                        self.num(&cg.cis_nonproline.to_string(), "non-Pro"),
                        a.paint(Role::Dim, &format!("· {} Pro", cg.cis_proline))
                    ),
                );
                g.row("twisted", self.num(&cg.twisted.to_string(), "peptides"));
            }
            groups.push(g);
        }
        if let Some(cg) = m.and_then(|m| m.covalent_geometry.as_ref()) {
            let mut g = Group::new("side chains", "");
            if let Some(p) = cg.rotamer_outlier_pct() {
                g.row("rotamer outliers", self.num(&format!("{p:.1}"), "%"));
            }
            g.row(
                "rotamers allowed",
                self.num(
                    &cg.rotamer_allowed.to_string(),
                    &format!("of {}", cg.rotamer_residues),
                ),
            );
            if let Some(p) = cg.cbeta_outlier_pct() {
                g.row("Cβ deviations", self.num(&format!("{p:.1}"), "%"));
            }
            groups.push(g);
            let mut g = Group::new("covalent", "RMSZ · beyond 4σ");
            let stat = |s: &proteus_core::geometry::RestraintStats| {
                format!(
                    "{} {}",
                    a.bold(&pad_to_width(
                        &s.rmsz.map_or("–".into(), |z| format!("{z:.2}")),
                        5
                    )),
                    a.paint(Role::Dim, &format!("{}/{}", s.outliers, s.n))
                )
            };
            g.row("bonds", stat(&cg.bonds));
            g.row("angles", stat(&cg.angles));
            g.row("chirality", stat(&cg.chiralities));
            g.row("planes", stat(&cg.planes));
            groups.push(g);
        }
        let numbers = self.columns(&groups, w);

        // The plot takes the room the numbers leave, square in terminal cells (a cell is about
        // twice as tall as it is wide), 9 rows at least.
        // When the numbers cannot fit under it anyway, the plot fills the first screen and the
        // numbers scroll below it.
        let fixed = 1
            + 2
            + 1
            + if numbers.is_empty() {
                0
            } else {
                numbers.len() + 1
            };
        let max_w = w.saturating_sub(8).max(9);
        let plot_room = if avail >= fixed + 9 {
            avail - fixed
        } else {
            avail.saturating_sub(1 + 2 + 1 + 1)
        };
        let mut plot_h = plot_room.clamp(9, 41);
        plot_h = plot_h.min(max_w / 2 + 1);
        if plot_h.is_multiple_of(2) {
            plot_h -= 1;
        }
        let mut plot_w = max_w.min(plot_h * 2 + 1).max(9);
        if plot_w.is_multiple_of(2) {
            plot_w -= 1;
        }
        let mut out = vec![self.rule("ramachandran", "φ ψ", w)];
        out.extend(self.rama_plot(data, plot_w, plot_h.max(5)));
        if let Some(r) = m.and_then(|m| m.ramachandran_stats.as_ref()) {
            let items = [
                format!(
                    "{} {} {}",
                    a.paint(Role::Accent, "●"),
                    a.paint(Role::Muted, "favoured"),
                    a.paint(Role::Text, &format!("{:.1}%", r.favored_fraction * 100.0))
                ),
                format!(
                    "{} {} {}",
                    a.paint(Role::Warm, "○"),
                    a.paint(Role::Muted, "allowed"),
                    a.paint(Role::Text, &format!("{:.1}%", r.allowed_fraction * 100.0))
                ),
                format!(
                    "{} {} {}",
                    a.paint(Role::Bad, "▲"),
                    a.paint(Role::Muted, "outliers"),
                    a.paint(Role::Text, &r.outlier_count.to_string())
                ),
            ];
            out.push(format!("       {}", fit_items(&items, w.saturating_sub(7))));
        }
        if !numbers.is_empty() {
            out.push(String::new());
            out.extend(numbers);
        }
        if m.is_none() {
            out.push(String::new());
            out.push(a.paint(Role::Dim, "– not measured"));
        }
        out.into_iter().map(|l| truncate_to_width(&l, w)).collect()
    }

    /// The φ/ψ plot, `plot_w` × `plot_h` cells inside a 7-column label gutter and its frame.
    fn rama_plot(&self, data: &DashboardData, plot_w: usize, plot_h: usize) -> Vec<String> {
        let a = &self.ansi;
        let mut grid: Vec<Option<(u8, char, Role)>> = vec![None; plot_w * plot_h];
        for (phi, psi, region) in &data.ramachandran_points {
            if let (Some(phi), Some(psi)) = (*phi, *psi) {
                let gx = (((phi + 180.0) / 360.0) * (plot_w as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_w - 1) as f64) as usize;
                let gy = (((180.0 - psi) / 360.0) * (plot_h as f64 - 1.0))
                    .round()
                    .clamp(0.0, (plot_h - 1) as f64) as usize;
                let idx = gy * plot_w + gx;
                let marker = rama_marker(*region);
                if grid[idx].is_none_or(|(rank, _, _)| marker.0 > rank) {
                    grid[idx] = Some(marker);
                }
            }
        }
        let mid_x = (plot_w - 1) / 2;
        let mid_y = (plot_h - 1) / 2;
        let line = a.fg(Role::Line);
        let dim = a.fg(Role::Dim);
        let mut out = Vec::with_capacity(plot_h + 2);
        for r in 0..plot_h {
            let label = if r == 0 {
                " +180°│"
            } else if r == mid_y / 2 && plot_h >= 9 {
                "  +90°│"
            } else if r == mid_y {
                "    0°├"
            } else if r == mid_y + (plot_h - 1 - mid_y) / 2 && plot_h >= 9 {
                "  -90°│"
            } else if r == plot_h - 1 {
                " -180°│"
            } else {
                "      │"
            };
            let right = if r == mid_y { "┤" } else { "│" };
            let mut row = String::with_capacity(plot_w * 4 + 16);
            let _ = write!(row, "{dim}{label}{RESET}");
            let psi_c = 180.0 - (r as f64 / (plot_h - 1) as f64) * 360.0;
            for c in 0..plot_w {
                if let Some((_, sym, role)) = grid[r * plot_w + c] {
                    let mut b = [0u8; 4];
                    row.push_str(&a.paint(role, sym.encode_utf8(&mut b)));
                    continue;
                }
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
                    row.push(' ');
                } else {
                    let _ = write!(row, "{line}{ch}{RESET}");
                }
            }
            let _ = write!(row, "{dim}{right}{RESET}");
            out.push(row);
        }
        // "      └" sits under the label gutter; the plot's own columns start after it.
        let mut axis = String::from("      └");
        for c in 0..plot_w {
            axis.push(if c == mid_x { '┴' } else { '─' });
        }
        axis.push('┘');
        out.push(format!("{dim}{axis}{RESET}"));
        // -180° under the first column, φ 0° centred on the middle, +180° ending at the last.
        let mut ticks = vec![' '; plot_w + 2];
        let put = |ticks: &mut Vec<char>, at: usize, s: &str| {
            for (k, ch) in s.chars().enumerate() {
                if let Some(t) = ticks.get_mut(at + k) {
                    *t = ch;
                }
            }
        };
        put(&mut ticks, 0, "-180°");
        put(&mut ticks, (mid_x + 1).saturating_sub(2), "φ 0°");
        put(&mut ticks, (plot_w + 1).saturating_sub(4), "+180°");
        out.push(format!(
            "      {dim}{}{RESET}",
            ticks.into_iter().collect::<String>()
        ));
        out
    }

    /// The predictor's numbers, a large PAE map with chain boundaries, and pLDDT per residue.
    fn confidence_page(&self, data: &DashboardData, w: usize, avail: usize) -> Vec<String> {
        let a = &self.ansi;
        let predicted = Self::predicted(data);
        let conf = data.confidence.as_ref().filter(|c| !c.is_empty());
        let lw = 12.min(w / 2);
        let room = w.saturating_sub(lw + 2);

        // Numbers first: they size what is left for the map.
        let mut numbers = Vec::new();
        if let Some(c) = conf {
            if let Some(x) = c.ptm {
                numbers.push(self.kv(
                    lw,
                    "pTM",
                    &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
                ));
            }
            if let Some(x) = c.iptm.filter(|_| data.chains.len() > 1) {
                numbers.push(self.kv(
                    lw,
                    "ipTM",
                    &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
                ));
            }
            if let Some(x) = c.confidence_score {
                numbers.push(self.kv(
                    lw,
                    "ranking",
                    &self.gauge_value(self.num(&format!("{x:.3}"), ""), x, room),
                ));
            }
            if let Some(p) = &c.pae {
                numbers.push(self.kv(lw, "mean PAE", &self.num(&format!("{:.1}", p.mean()), "Å")));
            }
            if c.chain_ptm.len() > 1 && c.chain_ptm.len() == data.chains.len() {
                let items: Vec<String> = data
                    .chains
                    .iter()
                    .zip(&c.chain_ptm)
                    .map(|((id, _), p)| {
                        format!(
                            "{} {}",
                            a.paint(Role::Muted, id),
                            a.bold(&format!("{p:.2}"))
                        )
                    })
                    .collect();
                numbers.push(self.kv(lw, "chain pTM", &fit_items(&items, room)));
            }
        }

        // pLDDT (or B-factor) block.
        let mut strip = Vec::new();
        let tw = w;
        if !data.plddts.is_empty() {
            if let Some(r) = self.chain_ruler(data, tw) {
                strip.push(r);
            }
            strip.push(self.plddt_cells(data, tw));
            strip.push(self.residue_axis(tw, data.plddts.len()));
            if predicted {
                strip.push(fit_items(&self.plddt_legend(), tw));
            }
            if let Some(m) = &data.metrics {
                let dim = |s: &str| a.paint(Role::Dim, s);
                let items: Vec<String> = match m.plddt() {
                    Some(p) => vec![
                        format!("{} {}", dim("mean"), a.bold(&format!("{:.1}", p.mean))),
                        format!("{} {}", dim("median"), a.bold(&format!("{:.1}", p.median))),
                        format!(
                            "{} {}",
                            dim("≥70"),
                            self.num(&format!("{:.0}", p.high_confidence_fraction * 100.0), "%")
                        ),
                        format!(
                            "{} {}",
                            dim("≥90"),
                            self.num(
                                &format!("{:.0}", p.very_high_confidence_fraction * 100.0),
                                "%"
                            )
                        ),
                    ],
                    None => vec![
                        format!(
                            "{} {}",
                            dim("mean"),
                            self.num(&format!("{:.1}", m.plddt_distribution.mean), "Å²")
                        ),
                        format!(
                            "{} {}",
                            dim("median"),
                            self.num(&format!("{:.1}", m.plddt_distribution.median), "Å²")
                        ),
                        dim("low to high: moss to khaki"),
                    ],
                };
                strip.push(fit_items(&items, tw));
            }
        }

        let mut out = Vec::new();
        let push_block = |out: &mut Vec<String>, block: Vec<String>| {
            if !out.is_empty() {
                out.push(String::new());
            }
            out.extend(block);
        };
        if !numbers.is_empty() {
            let mut b = vec![self.rule("predictor", "", w)];
            b.extend(numbers.iter().cloned());
            push_block(&mut out, b);
        }

        let pae = conf.and_then(|c| c.pae.as_ref());
        match pae {
            Some(p) => {
                // The map gets what the rest leaves, 6 rows at least.
                let rest = out.len()
                    + 1
                    + if strip.is_empty() { 0 } else { strip.len() + 2 }
                    + 1 // rule
                    + 3; // chain ruler, axis, legend
                let rows = avail.saturating_sub(rest).clamp(6, 40);
                let side = (w.saturating_sub(4)).min(rows * 2).min(p.n).max(2);
                let mut b = vec![self.rule("predicted aligned error", "aligned ↓ scored →", w)];
                b.extend(self.pae_map(p, &data.chains, side, w));
                push_block(&mut out, b);
            }
            None => {
                let why = if !predicted {
                    "none: an experimental structure"
                } else if conf.is_some() {
                    "only the predictor's summary numbers were found"
                } else {
                    "none found beside this model"
                };
                push_block(
                    &mut out,
                    vec![
                        self.rule("predicted aligned error", "", w),
                        a.paint(Role::Dim, &format!("– {why}")),
                    ],
                );
            }
        }
        if !strip.is_empty() {
            let (title, note) = if predicted {
                ("plddt", "per residue, AlphaFold bands")
            } else {
                ("b-factor", "per residue, not a confidence")
            };
            let mut b = vec![self.rule(title, note, w)];
            b.extend(strip);
            push_block(&mut out, b);
        }
        out.into_iter().map(|l| truncate_to_width(&l, w)).collect()
    }

    /// A `side` × `side` pixel PAE map in half-blocks (each cell two pixels high) behind a
    /// three-column gutter of chain IDs, chain boundaries drawn in the hairline colour.
    fn pae_map(
        &self,
        p: &proteus_core::pae::PredictedAlignedError,
        chains: &[(String, usize)],
        side: usize,
        w: usize,
    ) -> Vec<String> {
        use crate::rasterizer::shader::pae_color;
        let a = &self.ansi;
        const GUTTER: usize = 3;
        let n = p.n;
        let px = |i: usize| i * n / side;
        let cell = |r: usize, c: usize| -> f32 {
            let (i0, i1) = (px(r), px(r + 1).max(px(r) + 1).min(n));
            let (j0, j1) = (px(c), px(c + 1).max(px(c) + 1).min(n));
            let mut sum = 0.0f32;
            for i in i0..i1 {
                for j in j0..j1 {
                    sum += p.get(i, j);
                }
            }
            sum / ((i1 - i0) * (j1 - j0)).max(1) as f32
        };
        // Chain starts as pixel indices, when the chains cover the matrix's first residues.
        let residues: usize = chains.iter().map(|(_, k)| k).sum();
        let mut spans: Vec<(char, usize, usize)> = Vec::new();
        if chains.len() > 1 && residues <= n {
            let mut start = 0;
            for (id, k) in chains {
                let (s, e) = (start * side / n, (start + k) * side / n);
                spans.push((id.chars().next().unwrap_or('?'), s, e.max(s + 1)));
                start += k;
            }
        }
        let boundary = |i: usize| spans.iter().skip(1).any(|(_, s, _)| *s == i);
        let line_c = a.theme.line;

        let mut out = Vec::new();
        if !spans.is_empty() {
            let mut ruler = vec![' '; side];
            for (id, s, e) in &spans {
                if let Some(c) = ruler.get_mut((s + e) / 2) {
                    *c = *id;
                }
            }
            out.push(format!(
                "{}{}",
                " ".repeat(GUTTER),
                a.paint(Role::Muted, &ruler.into_iter().collect::<String>())
            ));
        }
        for r in (0..side).step_by(2) {
            let label = spans
                .iter()
                .find(|(_, s, e)| (s + e) / 2 / 2 == r / 2)
                .map_or(' ', |(id, _, _)| *id);
            let mut line = format!("{} ", a.paint(Role::Muted, &format!("{label:>2}")));
            for c in 0..side {
                let pix = |row: usize| -> ColorRGB {
                    if boundary(row) || boundary(c) {
                        line_c
                    } else {
                        pae_color(cell(row, c), p.max)
                    }
                };
                if a.backgrounds() {
                    let bottom = if r + 1 < side { pix(r + 1) } else { pix(r) };
                    line.push_str(&a.paint_rgb_on(pix(r), bottom, "▀"));
                } else if boundary(c) {
                    line.push('│');
                } else if boundary(r) || boundary(r + 1) {
                    line.push('─');
                } else {
                    let bottom = if r + 1 < side {
                        cell(r + 1, c)
                    } else {
                        cell(r, c)
                    };
                    let t = (cell(r, c) + bottom) / 2.0 / p.max;
                    line.push(match t {
                        t if t < 0.15 => '█',
                        t if t < 0.35 => '▓',
                        t if t < 0.6 => '▒',
                        _ => '░',
                    });
                }
            }
            out.push(line);
        }
        out.push(format!(
            "{}{}",
            " ".repeat(GUTTER),
            self.residue_axis(side, n)
        ));
        // The scale: low (confident) to high, as a ramp.
        let cells = 12usize;
        let mut ramp = String::new();
        for k in 0..cells {
            let v = p.max * k as f32 / (cells - 1) as f32;
            if a.colours() {
                ramp.push_str(&a.paint_rgb(pae_color(v, p.max), "█"));
            } else {
                ramp.push(['█', '▓', '▒', '░'][(k * 4 / cells).min(3)]);
            }
        }
        // The note is left out whole when it does not fit.
        let items = [
            format!(
                "{} {ramp} {}",
                a.paint(Role::Dim, "0"),
                a.paint(Role::Dim, &format!("{:.0} Å", p.max))
            ),
            a.paint(Role::Dim, "dark is a confident placement"),
        ];
        out.push(format!(
            "{}{}",
            " ".repeat(GUTTER),
            fit_items(&items, w.saturating_sub(GUTTER))
        ));
        out
    }

    /// Everything measured, grouped, in two columns where the panel is wide enough.
    fn measurements_page(&self, data: &DashboardData, w: usize) -> Vec<String> {
        let a = &self.ansi;
        let Some(m) = data.metrics.as_ref() else {
            return vec![
                self.rule("measurements", "", w),
                a.paint(Role::Dim, "– not measured"),
            ];
        };
        let f1 = |x: f64| format!("{x:.1}");
        let pct = |x: f64| format!("{:.1}", x * 100.0);
        let mut groups = Vec::new();

        let mut g = Group::new("size and surface", "");
        g.row("residues", self.num(&data.num_residues.to_string(), ""));
        if data.chains.len() > 1 {
            let ids: Vec<&str> = data.chains.iter().map(|(c, _)| c.as_str()).collect();
            g.row(
                "chains",
                format!(
                    "{} {}",
                    self.num(&data.chains.len().to_string(), ""),
                    a.paint(Role::Dim, &ids.join(" "))
                ),
            );
        }
        g.row(
            "radius of gyration",
            self.num(&format!("{:.2}", m.radius_of_gyration), "Å"),
        );
        g.row("Cα contacts ≤8 Å", self.num(&pct(m.contact_density), "%"));
        if let Some(s) = &m.sasa_metrics {
            g.row("SASA", self.num(&format!("{:.0}", s.total_sasa), "Å²"));
            g.row(
                "polar SASA",
                self.num(&format!("{:.0}", s.polar_sasa), "Å²"),
            );
            g.row(
                "apolar SASA",
                self.num(&format!("{:.0}", s.apolar_sasa), "Å²"),
            );
            g.row(
                "hydrophobic burial",
                self.num(&pct(s.hydrophobic_burial_ratio), "%"),
            );
        }
        groups.push(g);

        let mut g = Group::new("fold", "");
        if let Some(ss) = &m.secondary_structure_summary {
            g.row(
                "helix",
                self.num(&format!("{:.0}", ss.helix_fraction * 100.0), "%"),
            );
            g.row(
                "strand",
                self.num(&format!("{:.0}", ss.strand_fraction * 100.0), "%"),
            );
            g.row(
                "coil",
                self.num(&format!("{:.0}", ss.coil_fraction * 100.0), "%"),
            );
        }
        g.row(
            "disulfide bonds",
            if data.num_disulfides > 0 {
                format!(
                    "{} {}",
                    a.paint_rgb(crate::brand::structure::DISULFIDE, "■"),
                    self.num(&data.num_disulfides.to_string(), "pairs")
                )
            } else {
                a.paint(Role::Dim, "none")
            },
        );
        groups.push(g);

        let conf = data.confidence.as_ref();
        let mut g = match m.plddt() {
            Some(p) => {
                let mut g = Group::new("confidence", "");
                g.row("pLDDT mean", self.num(&f1(p.mean), ""));
                g.row("pLDDT median", self.num(&f1(p.median), ""));
                g.row("pLDDT ≥70", self.num(&pct(p.high_confidence_fraction), "%"));
                g.row(
                    "pLDDT ≥90",
                    self.num(&pct(p.very_high_confidence_fraction), "%"),
                );
                g
            }
            None => {
                let mut g = Group::new("b-factor", "not a confidence");
                g.row("mean", self.num(&f1(m.plddt_distribution.mean), "Å²"));
                g.row("median", self.num(&f1(m.plddt_distribution.median), "Å²"));
                g
            }
        };
        if let Some(c) = conf {
            if let Some(x) = c.ptm {
                g.row("pTM", self.num(&format!("{x:.3}"), ""));
            }
            if let Some(x) = c.iptm.filter(|_| data.chains.len() > 1) {
                g.row("ipTM", self.num(&format!("{x:.3}"), ""));
            }
            if let Some(p) = &c.pae {
                g.row("mean PAE", self.num(&f1(p.mean()), "Å"));
            }
            if let Some(x) = c.confidence_score {
                g.row("ranking score", self.num(&format!("{x:.3}"), ""));
            }
        }
        groups.push(g);

        if let Some(v) = Self::interface(data) {
            let im = &v.metrics;
            let n = |x: Option<f64>, d: usize| x.map_or("–".to_string(), |v| format!("{v:.d$}"));
            let mut g = Group::new(
                "interface",
                &format!(
                    "{} → {}",
                    data.name_chains(&im.binder_chains),
                    data.name_chains(&im.target_chains)
                ),
            );
            g.row("ipSAE min", self.num(&n(im.ipsae_min, 3), ""));
            g.row("ipSAE max", self.num(&n(im.ipsae_max, 3), ""));
            g.row("ipAE", self.num(&n(im.ipae, 1), "Å"));
            g.row("LIS", self.num(&n(im.lis, 3), ""));
            g.row("Sc", self.num(&n(im.shape_complementarity, 2), ""));
            g.row("dSASA", self.num(&format!("{:.0}", im.dsasa), "Å²"));
            g.row(
                "contacts",
                format!(
                    "{} {}",
                    self.num(&im.binder_interface_residues.to_string(), &im.binder_chains),
                    self.num(
                        &format!("· {}", im.target_interface_residues),
                        &im.target_chains
                    ),
                ),
            );
            g.row("H-bonds", self.num(&im.interface_hbonds.to_string(), ""));
            g.row(
                "salt bridges",
                self.num(&im.interface_salt_bridges.to_string(), ""),
            );
            groups.push(g);
        }

        let mut g = Group::new("geometry", "");
        if let Some(r) = &m.ramachandran_stats {
            g.row("Rama favoured", self.num(&pct(r.favored_fraction), "%"));
            g.row(
                "Rama outliers",
                format!(
                    "{} {}",
                    self.num(&r.outlier_count.to_string(), ""),
                    a.paint(Role::Dim, &format!("of {}", r.total_evaluated))
                ),
            );
        }
        if let Some(cg) = &m.covalent_geometry {
            if let Some(p) = cg.rotamer_outlier_pct() {
                g.row("rotamer outliers", self.num(&format!("{p:.1}"), "%"));
            }
            let z = |s: &proteus_core::geometry::RestraintStats| {
                format!(
                    "{} {}",
                    a.bold(&s.rmsz.map_or("–".into(), |z| format!("{z:.2}"))),
                    a.paint(Role::Dim, &format!("· {} > 4σ", s.outliers))
                )
            };
            g.row("bond RMSZ", z(&cg.bonds));
            g.row("angle RMSZ", z(&cg.angles));
            g.row("cis non-Pro", self.num(&cg.cis_nonproline.to_string(), ""));
        }
        if let Some(c) = &m.steric_overlap {
            g.row(
                "overlaps /1000",
                self.num(&f1(c.heavy_atom_overlap_score), "atoms"),
            );
            if c.clash_count > 0 {
                g.row(
                    "worst overlap",
                    format!(
                        "{} {}",
                        self.num(&format!("{:.2}", c.worst_overlap), "Å"),
                        a.paint(Role::Dim, &format!("of {}", c.clash_count))
                    ),
                );
            }
        }
        if !g.rows.is_empty() {
            groups.push(g);
        }

        if let Some(net) = &m.interaction_network {
            let s = &net.summary;
            let mut g = Group::new("interactions", "");
            g.row("H-bonds", self.num(&s.total_hbonds.to_string(), ""));
            g.row(
                "bb · bb–sc · sc",
                format!(
                    "{} {} {}",
                    self.num(&s.bb_bb_hbonds.to_string(), "·"),
                    self.num(&s.bb_sc_hbonds.to_string(), "·"),
                    a.bold(&s.sc_sc_hbonds.to_string())
                ),
            );
            g.row(
                "salt bridges",
                self.num(&s.total_salt_bridges.to_string(), ""),
            );
            g.row(
                "π–π stacks",
                self.num(&s.total_pi_pi_stacks.to_string(), ""),
            );
            g.row("cation–π", self.num(&s.total_cation_pi.to_string(), ""));
            g.row("per 100 residues", self.num(&f1(s.network_density), ""));
            groups.push(g);
        }

        if let Some(x) = m.candidate_fitness_score {
            let mut g = Group::new("triage", "");
            g.row("score", self.num(&f1(x), "/ 100"));
            groups.push(g);
        }
        self.columns(&groups, w)
            .into_iter()
            .map(|l| truncate_to_width(&l, w))
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::brand::ColorDepth;
    use proteus_core::models::PlddtDistribution;
    use proteus_core::structure::RamachandranStats;

    fn plain_text(line: &str) -> String {
        ansi_segments(line)
            .filter(|(esc, _)| !esc)
            .map(|(_, t)| t)
            .collect()
    }

    fn render(
        data: &DashboardData,
        page: DashboardPage,
        depth: ColorDepth,
        w: usize,
        h: usize,
    ) -> Vec<String> {
        let mut view = DashboardView { page, scroll: 0 };
        DashboardRenderer::with_ansi(Ansi::with_depth(depth)).generate_lines(data, &mut view, w, h)
    }

    fn text(lines: &[String]) -> String {
        lines
            .iter()
            .map(|l| plain_text(l))
            .collect::<Vec<_>>()
            .join("\n")
    }

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

    fn metrics() -> BiophysicalMetrics {
        BiophysicalMetrics {
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
            covalent_geometry: None,
            candidate_fitness_score: Some(87.4),
        }
    }

    fn sample_data(title: &str) -> DashboardData {
        DashboardData {
            title: title.to_string(),
            num_residues: 46,
            num_disulfides: 3,
            metrics: Some(metrics()),
            plddts: vec![92.0; 46],
            ramachandran_points: vec![
                (Some(-60.0), Some(-45.0), RamachandranRegion::Favored),
                (Some(-120.0), Some(135.0), RamachandranRegion::Favored),
            ],
            chains: vec![("A".into(), 46)],
            dssp: "HHHHHHHHHHEEEEE----".repeat(3)[..46].to_string(),
            chain_names: Vec::new(),
            ..Default::default()
        }
    }

    /// Two chains of 30 and 16 residues, a PAE low within each chain and high between them.
    fn complex_data() -> DashboardData {
        let n = 46;
        let mut values = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = if (i < 30) == (j < 30) { 2.0 } else { 25.0 };
            }
        }
        DashboardData {
            chains: vec![("A".into(), 30), ("B".into(), 16)],
            confidence: Some(proteus_core::pae::PredictionConfidence {
                pae: Some(proteus_core::pae::PredictedAlignedError {
                    n,
                    values,
                    max: 31.75,
                }),
                ptm: Some(0.81),
                iptm: Some(0.42),
                ..Default::default()
            }),
            ..sample_data("complex")
        }
    }

    /// Every page's every line fits its panel, at every width and height: a line one column
    /// too wide wraps into the next row. Measured in display columns — the pages carry `φ`,
    /// `ψ`, `Å` and chain IDs. Section rules reach the right margin exactly.
    #[test]
    fn every_page_fits_the_panel_at_every_size() {
        for data in [
            sample_data("1crn.pdb"),
            sample_data("a_very_long_structure_file_name_蛋白質_model_0001.cif"),
            complex_data(),
            DashboardData::default(),
        ] {
            for page in DashboardPage::ALL {
                for depth in [ColorDepth::TrueColor, ColorDepth::None] {
                    for width in 20..=100 {
                        for height in [10, 17, 18, 24, 30, 45, 55] {
                            let lines = render(&data, page, depth, width, height);
                            assert_eq!(lines.len(), height);
                            for (i, line) in lines.iter().enumerate() {
                                let lw = visible_width(line);
                                assert!(
                                    lw <= width,
                                    "{page:?} {width}x{height} line {i} is {lw} columns: {line:?}"
                                );
                                let plain = plain_text(line);
                                if (40..2 * MIN_COLUMN + 6).contains(&width)
                                    && plain.trim_start().starts_with('(')
                                    && plain.ends_with('─')
                                {
                                    assert_eq!(
                                        lw,
                                        width - MARGIN_R,
                                        "{page:?} {width}x{height} rule {i}: {plain}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn keys_pick_pages_and_the_selector_shows_which() {
        assert_eq!(DashboardPage::from_key('1'), Some(DashboardPage::Overview));
        assert_eq!(
            DashboardPage::from_key('4'),
            Some(DashboardPage::Measurements)
        );
        assert_eq!(DashboardPage::from_key('0'), None);
        assert_eq!(DashboardPage::from_key('5'), None);
        let data = sample_data("1crn");
        for page in DashboardPage::ALL {
            let lines = render(&data, page, ColorDepth::TrueColor, 80, 40);
            // The showing page is filled: ground on accent, then its name.
            let on = Ansi::with_depth(ColorDepth::TrueColor).tab_on(&format!(
                " {} {} ",
                page.name(),
                page.index() + 1
            ));
            assert!(lines[0].contains(&on), "{page:?}: {:?}", lines[0]);
            for other in DashboardPage::ALL.iter().filter(|p| **p != page) {
                assert!(plain_text(&lines[0]).contains(other.name()));
            }
            // Under NO_COLOR the showing page is still marked (reverse video), not by colour.
            let plain = render(&data, page, ColorDepth::None, 80, 40);
            assert!(plain[0].contains("\x1b[1;7m"), "{:?}", plain[0]);
        }
        // Narrow: the other pages keep their numbers, the showing one its name.
        let narrow = text(&render(&data, DashboardPage::Geometry, ColorDepth::None, 30, 20)[..1]);
        assert!(
            narrow.contains("geometry") && narrow.contains('1') && narrow.contains('4'),
            "{narrow}"
        );
    }

    #[test]
    fn sections_are_titled_and_spaced() {
        for page in DashboardPage::ALL {
            let lines = render(&complex_data(), page, ColorDepth::None, 70, 55);
            let t: Vec<String> = lines.iter().map(|l| plain_text(l)).collect();
            // Section titles, in either column: a `(` at the margin or after the column gap.
            let mut rules = Vec::new();
            for (i, l) in t.iter().enumerate() {
                let chars: Vec<char> = l.chars().collect();
                for (c, ch) in chars.iter().enumerate() {
                    let at_column = c == MARGIN_L || (c >= 3 && chars[c - 3..c] == [' '; 3]);
                    let titled = chars[c..].contains(&')') && chars[c..].contains(&'─');
                    if *ch == '(' && at_column && titled {
                        rules.push((i, c));
                    }
                }
            }
            assert!(!rules.is_empty(), "{page:?} {t:#?}");
            // A blank line above every section that is not at the top of its column.
            for &(i, c) in &rules {
                if i <= 3 {
                    continue;
                }
                let above: String = t[i - 1].chars().skip(c).take(8).collect();
                assert!(
                    above.trim().is_empty(),
                    "{page:?}: no space above {:?}",
                    t[i]
                );
            }
            // Nothing touches the separator: the left margin is blank on every line.
            assert!(t.iter().all(|l| l.is_empty() || l.starts_with("  ")));
        }
    }

    #[test]
    fn the_ramachandran_axis_ticks_line_up_with_the_plot() {
        let data = DashboardData {
            num_residues: 3,
            ..Default::default()
        };
        for (w, h) in [(40, 30), (70, 55), (50, 33)] {
            let plain: Vec<String> = render(&data, DashboardPage::Geometry, ColorDepth::None, w, h)
                .iter()
                .map(|l| plain_text(l))
                .collect();
            let cross = plain
                .iter()
                .find_map(|l| l.find('┼').map(|b| l[..b].chars().count()))
                .unwrap();
            let axis = plain.iter().find(|l| l.contains('└')).unwrap();
            let tick = axis[..axis.find('┴').unwrap()].chars().count();
            assert_eq!(tick, cross, "{axis}");
            let labels = plain.iter().find(|l| l.contains("φ 0°")).unwrap();
            let phi = labels[..labels.find('φ').unwrap()].chars().count();
            assert!(phi.abs_diff(cross) <= 2, "{labels}");
        }
    }

    /// The geometry page's plot grows with the panel: on a tall panel it is far larger than
    /// the old fixed 11 rows.
    #[test]
    fn the_ramachandran_plot_uses_the_room() {
        let data = sample_data("x");
        let rows = |h| {
            render(&data, DashboardPage::Geometry, ColorDepth::None, 80, h)
                .iter()
                .filter(|l| {
                    let p = plain_text(l);
                    p.contains("°│") || p.contains("°├") || p.trim_start().starts_with('│')
                })
                .count()
        };
        assert!(rows(53) >= 25, "{}", rows(53));
        assert!(rows(33) >= 9, "{}", rows(33));
    }

    #[test]
    fn the_pae_map_marks_its_chains() {
        let data = complex_data();
        let t = text(&render(
            &data,
            DashboardPage::Confidence,
            ColorDepth::None,
            70,
            50,
        ));
        assert!(t.contains("(predicted aligned error)"), "{t}");
        assert!(
            t.contains("pTM") && t.contains("0.810") && t.contains("ipTM"),
            "{t}"
        );
        // The boundary between the chains runs down and across the map.
        assert!(t.lines().any(|l| l.contains('│') && l.contains('█')), "{t}");
        assert!(
            t.lines()
                .any(|l| l.contains('│') && l.matches('─').count() > 10),
            "{t}"
        );
        // A structure without a PAE says why instead.
        let t = text(&render(
            &sample_data("x"),
            DashboardPage::Confidence,
            ColorDepth::None,
            70,
            40,
        ));
        assert!(t.contains("– none found beside this model"), "{t}");
    }

    #[test]
    fn a_tall_page_scrolls_and_says_so() {
        let data = complex_data();
        let mut view = DashboardView {
            page: DashboardPage::Measurements,
            scroll: 0,
        };
        let r = DashboardRenderer::with_ansi(Ansi::with_depth(ColorDepth::None));
        let t = text(&r.generate_lines(&data, &mut view, 40, 12));
        assert!(t.contains("more") && t.contains("pgdn"), "{t}");
        view.scroll = 10_000;
        let last = r.generate_lines(&data, &mut view, 40, 12);
        assert!(
            view.scroll > 0 && view.scroll < 10_000,
            "scroll clamped: {}",
            view.scroll
        );
        assert!(text(&last).contains("above"));
        // A page that fits is not scrolled.
        let mut view = DashboardView {
            page: DashboardPage::Overview,
            scroll: 5,
        };
        r.generate_lines(&sample_data("x"), &mut view, 80, 55);
        assert_eq!(view.scroll, 0);
    }

    #[test]
    fn colours_follow_the_depth() {
        let data = sample_data("1CRN");
        let full = text(&render(
            &data,
            DashboardPage::Measurements,
            ColorDepth::TrueColor,
            80,
            40,
        ));
        assert!(full.contains("9.76"), "{full}");
        // pLDDT 92 everywhere: the strip is the AlphaFold ≥90 blue, #0053D6, as in the ribbon.
        let conf = render(
            &data,
            DashboardPage::Confidence,
            ColorDepth::TrueColor,
            60,
            40,
        )
        .join("\n");
        assert!(conf.contains("\x1b[38;2;0;83;214m█"), "{conf:?}");
        // Without colour, not one colour escape, and every Ramachandran region still has its
        // own glyph.
        for page in DashboardPage::ALL {
            let plain = render(&data, page, ColorDepth::None, 60, 40).join("\n");
            assert!(
                !plain.contains("\x1b[38;") && !plain.contains("\x1b[48;"),
                "{page:?} {plain:?}"
            );
        }
        assert!(text(&render(
            &data,
            DashboardPage::Geometry,
            ColorDepth::None,
            60,
            40
        ))
        .contains('●'));
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
        assert_eq!(split_line("ab", "cd", 10, 2), "ab      cd");
        assert_eq!(split_line("abcd", "efgh", 9, 2), "abcd");
    }

    /// A complex shows its interface: the verdict as glyph and word, then the numbers.
    #[test]
    fn a_complex_shows_its_interface() {
        let s = crate::parse_pdb_structure(include_str!(
            "../../../proteus-core/tests/data/2ptc_EI.pdb"
        ))
        .unwrap();
        let data = DashboardData {
            title: "2ptc".into(),
            interfaces: s.interfaces.clone(),
            num_residues: s.num_residues,
            chains: vec![("E".into(), 223), ("I".into(), 58)],
            metrics: Some(metrics()),
            ..Default::default()
        };
        let t = text(&render(
            &data,
            DashboardPage::Overview,
            ColorDepth::None,
            60,
            40,
        ));
        // BPTI (58 residues) is the smaller chain, so it is the binder.
        assert!(t.contains("(interface) I → E"), "{t}");
        assert!(
            t.contains("– no verdict") && t.contains("no PAE beside this model"),
            "{t}"
        );
        assert!(t.contains("binder") && t.contains("58 residues"), "{t}");
        let t = text(&render(
            &data,
            DashboardPage::Measurements,
            ColorDepth::None,
            60,
            55,
        ));
        assert!(t.contains("Sc") && t.contains("0.7"), "{t}");
    }
}
