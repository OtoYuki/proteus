//! Drawing the home screen in the Proteus identity. Pure: it reads [`App`] and writes a frame,
//! so every screen can be rendered into a `TestBackend` in tests.
//!
//! The rules it follows (docs/design/2026-09-23-proteus-identity-design.md): colours only from
//! brand roles through [`Look`]; hairlines, not boxes; panel names in `(parentheses)`; a
//! state is always a glyph and a word, never a colour alone.

use super::app::{
    display_name, engine, model, short_id, Analysis, App, FieldKind, FormKind, Preview, Tab,
};
use super::style::Look;
use chrono::Utc;
use proteus_core::models::JobStatus;
use proteus_render::brand::{self, mark, matrix};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState, Wrap,
};
use ratatui::Frame;
use std::path::Path;

/// Rows the header takes: the bar and a hairline under it; a short terminal gets the bar only.
fn header_rows(area: Rect) -> u16 {
    if area.height >= 18 && area.width >= 60 {
        2
    } else {
        1
    }
}

pub fn draw(f: &mut Frame, app: &App) {
    let look = &app.look;
    f.render_widget(Block::new().style(look.base()), f.area());
    let [top, body, keys, status] = Layout::vertical([
        Constraint::Length(header_rows(f.area())),
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(f.area());

    draw_header(f, top, app);
    let body = body.inner(ratatui::layout::Margin::new(2, 1));
    match app.tab {
        Tab::Jobs => draw_jobs(f, body, app),
        Tab::Structures => draw_structures(f, body, app),
        Tab::Run => draw_run(f, body, app),
    }
    f.render_widget(Paragraph::new(key_hints(app, keys.width as usize)), keys);
    draw_status(f, status, app);
    if app.help {
        // A kitty picture sits above the text layer: placed, it would cover the help. No pane
        // asks for one while the help is up, so the loop takes it down.
        *app.preview_want.borrow_mut() = None;
        draw_help(f, f.area(), look);
    }
}

// ---------------------------------------------------------------------------------------------
// Header, key hints, status line

/// The tabs as one segmented control: the showing tab filled in the accent, the others quiet.
/// Icons where the terminal draws them; the number keys are in the key strip.
fn tab_spans(app: &App) -> Vec<Span<'static>> {
    let look = &app.look;
    let mut spans = Vec::new();
    for t in Tab::ALL.iter() {
        let name = t.title().to_lowercase();
        // A Nerd Font glyph is drawn a little wider than a cell: two spaces after it.
        let label = if look.icons {
            format!("  {}  {name}  ", t.icon())
        } else {
            format!("  {name}  ")
        };
        let style = if *t == app.tab {
            look.tab_on()
        } else {
            look.muted()
        };
        spans.push(Span::styled(label, style));
        spans.push(Span::raw(" "));
    }
    spans
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let [bar, rule] = if area.height >= 2 {
        Layout::vertical([Constraint::Length(1); 2]).areas(area)
    } else {
        [area, Rect::default()]
    };
    let mut spans = vec![
        Span::styled("  proteus", look.accent().add_modifier(Modifier::BOLD)),
        Span::raw("    "),
    ];
    spans.extend(tab_spans(app));
    f.render_widget(Paragraph::new(Line::from(spans)), bar);
    let mut right = job_summary(app);
    right.push(Span::styled("  ", look.dim()));
    f.render_widget(
        Paragraph::new(Line::from(right)).alignment(Alignment::Right),
        bar,
    );
    if rule.height > 0 {
        f.render_widget(
            Block::new().borders(Borders::TOP).border_style(look.line()),
            rule,
        );
    }
}

/// "● 1 running  ✓ 7 done  ✗ 4 failed", states with no jobs left out, in the order a
/// reader cares about: what is moving, what finished, what went wrong.
fn job_summary(app: &App) -> Vec<Span<'static>> {
    let jobs = &app.jobs.all;
    let count = |s: JobStatus| jobs.iter().filter(|j| j.job.status == s).count();
    let mut out = Vec::new();
    for s in [
        JobStatus::Running,
        JobStatus::Queued,
        JobStatus::Completed,
        JobStatus::Failed,
    ] {
        let n = count(s.clone());
        if n == 0 {
            continue;
        }
        let st = state(&app.look, &s, app.tick);
        let text = st.content.to_string();
        let (glyph, word) = text.split_once(' ').unwrap_or(("", &text));
        out.push(Span::styled(format!("{glyph} {n} {word}  "), st.style));
    }
    out
}

/// Key hints: the keys in the accent, their meaning dim.
fn key_hints(app: &App, width: usize) -> Line<'static> {
    let look = &app.look;
    let pairs: &[(&str, &str)] = if app.typing() {
        match app.tab {
            Tab::Jobs if app.jobs.renaming.is_some() => {
                &[("type", "a name"), ("⏎", "rename"), ("esc", "cancel")]
            }
            Tab::Jobs => &[("type", "to filter"), ("⏎", "keep"), ("esc", "clear")],
            _ => &[
                ("type", "into the field"),
                ("⏎ ↓", "next field"),
                ("esc", "stop typing"),
                ("alt 1-3", "tabs"),
            ],
        }
    } else {
        match app.tab {
            Tab::Jobs => &[
                ("↑↓", "move"),
                ("⏎", "view"),
                ("w", "in browser"),
                ("i", "report"),
                ("/", "filter"),
                ("s", "sort"),
                ("n", "rename"),
                ("x", "delete"),
                ("1 2 3", "tabs"),
                ("?", "keys"),
                ("q", "quit"),
            ],
            Tab::Structures => &[
                ("↑↓", "move"),
                ("⏎", "open"),
                ("←", "up"),
                ("w", "in browser"),
                ("a", "report"),
                ("~", "home"),
                ("1 2 3", "tabs"),
                ("?", "keys"),
                ("q", "quit"),
            ],
            Tab::Run => &[
                ("↑↓", "field"),
                ("⏎", "edit · run"),
                ("←→", "choose"),
                ("f", "other form"),
                ("ctrl-e", "example"),
                ("1 2 3", "tabs"),
                ("F1", "keys"),
                ("q", "quit"),
            ],
        }
    };
    // Too many for the width: drop pairs from the middle, keeping the first ones (the tab's
    // own actions) and the last two (help, quit), which every screen needs.
    let cost = |p: &(&str, &str)| p.0.chars().count() + p.1.chars().count() + 1 + GAP.len();
    let mut shown: Vec<&(&str, &str)> = pairs.iter().collect();
    while shown.len() > 3 && 2 + shown.iter().map(|p| cost(p)).sum::<usize>() > width {
        shown.remove(shown.len() - 3);
    }
    let mut spans = vec![Span::raw("  ")];
    for (k, v) in shown {
        spans.push(Span::styled(k.to_string(), look.accent()));
        if !v.is_empty() {
            spans.push(Span::styled(format!(" {v}"), look.muted()));
        }
        spans.push(Span::raw(GAP));
    }
    Line::from(spans)
}

/// Space between two key hints.
const GAP: &str = "   ";

/// The s1re.sh prompt: the last command run, what came of it, and the Clay cursor.
/// The bottom line: a question waiting on a key (rename, delete) or the result of the last
/// action on the left, and the command that produced it, dim, on the right when there is room.
fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let width = area.width as usize;
    if let Some(name) = &app.jobs.renaming {
        let id = app.jobs.current().map_or(String::new(), |j| {
            short_id(&j.job.id.to_string()).to_string()
        });
        let label = format!("  rename {id}  ");
        let room = width.saturating_sub(label.chars().count() + 2);
        let shown: String = {
            let n = name.chars().count();
            name.chars().skip(n.saturating_sub(room)).collect()
        };
        let line = Line::from(vec![
            Span::styled(label, look.accent()),
            Span::styled(shown, look.text()),
            Span::styled("▌", look.cursor()),
        ]);
        f.render_widget(Paragraph::new(line), area);
        return;
    }
    if let Some(id) = app.jobs.deleting {
        let name = app
            .jobs
            .all
            .iter()
            .find(|j| j.job.id == id)
            .map_or_else(String::new, |j| display_name(&j.header));
        let short = id.to_string()[..8].to_string();
        let ask = format!("  delete {name} ({short}) and its files?  ");
        let line = Line::from(vec![
            Span::styled(elide_end(&ask, width.saturating_sub(34)), look.bad()),
            Span::styled("y", look.accent()),
            Span::styled(" delete   ", look.muted()),
            Span::styled("any other key", look.accent()),
            Span::styled(" keep", look.muted()),
        ]);
        f.render_widget(Paragraph::new(line), area);
        return;
    }
    let (text, style) = match app.status.as_deref() {
        Some(s) if s.starts_with('✗') => (s.to_string(), look.bad()),
        Some(s) if s.starts_with('✓') => (s.to_string(), look.text()),
        Some(s) if s.ends_with('…') => (s.to_string(), look.muted()),
        Some(s) => (s.to_string(), look.muted()),
        None => (String::new(), look.muted()),
    };
    // The message comes first; the command gets what is left, and only a useful amount.
    let left = elide_end(&text, width.saturating_sub(4));
    let used = left.chars().count() + 2;
    let spare = width.saturating_sub(used + 6);
    let mut spans = vec![Span::raw("  ")];
    if let Some(glyph @ ('✓' | '✗')) = left.chars().next() {
        let glyph_style = if glyph == '✓' {
            look.accent()
        } else {
            look.bad()
        };
        spans.push(Span::styled(glyph.to_string(), glyph_style));
        spans.push(Span::styled(
            left.chars().skip(1).collect::<String>(),
            style,
        ));
    } else {
        spans.push(Span::styled(left, style));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
    if let Some(cmd) = &app.last_command {
        if spare >= 24 {
            let cmd = elide_middle(&format!("$ {cmd}"), spare.min(64));
            let right = Line::from(vec![Span::styled(cmd, look.dim()), Span::raw("  ")]);
            f.render_widget(Paragraph::new(right).alignment(Alignment::Right), area);
        }
    }
}

/// `s` cut to `max` characters, ending in `…` when it was cut.
fn elide_end(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// A section title: `(name)` on a hairline.
fn section(look: &Look, name: &str, extra: Option<String>) -> Block<'static> {
    let mut title = vec![Span::styled(format!("({name})"), look.muted())];
    if let Some(e) = extra {
        title.push(Span::styled(format!(" {e}"), look.dim()));
    }
    title.push(Span::raw(" "));
    Block::new()
        .borders(Borders::TOP)
        .border_style(look.line())
        .title(Line::from(title))
}

// ---------------------------------------------------------------------------------------------
// Jobs

/// A job state as glyph and word; the colour only repeats what the glyph says.
pub fn state(look: &Look, s: &JobStatus, tick: u64) -> Span<'static> {
    match s {
        JobStatus::Completed => Span::styled("✓ done", look.accent()),
        JobStatus::Failed => Span::styled("✗ failed", look.bad()),
        JobStatus::Cancelled => Span::styled("– cancelled", look.dim()),
        JobStatus::Running => {
            let glyph = if look.motion && tick % 2 == 1 {
                "◉"
            } else {
                "●"
            };
            Span::styled(format!("{glyph} running"), look.warm())
        }
        JobStatus::Queued => Span::styled("◌ queued", look.muted()),
        JobStatus::Pending => Span::styled("◌ pending", look.muted()),
    }
}

fn age(t: chrono::DateTime<Utc>) -> String {
    let s = (Utc::now() - t).num_seconds().max(0);
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        3600..86400 => format!("{}h", s / 3600),
        _ => format!("{}d", s / 86400),
    }
}

fn tier_slug(t: &proteus_core::models::PipelineTier) -> &'static str {
    use proteus_core::models::PipelineTier::*;
    match t {
        FastScreening => "fast",
        HighFidelity => "sota",
        FullValidation => "full",
    }
}

/// The mark, small, for empty states: braille, the chain in the accent and the ligand warm.
fn mark_lines(look: &Look, cols: usize, rows: usize) -> Vec<Line<'static>> {
    mark::braille(cols, rows, 1.0)
        .into_iter()
        .map(|row| {
            Line::from(
                row.into_iter()
                    .map(|(c, ink)| {
                        let style = match ink {
                            mark::Ink::Ligand => look.warm(),
                            _ => look.accent(),
                        };
                        Span::styled(c.to_string(), style)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn draw_jobs(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let jobs = &app.jobs;
    let visible = jobs.visible();
    let extra = if jobs.filter.is_empty() && !jobs.filtering {
        Some(format!("{} · {}", jobs.all.len(), jobs.sort.label()))
    } else {
        Some(format!(
            "{} of {} · {} · filter {}{}",
            visible.len(),
            jobs.all.len(),
            jobs.sort.label(),
            jobs.filter,
            if jobs.filtering { "▏" } else { "" }
        ))
    };

    if jobs.all.is_empty() {
        let mut lines = Vec::new();
        if area.height >= 16 && area.width >= 40 {
            lines.push(Line::from(""));
            lines.extend(mark_lines(look, 14, 7));
            lines.push(Line::from(""));
        }
        if let Some(e) = &jobs.error {
            lines.push(Line::from(Span::styled(
                "✗ the job database could not be read",
                look.bad(),
            )));
            lines.push(Line::from(Span::styled(e.clone(), look.muted())));
        } else if !jobs.loaded {
            lines.push(Line::from(Span::styled("reading…", look.dim())));
        } else {
            lines.push(Line::from(Span::styled(
                "No jobs yet.",
                look.text().add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Fold a sequence from (run), key 3, or from a shell:",
                look.muted(),
            )));
            lines.push(Line::from(vec![
                Span::styled("~ $ ", look.dim()),
                Span::styled(
                    "proteus submit --fasta $'>demo\\nMKTAYIAKQRQISFVKSHFSRQ'",
                    look.accent(),
                ),
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Structure files you already have are in (structures), key 2.",
                look.muted(),
            )));
            lines.push(Line::from(Span::styled(
                format!("jobs live in {}", tilde(&app.data_dir.to_string_lossy())),
                look.dim(),
            )));
        }
        f.render_widget(
            Paragraph::new(lines)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: false })
                .block(section(look, "jobs", extra)),
            area,
        );
        return;
    }

    // Wide: the list on the left, the selected job inspected on the right. Narrow: the list
    // takes what it needs and the inspector the rest.
    let wide = area.width >= 120 && area.height >= 16;
    let mut overview = Rect::default();
    let [list, detail] = if wide {
        let [l, _, d] = Layout::horizontal([
            Constraint::Percentage(44),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .areas(area);
        // The list takes what its rows need; the overview of all jobs fills below it.
        let want = visible.len() as u16 + 3;
        if l.height >= want + 14 {
            let [top, _, bottom] = Layout::vertical([
                Constraint::Length(want),
                Constraint::Length(2),
                Constraint::Min(0),
            ])
            .areas(l);
            overview = bottom;
            [top, d]
        } else {
            [l, d]
        }
    } else if area.height >= 16 {
        let want = (visible.len() as u16 + 3).max(6);
        Layout::vertical([Constraint::Max(want), Constraint::Min(8)])
            .spacing(1)
            .areas(area)
    } else {
        [area, Rect::default()]
    };

    // The model column only where the names keep room (at least 22 characters); names cut
    // with an ellipsis, never mid-word without a mark.
    let show_model = list.width >= 22 + 10 + 4 + 5 + 10 + 4 + 5 * 2 + 3;
    let name_w = (list.width as usize)
        .saturating_sub(10 + 4 + 5 + 4 + 3 + if show_model { 10 + 2 } else { 0 } + 4 * 2)
        .max(8);
    let rows = visible.iter().map(|j| {
        let eng = model(j);
        let eng_style = if engine(j) == proteus_engine::ENGINE_SIMULATED {
            look.warm()
        } else {
            look.muted()
        };
        let failed = j.job.status == JobStatus::Failed;
        Row::new(vec![
            Cell::from(state(look, &j.job.status, app.tick)),
            Cell::from(Span::styled(
                elide_end(&display_name(&j.header), name_w),
                if failed { look.muted() } else { look.text() },
            )),
            Cell::from(
                Line::from(Span::styled(j.length.to_string(), look.muted())).right_aligned(),
            ),
            Cell::from(
                Line::from(match j.plddt {
                    Some(p) => Span::styled(
                        format!("{p:.1}"),
                        look.data(proteus_render::rasterizer::shader::plddt_to_color(p as f32)),
                    ),
                    None => Span::styled("–", look.dim()),
                })
                .right_aligned(),
            ),
            Cell::from(Span::styled(
                if show_model { eng } else { String::new() },
                eng_style,
            )),
            Cell::from(Line::from(Span::styled(age(j.job.created_at), look.dim())).right_aligned()),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Fill(1),
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Length(if show_model { 10 } else { 0 }),
            Constraint::Length(4),
        ],
    )
    .header(
        Row::new([
            Cell::from("state"),
            Cell::from("name"),
            Cell::from(Line::from("len").right_aligned()),
            Cell::from(Line::from("pLDDT").right_aligned()),
            Cell::from(if show_model { "model" } else { "" }),
            Cell::from(Line::from("age").right_aligned()),
        ])
        .style(look.dim())
        .bottom_margin(1),
    )
    .column_spacing(2)
    .row_highlight_style(look.selected())
    .highlight_symbol(Text::from(Span::styled("▌", look.accent())))
    .block(section(look, "jobs", extra));
    let mut state_ = TableState::default().with_selected(Some(jobs.selected));
    f.render_stateful_widget(table, list, &mut state_);

    if detail.height > 0 {
        if let Some(j) = jobs.current() {
            draw_job_detail(f, detail, app, j);
        }
    }
    if overview.height > 0 {
        draw_overview(f, overview, app);
    }
}

/// All jobs at a glance: states, how confident the models are, how the complexes' interfaces
/// came out, and which models folded them.
fn draw_overview(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let jobs = &app.jobs.all;
    let w = area.width as usize;
    let mut out = vec![heading(look, "overview", w), Line::from("")];
    let count = |s: JobStatus| jobs.iter().filter(|j| j.job.status == s).count();
    let mut states = vec![Span::styled(format!("{:<14}", "jobs"), look.dim())];
    for (n, s, glyph, style) in [
        (count(JobStatus::Completed), "done", "✓", look.accent()),
        (count(JobStatus::Running), "running", "●", look.warm()),
        (
            count(JobStatus::Queued) + count(JobStatus::Pending),
            "queued",
            "○",
            look.muted(),
        ),
        (count(JobStatus::Failed), "failed", "✗", look.bad()),
    ] {
        if n > 0 {
            states.push(Span::styled(format!("{glyph} "), style));
            states.push(Span::styled(
                n.to_string(),
                look.text().add_modifier(Modifier::BOLD),
            ));
            states.push(Span::styled(format!(" {s}    "), look.muted()));
        }
    }
    out.push(Line::from(states));
    out.push(Line::from(""));

    // pLDDT in AlphaFold's bands, one bar per band scaled to the fullest.
    let bands: [(&str, f64, f64, f32); 4] = [
        ("≥ 90", 90.0, 101.0, 95.0),
        ("70–90", 70.0, 90.0, 80.0),
        ("50–70", 50.0, 70.0, 60.0),
        ("< 50", f64::NEG_INFINITY, 50.0, 40.0),
    ];
    let with_plddt: Vec<f64> = jobs.iter().filter_map(|j| j.plddt).collect();
    if !with_plddt.is_empty() {
        out.push(Line::from(Span::styled("confidence (pLDDT)", look.dim())));
        let most = bands
            .iter()
            .map(|(_, lo, hi, _)| {
                with_plddt
                    .iter()
                    .filter(|p| **p >= *lo && **p < *hi)
                    .count()
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let bar_w = w.saturating_sub(14 + 6).min(28);
        for (name, lo, hi, mid) in bands {
            let n = with_plddt.iter().filter(|p| **p >= lo && **p < hi).count();
            let c = proteus_render::rasterizer::shader::plddt_to_color(mid);
            let mut l = vec![Span::styled(format!("  {name:<12}"), look.muted())];
            l.extend(gauge(look, n as f64 / most as f64, bar_w, look.data(c)));
            l.push(Span::styled(
                format!(" {n}"),
                if n > 0 {
                    look.text().add_modifier(Modifier::BOLD)
                } else {
                    look.dim()
                },
            ));
            out.push(Line::from(l));
        }
        out.push(Line::from(""));
    }

    // Complexes: how many interfaces pass the ipSAE_min line, of those measured so far.
    let complexes: Vec<_> = jobs
        .iter()
        .filter(|j| j.chain_names.len() > 1 || j.header.contains(" + "))
        .collect();
    if !complexes.is_empty() {
        let (mut measured, mut confident, mut waiting) = (0, 0, 0);
        for j in &complexes {
            match j
                .pdb_path
                .as_ref()
                .and_then(|p| app.analyses.get(Path::new(p)))
            {
                Some(Analysis::Done {
                    verdict: Some((ok, _)),
                    ..
                }) => {
                    measured += 1;
                    confident += usize::from(*ok);
                }
                // No PAE to judge an interface by: not counted either way.
                Some(Analysis::Done { .. }) | Some(Analysis::Failed(_)) => {}
                _ if j.pdb_path.is_some() => waiting += 1,
                _ => {}
            }
        }
        let mut l = vec![
            Span::styled(format!("{:<14}", "interfaces"), look.dim()),
            Span::styled(
                format!("{confident}"),
                look.accent().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" of {measured} confident",), look.muted()),
        ];
        if waiting > 0 {
            l.push(Span::styled(
                format!("  · {waiting} measuring…"),
                look.dim(),
            ));
        }
        out.push(Line::from(l));
        out.push(Line::from(Span::styled(
            format!("{:<14}ipSAE_min above 0.61", ""),
            look.dim(),
        )));
        out.push(Line::from(""));
    }

    // Which models folded them.
    let mut models: Vec<(String, usize)> = Vec::new();
    for j in jobs.iter().filter(|j| j.pdb_path.is_some()) {
        let m = model(j);
        match models.iter_mut().find(|(k, _)| *k == m) {
            Some((_, n)) => *n += 1,
            None => models.push((m, 1)),
        }
    }
    models.sort_by_key(|m| std::cmp::Reverse(m.1));
    for (i, (m, n)) in models.iter().take(4).enumerate() {
        out.push(Line::from(vec![
            Span::styled(
                format!("{:<14}", if i == 0 { "models" } else { "" }),
                look.dim(),
            ),
            Span::styled(format!("{n:>3} "), look.text().add_modifier(Modifier::BOLD)),
            Span::styled(m.clone(), look.muted()),
        ]));
    }
    f.render_widget(Paragraph::new(out), area);
}

/// The selected job, inspected: its name and state, how it ran, why it failed if it did, and
/// for a model the preview, the headline gauges and every measurement.
fn draw_job_detail(
    f: &mut Frame,
    area: Rect,
    app: &App,
    j: &proteus_storage::repository::JobSummary,
) {
    let look = &app.look;
    let when = |t: chrono::DateTime<Utc>| {
        t.with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string()
    };
    let mut sub = vec![
        Span::styled(short_id(&j.job.id.to_string()).to_string(), look.muted()),
        Span::styled("  ", look.dim()),
        Span::styled(tier_slug(&j.job.tier).to_string(), look.text()),
    ];
    if !engine(j).is_empty() {
        // "sota · boltz+msa on oci": the model, then where a container ran it.
        sub.push(Span::styled(" · ", look.dim()));
        sub.push(Span::styled(model(j), look.text()));
        if engine(j) == proteus_engine::ENGINE_OCI && !model(j).starts_with("oci") {
            sub.push(Span::styled(" on ", look.dim()));
            sub.push(Span::styled(engine(j).to_string(), look.text()));
        }
    }
    // The date only where the line has room for it (the list's age column has it roughly).
    let date = if area.width >= 110 {
        format!("  ·  {}", when(j.job.created_at))
    } else {
        String::new()
    };
    sub.push(Span::styled(
        format!("  ·  {} residues{date}", j.length),
        look.dim(),
    ));
    if let Some(done) = j.job.completed_at {
        let secs = (done - j.job.started_at.unwrap_or(j.job.created_at))
            .num_seconds()
            .max(0);
        sub.push(Span::styled(format!("  ·  {secs} s"), look.dim()));
    }
    let mut notes = Vec::new();
    if engine(j) == proteus_engine::ENGINE_SIMULATED {
        notes.push(Line::from(Span::styled(
            "! simulated: a synthetic helix, not a prediction",
            look.warm(),
        )));
    }
    if let Some(d) = proteus_engine::tier_downgrade(j.metadata.as_ref()) {
        notes.push(Line::from(Span::styled(
            format!("! tier '{}' not honoured: {}", d.requested, d.reason),
            look.warm(),
        )));
    }
    if let Some(e) = &j.job.error_log {
        notes.push(Line::from(vec![
            Span::styled("✗ ", look.bad()),
            Span::styled(e.clone(), look.text()),
        ]));
    }
    if let Some(p) = &j.pdb_path {
        notes.push(Line::from(Span::styled(
            elide_middle(&tilde(p), area.width as usize),
            look.dim(),
        )));
    }
    let path = j.pdb_path.as_ref().map(std::path::PathBuf::from);
    draw_inspector(
        f,
        area,
        app,
        Inspect {
            title: j.header.clone(),
            badge: Some(state(look, &j.job.status, app.tick)),
            sub,
            notes,
            path: path.as_deref(),
            chain_names: &j.chain_names,
            empty: match j.job.status {
                JobStatus::Completed => None,
                JobStatus::Failed => Some("This job left no model."),
                _ => Some("The model appears here when the job finishes."),
            },
        },
    );
}

/// What the inspector shows about one structure (a job's model or a file).
struct Inspect<'a> {
    title: String,
    badge: Option<Span<'static>>,
    sub: Vec<Span<'static>>,
    notes: Vec<Line<'static>>,
    path: Option<&'a Path>,
    /// Said where the preview and measurements would be, when there is no structure.
    empty: Option<&'static str>,
    /// Chain ID → name, to name the binder and target (a job's complex).
    chain_names: &'a [(String, String)],
}

/// `A → B` as `PD-L1 (A) → PD-1 (B)`, for each chain the input named. Several chains on a side
/// (`A,C`) are named one by one.
pub fn name_chains(value: &str, names: &[(String, String)]) -> String {
    if names.is_empty() {
        return value.to_string();
    }
    let one = |id: &str| {
        let id = id.trim();
        names
            .iter()
            .find(|(c, _)| c == id)
            .map_or(id.to_string(), |(_, n)| format!("{n} ({id})"))
    };
    value
        .split(" → ")
        .map(|side| side.split(',').map(one).collect::<Vec<_>>().join(", "))
        .collect::<Vec<_>>()
        .join(" → ")
}

/// The inspector: title and badge, an identity line, notes; then the structure's preview beside
/// its headline card (the interface verdict and gauges); then every measurement, grouped.
fn draw_inspector(f: &mut Frame, area: Rect, app: &App, it: Inspect) {
    let look = &app.look;
    if area.width < 20 || area.height < 4 {
        return;
    }
    // Title line: the name in bold, the badge right-aligned; the identity line and notes under
    // it; a hairline closes the header off from the numbers.
    // A short pane gives the numbers every row: no rule, no gaps.
    let roomy = u16::from(area.height >= 20);
    let [title, sub, notes_area, _, rule, _, body] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(it.notes.len() as u16),
        Constraint::Length(roomy),
        Constraint::Length(roomy),
        Constraint::Length(roomy),
        Constraint::Min(0),
    ])
    .areas(area);
    f.render_widget(
        Paragraph::new(Span::styled("─".repeat(rule.width as usize), look.line())),
        rule,
    );
    f.render_widget(
        Paragraph::new(Span::styled(
            it.title.clone(),
            look.text().add_modifier(Modifier::BOLD),
        )),
        title,
    );
    if let Some(b) = it.badge {
        f.render_widget(
            Paragraph::new(Line::from(b)).alignment(Alignment::Right),
            title,
        );
    }
    f.render_widget(Paragraph::new(Line::from(it.sub)), sub);
    f.render_widget(Paragraph::new(it.notes), notes_area);

    let analysis = it.path.and_then(|p| app.analyses.get(p));
    let done = match analysis {
        Some(Analysis::Done { .. }) => analysis.unwrap(),
        Some(Analysis::Pending) => {
            f.render_widget(Paragraph::new(Span::styled("measuring…", look.dim())), body);
            return;
        }
        Some(Analysis::Failed(why)) => {
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        "✗ could not measure this structure",
                        look.bad(),
                    )),
                    Line::from(Span::styled(why.clone(), look.muted())),
                ])
                .wrap(Wrap { trim: false }),
                body,
            );
            return;
        }
        None => {
            if let Some(e) = it.empty {
                f.render_widget(Paragraph::new(Span::styled(e, look.dim())), body);
            }
            return;
        }
    };
    let Analysis::Done {
        residues,
        predicted,
        rows,
        interface,
        verdict,
        facts,
    } = done
    else {
        return;
    };
    // The binder and target by the names the input gave them.
    let named: Vec<[String; 2]> = interface
        .iter()
        .map(|[k, v]| {
            if k == "binder → target" {
                [k.clone(), name_chains(v, it.chain_names)]
            } else {
                [k.clone(), v.clone()]
            }
        })
        .collect();
    let interface = &named;

    // Preview and card share a band as tall as the preview wants to be (about square in
    // pixels: a cell is twice as tall as wide), capped at half the body.
    // Numbers first: the preview only where it leaves room for them, and only when there is a
    // structure to draw.
    let has_pic = body.height >= 16
        && it
            .path
            .is_some_and(|p| app.scenes.contains_key(p) || app.any_preview(p).is_some());
    let side = body.width >= 70;
    let card_w = if side && has_pic {
        body.width * 48 / 100 - 2
    } else {
        body.width
    };
    let card_lines = headline_card(
        look,
        *residues,
        *predicted,
        facts.as_deref(),
        verdict.as_ref(),
        card_w,
    );
    if !has_pic {
        let h = (card_lines.len() as u16).min(body.height);
        let [c, _, d] = Layout::vertical([
            Constraint::Length(h),
            Constraint::Length(u16::from(h > 0)),
            Constraint::Min(0),
        ])
        .areas(body);
        f.render_widget(Paragraph::new(card_lines), c);
        draw_details(f, d, look, rows, interface);
        return;
    }
    // The picture about square in pixels (a cell is about twice as tall as wide), the card
    // beside it; the band no taller than half the body, so the measurements keep their room.
    let band_h = if side {
        ((body.width as f32 * 0.48 / 2.2) as u16)
            .max(card_lines.len() as u16 + 2)
            .min(body.height / 2 + 1)
            .max(8.min(body.height))
    } else {
        (body.height / 3).clamp(6.min(body.height), 14)
    };
    let [band, _, details] = Layout::vertical([
        Constraint::Length(band_h),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(body);
    let (pic, card) = if side {
        let [p, _, c] = Layout::horizontal([
            Constraint::Percentage(48),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .areas(band);
        (p, Some(c))
    } else {
        (band, None)
    };
    if let Some(path) = it.path {
        draw_preview_pane(f, pic, app, path);
    }
    match card {
        Some(c) => {
            // The card sits in the middle of the band's height, beside the picture's middle.
            let top = c.height.saturating_sub(card_lines.len() as u16) / 2;
            let c = Rect {
                y: c.y + top.min(2),
                height: c.height - top.min(2),
                ..c
            };
            f.render_widget(Paragraph::new(card_lines), c);
            draw_details(f, details, look, rows, interface);
        }
        None => {
            let h = (card_lines.len() as u16).min(details.height);
            let [c, _, d] = Layout::vertical([
                Constraint::Length(h),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .areas(details);
            f.render_widget(Paragraph::new(card_lines), c);
            draw_details(f, d, look, rows, interface);
        }
    }
}

/// The inspector's headline: the interface verdict for a complex, then gauges for confidence,
/// backbone, fold and the triage score.
fn headline_card(
    look: &Look,
    residues: usize,
    predicted: bool,
    facts: Option<&super::app::Facts>,
    verdict: Option<&(bool, String)>,
    width: u16,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    if let Some((ok, text)) = verdict {
        let (mark, rest) = text.split_once(" · ").unwrap_or((text.as_str(), ""));
        out.push(Line::from(Span::styled(
            mark.to_string(),
            if *ok { look.accent() } else { look.warm() }.add_modifier(Modifier::BOLD),
        )));
        out.push(Line::from(Span::styled(rest.to_string(), look.muted())));
        out.push(Line::from(""));
    }
    let Some(fx) = facts else {
        return out;
    };
    // The text after each bar, long forms first; on a narrow card the short forms, so that no
    // line wraps (a wrapped value strands under the labels). The bars share one width, what the
    // labels (11) and the longest text leave, up to 18 cells.
    let outliers = match fx.rama_outliers {
        0 => String::new(),
        1 => " · 1 outlier".to_string(),
        n => format!(" · {n} outliers"),
    };
    let long = [
        format!(" {:.0} % favoured{outliers}", fx.rama_favored),
        format!(" α{:.0} β{:.0} coil {:.0} %", fx.helix, fx.strand, fx.coil),
    ];
    let short = [
        format!(" {:.0} %{outliers}", fx.rama_favored),
        format!(" α{:.0} β{:.0}", fx.helix, fx.strand),
    ];
    let room = |texts: &[String; 2]| {
        let longest = texts
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(0)
            .max(6);
        (width as usize).saturating_sub(11 + longest)
    };
    let (texts, bar) = if room(&long) >= 8 {
        let r = room(&long);
        (long, r)
    } else {
        let r = room(&short);
        (short, r)
    };
    #[allow(non_snake_case)]
    let W = bar.clamp(3, 18);
    let [rama_text, fold_text] = texts;
    let label = |s: &str| Span::styled(format!("{s:<11}"), look.dim());
    // pLDDT gauge in AlphaFold's own band colours, or the B-factor warning.
    match (fx.plddt, predicted) {
        (Some(p), true) => {
            let c = proteus_render::rasterizer::shader::plddt_to_color(p as f32);
            let mut l = vec![label("pLDDT")];
            l.extend(gauge(look, p / 100.0, W, look.data(c)));
            l.push(Span::styled(
                format!(" {p:.1}"),
                look.data(c).add_modifier(Modifier::BOLD),
            ));
            out.push(Line::from(l));
        }
        _ => out.push(Line::from(vec![
            label("pLDDT"),
            Span::styled("– experimental structure", look.dim()),
        ])),
    }
    if fx.ptm.is_some() || fx.iptm.is_some() {
        let n = |v: Option<f64>| v.map_or("–".into(), |v| format!("{v:.2}"));
        out.push(Line::from(vec![
            label("pTM · ipTM"),
            Span::styled(n(fx.ptm), look.text().add_modifier(Modifier::BOLD)),
            Span::styled(" · ", look.dim()),
            Span::styled(n(fx.iptm), look.text().add_modifier(Modifier::BOLD)),
        ]));
    }
    // Confidence above, the model's quality below.
    out.push(Line::from(""));
    let mut rama = vec![label("backbone")];
    let rama_style = if fx.rama_outliers == 0 {
        look.accent()
    } else {
        look.warm()
    };
    rama.extend(gauge(look, fx.rama_favored / 100.0, W, rama_style));
    let (value, flag) = rama_text.split_at(rama_text.find(" · ").unwrap_or(rama_text.len()));
    rama.extend(value_spans(look, value));
    if !flag.is_empty() {
        rama.push(Span::styled(flag.to_string(), look.warm()));
    }
    out.push(Line::from(rama));
    // Secondary structure as one stacked bar in the viewers' own colours.
    let mut ss = vec![label("fold")];
    let total = (fx.helix + fx.strand + fx.coil).max(1e-9);
    let mut used = 0usize;
    for (i, (v, c)) in [
        (fx.helix, proteus_render::brand::structure::HELIX),
        (fx.strand, proteus_render::brand::structure::STRAND),
        (fx.coil, proteus_render::brand::structure::COIL),
    ]
    .into_iter()
    .enumerate()
    {
        let n = if i == 2 {
            W - used
        } else {
            ((v / total) * W as f64).round() as usize
        };
        let n = n.min(W - used);
        used += n;
        ss.push(Span::styled("█".repeat(n), look.data(c)));
    }
    ss.extend(value_spans(look, &fold_text));
    out.push(Line::from(ss));
    let mut tri = vec![label("triage")];
    tri.extend(gauge(look, fx.triage / 100.0, W, look.accent()));
    tri.push(Span::styled(
        format!(" {:.0}", fx.triage),
        look.text().add_modifier(Modifier::BOLD),
    ));
    out.push(Line::from(tri));
    // The Rg ratio only when it fits on the line: wrapped, it strands under the labels (the
    // measurements below give it in full either way).
    let mut size = format!(
        "{residues} residues · {} chain{}",
        fx.chains,
        if fx.chains == 1 { "" } else { "s" },
    );
    let rg = format!(" · Rg ×{:.2}", fx.rg_ratio);
    if 11 + size.chars().count() + rg.chars().count() <= width as usize {
        size.push_str(&rg);
    } else if 11 + size.chars().count() > width as usize {
        size = format!("{residues} res · {} ch", fx.chains);
    }
    let mut l = vec![label("size")];
    l.extend(value_spans(look, &size));
    out.push(Line::from(l));
    out
}

#[cfg(test)]
pub fn headline_card_for_test(
    look: &Look,
    residues: usize,
    predicted: bool,
    facts: Option<&super::app::Facts>,
    width: u16,
) -> Vec<Line<'static>> {
    headline_card(look, residues, predicted, facts, None, width)
}

/// A bar of `width` cells, filled to `frac` with eighth-block precision.
fn gauge(look: &Look, frac: f64, width: usize, fill: Style) -> Vec<Span<'static>> {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let frac = frac.clamp(0.0, 1.0);
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    let (full, part) = (eighths / 8, eighths % 8);
    let mut s = "█".repeat(full);
    s.push_str(EIGHTHS[part]);
    let used = full + usize::from(part > 0);
    vec![
        Span::styled(s, fill),
        Span::styled("░".repeat(width.saturating_sub(used)), look.line()),
    ]
}

/// A value as spans: numbers bold (they are what is read), words quiet, a note in brackets dim,
/// `·` separators dimmer still.
pub fn value_spans(look: &Look, v: &str) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    for (i, tok) in v.split(' ').enumerate() {
        if i > 0 {
            out.push(Span::raw(" "));
        }
        let opens = tok.matches('(').count() as i32;
        let closes = tok.matches(')').count() as i32;
        let inside = depth > 0 || opens > 0;
        depth += opens - closes;
        let style = if inside || tok == "·" || tok == "/" || tok == "→" {
            look.dim()
        } else if tok.chars().any(|c| c.is_ascii_digit()) {
            look.text().add_modifier(Modifier::BOLD)
        } else {
            look.muted()
        };
        out.push(Span::styled(tok.to_string(), style));
    }
    out
}

/// A section heading: `(name)` and a hairline to `width`.
fn heading(look: &Look, name: &str, width: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("({name}) "),
            look.muted().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "─".repeat(width.saturating_sub(name.chars().count() + 3)),
            look.line(),
        ),
    ])
}

/// One group of label/value rows under a heading, labels in a column as wide as the longest
/// (at most 22), or stacked when the width cannot hold both.
fn kv_block(look: &Look, name: &str, items: &[[String; 2]], width: u16) -> Vec<Line<'static>> {
    let width = width as usize;
    // Short labels: the value gets the width (a wrapped value reads worse than a terse label).
    let items: Vec<[String; 2]> = items
        .iter()
        .map(|[k, v]| [short_label(k).to_string(), v.clone()])
        .collect();
    let items = &items;
    let label_w = items
        .iter()
        .map(|[k, _]| k.chars().count())
        .max()
        .unwrap_or(0)
        .min(22)
        + 3;
    let side = width >= label_w + 24;
    let mut out = vec![heading(look, name, width)];
    for [k, v] in items {
        if side {
            // A long value breaks at its own seams (` · `, a bracketed note), each piece
            // continuing under the value column, never back under the labels.
            for (i, piece) in wrap_value(v, width - label_w).into_iter().enumerate() {
                let lead = if i == 0 {
                    format!("{k:<label_w$}")
                } else {
                    " ".repeat(label_w)
                };
                let mut l = vec![Span::styled(lead, look.dim())];
                l.extend(value_spans(look, &piece));
                out.push(Line::from(l));
            }
        } else {
            out.push(Line::from(Span::styled(k.clone(), look.dim())));
            let mut l = vec![Span::raw("  ")];
            l.extend(value_spans(look, v));
            out.push(Line::from(l));
        }
    }
    out
}

/// A report row's label as the inspector shows it, where the full one is long.
fn short_label(k: &str) -> &str {
    match k {
        "Radius of gyration" => "Rg",
        "Heavy-atom overlaps" => "Overlaps",
        "Covalent geometry" => "Covalent",
        "Stereochemistry" => "Stereo",
        "binder → target" => "binder → target",
        other => other,
    }
}

/// `v` in lines of at most `width` characters, broken at ` · ` or before ` (`; a piece longer
/// than the width on its own is left whole (the paragraph wraps it).
pub fn wrap_value(v: &str, width: usize) -> Vec<String> {
    if v.chars().count() <= width {
        return vec![v.to_string()];
    }
    // Pieces, each keeping the separator that starts it.
    let mut pieces: Vec<String> = Vec::new();
    let mut rest = v;
    while !rest.is_empty() {
        let cut = [" · ", " ("]
            .iter()
            .filter_map(|sep| rest[1.min(rest.len())..].find(sep).map(|i| i + 1))
            .min();
        match cut {
            Some(i) => {
                pieces.push(rest[..i].to_string());
                rest = &rest[i..];
            }
            None => {
                pieces.push(rest.to_string());
                rest = "";
            }
        }
    }
    let mut lines: Vec<String> = Vec::new();
    for p in pieces {
        match lines.last_mut() {
            Some(l) if l.chars().count() + p.chars().count() <= width => l.push_str(&p),
            _ => {
                let p = p
                    .strip_prefix(" · ")
                    .map_or(p.trim_start().to_string(), |t| format!("· {t}"));
                lines.push(p);
            }
        }
    }
    lines
}

/// The measurements the card does not already show, in groups a reader looks for: shape,
/// geometry, contacts; anything else under its own heading.
pub fn measurement_groups(rows: &[[String; 2]]) -> Vec<(&'static str, Vec<[String; 2]>)> {
    // The card's gauges already show these.
    let shown = [
        "pLDDT",
        "Secondary structure",
        "Ramachandran",
        "Triage score",
    ];
    let group_of = |k: &str| {
        let k = k.to_lowercase();
        if k.contains("gyration")
            || k.contains("sasa")
            || k.contains("surface")
            || k.contains("density")
        {
            "shape"
        } else if k.contains("overlap")
            || k.contains("geometry")
            || k.contains("stereo")
            || k.contains("rotamer")
            || k.contains("disulfide")
            || k.contains("clash")
        {
            "geometry"
        } else if k.contains("interaction") || k.contains("bond") || k.contains("bridge") {
            "contacts"
        } else {
            "other"
        }
    };
    let mut out: Vec<(&'static str, Vec<[String; 2]>)> = ["shape", "geometry", "contacts", "other"]
        .into_iter()
        .map(|g| (g, Vec::new()))
        .collect();
    for r in rows.iter().filter(|[k, _]| !shown.contains(&k.as_str())) {
        let g = group_of(&r[0]);
        if let Some((_, v)) = out.iter_mut().find(|(n, _)| *n == g) {
            v.push(r.clone());
        }
    }
    out.retain(|(_, v)| !v.is_empty());
    out
}

/// Every measurement in its group; on a wide pane in two columns, balanced by height.
fn draw_details(
    f: &mut Frame,
    area: Rect,
    look: &Look,
    rows: &[[String; 2]],
    interface: &[[String; 2]],
) {
    let mut groups: Vec<(&str, Vec<[String; 2]>)> = Vec::new();
    if !interface.is_empty() {
        groups.push(("interface", interface.to_vec()));
    }
    groups.extend(measurement_groups(rows));
    let two = area.width >= 100 && groups.len() > 1;
    let cols: Vec<Rect> = if two {
        let [l, _, r] = Layout::horizontal([
            Constraint::Percentage(50),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .areas(area);
        vec![l, r]
    } else {
        vec![area]
    };
    // Greedy: each group to the shorter column, in order.
    let mut lines: Vec<Vec<Line<'static>>> = vec![Vec::new(); cols.len()];
    for (name, items) in &groups {
        let i = (0..cols.len()).min_by_key(|i| lines[*i].len()).unwrap_or(0);
        if !lines[i].is_empty() {
            lines[i].push(Line::from(""));
        }
        lines[i].extend(kv_block(look, name, items, cols[i].width));
    }
    for (c, mut l) in cols.into_iter().zip(lines) {
        // Cut short, say so rather than let a group vanish off the bottom.
        if l.len() > c.height as usize && c.height >= 2 {
            let keep = c.height as usize - 1;
            let hidden = l[keep..]
                .iter()
                .filter(|line| line.spans.iter().any(|s| s.content.starts_with('(')))
                .count();
            l.truncate(keep);
            l.push(Line::from(Span::styled(
                if hidden > 0 {
                    format!(
                        "⋯ {hidden} more group{} · i for the full report",
                        if hidden == 1 { "" } else { "s" }
                    )
                } else {
                    "⋯ more · i for the full report".to_string()
                },
                look.dim(),
            )));
        }
        f.render_widget(Paragraph::new(l), c);
    }
}

/// `text` in lines of at most `width` characters, broken at spaces.
pub fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(' ') {
        match out.last_mut() {
            Some(l) if l.chars().count() + 1 + word.chars().count() <= width => {
                l.push(' ');
                l.push_str(word);
            }
            _ => out.push(word.to_string()),
        }
    }
    out
}

/// The preview pane: under kitty graphics the pane is left empty and the picture is placed over
/// it after the frame; otherwise the still is drawn in half-block cells. Either way the pane
/// asks for a still of its own size, which the loop renders.
fn draw_preview_pane(f: &mut Frame, area: Rect, app: &App, path: &Path) {
    if area.width < 4 || area.height < 3 || app.look.pixel((0, 0, 0)).is_none() {
        return;
    }
    let key = super::app::PreviewKey {
        path: path.to_path_buf(),
        cols: area.width,
        rows: area.height,
        pixels: app.cell_pixels.is_some(),
    };
    *app.preview_want.borrow_mut() = Some((key.clone(), area));
    if key.pixels {
        return;
    }
    match app
        .previews
        .get(&key)
        .or_else(|| app.any_preview(&key.path))
    {
        Some(p) => draw_preview(f, area, p, &app.look),
        _ => f.render_widget(
            Paragraph::new(Span::styled("rendering…", app.look.dim())),
            area,
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Structures

/// A path under `$HOME` as `~/…`, the way a shell prompt shows it: shorter, and the same on
/// every machine.
pub fn tilde(p: &str) -> String {
    tilde_in(p, std::env::var("HOME").ok().as_deref())
}

fn tilde_in(p: &str, home: Option<&str>) -> String {
    match home.map(|h| h.trim_end_matches('/')) {
        Some(h) if !h.is_empty() => match p.strip_prefix(h) {
            Some("") => "~".into(),
            Some(rest) if rest.starts_with('/') => format!("~{rest}"),
            _ => p.to_string(),
        },
        _ => p.to_string(),
    }
}

/// A path for a command line the user may paste: `~/…` under `$HOME` (the tilde left unquoted
/// so the shell still expands it), otherwise quoted as a whole.
fn shell_path(p: &str) -> String {
    shell_path_in(p, std::env::var("HOME").ok().as_deref())
}

fn shell_path_in(p: &str, home: Option<&str>) -> String {
    let t = tilde_in(p, home);
    match t.strip_prefix("~/") {
        Some(rest) if t != p => format!("~/{}", super::app::shell_quote(rest)),
        _ if t == "~" && t != p => t,
        _ => super::app::shell_quote(p),
    }
}

#[cfg(test)]
#[test]
fn a_pasted_path_keeps_its_tilde_expandable() {
    let h = Some("/home/ada");
    assert_eq!(shell_path_in("/home/ada/runs/a", h), "~/runs/a");
    assert_eq!(shell_path_in("/home/ada/my runs", h), "~/'my runs'");
    assert_eq!(shell_path_in("/home/ada", h), "~");
    assert_eq!(shell_path_in("/srv/my runs", h), "'/srv/my runs'");
    // A real directory named "~" is not the home directory.
    assert_eq!(shell_path_in("~/x", h), "'~/x'");
}

#[cfg(test)]
#[test]
fn paths_under_home_are_shown_with_a_tilde() {
    let h = Some("/home/ada");
    assert_eq!(tilde_in("/home/ada/x.pdb", h), "~/x.pdb");
    assert_eq!(tilde_in("/home/ada", h), "~");
    assert_eq!(tilde_in("/home/adam/x.pdb", h), "/home/adam/x.pdb");
    assert_eq!(tilde_in("/srv/x.pdb", h), "/srv/x.pdb");
    assert_eq!(tilde_in("/home/ada/x.pdb", Some("/home/ada/")), "~/x.pdb");
    assert_eq!(tilde_in("/x.pdb", Some("/")), "/x.pdb");
    assert_eq!(tilde_in("/x.pdb", None), "/x.pdb");
}

fn human_size(n: u64) -> String {
    match n {
        0..1_000 => format!("{n} B"),
        1_000..1_000_000 => format!("{:.0} kB", n as f64 / 1e3),
        _ => format!("{:.1} MB", n as f64 / 1e6),
    }
}

fn draw_structures(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let files = &app.files;
    let wide = area.width >= 100;
    let [list_area, detail_area] = if wide {
        let [l, _, d] = Layout::horizontal([
            Constraint::Percentage(36),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .areas(area);
        [l, d]
    } else {
        Layout::vertical([Constraint::Percentage(45), Constraint::Min(8)])
            .spacing(1)
            .areas(area)
    };

    // Icons where the terminal draws them: a folder, a molecule for a structure file.
    let (dir_icon, up_icon, file_icon) = if look.icons {
        ("\u{F024B} ", "\u{F005D} ", "\u{F0BAC} ")
    } else {
        ("▸ ", "↑ ", "· ")
    };
    let inner_w = list_area.width.saturating_sub(4) as usize;
    let items: Vec<ListItem> = files
        .entries
        .iter()
        .map(|e| {
            if e.name == ".." {
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {up_icon}"), look.dim()),
                    Span::styled("up a folder", look.muted()),
                ]))
            } else if e.is_dir {
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {dir_icon}"), look.sea()),
                    Span::styled(e.name.clone(), look.sea()),
                ]))
            } else {
                let size = human_size(e.size);
                let name_w = inner_w.saturating_sub(size.len() + 6);
                let name = elide_middle(&e.name, name_w.max(8));
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {file_icon}"), look.accent()),
                    Span::styled(format!("{name:<name_w$}"), look.text()),
                    Span::styled(format!(" {size:>8}"), look.dim()),
                ]))
            }
        })
        .collect();
    let n_files = files.entries.iter().filter(|e| !e.is_dir).count();
    let n_dirs = files
        .entries
        .iter()
        .filter(|e| e.is_dir && e.name != "..")
        .count();
    let where_ = elide_middle(
        &tilde(&files.dir.to_string_lossy()),
        inner_w.saturating_sub(24).max(12),
    );
    let list = List::new(items)
        .block(section(
            look,
            "files",
            Some(format!(
                "{where_} · {n_files} structure{} · {n_dirs} folder{}",
                if n_files == 1 { "" } else { "s" },
                if n_dirs == 1 { "" } else { "s" }
            )),
        ))
        .highlight_style(look.selected())
        .highlight_symbol(Line::from(Span::styled("▌", look.accent())));
    let mut state_ = ListState::default().with_selected(Some(files.selected));
    f.render_stateful_widget(list, list_area, &mut state_);

    match (files.error.as_ref(), files.current()) {
        (Some(e), _) => f.render_widget(
            Paragraph::new(Span::styled(format!("✗ {e}"), look.bad())).wrap(Wrap { trim: false }),
            detail_area,
        ),
        (None, Some(e)) if e.is_dir => draw_folder_card(f, detail_area, app, e, file_icon),
        (None, Some(e)) => {
            let mut sub = vec![Span::styled(human_size(e.size), look.dim())];
            if let Some(Analysis::Done {
                residues,
                predicted,
                facts,
                ..
            }) = app.analyses.get(&e.path)
            {
                let chains = match facts {
                    Some(f) => format!(
                        "  ·  {} chain{}  ·  ",
                        f.chains,
                        if f.chains == 1 { "" } else { "s" }
                    ),
                    None => "  ·  ".into(),
                };
                sub = vec![
                    Span::styled(format!("{residues} residues"), look.text()),
                    Span::styled(chains, look.dim()),
                    if *predicted {
                        Span::styled("predicted model", look.text())
                    } else {
                        Span::styled("experimental", look.text())
                    },
                    Span::styled(format!("  ·  {}", human_size(e.size)), look.dim()),
                ];
            }
            draw_inspector(
                f,
                detail_area,
                app,
                Inspect {
                    title: e.name.clone(),
                    badge: None,
                    sub,
                    notes: vec![Line::from(Span::styled(
                        elide_middle(
                            &tilde(&e.path.to_string_lossy()),
                            detail_area.width as usize,
                        ),
                        look.dim(),
                    ))],
                    path: Some(&e.path),
                    empty: Some("Rest on a file to measure it."),
                    chain_names: &[],
                },
            );
        }
        (None, None) => f.render_widget(
            Paragraph::new(Span::styled(
                "This folder has no structure files or subfolders.",
                look.dim(),
            )),
            detail_area,
        ),
    }
}

/// A folder, before opening it: what it holds (its structure files by name and size), the
/// command that measures all of them, and where else to look when it holds none.
fn draw_folder_card(f: &mut Frame, area: Rect, app: &App, e: &super::app::Entry, file_icon: &str) {
    let look = &app.look;
    let w = area.width as usize;
    let up = e.name == "..";
    let peek = app.files.peek(&e.path);
    let title = if up {
        tilde(&e.path.to_string_lossy())
    } else {
        format!("{}/", e.name)
    };
    let mut out = vec![
        Line::from(Span::styled(
            title,
            look.text().add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled(
                peek.n_structures.to_string(),
                look.text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    " structure file{}  ·  ",
                    if peek.n_structures == 1 { "" } else { "s" }
                ),
                look.muted(),
            ),
            Span::styled(
                peek.n_dirs.to_string(),
                look.text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" folder{}", if peek.n_dirs == 1 { "" } else { "s" }),
                look.muted(),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled("─".repeat(w), look.line())),
        Line::from(""),
    ];
    if peek.n_structures > 0 {
        out.push(heading(look, "inside", w));
        let name_w = w.saturating_sub(14).min(60);
        for (name, size) in &peek.structures {
            out.push(Line::from(vec![
                Span::styled(format!(" {file_icon}"), look.accent()),
                Span::styled(
                    format!("{:<name_w$}", elide_middle(name, name_w)),
                    look.text(),
                ),
                Span::styled(format!("{:>9}", human_size(*size)), look.dim()),
            ]));
        }
        if peek.n_structures > peek.structures.len() {
            out.push(Line::from(Span::styled(
                format!("   and {} more", peek.n_structures - peek.structures.len()),
                look.dim(),
            )));
        }
        out.push(Line::from(""));
        out.push(heading(look, "measure them all", w));
        out.push(Line::from(vec![
            Span::styled("~ $ ", look.dim()),
            Span::styled(
                format!(
                    "proteus analyze {} --export qc.parquet",
                    shell_path(&e.path.to_string_lossy())
                ),
                look.accent(),
            ),
        ]));
        out.push(Line::from(Span::styled(
            "One row per structure: confidence, geometry, interfaces; sortable in any notebook.",
            look.dim(),
        )));
    } else {
        out.push(Line::from(Span::styled(
            "No structure files directly in here.",
            look.muted(),
        )));
        out.push(Line::from(Span::styled(
            "Proteus reads .pdb, .ent, .cif and .mmcif files, gzipped or not.",
            look.dim(),
        )));
        if !peek.below.is_empty() {
            out.push(Line::from(""));
            out.push(heading(look, "further down", w));
            for rel in &peek.below {
                let (folder, file) = rel.rsplit_once('/').unwrap_or(("", rel.as_str()));
                out.push(Line::from(vec![
                    Span::styled(format!(" {file_icon}"), look.accent()),
                    Span::styled(format!("{folder}/"), look.dim()),
                    Span::styled(file.to_string(), look.text()),
                ]));
            }
            if peek.n_below > peek.below.len() {
                out.push(Line::from(Span::styled(
                    format!("   and {} more", peek.n_below - peek.below.len()),
                    look.dim(),
                )));
            }
            out.push(Line::from(vec![
                Span::styled("~ $ ", look.dim()),
                Span::styled(
                    format!(
                        "proteus analyze {} --export qc.parquet",
                        shell_path(&e.path.to_string_lossy())
                    ),
                    look.accent(),
                ),
            ]));
        }
        out.push(Line::from(""));
        out.push(heading(look, "elsewhere", w));
        let jobs_n = app.jobs.all.iter().filter(|j| j.pdb_path.is_some()).count();
        let mut places = vec![
            ("1", format!("your {jobs_n} job models, measured, in jobs")),
            ("~", "your home folder".to_string()),
        ];
        if app.files.start != app.files.dir {
            places.push((
                ".",
                format!("back to {}", tilde(&app.files.start.to_string_lossy())),
            ));
        }
        for (k, what) in places {
            out.push(Line::from(vec![
                Span::styled(format!("  {k:<4}"), look.accent()),
                Span::styled(what, look.text()),
            ]));
        }
    }
    out.push(Line::from(""));
    out.push(Line::from(vec![
        Span::styled("⏎ ", look.accent()),
        Span::styled(if up { "goes up" } else { "opens it" }, look.muted()),
    ]));
    f.render_widget(Paragraph::new(out).wrap(Wrap { trim: false }), area);
}

/// `s` cut to `max` characters by replacing its middle with "…", keeping the end (a file name)
/// whole where it fits.
fn elide_middle(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max || max < 8 {
        return s.to_string();
    }
    let tail = (max * 2 / 3).min(max - 4);
    let head = max - tail - 1;
    let start: String = s.chars().take(head).collect();
    let end: String = s.chars().skip(n - tail).collect();
    format!("{start}…{end}")
}

#[test]
fn a_long_path_keeps_its_file_name() {
    let p = "~/.local/share/proteus/artifacts/ebe8a254/boltz_results_input/predictions/input/input_model_0.pdb";
    let e = elide_middle(p, 50);
    assert_eq!(e.chars().count(), 50);
    assert!(
        e.starts_with("~/.local") && e.ends_with("input_model_0.pdb"),
        "{e}"
    );
    assert_eq!(elide_middle("short.pdb", 50), "short.pdb");
}

/// Draw `p` into `area` as half-block cells (each cell two pixels high), scaled to fit with its
/// aspect kept and centred. Pixels nothing was drawn on leave the terminal's background alone.
fn draw_preview(f: &mut Frame, area: Rect, p: &Preview, look: &Look) {
    if area.width == 0 || area.height == 0 || p.width == 0 || p.height == 0 {
        return;
    }
    let (aw, ah) = (area.width as f32, area.height as f32 * 2.0);
    let scale = (aw / p.width as f32).min(ah / p.height as f32);
    let (w, h) = (
        (p.width as f32 * scale) as u16,
        (p.height as f32 * scale) as u16,
    );
    let x0 = area.x + (area.width.saturating_sub(w)) / 2;
    let y0 = area.y + (area.height.saturating_sub(h.div_ceil(2))) / 2;
    let sample = |x: u16, y: u16| -> Option<(u8, u8, u8)> {
        let sx = ((x as f32 + 0.5) / scale) as usize;
        let sy = ((y as f32 + 0.5) / scale) as usize;
        p.pixels
            .get(sy.min(p.height - 1) * p.width + sx.min(p.width - 1))
            .copied()
            .flatten()
    };
    let buf = f.buffer_mut();
    for cy in 0..h.div_ceil(2) {
        for cx in 0..w {
            let (top, bottom) = (sample(cx, cy * 2), sample(cx, cy * 2 + 1));
            let pos = (x0 + cx, y0 + cy);
            if pos.0 >= area.right() || pos.1 >= area.bottom() {
                continue;
            }
            let cell = &mut buf[pos];
            match (
                top.and_then(|c| look.pixel(c)),
                bottom.and_then(|c| look.pixel(c)),
            ) {
                (None, None) => {}
                (Some(t), None) => {
                    cell.set_symbol("▀").set_fg(t);
                }
                (None, Some(b)) => {
                    cell.set_symbol("▄").set_fg(b);
                }
                (Some(t), Some(b)) => {
                    cell.set_symbol("▀").set_fg(t).set_bg(b);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Run

/// What the selected option of a choice means, in a sentence.
fn option_note(label: &str, value: &str) -> &'static str {
    match (label, value) {
        ("Tier", "fast") => "ESMFold: one chain, seconds per sequence.",
        ("Tier", "sota") => {
            "Boltz-2 in your container, on the GPU if there is one: complexes, ligands, MSAs."
        }
        ("Tier", "full") => "sota plus relaxation. Not in this release: the job will say so.",
        ("Runner", "auto") => {
            "Your container if one is configured, otherwise the public ESMFold API."
        }
        ("Runner", "esm-api") => "Meta's public ESMFold API. The sequence leaves this machine.",
        ("Runner", "oci") => "A local container (PROTEUS_IMAGE_FAST / PROTEUS_IMAGE_SOTA).",
        ("Runner", "simulated") => "Offline and instant: a placeholder helix, not a prediction.",
        ("Mutations", "alanine") => "One alanine at each position in the window.",
        ("Mutations", "saturation") => "All 19 substitutions at each position in the window.",
        ("Rank by", "structure") => "The fold's triage score.",
        ("Rank by", "esm2") => "ESM-2 zero-shot likelihood of each substitution.",
        ("Rank by", "hybrid") => "Both, the structure weighted 0.7.",
        _ => "",
    }
}

fn draw_run(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let run = &app.run;
    let fields = run.fields();
    let [forms, _, main] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);

    // The two forms as a segmented control, like the tabs.
    let pick = |k: FormKind| {
        if run.form == k {
            Span::styled(format!(" {} ", k.title()), look.tab_on())
        } else {
            Span::styled(format!(" {} ", k.title()), look.muted())
        }
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            pick(FormKind::Fold),
            Span::raw("  "),
            pick(FormKind::Scan),
            Span::styled("    f switches", look.dim()),
        ])),
        forms,
    );

    let wide = main.width >= 110;
    let [form, card] = if wide {
        let [l, _, r] = Layout::horizontal([
            Constraint::Percentage(52),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .areas(main);
        [l, r]
    } else {
        let h = (fields.len() * 3 + 4) as u16;
        Layout::vertical([Constraint::Length(h), Constraint::Min(0)]).areas(main)
    };

    // The form: each field a label and its value; the focused one says what it is for (or,
    // for a choice, what the chosen option means) on the line under it.
    let label_w = fields.iter().map(|f| f.label.len()).max().unwrap_or(0) + 3;
    let mut lines = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let focused = run.focus == i;
        let marker = Span::styled(if focused { "▌ " } else { "  " }, look.accent());
        let label_style = if focused { look.accent() } else { look.muted() };
        let value = match &field.kind {
            FieldKind::Text(s) => {
                // A FASTA record reads as its name and length, not a wrapped wall of letters.
                let shown = match s.split_once('\n') {
                    Some((head, seq)) if head.starts_with('>') && !(focused && run.editing) => {
                        let seq: String = seq.split_whitespace().collect();
                        format!("{head}  ·  {} residues", seq.chars().count())
                    }
                    _ => s.replace('\n', " ⏎ "),
                };
                let editing = focused && run.editing;
                if shown.is_empty() && !editing {
                    vec![Span::styled("—", look.dim())]
                } else if editing {
                    vec![
                        Span::styled(shown, look.text()),
                        Span::styled("▌", look.cursor()),
                    ]
                } else {
                    vec![Span::styled(shown, look.text())]
                }
            }
            FieldKind::Choice { options, index } => {
                // Every option in a row, the chosen one filled.
                let mut v = Vec::new();
                for (k, o) in options.iter().enumerate() {
                    if k == *index {
                        v.push(Span::styled(
                            format!(" {o} "),
                            if focused {
                                look.tab_on()
                            } else {
                                look.text().add_modifier(Modifier::BOLD)
                            },
                        ));
                    } else {
                        v.push(Span::styled(format!(" {o} "), look.dim()));
                    }
                }
                v
            }
        };
        let mut spans = vec![
            marker,
            Span::styled(
                format!("{:<label_w$}", field.label.to_lowercase()),
                label_style,
            ),
        ];
        spans.extend(value);
        lines.push(Line::from(spans));
        let note = match &field.kind {
            FieldKind::Choice { .. } => option_note(field.label, field.value()),
            FieldKind::Text(_) if focused => field.hint,
            FieldKind::Text(_) => "",
        };
        // The note under its value, wrapped there rather than back under the labels.
        let room = (form.width as usize).saturating_sub(label_w + 2).max(20);
        for piece in wrap_words(note, room) {
            lines.push(Line::from(vec![
                Span::raw(" ".repeat(label_w + 2)),
                Span::styled(piece, if focused { look.muted() } else { look.dim() }),
            ]));
        }
        lines.push(Line::from(""));
    }
    let run_focused = run.focus == fields.len();
    let ready = run.command().is_ok();
    // Filled when it would run: the one action on this tab should look like one.
    let button = match (run_focused, ready) {
        (_, true) => look.tab_on(),
        (true, false) => look.selected(),
        (false, false) => look.dim(),
    };
    lines.push(Line::from(vec![
        Span::styled(if run_focused { "▌ " } else { "  " }, look.accent()),
        Span::styled(
            if look.icons {
                "  \u{F040A}  run  "
            } else {
                "  run ▸  "
            },
            button,
        ),
        Span::styled(
            if ready {
                "  ⏎ runs it"
            } else {
                "  fill in the sequence first"
            },
            look.dim(),
        ),
    ]));
    // Under the form: the latest jobs, so a run's result is one glance away.
    let form_h = (lines.len() as u16).min(form.height);
    let [form_top, _, recent_area] = Layout::vertical([
        Constraint::Length(form_h),
        Constraint::Length(2),
        Constraint::Min(0),
    ])
    .areas(form);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), form_top);
    if recent_area.height >= 4 && !app.jobs.all.is_empty() {
        let w = recent_area.width as usize;
        let mut out = vec![heading(look, "latest jobs", w)];
        let n = (recent_area.height as usize).saturating_sub(2).min(6);
        for j in app.jobs.all.iter().take(n) {
            let p = match j.plddt {
                Some(p) => Span::styled(
                    format!("{p:>6.1}"),
                    look.data(proteus_render::rasterizer::shader::plddt_to_color(p as f32)),
                ),
                None => Span::styled(format!("{:>6}", "–"), look.dim()),
            };
            let name_w = w.saturating_sub(10 + 6 + 6).max(8);
            out.push(Line::from(vec![
                Span::raw(" "),
                state(look, &j.job.status, app.tick),
                Span::raw("   "),
                Span::styled(
                    format!("{:<name_w$}", elide_end(&display_name(&j.header), name_w)),
                    look.text(),
                ),
                p,
                Span::styled(format!("{:>5}", age(j.job.created_at)), look.dim()),
            ]));
        }
        out.push(Line::from(Span::styled(
            " 1 shows them all, with every measurement",
            look.dim(),
        )));
        f.render_widget(Paragraph::new(out), recent_area);
    }

    // The right side: what happens (the command first: it is what a short terminal must still
    // show), then the tiers side by side, then where the result goes.
    let (what, after) = match run.form {
        FormKind::Fold => (
            "Folds one sequence and keeps the model as a job, with every measurement.",
            "It appears at the top of jobs (1): ⏎ opens the 3-D viewer, w the browser page.",
        ),
        FormKind::Scan => (
            "Writes the variants of the wild type, folds each and ranks them.",
            "The leaderboard prints when the last variant is folded; each is also a job.",
        ),
    };
    let w = card.width as usize;
    let mut info = vec![heading(look, "what happens", w), Line::from("")];
    match run.command() {
        Ok(spec) => {
            info.push(Line::from(vec![
                Span::styled("~ $ ", look.dim()),
                Span::styled(spec.display(), look.accent().add_modifier(Modifier::BOLD)),
            ]));
            info.push(Line::from(""));
            info.push(Line::from(Span::styled(what, look.text())));
            info.push(Line::from(Span::styled(after, look.muted())));
            info.push(Line::from(Span::styled(
                "The same line works in a script.",
                look.dim(),
            )));
        }
        Err(why) => {
            info.push(Line::from(vec![
                Span::styled("! ", look.warm()),
                Span::styled(why, look.warm()),
            ]));
            info.push(Line::from(""));
            info.push(Line::from(Span::styled(what, look.text())));
            info.push(Line::from(vec![
                Span::styled("ctrl-e ", look.accent()),
                Span::styled(
                    "puts in an example: ubiquitin, protein G B1, Trp-cage.",
                    look.muted(),
                ),
            ]));
        }
    }
    if run.form == FormKind::Fold {
        info.push(Line::from(""));
        info.push(heading(look, "tiers", w));
        let tier = fields
            .iter()
            .find(|f| f.label == "Tier")
            .map(|f| f.value())
            .unwrap_or("");
        for (name, engine, good, takes) in [
            ("fast", "ESMFold", "one chain", "seconds"),
            (
                "sota",
                "Boltz-2",
                "complexes, ligands, MSAs",
                "about a minute on a GPU",
            ),
            ("full", "sota + relax", "not in this release", "–"),
        ] {
            let on = name == tier;
            let mark = if on { "▌ " } else { "  " };
            info.push(Line::from(vec![
                Span::styled(mark, look.accent()),
                Span::styled(
                    format!("{name:<6}"),
                    if on {
                        look.accent().add_modifier(Modifier::BOLD)
                    } else {
                        look.muted()
                    },
                ),
                Span::styled(
                    format!("{engine:<14}"),
                    if on {
                        look.text().add_modifier(Modifier::BOLD)
                    } else {
                        look.text()
                    },
                ),
                Span::styled(format!("{good:<27}"), look.muted()),
                Span::styled(if w >= 75 { takes } else { "" }, look.dim()),
            ]));
        }
    }
    info.push(Line::from(""));
    info.push(heading(look, "input", w));
    for l in [
        "A bare sequence, a FASTA record, or the path of a FASTA file.",
        "For a complex, use the shell: one record per chain, e.g.",
    ] {
        info.push(Line::from(Span::styled(l, look.muted())));
    }
    info.push(Line::from(vec![
        Span::styled("~ $ ", look.dim()),
        Span::styled(
            "proteus submit --file pair.fasta --tier sota --msa server",
            look.accent(),
        ),
    ]));
    f.render_widget(Paragraph::new(info).wrap(Wrap { trim: false }), card);
}

// ---------------------------------------------------------------------------------------------
// Help

fn draw_help(f: &mut Frame, area: Rect, look: &Look) {
    let head =
        |s: &'static str| Line::from(Span::styled(s, look.accent().add_modifier(Modifier::BOLD)));
    let row = |k: &'static str, v: &'static str| {
        Line::from(vec![
            Span::styled(format!("  {k:<14}"), look.accent()),
            Span::styled(v, look.text()),
        ])
    };
    let text = vec![
        head("everywhere"),
        row("1 2 3 · tab", "switch tab (alt 1 2 3 while typing)"),
        row("? · F1", "this help (F1 also in a text field)"),
        row("q · ctrl-c", "quit"),
        Line::from(""),
        head("(jobs)"),
        row("↑↓ j k g G", "move"),
        row("⏎ v", "3-D viewer in the terminal"),
        row("w", "page in the browser"),
        row("i", "full report (proteus inspect)"),
        row("/ · r", "filter · refresh (also every 2 s)"),
        row("s", "sort: newest, name, state, pLDDT"),
        row("n · x", "rename · delete (asks first)"),
        Line::from(""),
        head("(structures)"),
        row("⏎ → l", "open a folder, or the 3-D viewer"),
        row("← h ⌫", "up a folder"),
        row("w · a", "browser page · full report (analyze)"),
        row("~ · .", "home folder · the folder proteus started in"),
        Line::from(""),
        head("(run)"),
        row("↑↓ · ⏎", "field · edit it, or run"),
        row("←→ · f", "change a choice · the other form"),
        row("esc", "stop typing; digits switch tabs again"),
        row("ctrl-e", "put in an example sequence"),
        Line::from(""),
        Line::from(Span::styled(
            "  Every action runs a proteus command and shows it first:",
            look.dim(),
        )),
        Line::from(Span::styled(
            "  the same line works in a script.",
            look.dim(),
        )),
    ];
    let w = area.width.min(84);
    // Rows as wrapped at the box's inner width (border and padding off), so the last line is
    // never cut off.
    let inner = w.saturating_sub(2 + 6).max(1) as usize;
    let rows: usize = text.iter().map(|l| l.width().max(1).div_ceil(inner)).sum();
    let h = area.height.min(rows as u16 + 2 + 2);
    let r = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(text).wrap(Wrap { trim: false }).block(
            Block::bordered()
                .border_style(look.line())
                .title(Line::from(Span::styled(" (keys) ", look.muted())))
                .padding(ratatui::widgets::Padding::new(3, 3, 1, 1))
                .style(look.base().patch(look.surface())),
        ),
        r,
    );
}

// ---------------------------------------------------------------------------------------------
// Launch

/// One frame of the launch: the chain folding into the mark (t ∈ [0, 1]), then the wordmark
/// and the signature once it has settled.
pub fn draw_launch(f: &mut Frame, look: &Look, t: f32) {
    let area = f.area();
    f.render_widget(Block::new().style(look.base()), area);
    let big = area.width >= 50 && area.height >= 22;
    let (mc, mr) = if big { (26usize, 12usize) } else { (14, 7) };
    let word: Vec<String> = if big {
        matrix::half_blocks("proteus")
    } else {
        matrix::braille("proteus")
    };
    let word_w = word.first().map_or(0, |l| l.chars().count());
    let total_h = mr + 1 + word.len() + 2;
    let top = area.height.saturating_sub(total_h as u16) / 2;
    let mut lines: Vec<Line> = Vec::new();
    for _ in 0..top {
        lines.push(Line::from(""));
    }
    for row in mark::braille(mc, mr, t) {
        lines.push(Line::from(
            row.into_iter()
                .map(|(c, ink)| {
                    Span::styled(
                        c.to_string(),
                        if ink == mark::Ink::Ligand {
                            look.warm()
                        } else {
                            look.accent()
                        },
                    )
                })
                .collect::<Vec<_>>(),
        ));
    }
    lines.push(Line::from(""));
    let settled = t >= 1.0;
    for l in &word {
        lines.push(Line::from(Span::styled(
            if settled {
                l.clone()
            } else {
                " ".repeat(word_w)
            },
            look.text(),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        if settled {
            brand::SIGNATURE.to_uppercase()
        } else {
            String::new()
        },
        look.dim(),
    )));
    f.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .style(look.base()),
        area,
    );
}
