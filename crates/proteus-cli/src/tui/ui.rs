//! Drawing the home screen in the Proteus identity. Pure: it reads [`App`] and writes a frame,
//! so every screen can be rendered into a `TestBackend` in tests.
//!
//! The rules it follows (docs/design/2026-09-23-proteus-identity-design.md): colours only from
//! brand roles through [`Look`]; hairlines, not boxes; panel names in `(parentheses)`; a
//! state is always a glyph and a word, never a colour alone.

use super::app::{engine, short_id, Analysis, App, FieldKind, FormKind, Preview, Tab};
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

/// Rows the full header takes (name line, tabs, hairline); a short terminal gets one row.
fn header_rows(area: Rect) -> u16 {
    if area.height >= 18 && area.width >= 60 {
        3
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
    let body = body.inner(ratatui::layout::Margin::new(1, 0));
    match app.tab {
        Tab::Jobs => draw_jobs(f, body, app),
        Tab::Structures => draw_structures(f, body, app),
        Tab::Run => draw_run(f, body, app),
    }
    f.render_widget(Paragraph::new(key_hints(app)), keys);
    f.render_widget(Paragraph::new(status_line(app)), status);
    if app.help {
        draw_help(f, f.area(), look);
    }
}

// ---------------------------------------------------------------------------------------------
// Header, key hints, status line

fn tab_spans(app: &App) -> Vec<Span<'static>> {
    let look = &app.look;
    let mut spans = Vec::new();
    for (i, t) in Tab::ALL.iter().enumerate() {
        let active = *t == app.tab;
        spans.push(Span::styled(format!("{} ", i + 1), look.dim()));
        let name = format!("({})", t.title().to_lowercase());
        spans.push(if active {
            Span::styled(
                name,
                look.accent()
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
        } else {
            Span::styled(name, look.muted())
        });
        spans.push(Span::raw("   "));
    }
    spans
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let name = vec![
        Span::styled(" proteus", look.accent().add_modifier(Modifier::BOLD)),
        Span::styled(concat!("  v", env!("CARGO_PKG_VERSION")), look.dim()),
    ];
    if area.height < 3 {
        let mut spans = name;
        spans.push(Span::raw("   "));
        spans.extend(tab_spans(app));
        f.render_widget(Paragraph::new(Line::from(spans)), area);
        return;
    }
    let [top, tabs, rule] = Layout::vertical([Constraint::Length(1); 3]).areas(area);
    f.render_widget(Paragraph::new(Line::from(name)), top);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{} ", brand::SIGNATURE),
            look.dim(),
        )))
        .alignment(Alignment::Right),
        top,
    );
    let mut spans = vec![Span::raw(" ")];
    spans.extend(tab_spans(app));
    f.render_widget(Paragraph::new(Line::from(spans)), tabs);
    // The jobs at a glance, right-aligned under the signature: each state as glyph and word.
    let summary = job_summary(app);
    if !summary.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(summary)).alignment(Alignment::Right),
            tabs,
        );
    }
    f.render_widget(
        Block::new().borders(Borders::TOP).border_style(look.line()),
        rule,
    );
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
fn key_hints(app: &App) -> Line<'static> {
    let look = &app.look;
    let pairs: &[(&str, &str)] = if app.typing() {
        match app.tab {
            Tab::Jobs => &[("type", "to filter"), ("⏎", "keep"), ("esc", "clear")],
            _ => &[("type", ""), ("⏎ ↓", "next field"), ("esc", "done")],
        }
    } else {
        match app.tab {
            Tab::Jobs => &[
                ("↑↓", "move"),
                ("⏎", "view"),
                ("w", "browser"),
                ("i", "inspect"),
                ("/", "filter"),
                ("?", "help"),
                ("q", "quit"),
            ],
            Tab::Structures => &[
                ("↑↓", "move"),
                ("⏎", "open"),
                ("←", "up"),
                ("w", "browser"),
                ("a", "analyze"),
                ("?", "help"),
                ("q", "quit"),
            ],
            Tab::Run => &[
                ("↑↓", "field"),
                ("⏎", "edit / run"),
                ("←→", "choose"),
                ("f", "other form"),
                ("F1", "help"),
                ("q", "quit"),
            ],
        }
    };
    let mut spans = vec![Span::raw(" ")];
    for (k, v) in pairs {
        spans.push(Span::styled(k.to_string(), look.accent()));
        if !v.is_empty() {
            spans.push(Span::styled(format!(" {v}"), look.dim()));
        }
        spans.push(Span::styled("  ·  ", look.line()));
    }
    spans.pop();
    Line::from(spans)
}

/// The s1re.sh prompt: the last command run, what came of it, and the Clay cursor.
fn status_line(app: &App) -> Line<'static> {
    let look = &app.look;
    let mut spans = vec![Span::styled(" ~ $ ", look.dim())];
    if let Some(cmd) = &app.last_command {
        spans.push(Span::styled(cmd.clone(), look.muted()));
        spans.push(Span::raw("  "));
    }
    if let Some(s) = &app.status {
        let failed = s.contains("failed") || s.starts_with("could not");
        let style = if failed { look.bad() } else { look.text() };
        let glyph = if failed {
            "✗ "
        } else if app.last_command.is_some() {
            "✓ "
        } else {
            ""
        };
        spans.push(Span::styled(format!("{glyph}{s}"), style));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled("▌", look.cursor()));
    Line::from(spans)
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
        Some(format!("{}", jobs.all.len()))
    } else {
        Some(format!(
            "{} of {} · filter {}{}",
            visible.len(),
            jobs.all.len(),
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

    // Wide: the list on the left, the selected job on the right with a still of its model and
    // its measurements. Narrow: the job under the list, filling what the list does not need.
    let wide = area.width >= 120 && area.height >= 16;
    let [list, detail] = if wide {
        Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)]).areas(area)
    } else if area.height >= 16 {
        let want = (visible.len() as u16 + 4).max(6);
        Layout::vertical([Constraint::Max(want), Constraint::Min(8)]).areas(area)
    } else {
        [area, Rect::default()]
    };

    let rows = visible.iter().map(|j| {
        let eng = engine(j);
        let eng_style = if eng == proteus_engine::ENGINE_SIMULATED {
            look.warm()
        } else {
            look.muted()
        };
        Row::new(vec![
            Cell::from(state(look, &j.job.status, app.tick)),
            Cell::from(Span::styled(
                short_id(&j.job.id.to_string()).to_string(),
                look.dim(),
            )),
            Cell::from(Span::styled(j.header.clone(), look.text())),
            Cell::from(Span::styled(j.length.to_string(), look.muted())),
            Cell::from(Span::styled(tier_slug(&j.job.tier), look.muted())),
            Cell::from(Span::styled(eng.to_string(), eng_style)),
            Cell::from(Span::styled(
                j.plddt.map_or(String::new(), |p| format!("{p:.1}")),
                look.text(),
            )),
            Cell::from(Span::styled(age(j.job.created_at), look.dim())),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(11),
            Constraint::Length(8),
            Constraint::Fill(1),
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(12),
            Constraint::Length(5),
            Constraint::Length(4),
        ],
    )
    .header(
        Row::new([
            "STATE", "ID", "NAME", "LEN", "TIER", "ENGINE", "PLDDT", "AGE",
        ])
        .style(look.dim())
        .bottom_margin(0),
    )
    .column_spacing(2)
    .row_highlight_style(look.selected())
    .highlight_symbol(Text::from(Span::styled("▌", look.accent())))
    .block(section(look, "jobs", extra));
    let mut state_ = TableState::default().with_selected(Some(jobs.selected));
    f.render_stateful_widget(table, list, &mut state_);

    if detail.height > 0 {
        if let Some(j) = jobs.current() {
            draw_job_detail(f, detail, app, j, wide);
        }
    }
}

/// The selected job: its state and times, what went wrong if it failed, and for a finished
/// model a preview and the same measurements as `proteus analyze`.
fn draw_job_detail(
    f: &mut Frame,
    area: Rect,
    app: &App,
    j: &proteus_storage::repository::JobSummary,
    wide: bool,
) {
    let look = &app.look;
    let when = |t: Option<chrono::DateTime<Utc>>| {
        t.map_or("—".to_string(), |t| {
            t.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
    };
    let kv = |k: &str, v: String, vs: Style| {
        Line::from(vec![
            Span::styled(format!("{k:<8}"), look.dim()),
            Span::styled(v, vs),
        ])
    };
    let mut info = vec![Line::from(vec![
        state(look, &j.job.status, app.tick),
        Span::raw("   "),
        Span::styled(j.job.id.to_string(), look.muted()),
    ])];
    let mut engine_line = engine(j).to_string();
    if let Some(p) = j.plddt {
        if !engine_line.is_empty() {
            engine_line.push_str(" · ");
        }
        engine_line.push_str(&format!("mean pLDDT {p:.1}"));
    }
    info.push(kv(
        "run",
        format!(
            "{} · {} residues · {}{}",
            tier_slug(&j.job.tier),
            j.length,
            when(Some(j.job.created_at)),
            j.job
                .completed_at
                .map(|t| format!(
                    " · took {}s",
                    (t - j.job.started_at.unwrap_or(j.job.created_at))
                        .num_seconds()
                        .max(0)
                ))
                .unwrap_or_default()
        ),
        look.text(),
    ));
    if !engine_line.is_empty() {
        info.push(kv("engine", engine_line, look.text()));
    }
    if engine(j) == proteus_engine::ENGINE_SIMULATED {
        info.push(kv(
            "",
            "! simulated: a synthetic helix, not a prediction".into(),
            look.warm(),
        ));
    }
    if let Some(d) = proteus_engine::tier_downgrade(j.metadata.as_ref()) {
        info.push(kv(
            "tier",
            format!("! '{}' not honoured: {}", d.requested, d.reason),
            look.warm(),
        ));
    }
    if let Some(e) = &j.job.error_log {
        info.push(kv("error", format!("✗ {e}"), look.bad()));
    }
    let block = section(look, &j.header, None);
    let inner = block.inner(area);
    if let Some(p) = &j.pdb_path {
        // One line: the start and the file name, the middle elided.
        info.push(kv(
            "file",
            elide_middle(&tilde(p), (inner.width as usize).saturating_sub(9)),
            look.dim(),
        ));
    }
    f.render_widget(block, area);
    let analysis = j
        .pdb_path
        .as_ref()
        .and_then(|p| app.analyses.get(std::path::Path::new(p)));
    // The info lines wrap; count what they take at this width.
    let info_h = info
        .iter()
        .map(|l| (l.width() as u16).div_ceil(inner.width.max(1)).max(1))
        .sum::<u16>();
    let [info_area, rest] =
        Layout::vertical([Constraint::Length(info_h + 1), Constraint::Min(0)]).areas(inner);
    f.render_widget(Paragraph::new(info).wrap(Wrap { trim: false }), info_area);
    match analysis {
        Some(Analysis::Done {
            rows,
            interface,
            verdict,
            preview,
            ..
        }) => {
            let lines = measurement_lines(look, rows, interface, verdict.as_ref(), rest.width);
            let text_h = lines.len() as u16;
            let room = rest.height.saturating_sub(text_h);
            let text_area = match preview
                .as_ref()
                .filter(|_| wide && room >= 8 && look.pixel((0, 0, 0)).is_some())
            {
                Some(p) => {
                    let [pic, txt] =
                        Layout::vertical([Constraint::Length(room.min(28)), Constraint::Min(0)])
                            .areas(rest);
                    draw_preview(f, pic, p, look);
                    txt
                }
                None => rest,
            };
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), text_area);
        }
        Some(Analysis::Pending) => {
            f.render_widget(Paragraph::new(Span::styled("measuring…", look.dim())), rest)
        }
        Some(Analysis::Failed(why)) => f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("✗ could not analyse the model", look.bad())),
                Line::from(Span::styled(why.clone(), look.muted())),
            ])
            .wrap(Wrap { trim: false }),
            rest,
        ),
        None => {}
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
    let wide = area.width >= 90;
    let [list_area, detail_area] = if wide {
        Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).areas(area)
    } else {
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area)
    };

    let name_w = files
        .entries
        .iter()
        .map(|e| e.name.chars().count())
        .max()
        .unwrap_or(0)
        .min(40);
    let items: Vec<ListItem> = files
        .entries
        .iter()
        .map(|e| {
            if e.is_dir {
                ListItem::new(Line::from(vec![
                    Span::styled("▸ ", look.dim()),
                    Span::styled(format!("{}/", e.name), look.sea()),
                ]))
            } else {
                ListItem::new(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{:<name_w$}", e.name), look.text()),
                    Span::styled(format!("  {:>7}", human_size(e.size)), look.dim()),
                ]))
            }
        })
        .collect();
    let n_files = files.entries.iter().filter(|e| !e.is_dir).count();
    let list = List::new(items)
        .block(section(
            look,
            "structures",
            Some(format!(
                "{n_files} in {}",
                tilde(&files.dir.to_string_lossy())
            )),
        ))
        .highlight_style(look.selected())
        .highlight_symbol(Line::from(Span::styled("▌", look.accent())));
    let mut state_ = ListState::default().with_selected(Some(files.selected));
    f.render_stateful_widget(list, list_area, &mut state_);

    let detail_area = if wide {
        detail_area.inner(ratatui::layout::Margin::new(2, 0))
    } else {
        detail_area
    };
    let body: Text = match (files.error.as_ref(), files.current()) {
        (Some(e), _) => Text::from(Span::styled(format!("✗ {e}"), look.bad())),
        (None, Some(e)) if e.is_dir => Text::from(vec![
            Line::from(Span::styled("⏎ opens the folder.", look.muted())),
            Line::from(""),
            Line::from(Span::styled(
                "Structure files are .pdb, .ent, .cif and .mmcif, also gzipped. A table over a whole folder:",
                look.dim(),
            )),
            Line::from(vec![
                Span::styled("~ $ ", look.dim()),
                Span::styled(
                    format!(
                        "proteus analyze {} --export qc.parquet",
                        shell_path(&e.path.to_string_lossy())
                    ),
                    look.accent(),
                ),
            ]),
        ]),
        (None, Some(e)) => match app.analyses.get(&e.path) {
            None | Some(Analysis::Pending) => Text::from(Span::styled("measuring…", look.dim())),
            Some(Analysis::Failed(why)) => Text::from(vec![
                Line::from(Span::styled("✗ could not analyse this file", look.bad())),
                Line::from(Span::styled(why.clone(), look.muted())),
            ]),
            Some(Analysis::Done {
                residues,
                predicted,
                rows,
                interface,
                verdict,
                ..
            }) => {
                let mut lines = vec![
                    Line::from(Span::styled(e.name.clone(), look.text().add_modifier(Modifier::BOLD))),
                    Line::from(Span::styled(
                        format!(
                            "{residues} residues · B-factor column is {}",
                            if *predicted {
                                "a predicted pLDDT"
                            } else {
                                "experimental (not a confidence)"
                            }
                        ),
                        look.dim(),
                    )),
                    Line::from(""),
                ];
                lines.extend(measurement_lines(look, rows, interface, verdict.as_ref(), detail_area.width));
                Text::from(lines)
            }
        },
        (None, None) => Text::from(Span::styled(
            "This folder has no structure files or subfolders.",
            look.dim(),
        )),
    };
    // A still of the structure above its numbers, when there is room and a picture.
    let preview = files
        .current()
        .and_then(|e| match app.analyses.get(&e.path) {
            Some(Analysis::Done {
                preview: Some(p), ..
            }) => Some(p),
            _ => None,
        })
        .filter(|_| detail_area.height >= 22 && look.pixel((0, 0, 0)).is_some());
    let text_area = match preview {
        Some(p) => {
            let h = (detail_area.height * 2 / 5).clamp(8, 20);
            let [pic, rest] =
                Layout::vertical([Constraint::Length(h), Constraint::Min(4)]).areas(detail_area);
            f.render_widget(section(look, "preview", None), pic);
            draw_preview(f, pic.inner(ratatui::layout::Margin::new(1, 1)), p, look);
            rest
        }
        None => detail_area,
    };
    f.render_widget(
        Paragraph::new(body)
            .wrap(Wrap { trim: false })
            .block(section(look, "measurements", None)),
        text_area,
    );
}

/// Measurement rows (label, value), the interface verdict and lines of a complex, laid out side
/// by side when `width` allows.
fn measurement_lines(
    look: &Look,
    rows: &[[String; 2]],
    interface: &[[String; 2]],
    verdict: Option<&(bool, String)>,
    width: u16,
) -> Vec<Line<'static>> {
    let label_w = 22;
    let side_by_side = width as usize >= label_w + 30;
    let mut lines = Vec::new();
    let push = |k: &str, v: &str, vs: Style, lines: &mut Vec<Line<'static>>| {
        if side_by_side {
            lines.push(Line::from(vec![
                Span::styled(format!("{k:<label_w$}"), look.dim()),
                Span::styled(v.to_string(), vs),
            ]));
        } else {
            lines.push(Line::from(Span::styled(k.to_string(), look.dim())));
            lines.push(Line::from(Span::styled(format!("  {v}"), vs)));
        }
    };
    if let Some((ok, text)) = verdict {
        lines.push(Line::from(Span::styled(
            text.clone(),
            if *ok { look.accent() } else { look.warm() },
        )));
    }
    for [k, v] in interface {
        push(k, v, look.text(), &mut lines);
    }
    if !interface.is_empty() {
        lines.push(Line::from(""));
    }
    for [k, v] in rows {
        push(k, v, look.text(), &mut lines);
    }
    lines
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

fn draw_run(f: &mut Frame, area: Rect, app: &App) {
    let look = &app.look;
    let run = &app.run;
    let fields = run.fields();
    let [forms, _, form, command] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(fields.len() as u16 + 3),
        Constraint::Min(0),
    ])
    .areas(area);

    let pick = |k: FormKind| {
        let name = format!("({})", k.title());
        if run.form == k {
            Span::styled(
                name,
                look.accent()
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
        } else {
            Span::styled(name, look.muted())
        }
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            pick(FormKind::Fold),
            Span::raw("   "),
            pick(FormKind::Scan),
            Span::styled("     f switches", look.dim()),
        ])),
        forms,
    );

    let label_w = fields.iter().map(|f| f.label.len()).max().unwrap_or(0) + 3;
    let mut lines = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let focused = run.focus == i;
        let marker = Span::styled(if focused { "▌ " } else { "  " }, look.accent());
        let label_style = if focused { look.accent() } else { look.dim() };
        let value = match &field.kind {
            FieldKind::Text(s) => {
                let shown = s.replace('\n', " ⏎ ");
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
                let arrows = if focused { look.accent() } else { look.dim() };
                vec![
                    Span::styled("‹ ", arrows),
                    Span::styled(options[*index].to_string(), look.text()),
                    Span::styled(" ›", arrows),
                ]
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
    }
    let run_focused = run.focus == fields.len();
    lines.push(Line::from(""));
    let button = if run_focused {
        look.accent()
            .add_modifier(Modifier::REVERSED | Modifier::BOLD)
    } else {
        look.accent()
    };
    lines.push(Line::from(vec![
        Span::styled(if run_focused { "▌ " } else { "  " }, look.accent()),
        Span::styled("[ run ▸ ]", button),
    ]));
    f.render_widget(Paragraph::new(lines), form);

    let mut info = Vec::new();
    if let Some(field) = fields.get(run.focus) {
        info.push(Line::from(Span::styled(field.hint, look.dim())));
        info.push(Line::from(""));
    }
    match run.command() {
        Ok(spec) => info.push(Line::from(vec![
            Span::styled("~ $ ", look.dim()),
            Span::styled(spec.display(), look.text()),
        ])),
        Err(why) => info.push(Line::from(Span::styled(format!("! {why}"), look.warm()))),
    }
    f.render_widget(
        Paragraph::new(info)
            .wrap(Wrap { trim: false })
            .block(section(look, "command", None)),
        command,
    );
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
        row("1 2 3 · tab", "switch tab"),
        row("? · F1", "this help (F1 also in a text field)"),
        row("q · ctrl-c", "quit"),
        Line::from(""),
        head("(jobs)"),
        row("↑↓ j k g G", "move"),
        row("⏎ v", "3-D viewer in the terminal"),
        row("w", "page in the browser"),
        row("i", "full report (proteus inspect)"),
        row("/ · r", "filter · refresh (also every 2 s)"),
        Line::from(""),
        head("(structures)"),
        row("⏎ → l", "open a folder, or the 3-D viewer"),
        row("← h ⌫", "up a folder"),
        row("w · a", "browser page · full report (analyze)"),
        Line::from(""),
        head("(run)"),
        row("↑↓ · ⏎", "field · edit it, or run"),
        row("←→ · f", "change a choice · the other form"),
        Line::from(""),
        Line::from(Span::styled(
            "  Every action runs a proteus command and shows it first; the same line works in a script.",
            look.dim(),
        )),
    ];
    let w = area.width.min(80);
    let h = area.height.min(text.len() as u16 + 2);
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
