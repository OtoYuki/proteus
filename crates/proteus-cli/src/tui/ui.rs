//! Drawing the home screen. Pure: it reads [`App`] and writes a frame, so it can be rendered
//! into a `TestBackend` in tests.

use super::app::{engine, short_id, Analysis, App, FieldKind, FormKind, Tab};
use chrono::Utc;
use proteus_core::models::JobStatus;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState, Tabs, Wrap,
};
use ratatui::Frame;

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;

pub fn draw(f: &mut Frame, app: &App) {
    let [top, body, keys, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(f.area());

    draw_tabs(f, top, app);
    match app.tab {
        Tab::Jobs => draw_jobs(f, body, app),
        Tab::Structures => draw_structures(f, body, app),
        Tab::Run => draw_run(f, body, app),
    }
    f.render_widget(
        Paragraph::new(key_hints(app)).style(Style::new().fg(DIM)),
        keys,
    );
    if let Some(s) = &app.status {
        f.render_widget(
            Paragraph::new(s.as_str()).style(Style::new().fg(ACCENT)),
            status,
        );
    }
    if app.help {
        draw_help(f, f.area());
    }
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let [name, tabs] = Layout::horizontal([Constraint::Length(17), Constraint::Min(0)]).areas(area);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" proteus ", Style::new().fg(Color::Black).bg(ACCENT).bold()),
            Span::styled(
                concat!(" ", env!("CARGO_PKG_VERSION")),
                Style::new().fg(DIM),
            ),
        ])),
        name,
    );
    let titles = Tab::ALL
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{} {}", i + 1, t.title()));
    f.render_widget(
        Tabs::new(titles)
            .select(Tab::ALL.iter().position(|t| *t == app.tab))
            .highlight_style(Style::new().fg(ACCENT).bold().underlined())
            .divider("·"),
        tabs,
    );
}

fn key_hints(app: &App) -> String {
    if app.typing() {
        return match app.tab {
            Tab::Jobs => " type to filter · Enter keep · Esc clear".into(),
            _ => " type · Enter/↓ next field · Esc done".into(),
        };
    }
    let tab = match app.tab {
        Tab::Jobs => "↑↓ move · Enter view · w browser · i inspect · / filter",
        Tab::Structures => "↑↓ move · Enter open · ← up · w browser · a analyze",
        Tab::Run => "↑↓ field · Enter edit/run · ←→ choose · f other form",
    };
    format!(" {tab} · ? help · q quit")
}

// ---------------------------------------------------------------------------------------------
// Jobs

fn status_style(s: &JobStatus) -> Style {
    match s {
        JobStatus::Completed => Style::new().fg(Color::Green),
        JobStatus::Failed | JobStatus::Cancelled => Style::new().fg(Color::Red),
        JobStatus::Running => Style::new().fg(Color::Yellow),
        _ => Style::new().fg(Color::Blue),
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

fn draw_jobs(f: &mut Frame, area: Rect, app: &App) {
    let jobs = &app.jobs;
    let visible = jobs.visible();
    let title = if jobs.filter.is_empty() && !jobs.filtering {
        format!(" Jobs ({}) ", jobs.all.len())
    } else {
        format!(
            " Jobs ({}/{}) · filter: {}{} ",
            visible.len(),
            jobs.all.len(),
            jobs.filter,
            if jobs.filtering { "▏" } else { "" }
        )
    };

    if jobs.all.is_empty() {
        let msg = if let Some(e) = &jobs.error {
            vec![
                Line::from("The job database could not be read:".red()),
                Line::from(e.as_str()),
            ]
        } else if !jobs.loaded {
            vec![Line::from("Loading…")]
        } else {
            vec![
                Line::from("No jobs yet.".bold()),
                Line::from(""),
                Line::from("Fold a sequence from the Run tab (press 3), or from a shell:"),
                Line::from(Span::styled(
                    "  proteus submit --fasta $'>demo\\nMKTAYIAKQRQISFVKSHFSRQ'",
                    Style::new().fg(ACCENT),
                )),
                Line::from(""),
                Line::from("Structure files you already have are in the Structures tab (2)."),
                Line::from(Span::styled(
                    format!("Jobs are kept in {}", app.data_dir.display()),
                    Style::new().fg(DIM),
                )),
            ]
        };
        f.render_widget(
            Paragraph::new(msg)
                .wrap(Wrap { trim: false })
                .block(Block::bordered().title(title)),
            area,
        );
        return;
    }

    let detail_rows = if area.height >= 16 { 8 } else { 0 };
    let [list, detail] =
        Layout::vertical([Constraint::Min(3), Constraint::Length(detail_rows)]).areas(area);

    let rows = visible.iter().map(|j| {
        Row::new(vec![
            Cell::from(short_id(&j.job.id.to_string()).to_string()),
            Cell::from(j.header.clone()),
            Cell::from(j.length.to_string()),
            Cell::from(tier_slug(&j.job.tier)),
            Cell::from(format!("{:?}", j.job.status)).style(status_style(&j.job.status)),
            Cell::from(engine(j).to_string()),
            Cell::from(j.plddt.map_or(String::new(), |p| format!("{p:.1}"))),
            Cell::from(age(j.job.created_at)),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Fill(1),
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(9),
            Constraint::Length(12),
            Constraint::Length(5),
            Constraint::Length(4),
        ],
    )
    .header(
        Row::new([
            "id", "name", "len", "tier", "status", "engine", "pLDDT", "age",
        ])
        .style(Style::new().fg(DIM)),
    )
    .row_highlight_style(Style::new().bg(Color::Rgb(30, 41, 59)).bold())
    .block(Block::bordered().title(title));
    let mut state = TableState::default().with_selected(Some(jobs.selected));
    f.render_stateful_widget(table, list, &mut state);

    if detail_rows > 0 {
        if let Some(j) = jobs.current() {
            let mut lines = vec![Line::from(vec![
                Span::styled(j.job.id.to_string(), Style::new().bold()),
                Span::raw("  "),
                Span::styled(format!("{:?}", j.job.status), status_style(&j.job.status)),
            ])];
            let when = |t: Option<chrono::DateTime<Utc>>| {
                t.map_or("—".to_string(), |t| {
                    t.with_timezone(&chrono::Local)
                        .format("%Y-%m-%d %H:%M:%S")
                        .to_string()
                })
            };
            lines.push(Line::from(format!(
                "created {} · started {} · finished {} (local time)",
                when(Some(j.job.created_at)),
                when(j.job.started_at),
                when(j.job.completed_at)
            )));
            if !engine(j).is_empty() {
                let mut l = format!("engine {}", engine(j));
                if let Some(p) = j.plddt {
                    l.push_str(&format!(" · mean pLDDT {p:.1}"));
                }
                if engine(j) == proteus_engine::ENGINE_SIMULATED {
                    l.push_str(" · SIMULATED: a synthetic helix, not a prediction");
                }
                lines.push(Line::from(l));
            }
            if let Some(d) = proteus_engine::tier_downgrade(j.metadata.as_ref()) {
                lines.push(Line::from(
                    format!("tier '{}' not honoured: {}", d.requested, d.reason).yellow(),
                ));
            }
            if let Some(e) = &j.job.error_log {
                lines.push(Line::from(e.as_str().red()));
            }
            if let Some(p) = &j.pdb_path {
                lines.push(Line::from(Span::styled(p.as_str(), Style::new().fg(DIM))));
            }
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .block(Block::bordered().title(format!(" {} ", j.header))),
                detail,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Structures

fn human_size(n: u64) -> String {
    match n {
        0..1_000 => format!("{n} B"),
        1_000..1_000_000 => format!("{:.0} kB", n as f64 / 1e3),
        _ => format!("{:.1} MB", n as f64 / 1e6),
    }
}

fn draw_structures(f: &mut Frame, area: Rect, app: &App) {
    let files = &app.files;
    let [list_area, detail_area] = if area.width >= 90 {
        Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area)
    } else {
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area)
    };

    let items: Vec<ListItem> = files
        .entries
        .iter()
        .map(|e| {
            if e.is_dir {
                ListItem::new(Line::from(format!("{}/", e.name).fg(Color::Blue)))
            } else {
                ListItem::new(Line::from(vec![
                    Span::raw(e.name.clone()),
                    Span::styled(format!("  {}", human_size(e.size)), Style::new().fg(DIM)),
                ]))
            }
        })
        .collect();
    let n_files = files.entries.iter().filter(|e| !e.is_dir).count();
    let title = format!(" {} · {} structure files ", files.dir.display(), n_files);
    let list = List::new(items)
        .block(Block::bordered().title(title))
        .highlight_style(Style::new().bg(Color::Rgb(30, 41, 59)).bold());
    let mut state = ListState::default().with_selected(Some(files.selected));
    f.render_stateful_widget(list, list_area, &mut state);

    let block = Block::bordered().title(" Measurements ");
    let body: Text = match (files.error.as_ref(), files.current()) {
        (Some(e), _) => Text::from(e.as_str().red()),
        (None, Some(e)) if e.is_dir => Text::from(vec![
            Line::from("Enter opens the folder.".fg(DIM)),
            Line::from(""),
            Line::from(
                "Structure files are .pdb, .ent, .cif and .mmcif, also gzipped. \
                 For a table over a whole folder:"
                    .fg(DIM),
            ),
            Line::from(Span::styled(
                format!(
                    "  proteus analyze {} --export qc.parquet",
                    super::app::shell_quote(&e.path.to_string_lossy())
                ),
                Style::new().fg(ACCENT),
            )),
        ]),
        (None, Some(e)) => match app.analyses.get(&e.path) {
            None | Some(Analysis::Pending) => Text::from("Measuring…".fg(DIM)),
            Some(Analysis::Failed(why)) => Text::from(vec![
                Line::from("Could not analyse this file:".red()),
                Line::from(why.as_str()),
            ]),
            Some(Analysis::Done {
                residues,
                predicted,
                rows,
            }) => {
                let mut lines = vec![
                    Line::from(e.name.as_str().bold()),
                    Line::from(
                        format!(
                            "{residues} residues · B-factor column is {}",
                            if *predicted {
                                "a predicted pLDDT"
                            } else {
                                "experimental (not a confidence)"
                            }
                        )
                        .fg(DIM),
                    ),
                    Line::from(""),
                ];
                for [k, v] in rows {
                    lines.push(Line::from(k.as_str().fg(DIM)));
                    lines.push(Line::from(format!("  {v}")));
                }
                Text::from(lines)
            }
        },
        (None, None) => Text::from("This folder has no structure files or subfolders.".fg(DIM)),
    };
    f.render_widget(
        Paragraph::new(body).wrap(Wrap { trim: false }).block(block),
        detail_area,
    );
}

// ---------------------------------------------------------------------------------------------
// Run

fn draw_run(f: &mut Frame, area: Rect, app: &App) {
    let run = &app.run;
    let fields = run.fields();
    let [forms, form, command] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(fields.len() as u16 + 4),
        Constraint::Min(0),
    ])
    .areas(area);

    let pick = |k: FormKind| {
        let s = format!(" {} ", k.title());
        if run.form == k {
            Span::styled(s, Style::new().fg(Color::Black).bg(ACCENT))
        } else {
            Span::styled(s, Style::new().fg(DIM))
        }
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            pick(FormKind::Fold),
            Span::raw(" "),
            pick(FormKind::Scan),
            Span::styled("   f switches", Style::new().fg(DIM)),
        ])),
        forms,
    );

    let label_w = fields.iter().map(|f| f.label.len()).max().unwrap_or(0) + 2;
    let mut lines = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let focused = run.focus == i;
        let marker = if focused { "▶ " } else { "  " };
        let value = match &field.kind {
            FieldKind::Text(s) => {
                let shown = s.replace('\n', " ⏎ ");
                let caret = if focused && run.editing { "▏" } else { "" };
                if shown.is_empty() && !(focused && run.editing) {
                    Span::styled("(empty)", Style::new().fg(DIM))
                } else {
                    Span::raw(format!("{shown}{caret}"))
                }
            }
            FieldKind::Choice { options, index } => Span::raw(format!("◀ {} ▶", options[*index])),
        };
        let style = if focused {
            Style::new().fg(ACCENT).bold()
        } else {
            Style::new()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{marker}{:<label_w$}", field.label), style),
            value,
        ]));
    }
    let run_focused = run.focus == fields.len();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "[ Run ]",
        if run_focused {
            Style::new().fg(Color::Black).bg(Color::Green).bold()
        } else {
            Style::new().fg(Color::Green)
        },
    )));
    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(format!(" {} ", run.form.title()))),
        form,
    );

    let mut info = Vec::new();
    if let Some(field) = fields.get(run.focus) {
        info.push(Line::from(field.hint.fg(DIM)));
        info.push(Line::from(""));
    }
    match run.command() {
        Ok(spec) => {
            info.push(Line::from("Runs:".fg(DIM)));
            info.push(Line::from(Span::styled(
                spec.display(),
                Style::new().fg(ACCENT),
            )));
        }
        Err(why) => info.push(Line::from(why.fg(Color::Yellow))),
    }
    f.render_widget(
        Paragraph::new(info)
            .wrap(Wrap { trim: false })
            .block(Block::bordered().title(" Command ")),
        command,
    );
}

// ---------------------------------------------------------------------------------------------
// Help

fn draw_help(f: &mut Frame, area: Rect) {
    let text = vec![
        Line::from("Everywhere".bold()),
        Line::from("  1 2 3 / Tab   switch tab        ?  this help      q  quit"),
        Line::from(""),
        Line::from("Jobs".bold()),
        Line::from("  ↑↓ j k g G    move              Enter v  3-D viewer in the terminal"),
        Line::from("  w             page in browser   i  full report (proteus inspect)"),
        Line::from("  /             filter            r  refresh (also every 2 s)"),
        Line::from(""),
        Line::from("Structures".bold()),
        Line::from("  Enter → l     open folder or 3-D viewer     ← h Backspace  up"),
        Line::from("  w             page in browser   a  full report (proteus analyze)"),
        Line::from(""),
        Line::from("Run".bold()),
        Line::from("  ↑↓            field             Enter  edit a field, or Run"),
        Line::from("  ←→            change a choice   f  the other form"),
        Line::from(""),
        Line::from(
            "Every action runs a proteus command, shown before it runs; the same command works \
             in a script."
                .fg(DIM),
        ),
    ];
    let w = area.width.min(78);
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
                .title(" Keys ")
                .style(Style::new().add_modifier(Modifier::empty())),
        ),
        r,
    );
}
