//! The home screen: what bare `proteus` opens in a terminal.
//!
//! Browsing (the job list, a structure file's measurements) happens here. Every action runs
//! the `proteus` binary itself as a child process, with the command line shown first, so the
//! home screen implements nothing twice and teaches the commands a script would use. See
//! `docs/design/2026-09-23-tui-home-design.md`.

pub mod app;
mod style;
mod ui;

use anyhow::{Context, Result};
use app::{Action, Analysis, App, CommandSpec, RunMode};
use crossterm::event::{self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyEventKind};
use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
use proteus_storage::repository::ProteusRepository;
use ratatui::layout::Rect;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How many jobs the list shows (newest first).
const JOB_LIMIT: i64 = 500;
const REFRESH_EVERY: Duration = Duration::from_secs(2);
/// How long a structure file must stay selected before it is measured.
const SETTLE: Duration = Duration::from_millis(150);

/// Whether bare `proteus` should open the home screen: only when a person is at a terminal on
/// both ends. In a pipe, a script or CI it keeps printing the usage error, as before.
pub fn wanted() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

pub async fn run(db_path: &Path, data_dir: &Path) -> Result<()> {
    let cwd = std::env::current_dir().context("Cannot read the current directory")?;
    let exe = std::env::current_exe().context("Cannot locate the proteus binary")?;
    let signal = watch_signals()?;

    let mut app = App::new(&cwd, data_dir);
    app.look = style::Look::detect();
    // Real pixels for previews in a local kitty-protocol terminal (not over SSH, not in tmux).
    if proteus_render::terminal::KittyRenderer::is_supported()
        && std::env::var_os("SSH_CONNECTION").is_none()
        && std::env::var_os("TMUX").is_none()
    {
        app.cell_pixels = crossterm::terminal::window_size().ok().and_then(|w| {
            (w.width > 0 && w.height > 0 && w.columns > 0 && w.rows > 0).then(|| {
                (
                    w.width as f32 / w.columns as f32,
                    w.height as f32 / w.rows as f32,
                )
            })
        });
    }
    let mut repo: Option<ProteusRepository> = None;
    type Scene = Option<Arc<proteus_render::StructureRenderData>>;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(PathBuf, Analysis, Scene)>();
    let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<(app::PreviewKey, app::Preview)>();
    // The preview being rendered, so a pane that keeps asking starts it once.
    let mut rendering: Option<app::PreviewKey> = None;
    let mut placed: Option<(app::PreviewKey, Rect)> = None;
    // The file selected and since when: only a selection that rests is measured, so scrolling
    // past a 5 MB file does not start (and wait for) its analysis.
    let mut resting: (Option<PathBuf>, Instant) = (None, Instant::now());

    // `try_init` installs a panic hook that restores the terminal; it is called once, and
    // re-entry after a child process only re-enables raw mode and the alternate screen.
    let mut terminal = ratatui::try_init().context("Cannot initialise the terminal")?;
    let restore = Restore;
    let _ = crossterm::execute!(std::io::stdout(), EnableBracketedPaste);
    launch(&mut terminal, &app.look)?;
    let started = Instant::now();

    // Measuring ahead uses up to four threads, leaving one for everything else.
    let workers = std::thread::available_parallelism()
        .map_or(2, |n| n.get().saturating_sub(1))
        .clamp(1, 4);
    let mut last_refresh: Option<Instant> = None;
    loop {
        if let n @ 1.. = signal.load(Ordering::SeqCst) {
            drop(restore);
            std::process::exit(128 + n);
        }
        if last_refresh.is_none_or(|t| t.elapsed() >= REFRESH_EVERY) {
            refresh_jobs(&mut app, db_path, &mut repo).await;
            last_refresh = Some(Instant::now());
        }
        while let Ok((path, analysis, scene)) = rx.try_recv() {
            if let Some(scene) = scene {
                app.scenes.insert(path.clone(), scene);
            }
            app.analyses.insert(path, analysis);
        }
        while let Ok((key, preview)) = prx.try_recv() {
            if rendering.as_ref() == Some(&key) {
                rendering = None;
            }
            app.keep_preview(key, preview);
        }
        // The pane the last frame drew asks for a still of its size; render it off the loop.
        // With nothing asked for, the neighbouring jobs are rendered at the same size, so the
        // next key press finds its picture ready.
        let want = app.preview_want.borrow().as_ref().map(|(k, _)| k.clone());
        if rendering.is_none() {
            let mut next = want.clone().filter(|k| !app.previews.contains_key(k));
            if next.is_none() && app.tab == app::Tab::Jobs {
                if let Some(k) = &want {
                    next = app
                        .neighbour_models()
                        .into_iter()
                        .map(|path| app::PreviewKey { path, ..k.clone() })
                        .find(|k| {
                            !app.previews.contains_key(k) && app.scenes.contains_key(&k.path)
                        });
                }
            }
            if let Some(key) = next {
                if let Some(scene) = app.scenes.get(&key.path).cloned() {
                    rendering = Some(key.clone());
                    let (ptx, cell) = (ptx.clone(), app.cell_pixels);
                    tokio::task::spawn_blocking(move || {
                        let p = render_preview(&scene, &key, cell);
                        let _ = ptx.send((key, p));
                    });
                }
            }
        }
        // What the screen shows is measured at once, a file in the browser once the selection
        // rests (scrolling past a 5 MB file should not wait on it). Jobs' models are measured
        // ahead, a few at a time, nearest the selection first.
        let wanted = app.wanted_analysis();
        if wanted != resting.0 {
            resting = (wanted.clone(), Instant::now());
        }
        let settle = if app.tab == app::Tab::Structures {
            SETTLE
        } else {
            Duration::ZERO
        };
        let mut starts: Vec<PathBuf> = Vec::new();
        if let Some(path) = wanted.filter(|_| resting.1.elapsed() >= settle) {
            starts.push(path);
        }
        let busy = app
            .analyses
            .values()
            .filter(|a| matches!(a, Analysis::Pending))
            .count();
        if busy + starts.len() < workers {
            starts.extend(
                app.prefetch()
                    .into_iter()
                    .take(workers - busy - starts.len()),
            );
        }
        for path in starts {
            if matches!(app.analyses.get(&path), Some(Analysis::Pending)) {
                continue;
            }
            app.analyses.insert(path.clone(), Analysis::Pending);
            app.stamps.insert(path.clone(), app::file_stamp(&path));
            let tx = tx.clone();
            tokio::task::spawn_blocking(move || {
                let (result, scene) = analyse(&path);
                let _ = tx.send((path, result, scene));
            });
        }

        app.tick = (started.elapsed().as_millis() / 500) as u64;
        *app.preview_want.borrow_mut() = None;
        terminal.draw(|f| ui::draw(f, &app))?;
        if app.cell_pixels.is_some() {
            place_picture(&app, &mut placed);
        }

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let action = match event::read()? {
            Event::Key(k) if k.kind == KeyEventKind::Press => app.handle_key(k),
            Event::Paste(text) => {
                app.handle_paste(&text);
                Action::None
            }
            _ => Action::None,
        };
        match action {
            Action::None => {}
            Action::Quit => break,
            Action::RefreshJobs => last_refresh = None,
            Action::Run(spec) => {
                // A quiet command needs no terminal: the home screen stays up while it runs.
                let leave = spec.mode != RunMode::Quiet;
                if leave && placed.is_some() {
                    // The picture would outlive the home screen on the child's screen.
                    *app.preview_want.borrow_mut() = None;
                    place_picture(&app, &mut placed);
                }
                if leave {
                    let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
                    ratatui::restore();
                } else {
                    app.status = Some(
                        spec.doing
                            .clone()
                            .unwrap_or_else(|| format!("running {}…", spec.display())),
                    );
                    terminal.draw(|f| ui::draw(f, &app))?;
                }
                let outcome = tokio::task::block_in_place(|| run_child(&exe, &spec));
                if leave {
                    enable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        EnterAlternateScreen,
                        EnableBracketedPaste
                    )?;
                    terminal.clear()?;
                }
                app.last_command = Some(spec.display());
                app.status = Some(match outcome {
                    // A failure keeps the child's own words; a success says what happened.
                    Ok((false, line)) => format!("✗ {line}"),
                    Ok((true, line)) => format!("✓ {}", spec.done.clone().unwrap_or(line)),
                    Err(e) => format!("✗ could not run `{}`: {e:#}", spec.display()),
                });
                // A job may have been added or finished; the folder may have a new file.
                last_refresh = None;
                app.files.rescan();
            }
        }
    }
    if placed.is_some() {
        *app.preview_want.borrow_mut() = None;
        place_picture(&app, &mut placed);
    }
    // End here rather than returning: dropping the runtime would wait for any analysis still
    // running on a blocking thread (a large file takes seconds; a FIFO never finishes).
    drop(restore);
    std::process::exit(0);
}

/// The launch: the chain folds into the mark (brand::mark::FOLD_SECONDS), the wordmark and
/// signature appear, and it holds briefly. Any key skips it; `NO_MOTION` turns it off.
fn launch(terminal: &mut ratatui::DefaultTerminal, look: &style::Look) -> Result<()> {
    if !look.motion {
        return Ok(());
    }
    // Quicker than the mark's own fold: this plays on every start, and a tool opens fast.
    let fold = Duration::from_secs_f32(proteus_render::brand::mark::FOLD_SECONDS * 0.6);
    let hold = Duration::from_millis(150);
    let start = Instant::now();
    loop {
        let t = start.elapsed().as_secs_f32() / fold.as_secs_f32();
        terminal.draw(|f| ui::draw_launch(f, look, t.min(1.0)))?;
        if start.elapsed() >= fold + hold {
            return Ok(());
        }
        // About 30 frames a second while folding; one wait for the hold.
        let wait = if t < 1.0 {
            Duration::from_millis(33)
        } else {
            (fold + hold).saturating_sub(start.elapsed())
        };
        if event::poll(wait)? {
            if let Event::Key(_) = event::read()? {
                return Ok(());
            }
        }
    }
}

/// Leaves the terminal as the shell expects it, however the loop ends.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
        ratatui::restore();
    }
}

/// Re-read the job list. The database is opened only once it exists: the home screen must not
/// create one just by being looked at.
async fn refresh_jobs(app: &mut App, db_path: &Path, repo: &mut Option<ProteusRepository>) {
    if repo.is_none() {
        if !db_path.exists() {
            app.jobs.loaded = true;
            return;
        }
        match proteus_storage::create_sqlite_pool(db_path).await {
            Ok(pool) => *repo = Some(ProteusRepository::new(pool)),
            Err(e) => {
                app.jobs.error = Some(e.to_string());
                return;
            }
        }
    }
    if let Some(r) = repo.as_ref() {
        match r.list_jobs(JOB_LIMIT).await {
            Ok(jobs) => app.jobs.replace(jobs),
            Err(e) => app.jobs.error = Some(e.to_string()),
        }
    }
}

/// The same analysis as `proteus analyze`, summarised as the browser page summarises it, plus
/// what the viewers would open on: the structure bundle (for previews) and, for a complex, its
/// interface.
fn analyse(path: &Path) -> (Analysis, Option<Arc<proteus_render::StructureRenderData>>) {
    let mut scene = None;
    let mut run = || -> Result<Analysis> {
        let qc = proteus_core::qc::structure_qc(path, None, None)?;
        let loaded = proteus_core::io::load_structure(path)?;
        let a = proteus_core::metrics::analyze_pdb_detailed_with_header(
            &loaded.pdb,
            None,
            Some(&loaded.header_preview),
        )?;
        let residues = a.plddts.len();
        let confidence = proteus_core::pae::read_confidence(path, None).ok();
        let (mut interface, mut verdict) = (Vec::new(), None);
        if let Ok(text) = proteus_core::io::read_structure_text(path) {
            if let Ok(mut s) = proteus_render::parse_pdb_structure(&text) {
                if let Some(c) = confidence.clone() {
                    s.attach_confidence(c);
                }
                if let Some(v) = s.default_interface() {
                    let m = &v.metrics;
                    let n =
                        |x: Option<f64>, d: usize| x.map_or("–".into(), |v| format!("{v:.d$}"));
                    verdict = m.ipsae_min.map(|x| {
                        let ok = x > 0.61;
                        (
                            ok,
                            format!(
                                "{} · ipSAE_min {x:.3} {} 0.61",
                                if ok {
                                    "● confident interface"
                                } else {
                                    "○ not a confident interface"
                                },
                                if ok { ">" } else { "≤" }
                            ),
                        )
                    });
                    interface.push([
                        "binder → target".into(),
                        format!("{} → {}", m.binder_chains, m.target_chains),
                    ]);
                    if m.ipsae_min.is_some() {
                        interface.push([
                            "ipSAE min / max".into(),
                            format!("{} / {}", n(m.ipsae_min, 3), n(m.ipsae_max, 3)),
                        ]);
                        interface.push([
                            "ipAE · LIS".into(),
                            format!("{} Å · {}", n(m.ipae, 1), n(m.lis, 3)),
                        ]);
                    }
                    interface.push([
                        "Sc · buried".into(),
                        format!("{} · {:.0} Å²", n(m.shape_complementarity, 2), m.dsasa),
                    ]);
                    interface.push([
                        "contacts".into(),
                        format!(
                            "{} + {} residues · {} H-bond{} · {} salt bridge{}",
                            m.binder_interface_residues,
                            m.target_interface_residues,
                            m.interface_hbonds,
                            if m.interface_hbonds == 1 { "" } else { "s" },
                            m.interface_salt_bridges,
                            if m.interface_salt_bridges == 1 {
                                ""
                            } else {
                                "s"
                            },
                        ),
                    ]);
                }
                scene = Some(Arc::new(s));
            }
        }
        let facts = app::Facts {
            chains: qc.n_chains,
            plddt: qc.plddt_mean,
            ptm: confidence.as_ref().and_then(|c| c.ptm),
            iptm: confidence.as_ref().and_then(|c| c.iptm),
            rama_favored: qc.rama_favored_pct,
            rama_outliers: qc.rama_outliers,
            helix: qc.helix_pct,
            strand: qc.strand_pct,
            coil: qc.coil_pct,
            rg_ratio: qc.rg_ratio,
            sasa: qc.sasa_total,
            burial: qc.hydrophobic_burial_pct,
            overlaps_per_1k: qc.heavy_atom_overlap_score,
            bond_outliers: qc.bond_outliers,
            angle_outliers: qc.angle_outliers,
            rotamer_outliers: qc.rotamer_outlier_pct,
            hbonds: qc.hbond_count,
            salt_bridges: qc.salt_bridge_count,
            pi: qc.pi_stacking_count + qc.cation_pi_count,
            triage: qc.fitness,
        };
        Ok(Analysis::Done {
            residues,
            predicted: a.metrics.confidence_source
                == proteus_core::confidence::ConfidenceSource::Predicted,
            rows: proteus_core::qc::summary_rows(&a.metrics, residues),
            interface,
            verdict,
            facts: Some(Box::new(facts)),
        })
    };
    let analysis = run().unwrap_or_else(|e| Analysis::Failed(format!("{e:#}")));
    (analysis, scene)
}

/// A still of `scene` for a pane of `key.cols` × `key.rows` cells: two pixels per cell high in
/// half-block, or the pane's real pixel size under kitty graphics.
fn render_preview(
    scene: &proteus_render::StructureRenderData,
    key: &app::PreviewKey,
    cell: Option<(f32, f32)>,
) -> app::Preview {
    let (w, h) = match (key.pixels, cell) {
        (true, Some((cw, ch))) => (
            ((key.cols as f32 * cw) as usize).min(1400),
            ((key.rows as f32 * ch) as usize).min(1400),
        ),
        _ => (key.cols as usize, key.rows as usize * 2),
    };
    let fb = proteus_render::preview_framebuffer(scene, w, h, scene.default_color_scheme());
    app::Preview {
        width: fb.width,
        height: fb.height,
        pixels: fb
            .colors
            .iter()
            .map(|c| {
                (*c != proteus_render::rasterizer::buffer::ColorRGB::BLACK)
                    .then_some((c.r, c.g, c.b))
            })
            .collect(),
    }
}

/// Place (or clear) the kitty picture over the preview pane after a frame: sent only when the
/// picture or its place changed, deleted when no pane wants one.
fn place_picture(app: &App, placed: &mut Option<(app::PreviewKey, Rect)>) {
    use std::io::Write;
    const ID: u32 = 0x5068; // "Ph"
    let want = app.preview_want.borrow().clone().filter(|(k, _)| k.pixels);
    let ready = want
        .as_ref()
        .and_then(|(k, r)| app.previews.get(k).map(|p| (k.clone(), *r, p)));
    let mut out = String::new();
    match ready {
        Some((k, r, p)) => {
            if placed.as_ref() != Some(&(k.clone(), r)) {
                let mut fb =
                    proteus_render::rasterizer::buffer::Framebuffer::new(p.width, p.height);
                for (i, px) in p.pixels.iter().enumerate() {
                    if let Some((r_, g, b)) = px {
                        fb.colors[i] =
                            proteus_render::rasterizer::buffer::ColorRGB::new(*r_, *g, *b);
                    }
                }
                out.push_str(&format!("\x1b7\x1b[{};{}H", r.y + 1, r.x + 1));
                proteus_render::terminal::KittyRenderer::frame(
                    &fb, r.width, r.height, ID, &mut out,
                );
                out.push_str("\x1b8");
                *placed = Some((k, r));
            }
        }
        None => {
            if placed.is_some() {
                proteus_render::terminal::KittyRenderer::delete(ID, &mut out);
                *placed = None;
            }
        }
    }
    if !out.is_empty() {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}

/// Run `spec` and return whether it succeeded and one line for the status bar. For [`RunMode::Quiet`] the output goes
/// to a temporary file rather than a pipe: `view --web` starts a browser through `xdg-open`,
/// which inherits the output and may hold a pipe open for as long as the browser runs, so
/// reading a pipe to its end could wait for the browser to close.
fn run_child(exe: &Path, spec: &CommandSpec) -> Result<(bool, String)> {
    let quiet = spec.mode == RunMode::Quiet;
    if !quiet {
        let a = proteus_render::brand::ansi::Ansi::detect();
        use proteus_render::brand::Role;
        println!(
            "{} {}",
            a.paint(Role::Dim, "~ $"),
            a.paint(Role::Muted, &spec.display())
        );
    }
    let capture = if quiet {
        Some(tempfile::tempfile().context("temporary file for the output")?)
    } else {
        None
    };
    let mut children = Vec::new();
    let mut previous_stdout: Option<std::process::ChildStdout> = None;
    let last = spec.stages.len() - 1;
    for (i, args) in spec.stages.iter().enumerate() {
        let mut cmd = Command::new(exe);
        cmd.args(args);
        if let Some(out) = previous_stdout.take() {
            cmd.stdin(Stdio::from(out));
        } else if spec.stdin.is_some() {
            cmd.stdin(Stdio::piped());
        } else if quiet {
            cmd.stdin(Stdio::null());
        }
        if i < last {
            cmd.stdout(Stdio::piped());
        }
        if let Some(file) = &capture {
            if i == last {
                cmd.stdout(file.try_clone()?);
            }
            cmd.stderr(file.try_clone()?);
        }
        let mut child = cmd.spawn().context("spawn")?;
        if let Ok(mut c) = CHILDREN.lock() {
            c.push(child.id());
        }
        if i == 0 {
            if let (Some(lines), Some(mut stdin)) = (&spec.stdin, child.stdin.take()) {
                let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
                // Written from a thread so a large input cannot deadlock against the pipe.
                std::thread::spawn(move || {
                    let _ = stdin.write_all(text.as_bytes());
                });
            }
        }
        if i < last {
            previous_stdout = child.stdout.take();
        }
        children.push(child);
    }

    let mut failure = None;
    for (i, mut child) in children.into_iter().enumerate() {
        let status = child.wait()?;
        if let Ok(mut c) = CHILDREN.lock() {
            c.retain(|pid| *pid != child.id());
        }
        if !status.success() && failure.is_none() {
            failure = Some((i, status));
        }
    }
    let mut captured = String::new();
    if let Some(mut file) = capture {
        use std::io::Seek as _;
        file.rewind()?;
        let _ = file.read_to_string(&mut captured);
    }

    let summary = match failure {
        None if quiet => last_line(&captured).unwrap_or("done").to_string(),
        None => "done".to_string(),
        Some((i, status)) => {
            let what = spec.stages[i].first().map_or("proteus", String::as_str);
            let detail = last_line(&captured).unwrap_or("");
            format!("proteus {what} failed ({status}) {detail}")
        }
    };
    if spec.mode == RunMode::Pause {
        let a = proteus_render::brand::ansi::Ansi::detect();
        use proteus_render::brand::Role;
        let line = match failure {
            None => a.paint(Role::Accent, &format!("✓ {summary}")),
            Some(_) => a.paint(Role::Bad, &format!("✗ {summary}")),
        };
        print!(
            "\n{line}\n{} ",
            a.paint(Role::Dim, "press enter to return to proteus…")
        );
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
    }
    Ok((failure.is_none(), strip_ansi(&summary).trim().to_string()))
}

fn last_line(s: &str) -> Option<&str> {
    s.lines().map(str::trim).rfind(|l| !l.is_empty())
}

/// Remove ANSI escape sequences so a captured line can be shown in the status bar.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Process ids of the child `proteus` commands running now, so a signal can stop them too.
static CHILDREN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

/// SIGTERM and SIGHUP end the home screen: the handler itself stops any running child, restores
/// the terminal and exits with 128 + the signal. It must not wait for the event loop, which may
/// never come back: once the terminal is gone (a closed window, a killed tmux pane) crossterm
/// retries the dead tty inside `event::poll` without returning, so a flag checked by the loop
/// was never seen and the process spun at full CPU. SIGINT is swallowed: in raw mode Ctrl-C is
/// a key, and while a child runs the child gets the terminal's Ctrl-C itself.
#[cfg(unix)]
fn watch_signals() -> Result<Arc<AtomicI32>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate())?;
    let mut hup = signal(SignalKind::hangup())?;
    let mut int = signal(SignalKind::interrupt())?;
    let received = Arc::new(AtomicI32::new(0));
    let flag = Arc::clone(&received);
    tokio::spawn(async move {
        let n = loop {
            tokio::select! {
                _ = term.recv() => break 15,
                _ = hup.recv() => break 1,
                _ = int.recv() => continue,
            }
        };
        flag.store(n, Ordering::SeqCst);
        let pids: Vec<u32> = CHILDREN.lock().map(|c| c.clone()).unwrap_or_default();
        for pid in pids {
            let _ = Command::new("kill")
                .arg(if n == 1 { "-HUP" } else { "-TERM" })
                .arg(pid.to_string())
                .status();
        }
        let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
        ratatui::restore();
        std::process::exit(128 + n);
    });
    Ok(received)
}

#[cfg(not(unix))]
fn watch_signals() -> Result<Arc<AtomicI32>> {
    Ok(Arc::new(AtomicI32::new(0)))
}

#[cfg(test)]
mod tests;
