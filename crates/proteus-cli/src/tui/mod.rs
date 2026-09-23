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
const SETTLE: Duration = Duration::from_millis(300);

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
    let mut repo: Option<ProteusRepository> = None;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(PathBuf, Analysis)>();
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
        while let Ok((path, analysis)) = rx.try_recv() {
            app.analyses.insert(path, analysis);
        }
        let wanted = app.wanted_analysis();
        if wanted != resting.0 {
            resting = (wanted, Instant::now());
        } else if let Some(path) = wanted {
            if resting.1.elapsed() >= SETTLE {
                app.analyses.insert(path.clone(), Analysis::Pending);
                let tx = tx.clone();
                tokio::task::spawn_blocking(move || {
                    let result = analyse(&path);
                    let _ = tx.send((path, result));
                });
            }
        }

        app.tick = (started.elapsed().as_millis() / 500) as u64;
        terminal.draw(|f| ui::draw(f, &app))?;

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
                if leave {
                    let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
                    ratatui::restore();
                } else {
                    app.status = Some(format!("running: {}", spec.display()));
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
                    Ok(line) => line,
                    Err(e) => format!("could not run `{}`: {e:#}", spec.display()),
                });
                // A job may have been added or finished; the folder may have a new file.
                last_refresh = None;
                app.files.rescan();
            }
        }
    }
    Ok(())
}

/// The launch: the chain folds into the mark (brand::mark::FOLD_SECONDS), the wordmark and
/// signature appear, and it holds briefly. Any key skips it; `NO_MOTION` turns it off.
fn launch(terminal: &mut ratatui::DefaultTerminal, look: &style::Look) -> Result<()> {
    if !look.motion {
        return Ok(());
    }
    let fold = Duration::from_secs_f32(proteus_render::brand::mark::FOLD_SECONDS);
    let hold = Duration::from_millis(450);
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

/// The same analysis as `proteus analyze`, summarised as the browser page summarises it.
fn analyse(path: &Path) -> Analysis {
    let run = || -> Result<Analysis> {
        let loaded = proteus_core::io::load_structure(path)?;
        let a = proteus_core::metrics::analyze_pdb_detailed_with_header(
            &loaded.pdb,
            None,
            Some(&loaded.header_preview),
        )?;
        let residues = a.plddts.len();
        Ok(Analysis::Done {
            residues,
            predicted: a.metrics.confidence_source
                == proteus_core::confidence::ConfidenceSource::Predicted,
            rows: proteus_core::qc::summary_rows(&a.metrics, residues),
        })
    };
    run().unwrap_or_else(|e| Analysis::Failed(format!("{e:#}")))
}

/// Run `spec` and return one line for the status bar. For [`RunMode::Quiet`] the output goes
/// to a temporary file rather than a pipe: `view --web` starts a browser through `xdg-open`,
/// which inherits the output and may hold a pipe open for as long as the browser runs, so
/// reading a pipe to its end could wait for the browser to close.
fn run_child(exe: &Path, spec: &CommandSpec) -> Result<String> {
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
    Ok(strip_ansi(&summary).trim().to_string())
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

/// SIGTERM and SIGHUP end the home screen with the terminal restored. SIGINT is swallowed:
/// in raw mode Ctrl-C is a key, and while a child runs a Ctrl-C is meant for the child (which
/// gets it from the terminal too), not for the home screen behind it.
#[cfg(unix)]
fn watch_signals() -> Result<Arc<AtomicI32>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate())?;
    let mut hup = signal(SignalKind::hangup())?;
    let mut int = signal(SignalKind::interrupt())?;
    let received = Arc::new(AtomicI32::new(0));
    let flag = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            let n = tokio::select! {
                _ = term.recv() => 15,
                _ = hup.recv() => 1,
                _ = int.recv() => continue,
            };
            flag.store(n, Ordering::SeqCst);
            break;
        }
    });
    Ok(received)
}

#[cfg(not(unix))]
fn watch_signals() -> Result<Arc<AtomicI32>> {
    Ok(Arc::new(AtomicI32::new(0)))
}

#[cfg(test)]
mod tests;
