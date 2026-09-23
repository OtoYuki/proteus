//! State and key handling of the home screen, free of terminal I/O so that every key path and
//! every command line it builds can be tested.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use proteus_core::models::JobStatus;
use proteus_storage::repository::JobSummary;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Jobs,
    Structures,
    Run,
}

impl Tab {
    pub const ALL: [Tab; 3] = [Tab::Jobs, Tab::Structures, Tab::Run];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Jobs => "Jobs",
            Tab::Structures => "Structures",
            Tab::Run => "Run",
        }
    }

    fn index(self) -> usize {
        Tab::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

/// How a child `proteus` process uses the terminal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunMode {
    /// Takes over the terminal (the interactive viewer); nothing to read afterwards.
    Interactive,
    /// Prints something to read: wait for Enter before coming back.
    Pause,
    /// Output is captured; its last line goes to the status bar.
    Quiet,
}

/// A command the home screen runs as a child `proteus` process (or a pipe of them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    /// Pipeline stages, each the arguments after the program name.
    pub stages: Vec<Vec<String>>,
    /// Lines fed to the first stage's standard input.
    pub stdin: Option<Vec<String>>,
    pub mode: RunMode,
}

impl CommandSpec {
    fn one(args: &[&str], mode: RunMode) -> Self {
        Self {
            stages: vec![args.iter().map(|a| a.to_string()).collect()],
            stdin: None,
            mode,
        }
    }

    /// The command line as a user would type it in a POSIX shell.
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        if let Some(lines) = &self.stdin {
            let quoted: Vec<String> = lines.iter().map(|l| shell_quote(l)).collect();
            parts.push(format!("printf '%s\\n' {}", quoted.join(" ")));
        }
        for stage in &self.stages {
            let args: Vec<String> = stage.iter().map(|a| shell_quote(a)).collect();
            parts.push(format!("proteus {}", args.join(" ")));
        }
        parts.join(" | ")
    }
}

/// Quote `s` for a shell: bare when it is only safe characters, in single quotes otherwise
/// (each `'` written as `'\''`), and as `$'…'` when it holds a newline, so the command stays on
/// one line (bash, zsh and ksh read `$'…'`).
pub fn shell_quote(s: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "_./:=@%+,-".contains(c);
    if !s.is_empty() && s.chars().all(safe) {
        s.to_string()
    } else if s.contains('\n') {
        let escaped = s
            .replace('\\', r"\\")
            .replace('\'', r"\'")
            .replace('\n', r"\n");
        format!("$'{escaped}'")
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// What the event loop must do after a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Quit,
    Run(CommandSpec),
    RefreshJobs,
}

// ---------------------------------------------------------------------------------------------
// Jobs

#[derive(Default)]
pub struct JobsView {
    pub all: Vec<JobSummary>,
    /// The database was read at least once (an empty list then means "no jobs").
    pub loaded: bool,
    /// Why the last read failed, if it did.
    pub error: Option<String>,
    pub filter: String,
    pub filtering: bool,
    pub selected: usize,
}

impl JobsView {
    /// Jobs matching the filter (case-insensitive, over id, name, status and engine).
    pub fn visible(&self) -> Vec<&JobSummary> {
        let needle = self.filter.to_lowercase();
        self.all
            .iter()
            .filter(|j| {
                needle.is_empty()
                    || j.job.id.to_string().contains(&needle)
                    || j.header.to_lowercase().contains(&needle)
                    || format!("{:?}", j.job.status)
                        .to_lowercase()
                        .contains(&needle)
                    || engine(j).to_lowercase().contains(&needle)
            })
            .collect()
    }

    pub fn current(&self) -> Option<&JobSummary> {
        self.visible().get(self.selected).copied()
    }

    /// Replace the list, keeping the same job selected when it is still there.
    pub fn replace(&mut self, jobs: Vec<JobSummary>) {
        let keep = self.current().map(|j| j.job.id);
        self.all = jobs;
        self.loaded = true;
        self.error = None;
        if let Some(id) = keep {
            if let Some(i) = self.visible().iter().position(|j| j.job.id == id) {
                self.selected = i;
            }
        }
        self.clamp();
    }

    fn clamp(&mut self) {
        let n = self.visible().len();
        self.selected = self.selected.min(n.saturating_sub(1));
    }
}

/// The engine that produced a job's structure, from the prediction's metadata.
pub fn engine(j: &JobSummary) -> &str {
    if j.pdb_path.is_none() {
        return "";
    }
    proteus_engine::engine_name(j.metadata.as_ref())
}

// ---------------------------------------------------------------------------------------------
// Structures

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
}

pub struct FilesView {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
    pub selected: usize,
    pub error: Option<String>,
}

impl FilesView {
    pub fn open(dir: &Path) -> Self {
        let mut view = Self {
            dir: dir.to_path_buf(),
            entries: Vec::new(),
            selected: 0,
            error: None,
        };
        view.rescan();
        view
    }

    /// Folders (not hidden) and structure files of `dir`, folders first, each sorted by name.
    /// Names that are not UTF-8 are skipped: they could not be passed on as arguments intact.
    pub fn rescan(&mut self) {
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        match std::fs::read_dir(&self.dir) {
            Ok(rd) => {
                self.error = None;
                for e in rd.flatten() {
                    let path = e.path();
                    let Some(name) = e.file_name().to_str().map(str::to_string) else {
                        continue;
                    };
                    if name.starts_with('.') {
                        continue;
                    }
                    let Ok(meta) = std::fs::metadata(&path) else {
                        continue;
                    };
                    if meta.is_dir() {
                        dirs.push(Entry {
                            name,
                            path,
                            is_dir: true,
                            size: 0,
                        });
                    } else if proteus_core::qc::is_structure_file_name(&path) {
                        files.push(Entry {
                            name,
                            path,
                            is_dir: false,
                            size: meta.len(),
                        });
                    }
                }
            }
            Err(e) => self.error = Some(format!("{}: {e}", self.dir.display())),
        }
        dirs.sort_by(|a, b| a.name.cmp(&b.name));
        files.sort_by(|a, b| a.name.cmp(&b.name));
        self.entries = Vec::new();
        if let Some(parent) = self.dir.parent() {
            self.entries.push(Entry {
                name: "..".into(),
                path: parent.to_path_buf(),
                is_dir: true,
                size: 0,
            });
        }
        self.entries.extend(dirs);
        self.entries.extend(files);
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
    }

    pub fn current(&self) -> Option<&Entry> {
        self.entries.get(self.selected)
    }

    fn enter(&mut self, dir: PathBuf) {
        let came_from = self.dir.clone();
        self.dir = dir;
        self.selected = 0;
        self.rescan();
        // Going up lands on the folder we came from, as in a file manager.
        if let Some(i) = self.entries.iter().position(|e| e.path == came_from) {
            self.selected = i;
        }
    }
}

/// The measurements of one structure file, as the Structures tab shows them.
#[derive(Clone, Debug, PartialEq)]
pub enum Analysis {
    Pending,
    Done {
        residues: usize,
        predicted: bool,
        rows: Vec<[String; 2]>,
    },
    Failed(String),
}

// ---------------------------------------------------------------------------------------------
// Run

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Text(String),
    Choice {
        options: &'static [&'static str],
        index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub label: &'static str,
    pub hint: &'static str,
    pub kind: FieldKind,
}

impl Field {
    fn text(label: &'static str, hint: &'static str) -> Self {
        Self {
            label,
            hint,
            kind: FieldKind::Text(String::new()),
        }
    }

    fn choice(label: &'static str, hint: &'static str, options: &'static [&'static str]) -> Self {
        Self {
            label,
            hint,
            kind: FieldKind::Choice { options, index: 0 },
        }
    }

    pub fn value(&self) -> &str {
        match &self.kind {
            FieldKind::Text(s) => s.trim(),
            FieldKind::Choice { options, index } => options[*index],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormKind {
    Fold,
    Scan,
}

impl FormKind {
    pub fn title(self) -> &'static str {
        match self {
            FormKind::Fold => "Fold a sequence",
            FormKind::Scan => "Scan a protein (mutate → fold → rank)",
        }
    }
}

const TIERS: &[&str] = &["fast", "sota", "full"];
const RUNNERS: &[&str] = &["auto", "esm-api", "oci", "simulated"];
const MODES: &[&str] = &["alanine", "saturation"];
const SCORERS: &[&str] = &["structure", "esm2", "hybrid"];

pub struct RunView {
    pub form: FormKind,
    /// Focused row: a field, or `fields().len()` for the Run button.
    pub focus: usize,
    pub editing: bool,
    pub fold: Vec<Field>,
    pub scan: Vec<Field>,
}

impl Default for RunView {
    fn default() -> Self {
        Self {
            form: FormKind::Fold,
            focus: 0,
            editing: false,
            fold: vec![
                Field::text(
                    "Sequence",
                    "amino-acid letters, a FASTA record, or the path of a FASTA file",
                ),
                Field::choice(
                    "Tier",
                    "fast = ESMFold; sota = Boltz (your own image); full = sota + checks",
                    TIERS,
                ),
                Field::choice(
                    "Runner",
                    "auto tries a container, then the ESMFold API; simulated is offline and fake",
                    RUNNERS,
                ),
            ],
            scan: vec![
                Field::text(
                    "Sequence",
                    "the wild type: amino-acid letters, a FASTA record, or a FASTA file path",
                ),
                Field::choice(
                    "Mutations",
                    "alanine = one Ala per position; saturation = all 19 per position",
                    MODES,
                ),
                Field::text("From residue", "optional, 1-based, inclusive"),
                Field::text("To residue", "optional, 1-based, inclusive"),
                Field::choice(
                    "Runner",
                    "auto tries a container, then the ESMFold API; simulated is offline and fake",
                    RUNNERS,
                ),
                Field::choice(
                    "Rank by",
                    "structure = fitness of the fold; esm2 = zero-shot ESM-2; hybrid = both",
                    SCORERS,
                ),
                Field::text("Export to", "optional .parquet, .csv or .json path"),
            ],
        }
    }
}

impl RunView {
    pub fn fields(&self) -> &[Field] {
        match self.form {
            FormKind::Fold => &self.fold,
            FormKind::Scan => &self.scan,
        }
    }

    fn fields_mut(&mut self) -> &mut Vec<Field> {
        match self.form {
            FormKind::Fold => &mut self.fold,
            FormKind::Scan => &mut self.scan,
        }
    }

    fn focused_mut(&mut self) -> Option<&mut Field> {
        let i = self.focus;
        self.fields_mut().get_mut(i)
    }

    /// The command the current form would run, or why it cannot run yet.
    pub fn command(&self) -> Result<CommandSpec, String> {
        let f = self.fields();
        let input = SequenceInput::parse(f[0].value())?;
        match self.form {
            FormKind::Fold => {
                let mut args = vec!["submit".to_string()];
                match input {
                    SequenceInput::File(p) => args.extend(["--file".into(), p]),
                    SequenceInput::Fasta(lines) => {
                        args.extend(["--fasta".into(), lines.join("\n")])
                    }
                }
                args.extend(["--tier".into(), f[1].value().into()]);
                args.extend(["--runner".into(), f[2].value().into()]);
                Ok(CommandSpec {
                    stages: vec![args],
                    stdin: None,
                    mode: RunMode::Pause,
                })
            }
            FormKind::Scan => {
                let mut mutate = vec!["mutate".to_string()];
                let stdin = match input {
                    SequenceInput::File(p) => {
                        mutate.push(p);
                        None
                    }
                    SequenceInput::Fasta(lines) => {
                        mutate.push("-".into());
                        Some(lines)
                    }
                };
                mutate.extend(["--mode".into(), f[1].value().into()]);
                for (field, flag) in [(&f[2], "--start"), (&f[3], "--end")] {
                    let v = field.value();
                    if !v.is_empty() {
                        match v.parse::<usize>() {
                            Ok(n) if n >= 1 => mutate.extend([flag.into(), n.to_string()]),
                            _ => return Err(format!("{} must be a whole number ≥ 1", field.label)),
                        }
                    }
                }
                let mut screen = vec!["screen".to_string(), "-".to_string()];
                screen.extend(["--runner".into(), f[4].value().into()]);
                screen.extend(["--scorer".into(), f[5].value().into()]);
                if !f[6].value().is_empty() {
                    screen.extend(["--export".into(), f[6].value().into()]);
                }
                Ok(CommandSpec {
                    stages: vec![mutate, screen],
                    stdin,
                    mode: RunMode::Pause,
                })
            }
        }
    }
}

/// What the Sequence field holds.
#[derive(Debug, PartialEq, Eq)]
enum SequenceInput {
    /// An existing file.
    File(String),
    /// A FASTA record, as lines (a bare sequence gets a `>query` header).
    Fasta(Vec<String>),
}

impl SequenceInput {
    fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("enter a sequence, a FASTA record or a FASTA file path".into());
        }
        if raw.starts_with('>') {
            if raw.lines().skip(1).any(|l| l.trim_start().starts_with('>')) {
                return Err(
                    "one sequence at a time here; put a library in a FASTA file instead".into(),
                );
            }
            let (header, seq) = match raw.split_once('\n') {
                // A pasted record keeps its lines: the first is the header.
                Some((header, rest)) => (header.trim().to_string(), rest.to_string()),
                // Typed on one line: the sequence is the trailing run of words that are all
                // capital letters, and the header is what comes before it.
                None => {
                    let words: Vec<&str> = raw.split_whitespace().collect();
                    let tail = words
                        .iter()
                        .rev()
                        .take_while(|w| w.chars().all(|c| c.is_ascii_uppercase()))
                        .count();
                    let split = words.len() - tail.min(words.len() - 1);
                    (words[..split].join(" "), words[split..].concat())
                }
            };
            let seq: String = seq.split_whitespace().collect();
            if seq.is_empty() {
                return Err("the FASTA record has a header but no sequence".into());
            }
            return Ok(Self::Fasta(vec![header, seq]));
        }
        let expanded = expand_home(raw);
        if Path::new(&expanded).is_file() {
            return Ok(Self::File(expanded));
        }
        let seq: String = raw.split_whitespace().collect();
        if !raw.contains('/') && seq.chars().all(|c| c.is_ascii_alphabetic()) {
            return Ok(Self::Fasta(vec![">query".into(), seq.to_ascii_uppercase()]));
        }
        Err(format!(
            "'{raw}' is neither a file nor a sequence of amino-acid letters"
        ))
    }
}

fn expand_home(p: &str) -> String {
    match (p.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => p.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// The whole screen

pub struct App {
    pub tab: Tab,
    pub jobs: JobsView,
    pub files: FilesView,
    pub analyses: HashMap<PathBuf, Analysis>,
    pub run: RunView,
    pub help: bool,
    /// One line for the status bar: the result of the last action, or an error.
    pub status: Option<String>,
    pub data_dir: PathBuf,
}

impl App {
    pub fn new(cwd: &Path, data_dir: &Path) -> Self {
        Self {
            tab: Tab::Jobs,
            jobs: JobsView::default(),
            files: FilesView::open(cwd),
            analyses: HashMap::new(),
            run: RunView::default(),
            help: false,
            status: None,
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// Whether keys are going into a text box rather than being commands.
    pub fn typing(&self) -> bool {
        (self.tab == Tab::Jobs && self.jobs.filtering) || (self.tab == Tab::Run && self.run.editing)
    }

    /// The structure file the Structures tab needs measurements for, if not already known.
    pub fn wanted_analysis(&self) -> Option<PathBuf> {
        if self.tab != Tab::Structures {
            return None;
        }
        let e = self.files.current()?;
        (!e.is_dir && !self.analyses.contains_key(&e.path)).then(|| e.path.clone())
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q')) {
            return Action::Quit;
        }
        if self.help {
            // Any key closes the help.
            self.help = false;
            return Action::None;
        }
        if self.typing() {
            return self.type_key(key);
        }
        match key.code {
            KeyCode::Char('q') => return Action::Quit,
            KeyCode::Char('?') => {
                self.help = true;
                return Action::None;
            }
            KeyCode::Tab => {
                self.tab = Tab::ALL[(self.tab.index() + 1) % Tab::ALL.len()];
                return Action::None;
            }
            KeyCode::BackTab => {
                self.tab = Tab::ALL[(self.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len()];
                return Action::None;
            }
            KeyCode::Char(c @ '1'..='3') => {
                self.tab = Tab::ALL[(c as u8 - b'1') as usize];
                return Action::None;
            }
            _ => {}
        }
        match self.tab {
            Tab::Jobs => self.jobs_key(key),
            Tab::Structures => self.files_key(key),
            Tab::Run => self.run_key(key),
        }
    }

    /// Pasted text (bracketed paste) goes into the text box being edited, or into a focused text
    /// field of the Run tab, and nowhere else: it is never read as keys, so a pasted sequence
    /// cannot trigger commands (a `q` in it would otherwise quit).
    pub fn handle_paste(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        if self.help {
            return;
        }
        match self.tab {
            Tab::Jobs if self.jobs.filtering => {
                self.jobs.filter.push_str(text.lines().next().unwrap_or(""));
                self.jobs.clamp();
            }
            Tab::Run => {
                if let Some(Field {
                    kind: FieldKind::Text(s),
                    ..
                }) = self.run.focused_mut()
                {
                    s.push_str(&text);
                    self.run.editing = true;
                }
            }
            _ => {}
        }
    }

    fn type_key(&mut self, key: KeyEvent) -> Action {
        let text: &mut String = if self.tab == Tab::Jobs {
            &mut self.jobs.filter
        } else {
            match self.run.focused_mut().map(|f| &mut f.kind) {
                Some(FieldKind::Text(s)) => s,
                _ => {
                    self.run.editing = false;
                    return Action::None;
                }
            }
        };
        match key.code {
            KeyCode::Char(c) => text.push(c),
            KeyCode::Backspace => {
                text.pop();
            }
            KeyCode::Enter | KeyCode::Down | KeyCode::Up | KeyCode::Tab => {
                self.jobs.filtering = false;
                self.run.editing = false;
                // Finishing a field moves to the next one (Up to the previous).
                if self.tab == Tab::Run {
                    let step = if key.code == KeyCode::Up {
                        KeyCode::Up
                    } else {
                        KeyCode::Down
                    };
                    return self.run_key(KeyEvent::new(step, KeyModifiers::NONE));
                }
            }
            KeyCode::Esc => {
                if self.tab == Tab::Jobs {
                    self.jobs.filter.clear();
                }
                self.jobs.filtering = false;
                self.run.editing = false;
            }
            _ => {}
        }
        self.jobs.clamp();
        Action::None
    }

    fn jobs_key(&mut self, key: KeyEvent) -> Action {
        let n = self.jobs.visible().len();
        move_selection(&mut self.jobs.selected, n, key.code);
        match key.code {
            KeyCode::Char('/') => self.jobs.filtering = true,
            KeyCode::Char('r') => return Action::RefreshJobs,
            KeyCode::Esc if !self.jobs.filter.is_empty() => {
                self.jobs.filter.clear();
                self.jobs.clamp();
            }
            KeyCode::Enter | KeyCode::Char('v') | KeyCode::Char('w') | KeyCode::Char('i') => {
                let Some(j) = self.jobs.current() else {
                    return Action::None;
                };
                let id = j.job.id.to_string();
                if key.code == KeyCode::Char('i') {
                    return Action::Run(CommandSpec::one(&["inspect", &id], RunMode::Pause));
                }
                if j.job.status != JobStatus::Completed || j.pdb_path.is_none() {
                    self.status = Some(format!(
                        "job {} has no structure to show ({:?})",
                        short_id(&id),
                        j.job.status
                    ));
                    return Action::None;
                }
                return Action::Run(if key.code == KeyCode::Char('w') {
                    CommandSpec::one(&["view", &id, "--web"], RunMode::Quiet)
                } else {
                    CommandSpec::one(&["view", &id, "--interactive"], RunMode::Interactive)
                });
            }
            _ => {}
        }
        Action::None
    }

    fn files_key(&mut self, key: KeyEvent) -> Action {
        move_selection(&mut self.files.selected, self.files.entries.len(), key.code);
        match key.code {
            KeyCode::Char('r') => {
                self.files.rescan();
                self.analyses.clear();
            }
            KeyCode::Backspace | KeyCode::Char('h') | KeyCode::Left => {
                if let Some(parent) = self.files.dir.parent().map(Path::to_path_buf) {
                    self.files.enter(parent);
                }
            }
            KeyCode::Enter
            | KeyCode::Right
            | KeyCode::Char('l')
            | KeyCode::Char('v')
            | KeyCode::Char('w')
            | KeyCode::Char('a') => {
                let Some(e) = self.files.current().cloned() else {
                    return Action::None;
                };
                if e.is_dir {
                    if matches!(
                        key.code,
                        KeyCode::Enter | KeyCode::Right | KeyCode::Char('l')
                    ) {
                        self.files.enter(e.path);
                    }
                    return Action::None;
                }
                let path = e.path.to_string_lossy().into_owned();
                return Action::Run(match key.code {
                    KeyCode::Char('w') => {
                        CommandSpec::one(&["view", &path, "--web"], RunMode::Quiet)
                    }
                    KeyCode::Char('a') => CommandSpec::one(&["analyze", &path], RunMode::Pause),
                    _ => CommandSpec::one(&["view", &path, "--interactive"], RunMode::Interactive),
                });
            }
            _ => {}
        }
        Action::None
    }

    fn run_key(&mut self, key: KeyEvent) -> Action {
        let rows = self.run.fields().len() + 1;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.run.focus = self.run.focus.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.run.focus = (self.run.focus + 1).min(rows - 1)
            }
            KeyCode::Char('f') => {
                self.run.form = match self.run.form {
                    FormKind::Fold => FormKind::Scan,
                    FormKind::Scan => FormKind::Fold,
                };
                self.run.focus = 0;
            }
            KeyCode::Left | KeyCode::Right => {
                if let Some(Field {
                    kind: FieldKind::Choice { options, index },
                    ..
                }) = self.run.focused_mut()
                {
                    let n = options.len();
                    *index = if key.code == KeyCode::Left {
                        (*index + n - 1) % n
                    } else {
                        (*index + 1) % n
                    };
                }
            }
            KeyCode::Enter => {
                if self.run.focus == rows - 1 {
                    return match self.run.command() {
                        Ok(spec) => Action::Run(spec),
                        Err(why) => {
                            self.status = Some(why);
                            Action::None
                        }
                    };
                }
                match self.run.focused_mut().map(|f| &mut f.kind) {
                    Some(FieldKind::Text(_)) => self.run.editing = true,
                    Some(FieldKind::Choice { options, index }) => {
                        *index = (*index + 1) % options.len()
                    }
                    None => {}
                }
            }
            _ => {}
        }
        Action::None
    }
}

fn move_selection(selected: &mut usize, n: usize, code: KeyCode) {
    let last = n.saturating_sub(1);
    *selected = match code {
        KeyCode::Down | KeyCode::Char('j') => (*selected + 1).min(last),
        KeyCode::Up | KeyCode::Char('k') => selected.saturating_sub(1),
        KeyCode::PageDown => (*selected + 10).min(last),
        KeyCode::PageUp => selected.saturating_sub(10),
        KeyCode::Home | KeyCode::Char('g') => 0,
        KeyCode::End | KeyCode::Char('G') => last,
        _ => *selected,
    };
}

pub fn short_id(id: &str) -> &str {
    &id[..id.len().min(8)]
}
