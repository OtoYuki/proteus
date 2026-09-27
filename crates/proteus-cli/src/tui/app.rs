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

    /// The tab's icon glyph.
    pub fn icon(self) -> char {
        use proteus_render::brand::icons::Icon;
        match self {
            Tab::Jobs => Icon::Jobs,
            Tab::Structures => Icon::Structures,
            Tab::Run => Icon::Run,
        }
        .glyph()
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
    /// What the status bar says while it runs and after it succeeds ("opening ubq…", "opened
    /// ubq in your browser"); without them it shows the command and its last line of output.
    pub doing: Option<String>,
    pub done: Option<String>,
}

impl CommandSpec {
    fn one(args: &[&str], mode: RunMode) -> Self {
        Self {
            stages: vec![args.iter().map(|a| a.to_string()).collect()],
            stdin: None,
            mode,
            doing: None,
            done: None,
        }
    }

    fn says(mut self, doing: impl Into<String>, done: impl Into<String>) -> Self {
        self.doing = Some(doing.into());
        self.done = Some(done.into());
        self
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
    } else if s.chars().any(char::is_control) {
        // $'…' with every control character escaped: a newline stays on one line, and an
        // escape sequence in a file name is shown, never sent to the terminal.
        let mut out = String::from("$'");
        for c in s.chars() {
            match c {
                '\\' => out.push_str(r"\\"),
                '\'' => out.push_str(r"\'"),
                '\n' => out.push_str(r"\n"),
                '\t' => out.push_str(r"\t"),
                c if c.is_control() => {
                    for b in c.to_string().bytes() {
                        out.push_str(&format!("\\x{b:02x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('\'');
        out
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
    pub sort: JobSort,
    /// The new name being typed for the selected job (`n`).
    pub renaming: Option<String>,
    /// The job a delete waits on `y` for (`x`).
    pub deleting: Option<uuid::Uuid>,
}

/// The order of the jobs list; `s` cycles through them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum JobSort {
    #[default]
    Newest,
    Name,
    State,
    Confidence,
}

impl JobSort {
    pub fn label(self) -> &'static str {
        match self {
            JobSort::Newest => "newest first",
            JobSort::Name => "by name",
            JobSort::State => "by state",
            JobSort::Confidence => "most confident first",
        }
    }

    fn next(self) -> Self {
        match self {
            JobSort::Newest => JobSort::Name,
            JobSort::Name => JobSort::State,
            JobSort::State => JobSort::Confidence,
            JobSort::Confidence => JobSort::Newest,
        }
    }
}

/// Where a state sorts: work in flight first, then failures to look at, then finished work.
fn state_rank(s: &JobStatus) -> u8 {
    match s {
        JobStatus::Running => 0,
        JobStatus::Queued | JobStatus::Pending => 1,
        JobStatus::Failed => 2,
        JobStatus::Completed => 3,
        JobStatus::Cancelled => 4,
    }
}

impl JobsView {
    /// Jobs matching the filter (case-insensitive, over id, name, status and engine).
    pub fn visible(&self) -> Vec<&JobSummary> {
        let needle = self.filter.to_lowercase();
        let mut v: Vec<&JobSummary> = self
            .all
            .iter()
            .filter(|j| {
                needle.is_empty()
                    || j.job.id.to_string().contains(&needle)
                    || j.header.to_lowercase().contains(&needle)
                    || format!("{:?}", j.job.status)
                        .to_lowercase()
                        .contains(&needle)
                    || state_word(&j.job.status).contains(&needle)
                    || engine(j).to_lowercase().contains(&needle)
                    || model(j).to_lowercase().contains(&needle)
            })
            .collect();
        // Stable sorts over the newest-first list, so ties stay newest first.
        match self.sort {
            JobSort::Newest => {}
            JobSort::Name => v.sort_by_cached_key(|j| j.header.to_lowercase()),
            JobSort::State => v.sort_by_key(|j| state_rank(&j.job.status)),
            JobSort::Confidence => v.sort_by(|a, b| {
                b.plddt
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&a.plddt.unwrap_or(f64::NEG_INFINITY))
            }),
        }
        v
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

/// The word the jobs list shows for a state (the filter matches it as well as the state's name).
pub fn state_word(s: &JobStatus) -> &'static str {
    match s {
        JobStatus::Completed => "done",
        JobStatus::Failed => "failed",
        JobStatus::Cancelled => "cancelled",
        JobStatus::Running => "running",
        JobStatus::Queued => "queued",
        JobStatus::Pending => "pending",
    }
}

/// The engine that produced a job's structure, from the prediction's metadata.
pub fn engine(j: &JobSummary) -> &str {
    if j.pdb_path.is_none() {
        return "";
    }
    proteus_engine::engine_name(j.metadata.as_ref())
}

/// The model that folded a job, for the list (`boltz+msa`); see `proteus_engine::model_name`.
pub fn model(j: &JobSummary) -> String {
    if j.pdb_path.is_none() {
        return String::new();
    }
    proteus_engine::model_name(j.metadata.as_ref())
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
    /// The folder proteus started in (`.` goes back to it).
    pub start: PathBuf,
    /// Every structure file below the folder in one list (`f`), rather than the folder.
    pub flat: bool,
    /// What each folder looked at holds, read once.
    peeks: std::cell::RefCell<HashMap<PathBuf, std::rc::Rc<DirPeek>>>,
}

/// What a folder holds, for the folder card: its structure files (the first few) and counts.
#[derive(Debug, Default)]
pub struct DirPeek {
    pub structures: Vec<(String, u64)>,
    pub n_structures: usize,
    pub n_dirs: usize,
    /// Structure files further down (up to 4 folders deep, at most 3 000 entries looked at),
    /// by path relative to the folder: the first 10, and how many there are.
    pub below: Vec<String>,
    pub n_below: usize,
}

/// Structure files under `dir`, breadth first, skipping hidden folders and build output: the
/// first `keep` by relative path and size, and how many there are. `top` counts the folder's
/// own files too.
fn find_below(dir: &Path, keep: usize, top: bool) -> (Vec<(String, u64)>, usize) {
    const SKIP: [&str; 4] = ["target", "node_modules", "__pycache__", "venv"];
    let mut queue = std::collections::VecDeque::from([(dir.to_path_buf(), 0usize)]);
    let (mut found, mut n, mut seen) = (Vec::new(), 0usize, 0usize);
    while let Some((d, depth)) = queue.pop_front() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            seen += 1;
            if seen > 3000 {
                return (found, n);
            }
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || SKIP.contains(&name.as_ref()) {
                continue;
            }
            let path = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() && depth < 4 => queue.push_back((path, depth + 1)),
                Ok(t)
                    if t.is_file()
                        && (depth > 0 || top)
                        && proteus_core::qc::is_structure_file_name(&path) =>
                {
                    n += 1;
                    if found.len() < keep {
                        if let Ok(rel) = path.strip_prefix(dir) {
                            let size = e.metadata().map_or(0, |m| m.len());
                            found.push((rel.to_string_lossy().into_owned(), size));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    (found, n)
}

impl FilesView {
    pub fn open(dir: &Path) -> Self {
        let mut view = Self {
            dir: dir.to_path_buf(),
            entries: Vec::new(),
            selected: 0,
            error: None,
            start: dir.to_path_buf(),
            flat: false,
            peeks: Default::default(),
        };
        view.rescan();
        view
    }

    /// Folders (not hidden) and structure files of `dir`, folders first, each sorted by name.
    /// Names that are not UTF-8 are skipped: they could not be passed on as arguments intact.
    pub fn rescan(&mut self) {
        self.peeks.borrow_mut().clear();
        if self.flat {
            let (found, _) = find_below(&self.dir, 500, true);
            self.error = None;
            self.entries = vec![Entry {
                name: "..".into(),
                path: self.dir.clone(),
                is_dir: true,
                size: 0,
            }];
            self.entries
                .extend(found.into_iter().map(|(rel, size)| Entry {
                    path: self.dir.join(&rel),
                    name: rel,
                    is_dir: false,
                    size,
                }));
            self.selected = self.selected.min(self.entries.len().saturating_sub(1));
            return;
        }
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
                    } else if meta.is_file() && proteus_core::qc::is_structure_file_name(&path) {
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

    /// What folder `dir` holds: read on first look (at most 5 000 entries), then remembered
    /// until a rescan.
    pub fn peek(&self, dir: &Path) -> std::rc::Rc<DirPeek> {
        if let Some(p) = self.peeks.borrow().get(dir) {
            return p.clone();
        }
        let mut p = DirPeek::default();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten().take(5000) {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                match e.file_type() {
                    Ok(t) if t.is_dir() => p.n_dirs += 1,
                    Ok(_) if proteus_core::qc::is_structure_file_name(&e.path()) => {
                        p.n_structures += 1;
                        let size = e.metadata().map_or(0, |m| m.len());
                        p.structures.push((name, size));
                    }
                    _ => {}
                }
            }
        }
        p.structures.sort();
        p.structures.truncate(12);
        if p.n_structures == 0 {
            let (below, n) = find_below(dir, 10, false);
            p.below = below.into_iter().map(|(rel, _)| rel).collect();
            p.n_below = n;
        }
        let p = std::rc::Rc::new(p);
        self.peeks.borrow_mut().insert(dir.to_path_buf(), p.clone());
        p
    }

    fn enter(&mut self, dir: PathBuf) {
        // Out of the flat list: ".." there goes back to the folder itself.
        if self.flat {
            self.flat = false;
            self.selected = 0;
            self.dir = dir;
            self.rescan();
            return;
        }
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

/// The measurements of one structure file, as the inspector shows them.
#[derive(Clone, Debug, PartialEq)]
pub enum Analysis {
    Pending,
    Done {
        residues: usize,
        predicted: bool,
        /// Every measurement as a labelled line (the `analyze` report, summarised).
        rows: Vec<[String; 2]>,
        /// For a complex: the interface the viewers open on, as label/value rows.
        interface: Vec<[String; 2]>,
        /// For a complex with the predictor's PAE: whether ipSAE_min clears 0.61, and the line.
        verdict: Option<(bool, String)>,
        /// The numbers the inspector draws as gauges.
        facts: Option<Box<Facts>>,
    },
    Failed(String),
}

/// The headline numbers of a structure, for the inspector's gauges and groups.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    pub chains: usize,
    pub plddt: Option<f64>,
    pub ptm: Option<f64>,
    pub iptm: Option<f64>,
    pub rama_favored: f64,
    pub rama_outliers: usize,
    pub helix: f64,
    pub strand: f64,
    pub coil: f64,
    pub rg_ratio: f64,
    pub sasa: f64,
    pub burial: f64,
    pub overlaps_per_1k: f64,
    pub bond_outliers: Option<usize>,
    pub angle_outliers: Option<usize>,
    pub rotamer_outliers: Option<f64>,
    pub hbonds: usize,
    pub salt_bridges: usize,
    pub pi: usize,
    pub triage: f64,
}

/// A still wanted for a preview pane: which file, how many cells, and whether the terminal
/// draws real pixels (kitty) or half-block cells.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreviewKey {
    pub path: PathBuf,
    pub cols: u16,
    pub rows: u16,
    pub pixels: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Option<(u8, u8, u8)>>,
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
    /// A number: digits typed on it are its value, not tab switches.
    pub numeric: bool,
}

impl Field {
    fn text(label: &'static str, hint: &'static str) -> Self {
        Self {
            label,
            hint,
            kind: FieldKind::Text(String::new()),
            numeric: false,
        }
    }

    fn number(label: &'static str, hint: &'static str) -> Self {
        Self {
            numeric: true,
            ..Self::text(label, hint)
        }
    }

    fn choice(label: &'static str, hint: &'static str, options: &'static [&'static str]) -> Self {
        Self {
            label,
            hint,
            kind: FieldKind::Choice { options, index: 0 },
            numeric: false,
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
            FormKind::Fold => "fold a sequence",
            FormKind::Scan => "scan a protein",
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
    /// The example ctrl-e put in last (it cycles).
    pub example: Option<usize>,
}

/// Sequences to try, as FASTA records (name, sequence): small, well known, and quick to fold.
pub const EXAMPLES: [(&str, &str); 3] = [
    (
        "ubiquitin",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    ),
    (
        "protein G B1",
        "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    ),
    ("Trp-cage", "NLYIQWLKDGGPSSGRPPPS"),
];

impl Default for RunView {
    fn default() -> Self {
        Self {
            form: FormKind::Fold,
            focus: 0,
            editing: false,
            example: None,
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
                Field::number("From residue", "optional, 1-based, inclusive"),
                Field::number("To residue", "optional, 1-based, inclusive"),
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
                    doing: None,
                    done: Some("folded; the job is at the top of Jobs".into()),
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
                    doing: None,
                    done: Some("scan finished".into()),
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
    /// A FASTA record, as lines (a bare sequence is named by its first residues).
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
                // Typed on one line: the last word is the sequence (any case) and the rest is
                // the header. A sequence broken by spaces needs a FASTA file or a paste.
                None => match raw.rsplit_once(char::is_whitespace) {
                    Some((header, seq)) if seq.chars().all(|c| c.is_ascii_alphabetic()) => {
                        (header.trim().to_string(), seq.to_ascii_uppercase())
                    }
                    _ => (raw.to_string(), String::new()),
                },
            };
            let seq: String = seq
                .split_whitespace()
                .collect::<String>()
                .to_ascii_uppercase();
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
            // Named by its first residues, so two pasted sequences tell apart in Jobs.
            let seq = seq.to_ascii_uppercase();
            let name: String = seq.chars().take(8).collect();
            return Ok(Self::Fasta(vec![format!(">{name}"), seq]));
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
    /// The (modification time, size) each measurement was taken at: a file changed since is
    /// measured again.
    pub stamps: HashMap<PathBuf, Option<(std::time::SystemTime, u64)>>,
    pub run: RunView,
    pub help: bool,
    /// One line for the status bar: the result of the last action, or an error.
    pub status: Option<String>,
    /// The last command run, shown at the prompt in the status line.
    pub last_command: Option<String>,
    pub data_dir: PathBuf,
    /// Colours at the terminal's depth, and whether motion is on.
    pub look: super::style::Look,
    /// Half-second ticks since start, for the running-state pulse.
    pub tick: u64,
    /// Parsed structures, kept to render previews at whatever size the pane has.
    pub scenes: HashMap<PathBuf, std::sync::Arc<proteus_render::StructureRenderData>>,
    /// The preview the last frame drew a pane for (set while drawing, read by the loop).
    pub preview_want: std::cell::RefCell<Option<(PreviewKey, ratatui::layout::Rect)>>,
    /// The latest rendered preview.
    pub previews: HashMap<PreviewKey, Preview>,
    /// Pixels per cell when the terminal draws kitty graphics, else `None`.
    pub cell_pixels: Option<(f32, f32)>,
}

impl App {
    pub fn new(cwd: &Path, data_dir: &Path) -> Self {
        Self {
            tab: Tab::Jobs,
            jobs: JobsView::default(),
            files: FilesView::open(cwd),
            analyses: HashMap::new(),
            stamps: HashMap::new(),
            run: RunView::default(),
            scenes: HashMap::new(),
            preview_want: std::cell::RefCell::new(None),
            previews: HashMap::new(),
            cell_pixels: None,
            help: false,
            status: None,
            last_command: None,
            data_dir: data_dir.to_path_buf(),
            look: super::style::Look::with_depth(proteus_render::brand::ColorDepth::TrueColor),
            tick: 0,
        }
    }

    /// Show tab `t`. A passing note in the status bar belongs to the tab it was said on; the
    /// result of a command (✓ or ✗) stays until the next one.
    fn switch_tab(&mut self, t: Tab) {
        if t != self.tab
            && self
                .status
                .as_deref()
                .is_some_and(|s| !s.starts_with('✓') && !s.starts_with('✗'))
        {
            self.status = None;
        }
        self.tab = t;
    }

    /// The name of job `id` as the list shows it.
    fn job_name(&self, id: uuid::Uuid) -> String {
        self.jobs.all.iter().find(|j| j.job.id == id).map_or_else(
            || id.to_string()[..8].to_string(),
            |j| display_name(&j.header),
        )
    }

    /// A still of `path` at any size: shown (in half-block cells) while the right size renders.
    pub fn any_preview(&self, path: &Path) -> Option<&Preview> {
        self.previews
            .iter()
            .find(|(k, _)| k.path == path)
            .map(|(_, p)| p)
    }

    /// Keep a rendered still, forgetting the others when there are many (each is a few MB of
    /// pixels at kitty resolution).
    pub fn keep_preview(&mut self, key: PreviewKey, p: Preview) {
        if self.previews.len() >= 40 {
            self.previews.clear();
        }
        self.previews.insert(key, p);
    }

    /// Structure files worth measuring before they are asked for: every job's model, the
    /// selected job's neighbours first, then newest first. Skips what is measured or under way.
    pub fn prefetch(&self) -> Vec<PathBuf> {
        let jobs = self.jobs.visible();
        let sel = self.jobs.selected;
        let mut order: Vec<usize> = Vec::new();
        for d in 1..=jobs.len() {
            if let Some(i) = sel.checked_add(d).filter(|i| *i < jobs.len()) {
                order.push(i);
            }
            if let Some(i) = sel.checked_sub(d) {
                order.push(i);
            }
        }
        order
            .into_iter()
            .filter_map(|i| jobs[i].pdb_path.as_ref().map(PathBuf::from))
            .filter(|p| !self.analyses.contains_key(p))
            .collect()
    }

    /// The jobs next to the selected one (for previews rendered ahead of a key press).
    pub fn neighbour_models(&self) -> Vec<PathBuf> {
        let jobs = self.jobs.visible();
        let sel = self.jobs.selected;
        [sel.checked_add(1), sel.checked_sub(1), sel.checked_add(2)]
            .into_iter()
            .flatten()
            .filter_map(|i| {
                jobs.get(i)
                    .and_then(|j| j.pdb_path.as_ref())
                    .map(PathBuf::from)
            })
            .collect()
    }

    /// Whether keys are going into a text box rather than being commands.
    pub fn typing(&self) -> bool {
        (self.tab == Tab::Jobs && (self.jobs.filtering || self.jobs.renaming.is_some()))
            || (self.tab == Tab::Run && self.run.editing)
    }

    /// The structure file the current tab needs measurements (and a preview) for, if not
    /// already known: the selected file in Structures, the selected job's model in Jobs.
    pub fn wanted_analysis(&self) -> Option<PathBuf> {
        let path = match self.tab {
            Tab::Structures => {
                let e = self.files.current()?;
                if e.is_dir {
                    return None;
                }
                e.path.clone()
            }
            Tab::Jobs => PathBuf::from(self.jobs.current()?.pdb_path.as_ref()?),
            _ => return None,
        };
        let fresh =
            self.analyses.contains_key(&path) && self.stamps.get(&path) == Some(&file_stamp(&path));
        (!fresh).then_some(path)
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q')) {
            return Action::Quit;
        }
        // A delete waits for `y`; any other key keeps the job.
        if let Some(id) = self.jobs.deleting.take() {
            if key.code == KeyCode::Char('y') {
                let short = id.to_string()[..8].to_string();
                let name = self.job_name(id);
                return Action::Run(
                    CommandSpec::one(&["delete", &short], RunMode::Quiet)
                        .says(format!("deleting {name}…"), format!("deleted {name}")),
                );
            }
            self.status = Some("kept it; nothing was deleted".into());
            return Action::None;
        }
        if self.help {
            // Any key closes the help.
            self.help = false;
            return Action::None;
        }
        // F1 opens the help from anywhere, even a text field where ? is just a character.
        if key.code == KeyCode::F(1) {
            self.help = true;
            return Action::None;
        }
        // Ctrl-e in the Run tab puts the next example into the sequence field.
        if ctrl && key.code == KeyCode::Char('e') && self.tab == Tab::Run {
            let i = self.run.example.map_or(0, |i| (i + 1) % EXAMPLES.len());
            self.run.example = Some(i);
            let (name, seq) = EXAMPLES[i];
            let fields = match self.run.form {
                FormKind::Fold => &mut self.run.fold,
                FormKind::Scan => &mut self.run.scan,
            };
            fields[0].kind = FieldKind::Text(format!(">{name}\n{seq}"));
            self.run.editing = false;
            self.status = Some(format!("example: {name}; ctrl-e for another"));
            return Action::None;
        }
        // Alt+1..3 switches tabs from anywhere, even mid-edit; the edit is kept.
        if let KeyCode::Char(c @ '1'..='3') = key.code {
            if key.modifiers.contains(KeyModifiers::ALT) {
                self.jobs.filtering = false;
                self.jobs.renaming = None;
                self.run.editing = false;
                self.switch_tab(Tab::ALL[(c as u8 - b'1') as usize]);
                return Action::None;
            }
        }
        if self.typing() {
            return self.type_key(key);
        }
        // On a focused text field of the Run form, a printable key is text: it starts editing
        // rather than acting as a command (a sequence typed straight in used to quit at a q,
        // losing the form). Digits are the exception: they switch tabs, as everywhere else,
        // unless the field holds a number.
        if let KeyCode::Char(c) = key.code {
            let tab_digit = matches!(c, '1'..='3');
            if self.tab == Tab::Run && !ctrl && !c.is_control() {
                if let Some(Field {
                    kind: FieldKind::Text(s),
                    numeric,
                    ..
                }) = self.run.focused_mut()
                {
                    if !tab_digit || *numeric {
                        s.push(c);
                        self.run.editing = true;
                        return Action::None;
                    }
                }
            }
        }
        match key.code {
            KeyCode::Char('q') => return Action::Quit,
            // Esc backs out of whatever is open, and quits when nothing is.
            KeyCode::Esc if self.tab != Tab::Jobs || self.jobs.filter.is_empty() => {
                return Action::Quit
            }
            KeyCode::Char('?') => {
                self.help = true;
                return Action::None;
            }
            KeyCode::Tab => {
                self.switch_tab(Tab::ALL[(self.tab.index() + 1) % Tab::ALL.len()]);
                return Action::None;
            }
            KeyCode::BackTab => {
                self.switch_tab(Tab::ALL[(self.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]);
                return Action::None;
            }
            KeyCode::Char(c @ '1'..='3') => {
                self.switch_tab(Tab::ALL[(c as u8 - b'1') as usize]);
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
        if self.tab == Tab::Jobs {
            if let Some(name) = self.jobs.renaming.as_mut() {
                match key.code {
                    KeyCode::Char(c) => name.push(c),
                    KeyCode::Backspace => {
                        name.pop();
                    }
                    KeyCode::Esc => self.jobs.renaming = None,
                    KeyCode::Enter => {
                        let name = self.jobs.renaming.take().unwrap_or_default();
                        let name = name.trim();
                        let Some(j) = self.jobs.current() else {
                            return Action::None;
                        };
                        if name.is_empty() || name == j.header {
                            self.status = Some("name unchanged".into());
                            return Action::None;
                        }
                        let short = short_id(&j.job.id.to_string()).to_string();
                        return Action::Run(
                            CommandSpec::one(&["rename", &short, name], RunMode::Quiet)
                                .says("renaming…", format!("renamed to {name}")),
                        );
                    }
                    _ => {}
                }
                return Action::None;
            }
        }
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
            KeyCode::Char('s') => {
                self.jobs.sort = self.jobs.sort.next();
                self.jobs.selected = 0;
            }
            KeyCode::Char('n') => {
                if let Some(j) = self.jobs.current() {
                    self.jobs.renaming = Some(j.header.clone());
                }
            }
            KeyCode::Char('x') | KeyCode::Delete => {
                if let Some(j) = self.jobs.current() {
                    if j.job.status == JobStatus::Running {
                        self.status =
                            Some("that job is still running; delete it when it ends".into());
                    } else {
                        self.jobs.deleting = Some(j.job.id);
                    }
                }
            }
            KeyCode::Esc if !self.jobs.filter.is_empty() => {
                self.jobs.filter.clear();
                self.jobs.clamp();
            }
            KeyCode::Enter | KeyCode::Char('v') | KeyCode::Char('w') | KeyCode::Char('i') => {
                let Some(j) = self.jobs.current() else {
                    return Action::None;
                };
                let full = j.job.id.to_string();
                // The short id: `proteus` takes any unique prefix, and it reads at a glance.
                let id = short_id(&full).to_string();
                let name = display_name(&j.header);
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
                    CommandSpec::one(&["view", &id, "--web"], RunMode::Quiet).says(
                        format!("opening {name} in your browser…"),
                        format!("opened {name} in your browser"),
                    )
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
                if self.files.flat {
                    let dir = self.files.dir.clone();
                    self.files.enter(dir);
                } else if let Some(parent) = self.files.dir.parent().map(Path::to_path_buf) {
                    self.files.enter(parent);
                }
            }
            KeyCode::Char('f') => {
                self.files.flat = !self.files.flat;
                self.files.selected = 0;
                self.files.rescan();
            }
            KeyCode::Char('~') => {
                if let Some(home) = std::env::var_os("HOME") {
                    self.files.enter(PathBuf::from(home));
                    self.files.selected = 0;
                }
            }
            KeyCode::Char('.') => {
                let start = self.files.start.clone();
                self.files.enter(start);
                self.files.selected = 0;
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
                // Relative to the folder proteus started in (the children's working directory),
                // so the command reads short.
                let shown = std::env::current_dir()
                    .ok()
                    .and_then(|cwd| e.path.strip_prefix(cwd).ok().map(Path::to_path_buf))
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| e.path.clone());
                let path = shown.to_string_lossy().into_owned();
                let name = e.name.clone();
                return Action::Run(match key.code {
                    KeyCode::Char('w') => {
                        CommandSpec::one(&["view", &path, "--web"], RunMode::Quiet).says(
                            format!("opening {name} in your browser…"),
                            format!("opened {name} in your browser"),
                        )
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

/// A file's modification time and size, to tell whether it changed since it was measured.
pub fn file_stamp(path: &Path) -> Option<(std::time::SystemTime, u64)> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
}

/// A job's name as the lists show it: the header, or "untitled" when there is none.
pub fn display_name(header: &str) -> String {
    let h = header.trim();
    if h.is_empty() {
        "untitled".into()
    } else {
        h.to_string()
    }
}

pub fn short_id(id: &str) -> &str {
    &id[..id.len().min(8)]
}
