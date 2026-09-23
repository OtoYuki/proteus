//! Tests of the home screen: key handling, the command lines it builds, and rendering.

use super::app::*;
use super::ui;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use proteus_core::models::{JobStatus, PipelineJob, PipelineTier};
use proteus_storage::repository::JobSummary;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use std::path::Path;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn press(app: &mut App, keys: &str) -> Action {
    let mut last = Action::None;
    for c in keys.chars() {
        last = app.handle_key(key(KeyCode::Char(c)));
    }
    last
}

fn job(header: &str, status: JobStatus, with_structure: bool) -> JobSummary {
    JobSummary {
        job: PipelineJob {
            id: uuid::Uuid::new_v4(),
            sequence_id: uuid::Uuid::new_v4(),
            tier: PipelineTier::FastScreening,
            status,
            priority: 1,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error_log: None,
        },
        header: header.into(),
        length: 20,
        pdb_path: with_structure.then(|| "/data/x.pdb".into()),
        plddt: with_structure.then_some(84.2),
        metadata: with_structure.then(|| serde_json::json!({ "engine": "esmfold-api" })),
    }
}

fn app_in(dir: &Path) -> App {
    App::new(dir, Path::new("/nonexistent-data"))
}

fn rendered(app: &App, w: u16, h: u16) -> String {
    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
    t.draw(|f| ui::draw(f, app)).unwrap();
    let buf = t.backend().buffer();
    let mut out = String::new();
    for y in 0..h {
        for x in 0..w {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// ---- command lines ------------------------------------------------------------------------

#[test]
fn quoting_is_bare_single_or_dollar_quoted() {
    assert_eq!(shell_quote("view"), "view");
    assert_eq!(shell_quote("/data/a b.pdb"), "'/data/a b.pdb'");
    assert_eq!(shell_quote("it's"), r"'it'\''s'");
    assert_eq!(shell_quote(""), "''");
    assert_eq!(shell_quote(">q\nMKT"), r"$'>q\nMKT'");
    assert_eq!(shell_quote("a'b\\c\nd"), r"$'a\'b\\c\nd'");
    // An escape sequence in a file name is shown, never sent to the terminal.
    let q = shell_quote("a\x1b[41mRED.pdb");
    assert_eq!(q, r"$'a\x1b[41mRED.pdb'");
    assert!(!q.contains('\x1b'));
}

/// What `display` prints must be what the home screen runs: parse it back with the shell.
#[test]
fn displayed_commands_parse_back_to_the_same_arguments() {
    if std::process::Command::new("bash")
        .arg("-c")
        .arg("true")
        .status()
        .is_err()
    {
        eprintln!("skipped: no bash");
        return;
    }
    for arg in [
        "plain",
        "with space",
        "it's",
        ">q\nMKT",
        "a'b\\c\nd",
        "$HOME `x` \"y\"",
        "a\x1b]0;TITLE\x07\x1b[41mRED.pdb",
    ] {
        let line = format!("printf '%s\\0' {}", shell_quote(arg));
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(&line)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8(out.stdout).unwrap(),
            format!("{arg}\0"),
            "{line}"
        );
    }
}

#[test]
fn the_fold_form_builds_a_submit_command() {
    let mut app = app_in(Path::new("/"));
    app.tab = Tab::Run;
    // Typing on the focused Sequence field is text straight away, digits and q included.
    press(&mut app, "mktayiakq123");
    assert_eq!(app.tab, Tab::Run);
    assert_eq!(app.run.fold[0].value(), "mktayiakq123");
    for _ in 0..3 {
        app.handle_key(key(KeyCode::Backspace));
    }
    app.handle_key(key(KeyCode::Enter)); // done editing, focus moves on
    app.handle_key(key(KeyCode::Right)); // tier fast → sota
    app.handle_key(key(KeyCode::Down));
    app.handle_key(key(KeyCode::Right)); // runner auto → esm-api
    app.handle_key(key(KeyCode::Down));
    let Action::Run(spec) = app.handle_key(key(KeyCode::Enter)) else {
        panic!("Run did not run: {:?}", app.status);
    };
    assert_eq!(
        spec.stages,
        vec![args(&[
            "submit",
            "--fasta",
            ">query\nMKTAYIAKQ",
            "--tier",
            "sota",
            "--runner",
            "esm-api"
        ])]
    );
    assert_eq!(spec.mode, RunMode::Pause);
    assert_eq!(
        spec.display(),
        r"proteus submit --fasta $'>query\nMKTAYIAKQ' --tier sota --runner esm-api"
    );
}

#[test]
fn a_path_to_an_existing_file_is_passed_as_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("wt.fasta");
    std::fs::write(&fasta, ">wt\nMKT\n").unwrap();
    let mut run = RunView::default();
    run.fold[0].kind = FieldKind::Text(fasta.to_string_lossy().into_owned());
    assert_eq!(
        run.command().unwrap().stages[0][1..3],
        args(&["--file", &fasta.to_string_lossy()])
    );

    run.form = FormKind::Scan;
    run.scan[0].kind = FieldKind::Text(fasta.to_string_lossy().into_owned());
    let spec = run.command().unwrap();
    assert_eq!(spec.stdin, None);
    assert_eq!(spec.stages[0][1], fasta.to_string_lossy());
}

#[test]
fn the_scan_form_pipes_mutate_into_screen() {
    let mut run = RunView {
        form: FormKind::Scan,
        ..RunView::default()
    };
    run.scan[0].kind = FieldKind::Text("MKTAYIAKQR".into());
    run.scan[2].kind = FieldKind::Text("3".into());
    run.scan[3].kind = FieldKind::Text("7".into());
    run.scan[6].kind = FieldKind::Text("out dir/scan.parquet".into());
    let spec = run.command().unwrap();
    assert_eq!(
        spec.stages,
        vec![
            args(&["mutate", "-", "--mode", "alanine", "--start", "3", "--end", "7"]),
            args(&[
                "screen",
                "-",
                "--runner",
                "auto",
                "--scorer",
                "structure",
                "--export",
                "out dir/scan.parquet"
            ]),
        ]
    );
    assert_eq!(spec.stdin, Some(args(&[">query", "MKTAYIAKQR"])));
    assert_eq!(
        spec.display(),
        "printf '%s\\n' '>query' MKTAYIAKQR | proteus mutate - --mode alanine --start 3 --end 7 \
         | proteus screen - --runner auto --scorer structure --export 'out dir/scan.parquet'"
    );

    run.scan[2].kind = FieldKind::Text("0".into());
    assert!(run.command().unwrap_err().contains("From residue"));
    run.scan[2].kind = FieldKind::Text("x".into());
    assert!(run.command().is_err());
}

#[test]
fn sequence_input_forms() {
    let mut run = RunView::default();
    let fasta_of = |run: &mut RunView, text: &str| {
        run.fold[0].kind = FieldKind::Text(text.into());
        run.command().map(|s| s.stages[0][2].clone())
    };
    // A pasted record keeps its header line.
    assert_eq!(
        fasta_of(
            &mut run,
            ">sp|P69905 Hemoglobin alpha\nMVLSPADKTN\nVKAAWGKVGA\n"
        )
        .unwrap(),
        ">sp|P69905 Hemoglobin alpha\nMVLSPADKTNVKAAWGKVGA"
    );
    // Typed on one line: the last word is the sequence, in any case.
    assert_eq!(
        fasta_of(&mut run, ">GFP EGFP MVSKGEELFT").unwrap(),
        ">GFP EGFP\nMVSKGEELFT"
    );
    assert_eq!(fasta_of(&mut run, ">q mktayiak").unwrap(), ">q\nMKTAYIAK");
    assert_eq!(
        fasta_of(&mut run, ">wt\nmktayiak\nqr\n").unwrap(),
        ">wt\nMKTAYIAKQR"
    );
    assert!(fasta_of(&mut run, ">only-a-header").is_err());
    assert!(fasta_of(&mut run, ">a\nMKT\n>b\nMKV").is_err());
    assert!(fasta_of(&mut run, "").is_err());
    assert!(fasta_of(&mut run, "MKT 42").is_err());
    assert!(fasta_of(&mut run, "no/such/file.fasta").is_err());
}

// ---- keys ---------------------------------------------------------------------------------

#[test]
fn job_keys_move_filter_and_open() {
    let mut app = app_in(Path::new("/"));
    app.jobs.replace(vec![
        job("lysozyme", JobStatus::Completed, true),
        job("insulin", JobStatus::Failed, false),
        job("ubiquitin", JobStatus::Completed, true),
    ]);
    press(&mut app, "jjjjj");
    assert_eq!(app.jobs.selected, 2, "clamped at the last row");
    press(&mut app, "g");
    assert_eq!(app.jobs.selected, 0);

    let Action::Run(spec) = app.handle_key(key(KeyCode::Enter)) else {
        panic!()
    };
    let id = app.jobs.all[0].job.id.to_string();
    assert_eq!(spec.stages, vec![args(&["view", &id, "--interactive"])]);
    assert_eq!(spec.mode, RunMode::Interactive);
    let Action::Run(spec) = press(&mut app, "w") else {
        panic!()
    };
    assert_eq!(spec.stages, vec![args(&["view", &id, "--web"])]);
    assert_eq!(spec.mode, RunMode::Quiet);

    // A failed job has nothing to view, and says so instead of running a failing command.
    press(&mut app, "j");
    assert_eq!(app.handle_key(key(KeyCode::Enter)), Action::None);
    assert!(app.status.as_deref().unwrap().contains("no structure"));
    let Action::Run(spec) = press(&mut app, "i") else {
        panic!()
    };
    assert_eq!(spec.stages[0][0], "inspect");

    // `/` filters; while filtering, letters are text, not commands (q does not quit).
    press(&mut app, "/");
    assert!(app.typing());
    assert_eq!(press(&mut app, "ubq"), Action::None);
    app.handle_key(key(KeyCode::Backspace));
    assert_eq!(app.jobs.visible().len(), 1);
    assert_eq!(app.jobs.current().unwrap().header, "ubiquitin");
    app.handle_key(key(KeyCode::Esc));
    assert!(!app.typing());
    assert_eq!(app.jobs.visible().len(), 3);
    assert_eq!(press(&mut app, "q"), Action::Quit);
}

#[test]
fn a_refresh_keeps_the_selected_job() {
    let mut app = app_in(Path::new("/"));
    let jobs = vec![
        job("a", JobStatus::Completed, true),
        job("b", JobStatus::Running, false),
    ];
    app.jobs.replace(jobs.clone());
    press(&mut app, "j");
    let mut newer = vec![job("new", JobStatus::Queued, false)];
    newer.extend(jobs);
    app.jobs.replace(newer);
    assert_eq!(app.jobs.current().unwrap().header, "b");
}

#[test]
fn pasted_text_is_never_read_as_keys() {
    let mut app = app_in(Path::new("/"));
    app.handle_paste("q");
    assert_eq!(app.tab, Tab::Jobs);
    app.tab = Tab::Run;
    app.handle_paste(">wt\r\nMKTQ\r\n");
    assert_eq!(app.run.fold[0].value(), ">wt\nMKTQ");
    assert!(app.run.editing);
    assert!(app.run.command().is_ok());
}

#[test]
fn tabs_help_and_ctrl_c() {
    let mut app = app_in(Path::new("/"));
    press(&mut app, "2");
    assert_eq!(app.tab, Tab::Structures);
    app.handle_key(key(KeyCode::Tab));
    assert_eq!(app.tab, Tab::Run);
    app.handle_key(key(KeyCode::Tab));
    assert_eq!(app.tab, Tab::Jobs);
    app.handle_key(key(KeyCode::BackTab));
    assert_eq!(app.tab, Tab::Run);
    // On the Run form's text field ? is text; F1 opens the help anywhere.
    press(&mut app, "?");
    assert!(!app.help);
    assert_eq!(app.run.fold[0].value(), "?");
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Esc));
    app.handle_key(key(KeyCode::F(1)));
    assert!(app.help);
    app.help = false;
    app.handle_key(key(KeyCode::BackTab));
    assert_eq!(app.tab, Tab::Structures);
    press(&mut app, "?");
    assert!(app.help);
    assert_eq!(
        press(&mut app, "q"),
        Action::None,
        "a key closes the help first"
    );
    assert!(!app.help);
    // Ctrl-C quits even while typing.
    press(&mut app, "3");
    press(&mut app, "x");
    assert!(app.typing());
    assert_eq!(
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Action::Quit
    );
}

#[test]
fn the_file_browser_lists_folders_then_structures() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("models")).unwrap();
    std::fs::create_dir(dir.path().join(".cache")).unwrap();
    for f in ["b.cif", "a.pdb", "notes.txt", ".hidden.pdb", "c.pdb.gz"] {
        std::fs::write(dir.path().join(f), "x").unwrap();
    }
    let mut app = app_in(dir.path());
    let names: Vec<_> = app.files.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["..", "models", "a.pdb", "b.cif", "c.pdb.gz"]);

    assert_eq!(
        app.wanted_analysis(),
        None,
        "only measured on the Structures tab"
    );
    press(&mut app, "2jj");
    assert_eq!(app.wanted_analysis(), Some(dir.path().join("a.pdb")));
    let Action::Run(spec) = press(&mut app, "a") else {
        panic!()
    };
    assert_eq!(
        spec.stages,
        vec![args(&[
            "analyze",
            &dir.path().join("a.pdb").to_string_lossy()
        ])]
    );

    // Into a folder and back out lands on that folder again.
    press(&mut app, "k");
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(app.files.dir, dir.path().join("models"));
    app.handle_key(key(KeyCode::Backspace));
    assert_eq!(app.files.dir, dir.path());
    assert_eq!(app.files.current().unwrap().name, "models");
}

// ---- rendering ----------------------------------------------------------------------------

#[test]
fn an_empty_database_says_how_to_start() {
    let mut app = app_in(Path::new("/"));
    app.jobs.loaded = true;
    let screen = rendered(&app, 100, 24);
    assert!(screen.contains("No jobs yet."), "{screen}");
    assert!(screen.contains("proteus submit"), "{screen}");
    assert!(screen.contains("(jobs)"));
    assert!(screen.contains("q quit"));
}

#[test]
fn every_tab_renders_at_every_size_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.pdb"), "x").unwrap();
    let mut app = app_in(dir.path());
    app.jobs.replace(vec![
        job("lysozyme", JobStatus::Completed, true),
        job(
            "a-very-long-sequence-name-that-will-not-fit-anywhere",
            JobStatus::Failed,
            false,
        ),
    ]);
    app.jobs.all[1].job.error_log = Some("ESMFold API: HTTP 503".into());
    app.status = Some("finished: proteus submit …".into());
    for tab in Tab::ALL {
        app.tab = tab;
        for help in [false, true] {
            app.help = help;
            for (w, h) in [(120, 40), (80, 24), (40, 12), (12, 4), (1, 1), (0, 0)] {
                rendered(&app, w, h);
            }
        }
    }
    app.help = false;
    app.tab = Tab::Jobs;
    // A job's name is data: never re-cased (I6A is a mutation, i6a is not).
    app.jobs.all[0].header = "query_I6A [mutation=I6A]".into();
    let screen = rendered(&app, 120, 30);
    assert!(!screen.contains("i6a"), "{screen}");
    assert!(
        screen.matches("query_I6A [mutation=I6A]").count() >= 2,
        "{screen}"
    );
    assert!(screen.contains("esmfold-api"), "{screen}");
    assert!(screen.contains("84.2"), "{screen}");
}

#[test]
fn the_structures_tab_shows_the_measurements() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.pdb"), "x").unwrap();
    let mut app = app_in(dir.path());
    press(&mut app, "2j");
    let path = dir.path().join("a.pdb");
    app.analyses.insert(path.clone(), Analysis::Pending);
    assert!(rendered(&app, 100, 24).contains("measuring"));
    app.analyses.insert(
        path,
        Analysis::Done {
            residues: 46,
            predicted: false,
            rows: vec![["Radius of gyration".into(), "9.6 Å".into()]],
        },
    );
    let screen = rendered(&app, 100, 24);
    assert!(screen.contains("46 residues"), "{screen}");
    assert!(screen.contains("experimental"), "{screen}");
    assert!(screen.contains("9.6 Å"), "{screen}");
}

#[test]
fn the_run_tab_shows_the_command_it_will_run() {
    let mut app = app_in(Path::new("/"));
    app.tab = Tab::Run;
    let screen = rendered(&app, 100, 24);
    assert!(screen.contains("enter a sequence"), "{screen}");
    assert!(
        screen.contains("[ run ▸ ]"),
        "the Run button fits: {screen}"
    );
    app.handle_key(key(KeyCode::Enter));
    press(&mut app, "MKT");
    let screen = rendered(&app, 100, 24);
    assert!(screen.contains("proteus submit --fasta"), "{screen}");
}

/// The real structure analysis the tab runs, on a real file.
#[test]
fn analysing_crambin_gives_its_residue_count_and_rows() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validate/corpus/1crn.pdb");
    if !path.exists() {
        eprintln!("skipped: {} not present", path.display());
        return;
    }
    match super::analyse(&path) {
        Analysis::Done {
            residues,
            predicted,
            rows,
        } => {
            assert_eq!(residues, 46);
            assert!(!predicted);
            assert!(rows.iter().any(|[k, _]| k == "Radius of gyration"));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        super::analyse(Path::new("/definitely/not/here.pdb")),
        Analysis::Failed(_)
    ));
}

#[test]
fn esc_quits_when_nothing_is_open_and_the_filter_knows_done() {
    let mut app = app_in(Path::new("/"));
    app.jobs.replace(vec![
        job("a", JobStatus::Completed, true),
        job("b", JobStatus::Failed, false),
    ]);
    press(&mut app, "/");
    press(&mut app, "done");
    assert_eq!(
        app.jobs.visible().len(),
        1,
        "the list says done, so the filter matches it"
    );
    // Esc clears the filter first, then quits.
    assert_eq!(app.handle_key(key(KeyCode::Esc)), Action::None);
    assert_eq!(app.jobs.visible().len(), 2);
    assert_eq!(app.handle_key(key(KeyCode::Esc)), Action::Quit);
}

#[test]
fn only_regular_files_are_listed_and_a_changed_file_is_measured_again() {
    let dir = tempfile::tempdir().unwrap();
    let pdb = dir.path().join("a.pdb");
    std::fs::write(&pdb, "x").unwrap();
    #[cfg(unix)]
    assert!(std::process::Command::new("mkfifo")
        .arg(dir.path().join("pipe.pdb"))
        .status()
        .unwrap()
        .success());
    let mut app = app_in(dir.path());
    let names: Vec<_> = app.files.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["..", "a.pdb"], "a FIFO would hang its analysis");

    press(&mut app, "2j");
    assert_eq!(app.wanted_analysis(), Some(pdb.clone()));
    app.analyses.insert(pdb.clone(), Analysis::Pending);
    app.stamps.insert(pdb.clone(), file_stamp(&pdb));
    assert_eq!(app.wanted_analysis(), None, "measured and unchanged");
    std::fs::write(&pdb, "a longer file").unwrap();
    assert_eq!(
        app.wanted_analysis(),
        Some(pdb),
        "the file changed: measure it again"
    );
}
