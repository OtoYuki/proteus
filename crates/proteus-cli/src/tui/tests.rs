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
        chain_names: Vec::new(),
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
            ">MKTAYIAK\nMKTAYIAKQ",
            "--tier",
            "sota",
            "--runner",
            "esm-api"
        ])]
    );
    assert_eq!(spec.mode, RunMode::Pause);
    assert_eq!(
        spec.display(),
        r"proteus submit --fasta $'>MKTAYIAK\nMKTAYIAKQ' --tier sota --runner esm-api"
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
    assert_eq!(spec.stdin, Some(args(&[">MKTAYIAK", "MKTAYIAKQR"])));
    assert_eq!(
        spec.display(),
        "printf '%s\\n' '>MKTAYIAK' MKTAYIAKQR | proteus mutate - --mode alanine --start 3 --end 7 \
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
    // The short id: it reads at a glance, and proteus takes any unique prefix.
    let id = app.jobs.all[0].job.id.to_string()[..8].to_string();
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
    assert!(screen.contains("esmfold"), "{screen}");
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
            interface: vec![],
            verdict: None,
            facts: None,
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
    assert!(screen.contains("run ▸"), "the Run button fits: {screen}");
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
    let (analysis, scene) = super::analyse(&path);
    match analysis {
        Analysis::Done {
            residues,
            predicted,
            rows,
            facts,
            ..
        } => {
            assert_eq!(residues, 46);
            assert!(!predicted);
            assert!(rows.iter().any(|[k, _]| k == "Radius of gyration"));
            let f = facts.expect("facts");
            assert_eq!(f.chains, 1);
            assert!(f.rama_favored > 90.0 && f.helix > 0.0);
        }
        other => panic!("{other:?}"),
    }
    // The parsed scene comes back for previews, and renders at any pane size.
    let scene = scene.expect("scene");
    let key = super::app::PreviewKey {
        path: path.clone(),
        cols: 40,
        rows: 20,
        pixels: false,
    };
    let p = super::render_preview(&scene, &key, None);
    assert_eq!((p.width, p.height), (40, 40));
    assert!(p.pixels.iter().any(|x| x.is_some()));
    assert!(matches!(
        super::analyse(Path::new("/definitely/not/here.pdb")).0,
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

/// The jobs tab shows the selected job's findings: a complex's interface verdict and its
/// measurements, beside the list on a wide terminal.
#[test]
fn the_jobs_tab_shows_the_selected_jobs_findings() {
    use proteus_core::models::{JobStatus, PipelineJob, PipelineTier};
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_in(dir.path());
    let model = dir.path().join("model_0.pdb");
    let job = PipelineJob {
        id: uuid::Uuid::new_v4(),
        sequence_id: uuid::Uuid::new_v4(),
        tier: PipelineTier::HighFidelity,
        status: JobStatus::Completed,
        priority: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error_log: None,
    };
    app.jobs
        .replace(vec![proteus_storage::repository::JobSummary {
            job,
            header: "binder_7".into(),
            length: 150,
            pdb_path: Some(model.to_string_lossy().into_owned()),
            plddt: Some(91.0),
            metadata: None,
            chain_names: Vec::new(),
        }]);
    assert_eq!(app.wanted_analysis(), Some(model.clone()));
    app.analyses.insert(
        model,
        Analysis::Done {
            residues: 150,
            predicted: true,
            rows: vec![["Radius of gyration".into(), "14.2 Å".into()]],
            interface: vec![["Interface".into(), "A → B".into()]],
            verdict: Some((
                true,
                "● confident interface · ipSAE_min 0.750 > 0.61".into(),
            )),
            facts: None,
        },
    );
    let screen = rendered(&app, 160, 40);
    assert!(screen.contains("● confident interface"), "{screen}");
    assert!(screen.contains("A → B"), "{screen}");
    assert!(screen.contains("14.2 Å"), "{screen}");
}

/// The redesigned home: the tab control, the inspector's gauges, and the run tab's guidance.
#[test]
fn the_home_screen_shows_tabs_gauges_and_guidance() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.pdb"), "x").unwrap();
    let mut app = app_in(dir.path());
    press(&mut app, "2j");
    let path = dir.path().join("a.pdb");
    app.analyses.insert(
        path,
        Analysis::Done {
            residues: 76,
            predicted: true,
            rows: vec![
                ["pLDDT".into(), "mean 91.0".into()],
                ["Radius of gyration".into(), "11.8 Å".into()],
            ],
            interface: vec![],
            verdict: None,
            facts: Some(Box::new(super::app::Facts {
                chains: 1,
                plddt: Some(91.0),
                rama_favored: 98.0,
                helix: 30.0,
                strand: 40.0,
                coil: 30.0,
                rg_ratio: 1.02,
                triage: 90.0,
                ..Default::default()
            })),
        },
    );
    let screen = rendered(&app, 160, 40);
    assert!(screen.contains("⡮⠕ structures 2"), "{screen}");
    assert!(screen.contains("98 % favoured"), "{screen}");
    assert!(screen.contains("α30 β40 coil 30 %"), "{screen}");
    assert!(screen.contains("11.8 Å"), "{screen}");
    // The card shows pLDDT; the details do not repeat it.
    assert!(!screen.contains("mean 91.0"), "{screen}");
    press(&mut app, "3");
    let screen = rendered(&app, 160, 40);
    assert!(screen.contains("ESMFold: one chain"), "{screen}");
    assert!(screen.contains("what happens"), "{screen}");
    assert!(screen.contains("human ubiquitin"), "{screen}");
}

// ---- tabs from the Run form, sorting, renaming, deleting ----------------------------------

#[test]
fn digits_switch_tabs_from_the_run_form_except_on_a_number_field() {
    let mut app = app_in(Path::new("/"));
    app.tab = Tab::Run;
    // The sequence field is focused: letters are text, a digit is a tab key.
    press(&mut app, "2");
    assert_eq!(
        app.tab,
        Tab::Structures,
        "a digit on a text field switched tabs"
    );
    press(&mut app, "3");
    assert_eq!(app.tab, Tab::Run);
    press(&mut app, "MKT");
    assert!(app.run.editing);
    assert_eq!(app.run.fold[0].value(), "MKT");
    // Esc stops typing; the digit then switches tabs again.
    app.handle_key(key(KeyCode::Esc));
    assert!(!app.run.editing);
    press(&mut app, "1");
    assert_eq!(app.tab, Tab::Jobs);

    // On a number field, digits are the number.
    press(&mut app, "3");
    app.run.form = FormKind::Scan;
    app.run.focus = 2;
    press(&mut app, "12");
    assert_eq!(app.tab, Tab::Run);
    assert_eq!(app.run.scan[2].value(), "12");

    // Alt+digit switches from anywhere, even mid-edit, and keeps the edit.
    let alt1 = KeyEvent::new(KeyCode::Char('1'), KeyModifiers::ALT);
    app.handle_key(alt1);
    assert_eq!(app.tab, Tab::Jobs);
    assert!(!app.typing());
    assert_eq!(app.run.scan[2].value(), "12");
}

#[test]
fn s_cycles_the_order_of_the_jobs() {
    let mut app = app_in(Path::new("/"));
    let mut low = job("zeta", JobStatus::Completed, true);
    low.plddt = Some(40.0);
    let mut high = job("alpha", JobStatus::Completed, true);
    high.plddt = Some(90.0);
    app.jobs
        .replace(vec![low, job("mid", JobStatus::Running, false), high]);
    let names = |app: &App| -> Vec<String> {
        app.jobs
            .visible()
            .iter()
            .map(|j| j.header.clone())
            .collect()
    };
    assert_eq!(names(&app), ["zeta", "mid", "alpha"]);
    press(&mut app, "s");
    assert_eq!(app.jobs.sort, JobSort::Name);
    assert_eq!(names(&app), ["alpha", "mid", "zeta"]);
    press(&mut app, "s");
    assert_eq!(names(&app), ["mid", "zeta", "alpha"], "running first");
    press(&mut app, "s");
    assert_eq!(names(&app), ["alpha", "zeta", "mid"], "no pLDDT last");
    assert!(rendered(&app, 120, 30).contains("most confident first"));
    press(&mut app, "s");
    assert_eq!(app.jobs.sort, JobSort::Newest);
}

#[test]
fn n_renames_the_selected_job() {
    let mut app = app_in(Path::new("/"));
    app.jobs
        .replace(vec![job("A +1 chain(s)", JobStatus::Completed, true)]);
    let id = app.jobs.all[0].job.id.to_string()[..8].to_string();
    press(&mut app, "n");
    assert!(app.typing());
    assert_eq!(app.jobs.renaming.as_deref(), Some("A +1 chain(s)"));
    // Keys are text now: q does not quit, digits do not switch tabs.
    for _ in 0..13 {
        app.handle_key(key(KeyCode::Backspace));
    }
    assert_eq!(press(&mut app, "PD-L1 q2"), Action::None);
    assert_eq!(app.tab, Tab::Jobs);
    assert!(rendered(&app, 120, 30).contains(&format!("rename {id}  PD-L1 q2")));
    let Action::Run(spec) = app.handle_key(key(KeyCode::Enter)) else {
        panic!("Enter did not rename")
    };
    assert_eq!(spec.stages, vec![args(&["rename", &id, "PD-L1 q2"])]);
    assert_eq!(spec.mode, RunMode::Quiet);
    assert_eq!(spec.done.as_deref(), Some("renamed to PD-L1 q2"));
    assert!(!app.typing());

    // Esc cancels; an unchanged name runs nothing.
    press(&mut app, "n");
    app.handle_key(key(KeyCode::Esc));
    assert!(app.jobs.renaming.is_none());
    press(&mut app, "n");
    assert_eq!(app.handle_key(key(KeyCode::Enter)), Action::None);
}

#[test]
fn x_deletes_only_after_y() {
    let mut app = app_in(Path::new("/"));
    app.jobs.replace(vec![
        job("ubq", JobStatus::Completed, true),
        job("busy", JobStatus::Running, false),
    ]);
    let id = app.jobs.all[0].job.id.to_string()[..8].to_string();
    press(&mut app, "x");
    let screen = rendered(&app, 120, 30);
    assert!(
        screen.contains(&format!("delete ubq ({id}) and its files?")),
        "{screen}"
    );
    // Any other key keeps it, and is not acted on.
    assert_eq!(press(&mut app, "q"), Action::None);
    assert!(app.jobs.deleting.is_none());
    assert!(app
        .status
        .as_deref()
        .unwrap()
        .contains("nothing was deleted"));

    press(&mut app, "x");
    let Action::Run(spec) = press(&mut app, "y") else {
        panic!("y did not delete")
    };
    assert_eq!(spec.stages, vec![args(&["delete", &id])]);
    assert_eq!(spec.done.as_deref(), Some("deleted ubq"));

    // A running job is not deleted from under its runner.
    press(&mut app, "j");
    press(&mut app, "x");
    assert!(app.jobs.deleting.is_none());
}

#[test]
fn the_status_line_says_what_happened_not_the_raw_output() {
    let mut app = app_in(Path::new("/"));
    app.last_command = Some("proteus view 73d5e504 --web".into());
    app.status = Some("✓ opened ubq in your browser".into());
    let screen = rendered(&app, 120, 30);
    let last = screen.lines().last().unwrap();
    assert!(last.contains("✓ opened ubq in your browser"), "{last}");
    assert!(last.contains("$ proteus view 73d5e504 --web"), "{last}");
    // Narrow: the message wins and the command gives way.
    let screen = rendered(&app, 40, 30);
    let last = screen.lines().last().unwrap();
    assert!(last.contains("✓ opened ubq"), "{last}");
    assert!(!last.contains("$ proteus"), "{last}");
    // A long message is cut with an ellipsis, never wrapped or run off the edge.
    app.status = Some(format!("✗ {}", "x".repeat(200)));
    let screen = rendered(&app, 60, 30);
    assert!(screen.lines().last().unwrap().trim_end().ends_with('…'));
}

#[test]
fn the_help_takes_down_the_preview_picture() {
    // A kitty picture is drawn above the text, so it would sit on top of the help box.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.pdb"), "x").unwrap();
    let mut app = app_in(dir.path());
    app.tab = Tab::Structures;
    app.cell_pixels = Some((10.0, 20.0));
    app.help = true;
    *app.preview_want.borrow_mut() = Some((
        PreviewKey {
            path: dir.path().join("a.pdb"),
            cols: 10,
            rows: 5,
            pixels: true,
        },
        ratatui::layout::Rect::new(0, 0, 10, 5),
    ));
    let screen = rendered(&app, 120, 40);
    assert!(screen.contains("(keys)"));
    assert!(
        screen.contains("the same line works in a script."),
        "{screen}"
    );
    assert!(app.preview_want.borrow().is_none());
}

#[test]
fn the_list_names_the_model_and_whether_it_had_an_alignment() {
    let mut j = job("barnase", JobStatus::Completed, true);
    assert_eq!(model(&j), "esmfold");
    j.metadata = Some(serde_json::json!({
        "engine": "oci", "image": "ghcr.io/jwohlwend/boltz:latest", "msa": "none"
    }));
    assert_eq!(model(&j), "boltz");
    j.metadata = Some(serde_json::json!({
        "engine": "oci", "image": "ghcr.io/jwohlwend/boltz:latest", "msa": "server"
    }));
    assert_eq!(model(&j), "boltz+msa");
    j.metadata = Some(serde_json::json!({ "engine": "oci" }));
    assert_eq!(model(&j), "oci");
    j.metadata = Some(serde_json::json!({ "engine": "simulated" }));
    assert_eq!(model(&j), "simulated");
}

#[test]
fn the_card_fits_its_lines_instead_of_wrapping_them() {
    let fx = super::app::Facts {
        chains: 2,
        plddt: Some(94.1),
        rama_favored: 97.0,
        rama_outliers: 1,
        rg_ratio: 1.08,
        triage: 94.0,
        ..Default::default()
    };
    let look = &app_in(Path::new("/")).look;
    for width in [31u16, 34, 40, 48, 60, 90] {
        let lines = super::ui::headline_card_for_test(look, 233, true, Some(&fx), width);
        for l in &lines {
            assert!(l.width() <= width as usize, "{width}: {l:?}");
        }
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("1 outlier") && !text.contains("1 outliers"),
            "{text}"
        );
    }
}

#[test]
fn the_interface_names_its_chains() {
    let names = vec![
        ("A".to_string(), "PD-L1".to_string()),
        ("B".to_string(), "PD-1".to_string()),
    ];
    assert_eq!(ui::name_chains("A → B", &names), "PD-L1 (A) → PD-1 (B)");
    assert_eq!(
        ui::name_chains("A,C → B", &names),
        "PD-L1 (A), C → PD-1 (B)"
    );
    assert_eq!(ui::name_chains("A → B", &[]), "A → B");
}
