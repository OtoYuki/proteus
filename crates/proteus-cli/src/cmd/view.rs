//! `proteus view` — Render a structure: terminal back-ends, or a self-contained browser page.

use super::prelude::*;

/// Arguments of `proteus view`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Target PDB file path, or a job UUID or unique prefix of one
    target: String,

    /// Reference structure to superpose onto, with C-alpha RMSD. Residues are paired by
    /// chain ID, residue number and insertion code; the two are drawn in fixed colours, so
    /// --color and --dashboard do not apply
    #[arg(long, conflicts_with_all = ["color", "dashboard"])]
    compare: Option<PathBuf>,

    /// Run the interactive TUI viewer with orbit camera controls
    #[arg(short, long)]
    interactive: bool,

    /// Enable side-by-side live biophysical telemetry dashboard (Ramachandran, pLDDT, SASA)
    #[arg(long)]
    dashboard: bool,

    /// Terminal rendering backend
    #[arg(short, long, value_enum, default_value_t = CliBackend::HalfBlock)]
    backend: CliBackend,

    /// Color scheme. Default: pLDDT for predicted models, secondary structure for
    /// experimental ones (their B-factors are not confidences).
    #[arg(short, long, value_enum)]
    color: Option<CliColorScheme>,

    /// Terminal viewport width in cells, 1–4096 (defaults to terminal width or 80)
    #[arg(long, value_parser = viewport_cells())]
    width: Option<usize>,

    /// Terminal viewport height in cells, 1–4096 (defaults to terminal height or 30)
    #[arg(long, value_parser = viewport_cells())]
    height: Option<usize>,

    /// Open the structure in a browser: a self-contained 3Dmol.js page, no network needed
    #[arg(long)]
    web: bool,

    /// Write that self-contained 3Dmol.js page (with our DSSP assignment) to a file
    #[arg(long)]
    html: Option<PathBuf>,
}

/// `--width`/`--height`: a cell count the renderer can allocate a framebuffer for.
fn viewport_cells() -> clap::builder::RangedU64ValueParser<usize> {
    clap::builder::RangedU64ValueParser::<usize>::new()
        .range(1..=proteus_render::MAX_VIEWPORT_CELLS as u64)
}

pub async fn run(args: Args, db_path: &std::path::Path) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let Args {
        target,
        compare,
        interactive,
        dashboard,
        backend,
        color,
        width,
        height,
        web,
        html,
    } = args;
    let target_path = PathBuf::from(&target);
    let (pdb_content, title) = if target_path.exists() {
        let content = proteus_core::io::read_structure_text(&target_path)
            .with_context(|| format!("Failed to read structure file at {:?}", target_path))?;
        let name = target_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("PDB Structure")
            .to_string();
        (content, name)
    } else {
        // A path that exists wins; only then is the argument treated as a job reference, so a
        // file literally named like a UUID is still openable. Opening the job database creates
        // it (with its -wal and -shm files), which a mistyped file name must not do: look it
        // up only for something shaped like a job ID, and only in a database that exists.
        if !looks_like_job_ref(&target) {
            bail!("No such file: '{target}'");
        }
        if !db_path.exists() {
            bail!(
                "'{target}' is neither an existing file path nor a known job (there is no job \
                 database at {})",
                db_path.display()
            );
        }
        let pool = create_sqlite_pool(&db_path).await?;
        let repo = ProteusRepository::new(pool);
        let job_id = job_ref::resolve(&repo, &target).await.with_context(|| {
            format!("'{target}' is neither an existing file path nor a known job")
        })?;
        let pred = repo
            .get_prediction_by_job(job_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Prediction for job {} not found", job_id))?;
        let content = tokio::fs::read_to_string(&pred.pdb_path)
            .await
            .with_context(|| format!("Failed to read PDB at {:?}", pred.pdb_path))?;
        // The title is the only provenance the viewers show, so say what the file is.
        let engine = proteus_engine::engine_name(pred.metadata.as_ref());
        let title = if engine == proteus_engine::ENGINE_SIMULATED {
            format!("Job {job_id} — SIMULATED: synthetic helix, not a prediction")
        } else if let Some(d) = proteus_engine::tier_downgrade(pred.metadata.as_ref()) {
            format!(
                "Job {job_id} ({engine}; tier '{}' not honoured)",
                d.requested
            )
        } else {
            format!("Job {job_id} ({engine})")
        };
        (content, title)
    };

    if web || html.is_some() {
        let format = proteus_core::io::sniff_format(
            &pdb_content,
            target_path.file_name().and_then(|n| n.to_str()),
        );
        let (color, secondary_structure) =
            match proteus_core::io::open_structure_bytes(pdb_content.as_bytes(), None) {
                Ok(pdb) => {
                    let scale = proteus_render::parse_pdb_structure(&pdb_content)
                        .map(|sd| sd.plddt_scale)
                        .unwrap_or(1.0);
                    let source = proteus_core::metrics::analyze_pdb_detailed(&pdb, None)
                        .ok()
                        .map(|a| a.metrics.confidence_source);
                    (
                        proteus_core::webview::WebColorScheme::from_provenance(source, scale),
                        proteus_core::webview::dssp_by_residue(&pdb),
                    )
                }
                Err(_) => (
                    proteus_core::webview::WebColorScheme::SecondaryStructure,
                    Vec::new(),
                ),
            };
        let html_content = proteus_core::webview::WebViewPage {
            title: "Proteus structure viewer",
            caption: &title,
            structure: &pdb_content,
            format,
            color,
            secondary_structure: &secondary_structure,
        }
        .render();

        let html_path = match html {
            // The user named the file: write exactly there.
            Some(path) => {
                tokio::fs::write(&path, html_content)
                    .await
                    .with_context(|| format!("Failed to write HTML file to {:?}", path))?;
                path
            }
            None => {
                let (mut file, path) = create_web_page_file(&std::env::temp_dir(), &title)?;
                std::io::Write::write_all(&mut file, html_content.as_bytes())
                    .with_context(|| format!("Failed to write HTML file to {:?}", path))?;
                path
            }
        };

        println!(
            "Generated standalone 3D WebGL viewer HTML -> {:?}",
            html_path
        );

        if web {
            println!("Launching default browser via xdg-open...");
            let _ = std::process::Command::new("xdg-open")
                .arg(&html_path)
                .spawn();
        }
        return Ok(());
    }

    let render_backend: proteus_render::terminal::TerminalBackend = backend.into();
    // The kitty protocol writes raw pixel escapes; a terminal that does not speak it prints
    // the payload as garbage. Say so once rather than letting the user think it is corrupt.
    {
        use proteus_render::terminal::{KittyRenderer, SixelRenderer, TerminalBackend};
        let unsupported = match render_backend {
            TerminalBackend::Kitty if !KittyRenderer::is_supported() => {
                Some(("kitty graphics", "kitty, ghostty, wezterm"))
            }
            TerminalBackend::Sixel if !SixelRenderer::is_supported() => Some((
                "Sixel",
                "xterm -ti vt340, mlterm, foot, contour, WezTerm, Windows Terminal",
            )),
            _ => None,
        };
        if let Some((protocol, terminals)) = unsupported {
            eprintln!(
                "note: $TERM does not look like a {protocol} terminal ({terminals}); if the \
                 output is garbage, use --backend halfblock or braille"
            );
        }
    }
    // Parsed once here; the snapshot/interactive paths reuse it.
    let structure_data = if compare.is_none() {
        Some(
            proteus_render::parse_pdb_structure(&pdb_content)
                .context("Failed to parse structure for 3D rendering")?,
        )
    } else {
        None
    };
    let render_color: proteus_render::rasterizer::ColorScheme = match (color, &structure_data) {
        (Some(c), _) => c.into(),
        (None, Some(sd)) => {
            let scheme = sd.default_color_scheme();
            if scheme == proteus_render::rasterizer::ColorScheme::SecondaryStructure {
                eprintln!(
                    "note: colouring by secondary structure (B-factor column is not a \
                     pLDDT confidence); pass --color plddt to force"
                );
            }
            scheme
        }
        (None, None) => proteus_render::rasterizer::ColorScheme::Plddt,
    };

    let (term_cols, term_rows): (u16, u16) = crossterm::terminal::size().unwrap_or((80, 24));
    let w = width.unwrap_or(term_cols as usize);
    let h = height.unwrap_or(term_rows.saturating_sub(4).max(16) as usize);

    // Say when the viewport cannot resolve what the structure contains. A picture that cannot
    // separate neighbouring residues should not be read as if it could.
    if let Some(sd) = structure_data.as_ref() {
        if let Some(note) = sd.resolution_note(w, h, render_backend) {
            eprintln!("{note}");
        }
    }

    if let Some(ref_path) = compare {
        let ref_content = proteus_core::io::read_structure_text(&ref_path)
            .with_context(|| format!("Failed to read reference structure at {:?}", ref_path))?;

        if interactive {
            let sup_data =
                proteus_render::prepare_superposition_for_rendering(&pdb_content, &ref_content)
                    .context("Failed to superimpose structures for 3D rendering")?;

            let ref_name = ref_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("Reference");
            let stats = sup_data.stats;
            if let Some(warning) = superposition_warning(&stats) {
                eprintln!("{warning}");
            }
            let dual_title = format!(
                "{title} (Cyan) vs {ref_name} (Ruby), {} Cα paired",
                stats.paired
            );
            let config = proteus_render::tui::ViewerConfig {
                title: dual_title,
                initial_color_scheme: proteus_render::rasterizer::ColorScheme::Solid(
                    proteus_render::rasterizer::ColorRGB::new(6, 182, 212),
                ),
                auto_rotate: true,
                secondary_mesh: Some((
                    sup_data.ref_mesh,
                    proteus_render::rasterizer::ColorRGB::new(244, 63, 94),
                )),
                rmsd: Some(sup_data.rmsd),
                disulfide_mesh: None,
                dashboard_enabled: false,
                dashboard_data: None,
                stop: None,
            };
            run_viewer(&sup_data.target_mesh, sup_data.camera, config)
                .context("Interactive dual-structure 3D viewer error")?;
        } else {
            let (snapshot, stats) = proteus_render::render_superposition_snapshot(
                &pdb_content,
                &ref_content,
                w,
                h,
                render_backend,
            )
            .context("Failed to render superposition snapshot")?;

            println!("{snapshot}");
            println!("{}", superposition_summary(&stats));
            if let Some(warning) = superposition_warning(&stats) {
                eprintln!("{warning}");
            }
        }
    } else if interactive || dashboard {
        let structure_data = structure_data.expect("parsed above when --compare is absent");

        let dashboard_data = Some(proteus_render::tui::DashboardData {
            title: title.clone(),
            num_residues: structure_data.num_residues,
            num_disulfides: structure_data.num_disulfides,
            metrics: structure_data.metrics,
            plddts: structure_data.plddts,
            ramachandran_points: structure_data.ramachandran_points,
        });

        let config = proteus_render::tui::ViewerConfig {
            title,
            initial_color_scheme: render_color,
            auto_rotate: true,
            secondary_mesh: None,
            rmsd: None,
            disulfide_mesh: structure_data.disulfide_mesh,
            dashboard_enabled: dashboard || term_cols >= 100,
            dashboard_data,
            stop: None,
        };
        run_viewer(&structure_data.ribbon_mesh, structure_data.camera, config)
            .context("Interactive 3D viewer error")?;
    } else {
        let structure_data = structure_data.expect("parsed above when --compare is absent");
        let snapshot = proteus_render::render_structure_snapshot(
            &structure_data,
            w,
            h,
            render_backend,
            render_color,
        )
        .context("Failed to render 3D snapshot")?;

        println!("{snapshot}");
    }
    Ok(())
}

/// The RMSD line, with how many residues it covers. Green only when the two structures are
/// the same sequence residue for residue; otherwise yellow, since the number then describes
/// only the paired part.
fn superposition_summary(s: &proteus_render::SuperpositionStats) -> String {
    let colour = if s.same_sequence() { "32" } else { "33" };
    format!(
        "\x1b[1mSuperposition:\x1b[0m Target (Cyan) vs Reference (Ruby) | \x1b[{colour}mRMSD: \
         {:.3} Å over {} Cα pairs\x1b[0m (target {}/{}, reference {}/{} residues paired)",
        s.rmsd, s.paired, s.paired, s.target_residues, s.paired, s.reference_residues
    )
}

/// A warning when the structures are not the same sequence, saying what differs.
fn superposition_warning(s: &proteus_render::SuperpositionStats) -> Option<String> {
    if s.same_sequence() {
        return None;
    }
    let mut parts = Vec::new();
    let unpaired_t = s.target_residues - s.paired;
    let unpaired_r = s.reference_residues - s.paired;
    if unpaired_t > 0 || unpaired_r > 0 {
        parts.push(format!(
            "{unpaired_t} target and {unpaired_r} reference residue(s) have no partner with the \
             same chain, number and insertion code"
        ));
    }
    if s.mismatched_names > 0 {
        parts.push(format!(
            "{} paired residue(s) have a different amino acid",
            s.mismatched_names
        ));
    }
    Some(format!(
        "warning: the sequences differ: {}; the RMSD covers the {} paired Cα only",
        parts.join(", and "),
        s.paired
    ))
}

/// Whether `target` could be a job UUID or a prefix of one: hex digits and hyphens only. A
/// file name (`typo.pdb`, `models/x.cif`) never is.
fn looks_like_job_ref(target: &str) -> bool {
    !target.is_empty() && target.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

/// Create the `--web` page as a new file with an unpredictable name in `dir`.
///
/// The name used to be fixed (`proteus_view_<title>.html` in the shared temp directory), so
/// another user could plant a symlink there and have the page written through it into any
/// file the viewer's user can write. `tempfile` creates the file with `O_CREAT | O_EXCL`
/// (never following a link) under a random name, readable only by its owner; the file is kept
/// so the browser can open it after this process exits.
fn create_web_page_file(dir: &Path, title: &str) -> Result<(std::fs::File, PathBuf)> {
    let sanitized: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .take(48)
        .collect();
    let file = tempfile::Builder::new()
        .prefix(&format!("proteus_view_{sanitized}_"))
        .suffix(".html")
        .tempfile_in(dir)
        .with_context(|| format!("Failed to create an HTML file in {}", dir.display()))?;
    let (file, path) = file
        .keep()
        .with_context(|| format!("Failed to keep the HTML file in {}", dir.display()))?;
    Ok((file, path))
}

/// Run the interactive viewer with SIGTERM, SIGHUP and SIGINT turned into a clean exit.
///
/// In raw mode Ctrl-C is a key press, but a signal from outside (`kill`, a closed SSH session,
/// a job-control timeout) would otherwise end the process with the terminal still raw, on the
/// alternate screen and with the cursor hidden. The handler asks the viewer to stop, the viewer
/// restores the terminal on its way out, and the process then exits with the conventional
/// `128 + signal` status.
fn run_viewer(
    mesh: &proteus_render::geometry::mesh::TriangleMesh,
    camera: proteus_render::rasterizer::OrbitCamera,
    mut config: proteus_render::tui::ViewerConfig,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let stop = Arc::new(AtomicBool::new(false));
    let received = watch_termination_signals(Arc::clone(&stop))?;
    config.stop = Some(stop);
    let run = || proteus_render::tui::run_interactive_viewer(mesh, camera, config);
    // The viewer blocks; on the multi-threaded runtime let the signal task run elsewhere.
    let result = match tokio::runtime::Handle::current().runtime_flavor() {
        tokio::runtime::RuntimeFlavor::MultiThread => tokio::task::block_in_place(run),
        _ => run(),
    };
    let signal = received.load(Ordering::SeqCst);
    if signal != 0 {
        std::process::exit(128 + signal);
    }
    Ok(result?)
}

/// Listen for SIGTERM, SIGHUP and SIGINT; on the first, record its number and set `stop`.
#[cfg(unix)]
fn watch_termination_signals(
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> std::io::Result<Arc<std::sync::atomic::AtomicI32>> {
    use std::sync::atomic::{AtomicI32, Ordering};
    use tokio::signal::unix::{signal, SignalKind};
    // Registered before returning, so a signal that arrives once this has returned is caught.
    let mut term = signal(SignalKind::terminate())?;
    let mut hup = signal(SignalKind::hangup())?;
    let mut int = signal(SignalKind::interrupt())?;
    let received = Arc::new(AtomicI32::new(0));
    let flag = Arc::clone(&received);
    tokio::spawn(async move {
        // POSIX numbers, identical on Linux and macOS.
        let n = tokio::select! {
            _ = term.recv() => 15,
            _ = hup.recv() => 1,
            _ = int.recv() => 2,
        };
        flag.store(n, Ordering::SeqCst);
        stop.store(true, Ordering::SeqCst);
    });
    Ok(received)
}

#[cfg(not(unix))]
fn watch_termination_signals(
    _stop: Arc<std::sync::atomic::AtomicBool>,
) -> std::io::Result<Arc<std::sync::atomic::AtomicI32>> {
    Ok(Arc::new(std::sync::atomic::AtomicI32::new(0)))
}

#[cfg(test)]
mod tests {
    use super::Args;
    use crate::cli::{Cli, CliColorScheme, Commands};
    use clap::Parser;

    /// `--color` is optional: without it the scheme follows the structure's provenance, so a
    /// crystal structure is not painted "very low confidence" from small B-factors.
    #[test]
    fn view_colour_is_optional() {
        let color_of = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            Commands::View(Args { color, .. }) => color,
            _ => unreachable!(),
        };
        assert_eq!(color_of(&["proteus", "view", "x.pdb"]), None);
        assert_eq!(
            color_of(&["proteus", "view", "x.pdb", "--color", "rainbow"]),
            Some(CliColorScheme::Rainbow)
        );
    }

    /// `--web` must not write through a file planted at a predictable name in the shared temp
    /// directory: every page gets a fresh file, and a symlink waiting at the old fixed name is
    /// left alone along with its target.
    #[cfg(unix)]
    #[test]
    fn web_page_is_a_new_file_not_a_planted_link() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "precious").unwrap();
        // The name the page used to get for this title.
        let planted = dir.path().join("proteus_view_1crn_pdb.html");
        std::os::unix::fs::symlink(&victim, &planted).unwrap();

        let (mut file, path) = super::create_web_page_file(dir.path(), "1crn.pdb").unwrap();
        file.write_all(b"<html></html>").unwrap();
        let (_, second) = super::create_web_page_file(dir.path(), "1crn.pdb").unwrap();

        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
        assert_ne!(path, planted);
        assert_ne!(path, second, "two pages got the same name");
        assert!(path.starts_with(dir.path()));
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(meta.file_type().is_file(), "the page is not a regular file");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "<html></html>");
    }

    /// `proteus view typo.pdb` used to create `proteus.db` (and its -wal/-shm files) before
    /// failing, because the argument fell through to the job lookup, which opens the database
    /// read-write. A missing file must fail without touching the data directory, and so must a
    /// job ID when there is no database to look it up in.
    #[tokio::test]
    async fn a_missing_file_does_not_create_the_job_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("proteus.db");
        for target in ["typo.pdb", "models/missing.cif", "deadbeef"] {
            let Commands::View(args) = Cli::try_parse_from(["proteus", "view", target])
                .unwrap()
                .command
            else {
                unreachable!()
            };
            let err = super::run(args, &db).await.expect_err(target).to_string();
            assert!(err.contains(target), "{target}: {err}");
            let left: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
            assert!(left.is_empty(), "{target}: created {left:?}");
        }
        assert!(super::looks_like_job_ref("8c716e90"));
        assert!(super::looks_like_job_ref(
            "8c716e90-5b74-4a0e-9f3b-0123456789ab"
        ));
        assert!(!super::looks_like_job_ref("typo.pdb"));
        assert!(!super::looks_like_job_ref(""));
    }

    /// `--compare` draws both structures in fixed colours and has no dashboard, so asking for
    /// either is refused instead of silently ignored.
    #[test]
    fn compare_rejects_options_it_cannot_honour() {
        for extra in [["--color", "rainbow"], ["--dashboard", "--interactive"]] {
            let mut argv = vec!["proteus", "view", "a.pdb", "--compare", "b.pdb"];
            argv.extend(extra);
            let Err(err) = Cli::try_parse_from(&argv) else {
                panic!("{argv:?} was accepted");
            };
            assert!(err.to_string().contains("cannot be used with"), "{err}");
        }
        assert!(Cli::try_parse_from(["proteus", "view", "a.pdb", "--compare", "b.pdb"]).is_ok());
    }

    /// The RMSD is green only for the same sequence; otherwise it says how much was paired
    /// and warns what differs.
    #[test]
    fn superposition_report_says_what_was_paired() {
        let same = proteus_render::SuperpositionStats {
            rmsd: 0.5,
            paired: 46,
            target_residues: 46,
            reference_residues: 46,
            mismatched_names: 0,
        };
        let line = super::superposition_summary(&same);
        assert!(
            line.contains("\x1b[32mRMSD: 0.500 Å over 46 Cα pairs"),
            "{line}"
        );
        assert!(super::superposition_warning(&same).is_none());

        let shorter = proteus_render::SuperpositionStats {
            paired: 45,
            target_residues: 45,
            ..same
        };
        let line = super::superposition_summary(&shorter);
        assert!(
            !line.contains("\x1b[32m"),
            "differing sequences reported in green: {line}"
        );
        assert!(line.contains("reference 45/46"), "{line}");
        let warning = super::superposition_warning(&shorter).unwrap();
        assert!(warning.contains("0 target and 1 reference"), "{warning}");

        let mutant = proteus_render::SuperpositionStats {
            mismatched_names: 2,
            ..same
        };
        assert!(super::superposition_warning(&mutant)
            .unwrap()
            .contains("2 paired residue(s) have a different amino acid"));
    }

    /// A viewport the renderer cannot allocate is refused at the argument parser, with the
    /// limit in the message, instead of aborting (100000×100000) or printing nothing (0).
    #[test]
    fn viewport_size_is_bounded() {
        let parse = |w: &str| Cli::try_parse_from(["proteus", "view", "x.pdb", "--width", w]);
        for bad in ["0", "4097", "100000", "18446744073709551615"] {
            let Err(err) = parse(bad) else {
                panic!("--width {bad} was accepted")
            };
            let err = err.to_string();
            assert!(err.contains("--width"), "{bad}: {err}");
        }
        assert!(parse("1").is_ok());
        assert!(parse("4096").is_ok());
        assert!(Cli::try_parse_from(["proteus", "view", "x.pdb", "--height", "5000"]).is_err());
    }

    /// SIGTERM while the viewer runs must reach the viewer as a stop request (it then restores
    /// the terminal) rather than kill the process with the terminal left raw.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sigterm_asks_the_viewer_to_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let received = super::watch_termination_signals(Arc::clone(&stop)).unwrap();
        let status = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        for _ in 0..200 {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            stop.load(Ordering::SeqCst),
            "SIGTERM did not stop the viewer"
        );
        assert_eq!(received.load(Ordering::SeqCst), 15);
    }
}
