//! `proteus analyze` — All-atom biophysics of structure files, with no database involved.
//!
//! One file prints the full report. Several files, a directory, `--export` or `--json` switch
//! to the table form: one row per structure, analysed in parallel, written as Parquet/CSV/JSON
//! so a folder of predicted models can be triaged in DuckDB, Polars or pandas.

use super::prelude::*;
use proteus_core::qc::{is_structure_file_name, structure_qc, StructureQc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Arguments of `proteus analyze`.
#[derive(clap::Args, Debug)]
#[command(after_help = "Examples:
  proteus analyze model.pdb
  proteus analyze models/ --export qc.parquet
  proteus analyze designs/*.cif --reference target.pdb --json | jq .rmsd_to_reference")]
pub struct Args {
    /// Structure files (PDB or mmCIF, optionally gzipped) or directories to search recursively
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Structure file (the same as a PATH argument; kept for older scripts)
    #[arg(short, long, value_name = "PATH")]
    pdb: Vec<PathBuf>,

    /// Reference structure: adds the Kabsch C-alpha RMSD of every input against it
    #[arg(short, long)]
    reference: Option<PathBuf>,

    /// Treat the B-factor column as pLDDT (predicted) or as experimental B-factors;
    /// `auto` inspects the header and the value distribution.
    #[arg(long, value_enum, default_value_t = ConfidenceSourceArg::Auto)]
    confidence_source: ConfidenceSourceArg,

    /// Write one row per structure to a .parquet, .csv or .json file
    #[arg(short, long, value_name = "FILE")]
    export: Option<PathBuf>,

    /// Print one JSON object per structure to stdout (JSON Lines) instead of a table
    #[arg(long)]
    json: bool,

    /// Structures analysed in parallel [default: the number of CPUs]
    #[arg(short, long)]
    jobs: Option<usize>,

    /// Rows shown in the terminal summary, best fitness first (the export always has all)
    #[arg(long, default_value_t = 20)]
    top: usize,
}

/// Expand directories into the structure files beneath them, sorted, keeping explicit file
/// arguments in the order given. An explicit file is taken whatever its extension. Each
/// directory is walked once by its canonical path, so a symlink loop (`models/loop -> ..`)
/// or a directory reachable twice does not repeat files, and each file is listed once however
/// it was reached. A subdirectory that cannot be read is reported, not fatal.
/// A path that could not be analysed, and why.
type Failure = (PathBuf, String);

fn collect_inputs(paths: &[PathBuf]) -> Result<(Vec<PathBuf>, Vec<Failure>)> {
    use std::collections::HashSet;
    fn walk(
        dir: &Path,
        out: &mut Vec<PathBuf>,
        unreadable: &mut Vec<Failure>,
        seen_dirs: &mut HashSet<PathBuf>,
    ) {
        if let Ok(real) = dir.canonicalize() {
            if !seen_dirs.insert(real) {
                return;
            }
        }
        let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(e) => {
                unreadable.push((dir.to_path_buf(), format!("cannot read directory: {e}")));
                return;
            }
        };
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out, unreadable, seen_dirs);
            } else if is_structure_file_name(&p) {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    let mut unreadable = Vec::new();
    let mut seen_dirs = HashSet::new();
    for p in paths {
        if p.is_dir() {
            let before = out.len();
            walk(p, &mut out, &mut unreadable, &mut seen_dirs);
            if out.len() == before && unreadable.is_empty() {
                bail!(
                    "no structure files (.pdb, .ent, .cif, .mmcif, optionally .gz) under {}",
                    p.display()
                );
            }
        } else if p.exists() {
            out.push(p.clone());
        } else {
            bail!("no such file or directory: {}", p.display());
        }
    }
    let mut seen_files = HashSet::new();
    out.retain(|p| seen_files.insert(p.canonicalize().unwrap_or_else(|_| p.clone())));
    Ok((out, unreadable))
}

/// Refuse an export target that cannot be written before any structure is analysed: an
/// unsupported extension, an existing directory, or a parent that is a file.
fn check_export_target(out: &Path) -> Result<()> {
    proteus_storage::export::check_export_path(out)?;
    if out.is_dir() {
        bail!(
            "--export {} is a directory; give a file name",
            out.display()
        );
    }
    let mut parent = out.parent();
    while let Some(p) = parent.filter(|p| !p.as_os_str().is_empty()) {
        if p.exists() {
            if !p.is_dir() {
                bail!(
                    "--export {}: {} is not a directory",
                    out.display(),
                    p.display()
                );
            }
            break;
        }
        parent = p.parent();
    }
    Ok(())
}

fn forced_source(arg: ConfidenceSourceArg) -> Option<ConfidenceSource> {
    match arg {
        ConfidenceSourceArg::Auto => None,
        ConfidenceSourceArg::Predicted => Some(ConfidenceSource::Predicted),
        ConfidenceSourceArg::Experimental => Some(ConfidenceSource::ExperimentalBFactor),
    }
}

/// Analyse `inputs` on `jobs` threads. Results come back in input order.
fn analyze_many(
    inputs: &[PathBuf],
    reference: Option<&pdbtbx::PDB>,
    confidence: Option<ConfidenceSource>,
    jobs: usize,
) -> Vec<std::result::Result<StructureQc, String>> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<std::result::Result<StructureQc, String>>>> =
        Mutex::new(vec![None; inputs.len()]);
    let bar = ProgressBar::new(inputs.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} structures  {elapsed} (eta {eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    if inputs.len() < 2 {
        bar.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    std::thread::scope(|s| {
        for _ in 0..jobs.clamp(1, inputs.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(path) = inputs.get(i) else { break };
                let r = structure_qc(path, reference, confidence).map_err(|e| e.to_string());
                results.lock().unwrap_or_else(|p| p.into_inner())[i] = Some(r);
                bar.inc(1);
            });
        }
    });
    bar.finish_and_clear();
    results
        .into_inner()
        .unwrap_or_else(|p| p.into_inner())
        .into_iter()
        .map(|r| r.unwrap_or_else(|| Err("not analysed".into())))
        .collect()
}

fn opt(v: Option<f64>, prec: usize) -> String {
    v.map(|v| format!("{v:.prec$}"))
        .unwrap_or_else(|| "–".into())
}

fn print_summary(rows: &[StructureQc], top: usize, exported: bool) {
    if top == 0 || rows.is_empty() {
        return;
    }
    let mut order: Vec<&StructureQc> = rows.iter().collect();
    order.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let mut table = Table::new();
    table.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    table.set_header(vec![
        "file",
        "res",
        "ch",
        "pLDDT",
        "Rama fav %",
        "outliers",
        "overlap/1k",
        "Rg/Rg₀",
        "H %",
        "E %",
        "RMSD",
        "fitness",
    ]);
    for r in order.iter().take(top) {
        let name = Path::new(&r.file)
            .file_name()
            .map_or_else(|| r.file.clone(), |n| n.to_string_lossy().into_owned());
        table.add_row(vec![
            name,
            r.n_residues.to_string(),
            r.n_chains.to_string(),
            opt(r.plddt_mean, 1),
            format!("{:.1}", r.rama_favored_pct),
            r.rama_outliers.to_string(),
            format!("{:.1}", r.heavy_atom_overlap_score),
            format!("{:.2}", r.rg_ratio),
            format!("{:.0}", r.helix_pct),
            format!("{:.0}", r.strand_pct),
            opt(r.rmsd_to_reference, 2),
            format!("{:.1}", r.fitness),
        ]);
    }
    println!("{table}");
    if rows.len() > top {
        let hint = if exported {
            "the export has every row"
        } else {
            "--export writes every row"
        };
        println!("{} more not shown (--top {top}); {hint}.", rows.len() - top);
    }
}

pub async fn run(args: Args) -> Result<()> {
    let Args {
        mut paths,
        pdb,
        reference,
        confidence_source,
        export,
        json,
        jobs,
        top,
    } = args;
    paths.extend(pdb);
    if paths.is_empty() {
        bail!("give at least one structure file or directory, e.g. `proteus analyze model.pdb`");
    }
    if let Some(ref out) = export {
        check_export_target(out)?;
    }
    let (inputs, unreadable) = collect_inputs(&paths)?;
    let single_report =
        inputs.len() == 1 && paths.len() == 1 && paths[0].is_file() && export.is_none() && !json;
    if single_report {
        return print_report(&inputs[0], reference.as_deref(), confidence_source);
    }

    let reference_pdb = reference
        .as_deref()
        .map(proteus_core::io::open_structure)
        .transpose()
        .context("cannot read the reference structure")?;
    let jobs = jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let started = std::time::Instant::now();
    let results = analyze_many(
        &inputs,
        reference_pdb.as_ref(),
        forced_source(confidence_source),
        jobs,
    );
    let elapsed = started.elapsed();

    let mut rows = Vec::with_capacity(results.len());
    let mut failures: Vec<Failure> = unreadable;
    for (path, r) in inputs.iter().zip(results) {
        match r {
            Ok(row) => rows.push(row),
            Err(e) => failures.push((path.clone(), e)),
        }
    }

    // The export is written before anything goes to stdout: a reader that stops early
    // (`| head -1`) ends the process on its next write, and must not cost the file.
    if let Some(ref out) = export {
        proteus_storage::save_qc_table(&rows, out)
            .with_context(|| format!("cannot write {}", out.display()))?;
    }
    if json {
        for r in &rows {
            println!("{}", serde_json::to_string(r)?);
        }
    } else {
        print_summary(&rows, top, export.is_some());
    }
    eprintln!(
        "{} of {} structures analysed in {:.2} s{}",
        rows.len(),
        inputs.len(),
        elapsed.as_secs_f64(),
        export
            .as_ref()
            .map(|p| format!(", written to {}", p.display()))
            .unwrap_or_default()
    );
    for (path, e) in &failures {
        eprintln!("  failed: {}: {e}", path.display());
    }
    if !failures.is_empty() {
        bail!(
            "{} of {} structures could not be analysed",
            failures.len(),
            rows.len() + failures.len()
        );
    }
    Ok(())
}

/// The full single-structure report.
fn print_report(
    pdb: &Path,
    reference: Option<&Path>,
    confidence_source: ConfidenceSourceArg,
) -> Result<()> {
    println!("Analyzing structure file: {:?}", pdb);
    let loaded = proteus_core::io::load_structure(pdb).context("Biophysical analysis failed")?;
    let reference = reference
        .map(proteus_core::io::open_structure)
        .transpose()
        .context("cannot read the reference structure")?;
    let metrics = proteus_core::metrics::analyze_pdb_detailed_with_source(
        &loaded.pdb,
        reference.as_ref(),
        Some(&loaded.header_preview),
        forced_source(confidence_source),
    )
    .context("Biophysical analysis failed")?
    .metrics;

    let mut table = Table::new();
    table.load_style(UTF8_FULL);
    table.set_header(vec!["Biophysical Metric", "Value"]);

    table.add_row(vec![
        Cell::new("Radius of Gyration (Rg)"),
        Cell::new(format!("{:.3} Å", metrics.radius_of_gyration)),
    ]);
    table.add_row(vec![
        Cell::new("Contact Density (C-alpha <= 8Å)"),
        Cell::new(format!("{:.2}%", metrics.contact_density * 100.0)),
    ]);
    match metrics.plddt() {
        Some(p) => {
            table.add_row(vec![
                Cell::new("Mean pLDDT"),
                Cell::new(format!("{:.2}", p.mean)),
            ]);
            table.add_row(vec![
                Cell::new("Median pLDDT"),
                Cell::new(format!("{:.2}", p.median)),
            ]);
            table.add_row(vec![
                Cell::new("Fraction High Conf (pLDDT >= 70)"),
                Cell::new(format!("{:.1}%", p.high_confidence_fraction * 100.0)),
            ]);
            table.add_row(vec![
                Cell::new("Fraction Very High Conf (pLDDT >= 90)"),
                Cell::new(format!("{:.1}%", p.very_high_confidence_fraction * 100.0)),
            ]);
        }
        None => {
            table.add_row(vec![
                Cell::new("pLDDT"),
                Cell::new("n/a (experimental structure; B-factor column is not a confidence)"),
            ]);
        }
    }

    if let Some(rmsd) = metrics.rmsd_to_reference {
        table.add_row(vec![
            Cell::new("Kabsch RMSD to Reference"),
            Cell::new(format!("{:.3} Å", rmsd)),
        ]);
    }

    if let Some(ref ss) = metrics.secondary_structure_summary {
        table.add_row(vec![
            Cell::new("Secondary Structure Composition"),
            Cell::new(format!(
                "α-Helix: {:.1}% | β-Strand: {:.1}% | Coil: {:.1}%",
                ss.helix_fraction * 100.0,
                ss.strand_fraction * 100.0,
                ss.coil_fraction * 100.0
            )),
        ]);
    }

    if let Some(ref rama) = metrics.ramachandran_stats {
        table.add_row(vec![
            Cell::new("Ramachandran Conformation"),
            Cell::new(format!(
                "Favored: {:.1}% | Allowed: {:.1}% | Outliers: {}",
                rama.favored_fraction * 100.0,
                rama.allowed_fraction * 100.0,
                rama.outlier_count
            )),
        ]);
    }

    if let Some(ref sasa) = metrics.sasa_metrics {
        table.add_row(vec![
            Cell::new("Solvent Accessible Surface Area"),
            Cell::new(format!(
                "Total: {:.1} Å² (Hydrophobic Burial: {:.1}%)",
                sasa.total_sasa,
                sasa.hydrophobic_burial_ratio * 100.0
            )),
        ]);
    }

    if let Some(ref clash) = metrics.steric_overlap {
        table.add_row(vec![
            Cell::new("Heavy-atom steric overlap (>0.4 Å, no H)"),
            Cell::new(format!(
                "{:.1} per 1k atoms ({} overlaps)",
                clash.heavy_atom_overlap_score, clash.clash_count
            )),
        ]);
    }

    if let Some(ref net) = metrics.interaction_network {
        table.add_row(vec![
            Cell::new("Hydrogen Bonds (H-Bonds)"),
            Cell::new(format!(
                "{} total ({} BB-BB, {} BB-SC, {} SC-SC)",
                net.summary.total_hbonds,
                net.summary.bb_bb_hbonds,
                net.summary.bb_sc_hbonds,
                net.summary.sc_sc_hbonds
            )),
        ]);
        table.add_row(vec![
            Cell::new("Ionic Salt Bridges (≤4.0Å)"),
            Cell::new(format!(
                "{} detected{}",
                net.summary.total_salt_bridges,
                if let Some(s) = net.salt_bridges.first() {
                    format!(
                        " (closest: {}{}:{}-{}{}:{} {:.2}Å)",
                        s.cation_res_name,
                        s.cation_res_seq,
                        s.cation_atom_name,
                        s.anion_res_name,
                        s.anion_res_seq,
                        s.anion_atom_name,
                        s.distance
                    )
                } else {
                    "".to_string()
                }
            )),
        ]);
        table.add_row(vec![
            Cell::new("Aromatic π-π Stacking"),
            Cell::new(format!(
                "{} conjugated pairs ({} parallel, {} T-shaped)",
                net.summary.total_pi_pi_stacks,
                net.pi_pi_stacks
                    .iter()
                    .filter(|p| p.category == proteus_core::PiStackingCategory::Parallel)
                    .count(),
                net.pi_pi_stacks
                    .iter()
                    .filter(|p| p.category == proteus_core::PiStackingCategory::TShaped)
                    .count(),
            )),
        ]);
        table.add_row(vec![
            Cell::new("Cation-π Interactions"),
            Cell::new(format!(
                "{} active interactions",
                net.summary.total_cation_pi
            )),
        ]);
        table.add_row(vec![
            Cell::new("Non-Covalent Network Density"),
            Cell::new(format!(
                "{:.1} contacts / 100 res",
                net.summary.network_density
            )),
        ]);
    }

    if let Some(fitness) = metrics.candidate_fitness_score {
        table.add_row(vec![
            Cell::new("Candidate Fitness Score"),
            Cell::new(format!("{:.1} / 100", fitness)),
        ]);
    }

    println!("{table}");
    Ok(())
}
