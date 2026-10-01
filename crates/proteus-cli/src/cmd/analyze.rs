//! `proteus analyze` — All-atom biophysics of structure files, with no database involved.
//!
//! One file prints the full report. Several files, a directory, `--export` or `--json` switch
//! to the table form: one row per structure, analysed in parallel, written as Parquet/CSV/JSON
//! so a folder of predicted models can be triaged in DuckDB, Polars or pandas.

use super::prelude::*;
use proteus_core::interface::InterfaceSpec;
use proteus_core::qc::{is_structure_file_name, structure_qc_with, QcOptions, StructureQc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Arguments of `proteus analyze`.
#[derive(clap::Args, Debug)]
#[command(after_help = "Examples:
  proteus analyze model.pdb
  proteus analyze models/ --export qc.parquet
  proteus analyze designs/*.cif --reference target.pdb --json | jq .rmsd_to_reference
  proteus analyze boltz_results/ --interface A:B --export triage.parquet")]
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

    /// Measure a binder–target interface: `A` (chain A against every other chain), `A:B`,
    /// `H,L:A`; with no value, the first chain against the rest. Adds contacts, buried surface
    /// (dSASA), shape complementarity, cross-interface H-bonds and salt bridges and, when the
    /// predictor's PAE and scores files sit next to the model (Boltz, AlphaFold 3, Protenix,
    /// OpenFold3, ColabFold; Chai-1 gives ipTM only), ipTM, ipAE, ipSAE and LIS
    #[arg(long, value_name = "BINDER[:TARGET]", num_args = 0..=1, default_missing_value = "")]
    interface: Option<String>,

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
            let (files, failed) = (out.len(), unreadable.len());
            walk(p, &mut out, &mut unreadable, &mut seen_dirs);
            if out.len() == files && unreadable.len() == failed {
                unreadable.push((
                    p.clone(),
                    "no structure files (.pdb, .ent, .cif, .mmcif, optionally .gz) under it".into(),
                ));
            }
        } else if p.exists() {
            out.push(p.clone());
        } else {
            // One mistyped path among many is a failure to report, not a reason to drop the rest.
            unreadable.push((p.clone(), "no such file or directory".into()));
        }
    }
    if out.is_empty() {
        if let [(p, why)] = unreadable.as_slice() {
            bail!("{}: {why}", p.display());
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
    opts: &QcOptions,
    jobs: usize,
) -> Vec<std::result::Result<StructureQc, String>> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<std::result::Result<StructureQc, String>>>> =
        Mutex::new(vec![None; inputs.len()]);
    let bar = ProgressBar::new(inputs.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(&format!(
            "{{bar:40.{}/{}}} {{pos}}/{{len}} structures  {{elapsed}} (eta {{eta}})",
            crate::cli::tint(proteus_render::brand::Role::Accent),
            crate::cli::tint(proteus_render::brand::Role::Line)
        ))
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
                let r = structure_qc_with(path, opts).map_err(|e| e.to_string());
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
    if rows.iter().any(|r| r.interface.interface_binder.is_some()) {
        return print_interface_summary(rows, top, exported);
    }
    let mut order: Vec<&StructureQc> = rows.iter().collect();
    order.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let mut table = Table::new();
    table.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    crate::cli::fit_table(&mut table);
    table.set_header(vec![
        "file",
        "res",
        "ch",
        "pLDDT",
        "Rama fav %",
        "outliers",
        "overlap/1k",
        "bond Z",
        "rota out %",
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
            opt(r.bond_rmsz, 2),
            opt(r.rotamer_outlier_pct, 1),
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

/// The table under `--interface`: the interface columns, best ipSAE_min first when the
/// predictor's PAE was found (AlphaFold 3's ipSAE outperformed ipAE and ipTM at predicting
/// binding among 3,766 tested designs; Overath et al. 2025), otherwise by buried surface.
fn print_interface_summary(rows: &[StructureQc], top: usize, exported: bool) {
    let by_ipsae = rows.iter().any(|r| r.interface.ipsae_min.is_some());
    let key = |r: &StructureQc| {
        if by_ipsae {
            r.interface.ipsae_min
        } else {
            r.interface.interface_dsasa
        }
        .unwrap_or(f64::NEG_INFINITY)
    };
    let mut order: Vec<&StructureQc> = rows.iter().collect();
    order.sort_by(|a, b| key(b).total_cmp(&key(a)));
    let mut table = Table::new();
    table.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    crate::cli::fit_table(&mut table);
    table.set_header(vec![
        "file",
        "binder:target",
        "pLDDT",
        "ipTM",
        "ipSAE min",
        "ipAE",
        "LIS",
        "Sc",
        "dSASA Å²",
        "contacts",
        "H-bonds",
        "salt",
        "bond Z",
    ]);
    let n = |v: Option<usize>| v.map_or_else(|| "–".into(), |v| v.to_string());
    for r in order.iter().take(top) {
        let i = &r.interface;
        let name = Path::new(&r.file)
            .file_name()
            .map_or_else(|| r.file.clone(), |n| n.to_string_lossy().into_owned());
        table.add_row(vec![
            name,
            format!(
                "{}:{}",
                i.interface_binder.as_deref().unwrap_or("–"),
                i.interface_target.as_deref().unwrap_or("–")
            ),
            opt(r.plddt_mean, 1),
            opt(i.iptm, 2),
            opt(i.ipsae_min, 3),
            opt(i.ipae, 1),
            opt(i.lis, 3),
            opt(i.interface_sc, 2),
            opt(i.interface_dsasa, 0),
            format!(
                "{}/{}",
                n(i.interface_binder_residues),
                n(i.interface_target_residues)
            ),
            n(i.interface_hbonds),
            n(i.interface_salt_bridges),
            opt(r.bond_rmsz, 2),
        ]);
    }
    println!("{table}");
    println!(
        "sorted by {}",
        if by_ipsae {
            "ipSAE_min"
        } else {
            "dSASA (no PAE file found next to the models, so no ipSAE)"
        }
    );
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
        interface,
        confidence_source,
        export,
        json,
        jobs,
        top,
    } = args;
    let interface = interface.as_deref().map(InterfaceSpec::parse).transpose()?;
    paths.extend(pdb);
    if paths.is_empty() {
        bail!("give at least one structure file or directory, e.g. `proteus analyze model.pdb`");
    }
    if let Some(ref out) = export {
        check_export_target(out)?;
    }
    let (inputs, unreadable) = collect_inputs(&paths)?;
    // The interface columns live in the table form, so --interface on one file prints a row.
    let single_report = inputs.len() == 1
        && paths.len() == 1
        && paths[0].is_file()
        && export.is_none()
        && !json
        && interface.is_none();
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
        &QcOptions {
            reference: reference_pdb.as_ref(),
            confidence: forced_source(confidence_source),
            interface: interface.as_ref(),
        },
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
    // Confidence files that were found but could not be used; say what was lost, briefly.
    let noted: Vec<_> = rows
        .iter()
        .filter_map(|r| r.interface.interface_note.as_ref().map(|n| (&r.file, n)))
        .collect();
    for (file, note) in noted.iter().take(5) {
        eprintln!("  {file}: {note}");
    }
    if noted.len() > 5 {
        eprintln!("  … and {} more with confidence notes", noted.len() - 5);
    }
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

    crate::cli::fit_table(&mut table);
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

    if let Some(ref g) = metrics.covalent_geometry {
        let rmsz = |v: Option<f64>| v.map_or_else(|| "–".to_string(), |v| format!("{v:.2}"));
        table.add_row(vec![
            Cell::new("Bond lengths (geostd + CDL)"),
            Cell::new(format!(
                "RMSZ {} | {} of {} beyond 4σ",
                rmsz(g.bonds.rmsz),
                g.bonds.outliers,
                g.bonds.n
            )),
        ]);
        table.add_row(vec![
            Cell::new("Bond angles (geostd + CDL)"),
            Cell::new(format!(
                "RMSZ {} | {} of {} beyond 4σ",
                rmsz(g.angles.rmsz),
                g.angles.outliers,
                g.angles.n
            )),
        ]);
        table.add_row(vec![
            Cell::new("Chirality | planarity"),
            Cell::new(format!(
                "{} chiral outliers ({} inverted) | {} planar groups beyond 4σ",
                g.chiralities.outliers, g.handedness_swaps, g.planes.outliers
            )),
        ]);
        table.add_row(vec![
            Cell::new("Cβ deviation (≥0.25 Å)"),
            Cell::new(format!(
                "{} of {} residues",
                g.cbeta_outliers, g.cbeta_residues
            )),
        ]);
        table.add_row(vec![
            Cell::new("Peptide ω"),
            Cell::new(format!(
                "{} cis-Pro | {} cis non-Pro | {} twisted (of {})",
                g.cis_proline, g.cis_nonproline, g.twisted, g.peptides
            )),
        ]);
        table.add_row(vec![
            Cell::new("Rotamers (Top8000)"),
            Cell::new(format!(
                "outliers {} ({} of {}) | allowed {}",
                g.rotamer_outlier_pct()
                    .map_or_else(|| "–".to_string(), |p| format!("{p:.1}%")),
                g.rotamer_outliers,
                g.rotamer_residues,
                g.rotamer_allowed
            )),
        ]);
        if g.symmetric_flips > 0 {
            table.add_row(vec![
                Cell::new("Symmetric atoms renamed"),
                Cell::new(format!(
                    "{} residues named against the IUPAC convention (swapped before checking)",
                    g.symmetric_flips
                )),
            ]);
        }
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
    if let Some(ref g) = metrics.covalent_geometry {
        print_geometry_outliers(g, WORST_OUTLIERS_SHOWN);
    }
    Ok(())
}

/// Outliers listed under the single-structure report.
const WORST_OUTLIERS_SHOWN: usize = 10;

fn print_geometry_outliers(g: &proteus_core::geometry::CovalentGeometry, n: usize) {
    use proteus_core::geometry::OutlierKind;
    if g.outliers.is_empty() {
        return;
    }
    let mut table = Table::new();
    table.load_style(UTF8_FULL);
    crate::cli::fit_table(&mut table);
    table.set_header(vec![
        "Worst geometry outliers",
        "Atoms",
        "Ideal",
        "Model",
        "Z",
    ]);
    for o in g.outliers.iter().take(n) {
        let what = match o.kind {
            OutlierKind::Bond => "bond",
            OutlierKind::Angle => "angle",
            OutlierKind::Tetrahedral => "chiral volume",
            OutlierKind::HandednessSwap => "inverted chiral centre",
            OutlierKind::PseudochiralNaming => "methyls misnamed",
            OutlierKind::Planarity => "planar group",
            OutlierKind::Cbeta => "Cβ deviation",
            OutlierKind::CisPeptide => "cis peptide (non-Pro)",
            OutlierKind::TwistedPeptide => "twisted peptide",
            OutlierKind::Rotamer => "rotamer outlier",
            OutlierKind::Degenerate => "coincident atoms (undefined)",
        };
        table.add_row(vec![
            Cell::new(what),
            Cell::new(o.atoms.join(" – ")),
            Cell::new(format!("{:.3}", o.ideal)),
            Cell::new(format!("{:.3}", o.model)),
            Cell::new(o.z.map_or_else(|| "–".to_string(), |z| format!("{z:+.1}"))),
        ]);
    }
    println!("{table}");
    let total = g.outliers_total.max(g.outliers.len());
    if total > n {
        println!("{} more outliers not shown.", total - n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_path_is_a_failure_not_an_abort() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("a.pdb");
        std::fs::write(&good, "END\n").unwrap();
        let missing = dir.path().join("no-such.pdb");
        let (inputs, failed) = collect_inputs(&[good.clone(), missing.clone()]).unwrap();
        assert_eq!(inputs, vec![good]);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, missing);
        assert!(failed[0].1.contains("no such file"));
    }

    #[test]
    fn a_lone_missing_path_is_still_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("typo.pdb");
        let err = collect_inputs(&[missing]).unwrap_err().to_string();
        assert!(err.contains("typo.pdb: no such file or directory"), "{err}");
    }

    /// An empty folder among other inputs is reported like a missing path; alone, it is still
    /// an error.
    #[test]
    fn an_empty_folder_is_a_failure_not_an_abort() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("a.pdb");
        std::fs::write(&good, "END\n").unwrap();
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        let (inputs, failed) = collect_inputs(&[good.clone(), empty.clone()]).unwrap();
        assert_eq!(inputs, vec![good]);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, empty);
        let err = collect_inputs(&[empty]).unwrap_err().to_string();
        assert!(err.contains("no structure files"), "{err}");
    }
}
