//! `proteus diff` — what each design candidate changed relative to its parent: the changed
//! residues, how far the structure moved around them, the predictor's confidence there, and
//! the non-covalent interactions lost and gained (see `proteus_core::variant`).

use super::analyze::collect_inputs;
use super::prelude::*;
use proteus_core::variant::{
    diff, Change, DiffOptions, InteractionKind, PreparedStructure, VariantDiff,
};
use std::io::Write;

/// Arguments of `proteus diff`.
#[derive(clap::Args, Debug)]
#[command(after_help = "Examples:
  proteus diff --parent wt.pdb design_07.pdb
  proteus diff --parent scaffold.cif designs/ --csv triage.csv
  proteus diff --parent 1b27.pdb 1b3s.pdb --chains A,D --json | jq .summary")]
pub struct Args {
    /// The structure the candidates are compared with (wild type, scaffold, previous best)
    #[arg(long, value_name = "FILE")]
    pub parent: PathBuf,

    /// Candidate structures (PDB or mmCIF, optionally gzipped) or directories of them
    #[arg(value_name = "CANDIDATE", required = true)]
    pub candidates: Vec<PathBuf>,

    /// Compare only these chains, in both structures (e.g. one copy of a crystal's complex)
    #[arg(long, value_delimiter = ',', value_name = "IDS")]
    pub chains: Option<Vec<String>>,

    /// Interaction changes within this many Å of a changed residue are local; the rest are
    /// counted as distal and not listed
    #[arg(long, default_value_t = proteus_core::variant::DEFAULT_LOCAL_RADIUS, value_name = "Å")]
    pub local_radius: f64,

    /// Print one JSON object per candidate (JSON Lines) instead of tables
    #[arg(long)]
    pub json: bool,

    /// List marginal local changes too (geometry within 0.5 Å of a cut-off, or only an angle
    /// changed), not only the clear ones
    #[arg(long)]
    pub all: bool,

    /// Write one row per candidate to a CSV file (the columns a ranking can sort on)
    #[arg(long, value_name = "FILE")]
    pub csv: Option<PathBuf>,
}

pub async fn run(args: Args) -> Result<()> {
    let opts = DiffOptions {
        chains: args.chains.clone(),
        local_radius: args.local_radius,
    };
    let parent = prepare(&args.parent, &opts)?;
    let (files, unreadable) = collect_inputs(&args.candidates)?;
    for (p, why) in &unreadable {
        eprintln!("skipped {}: {why}", p.display());
    }
    let mut rows: Vec<(PathBuf, VariantDiff)> = Vec::new();
    let mut failed = unreadable.len();
    for f in &files {
        match prepare(f, &opts).and_then(|c| Ok(diff(&parent, &c, &opts)?)) {
            Ok(d) => rows.push((f.clone(), d)),
            Err(e) => {
                failed += 1;
                eprintln!("{}: {e:#}", f.display());
            }
        }
    }
    if let Some(csv) = &args.csv {
        write_csv(csv, &rows)?;
        eprintln!("wrote {} row(s) to {}", rows.len(), csv.display());
    }
    if args.json {
        let out = std::io::stdout();
        let mut out = out.lock();
        for (path, d) in &rows {
            let mut v = serde_json::to_value(d)?;
            v["candidate"] = serde_json::Value::String(path.display().to_string());
            v["parent"] = serde_json::Value::String(args.parent.display().to_string());
            writeln!(out, "{}", serde_json::to_string(&v)?)?;
        }
    } else if rows.len() == 1 {
        print_one(&rows[0].0, &rows[0].1, args.all);
    } else if !rows.is_empty() {
        print_table(&rows);
    }
    if rows.is_empty() {
        bail!("no candidate could be compared with the parent");
    }
    if failed > 0 {
        eprintln!("{failed} input(s) could not be compared");
    }
    Ok(())
}

pub(crate) fn prepare(path: &Path, opts: &DiffOptions) -> Result<PreparedStructure> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let name = path.file_name().and_then(|n| n.to_str());
    PreparedStructure::from_bytes(&bytes, name, opts.chains.as_deref())
        .with_context(|| format!("{}", path.display()))
}

/// `−2 H-bonds −1 salt bridge`, for one direction of change; empty when nothing changed.
pub(crate) fn counts_phrase(d: &VariantDiff, change: Change) -> String {
    let counts = match change {
        Change::Lost => &d.summary.lost_local,
        Change::Gained => &d.summary.gained_local,
    };
    let sign = match change {
        Change::Lost => '−',
        Change::Gained => '+',
    };
    InteractionKind::ALL
        .iter()
        .filter(|k| counts.get(**k) > 0)
        .map(|k| format!("{sign}{} {}", counts.get(*k), k.label(counts.get(*k))))
        .collect::<Vec<_>>()
        .join(" ")
}

fn confidence_cell(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.0}")).unwrap_or_else(|| "–".into())
}

fn print_one(path: &Path, d: &VariantDiff, all: bool) {
    println!(
        "{}: {} changed residue(s), C-alpha RMSD {:.2} Å over {} paired ({}; identity {:.0} %, \
         coverage {:.0} %)",
        path.display(),
        d.sites.len(),
        d.global_ca_rmsd,
        d.paired,
        d.pairing,
        d.identity * 100.0,
        d.coverage * 100.0
    );
    if d.coverage < 0.9 || d.identity < 0.3 {
        println!("  note: the two structures share little; check that the parent is the right one");
    }
    if !d.candidate_confidence.is_predicted() {
        println!("  confidence: the candidate's B-factor column is not a predicted confidence");
    }
    if d.sites.is_empty() {
        println!("  no residue differs from the parent");
    } else {
        let mut t = Table::new();
        t.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
        t.set_header(vec![
            "site",
            "chain",
            "group",
            "local RMSD Å",
            "pLDDT parent",
            "pLDDT cand.",
        ]);
        for s in &d.sites {
            t.add_row(vec![
                Cell::new(&s.notation),
                Cell::new(&s.candidate_chain_id),
                Cell::new(s.group + 1),
                Cell::new(format!("{:.2}", s.local_ca_rmsd)),
                Cell::new(confidence_cell(s.parent_confidence)),
                Cell::new(confidence_cell(s.candidate_confidence)),
            ]);
        }
        fit_table(&mut t);
        println!("{t}");
    }
    let local: Vec<_> = d
        .changes
        .iter()
        .filter(|c| c.local && (all || c.clear))
        .collect();
    if local.is_empty() {
        println!(
            "  no interaction changed within {:.0} Å of a site",
            d.local_radius
        );
    } else {
        let mut t = Table::new();
        t.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
        t.set_header(vec![
            "",
            "interaction",
            "between",
            "Å here",
            "Å in other",
            "note",
        ]);
        for c in local {
            let other = match c.change {
                Change::Lost => "candidate",
                Change::Gained => "parent",
            };
            let mut notes = Vec::new();
            if !c.removed.is_empty() {
                notes.push(format!("removed: {}", c.removed.join(", ")));
            }
            if !c.unmodelled.is_empty() {
                notes.push(format!(
                    "not modelled in {other}: {}",
                    c.unmodelled.join(", ")
                ));
            }
            let note = notes.join("; ");
            t.add_row(vec![
                Cell::new(match (c.change, c.clear) {
                    (Change::Lost, true) => "lost",
                    (Change::Gained, true) => "gained",
                    (Change::Lost, false) => "lost?",
                    (Change::Gained, false) => "gained?",
                }),
                Cell::new(format!(
                    "{}{}",
                    c.kind.label(1),
                    if c.interface { " (interface)" } else { "" }
                )),
                Cell::new(format!("{} · {}", c.a.label(), c.b.label())),
                Cell::new(format!("{:.1}", c.distance)),
                Cell::new(
                    c.other_distance
                        .map(|x| format!("{x:.1}"))
                        .unwrap_or_else(|| "–".into()),
                ),
                Cell::new(note),
            ]);
        }
        fit_table(&mut t);
        println!("{t}");
    }
    if !all && d.summary.marginal_local > 0 {
        println!(
            "  {} marginal local change(s) not listed: the same atoms are within 0.5 Å of the \
             cut-off, only an angle changed, or one file did not model the side chain (--all \
             lists them)",
            d.summary.marginal_local
        );
    }
    let distal = d.summary.lost_distal.total() + d.summary.gained_distal.total();
    if distal > 0 {
        println!(
            "  {distal} interaction change(s) farther than {:.0} Å from any site (not listed: at \
             that distance they are usually coordinate noise)",
            d.local_radius
        );
    }
}

fn print_table(rows: &[(PathBuf, VariantDiff)]) {
    let mut t = Table::new();
    t.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    t.set_header(vec![
        "candidate",
        "sites",
        "RMSD Å",
        "lost (local)",
        "gained (local)",
        "interface −/+",
        "min site pLDDT",
    ]);
    for (p, d) in rows {
        t.add_row(vec![
            Cell::new(
                p.file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default(),
            ),
            Cell::new(d.sites.len()),
            Cell::new(format!("{:.2}", d.global_ca_rmsd)),
            Cell::new(counts_phrase(d, Change::Lost)),
            Cell::new(counts_phrase(d, Change::Gained)),
            Cell::new(format!(
                "{}/{}",
                d.summary.interface_lost_local, d.summary.interface_gained_local
            )),
            Cell::new(confidence_cell(d.summary.min_site_confidence)),
        ]);
    }
    fit_table(&mut t);
    println!("{t}");
}

/// The CSV columns, in order. Counts are local unless named distal.
const CSV_COLUMNS: usize = 24;
const CSV_HEADER: &str = "candidate,pairing,paired,identity,coverage,inserted,deleted,sites,\
global_ca_rmsd,lost_hbonds,lost_salt_bridges,lost_pi_stacking,lost_cation_pi,gained_hbonds,\
gained_salt_bridges,gained_pi_stacking,gained_cation_pi,interface_lost,interface_gained,\
marginal_local,distal_lost,distal_gained,min_site_plddt,mean_site_plddt_change";

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn write_csv(path: &Path, rows: &[(PathBuf, VariantDiff)]) -> Result<()> {
    let opt = |v: Option<f64>| v.map(|x| format!("{x:.2}")).unwrap_or_default();
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for (p, d) in rows {
        let s = &d.summary;
        let fields: [String; CSV_COLUMNS] = [
            csv_field(&p.display().to_string()),
            csv_field(&d.pairing),
            d.paired.to_string(),
            format!("{:.4}", d.identity),
            format!("{:.4}", d.coverage),
            d.inserted.to_string(),
            d.deleted.to_string(),
            d.sites.len().to_string(),
            format!("{:.3}", d.global_ca_rmsd),
            s.lost_local.hydrogen_bonds.to_string(),
            s.lost_local.salt_bridges.to_string(),
            s.lost_local.pi_stacking.to_string(),
            s.lost_local.cation_pi.to_string(),
            s.gained_local.hydrogen_bonds.to_string(),
            s.gained_local.salt_bridges.to_string(),
            s.gained_local.pi_stacking.to_string(),
            s.gained_local.cation_pi.to_string(),
            s.interface_lost_local.to_string(),
            s.interface_gained_local.to_string(),
            s.marginal_local.to_string(),
            s.lost_distal.total().to_string(),
            s.gained_distal.total().to_string(),
            opt(s.min_site_confidence),
            opt(s.mean_site_confidence_change),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_and_rows_have_the_same_width() {
        // Rows are `[String; CSV_COLUMNS]`, so the compiler holds them to the count.
        assert_eq!(CSV_HEADER.split(',').count(), CSV_COLUMNS);
    }

    #[test]
    fn csv_fields_with_commas_are_quoted() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"x\", ok"), "\"say \"\"x\"\", ok\"");
        assert_eq!(csv_field("plain"), "plain");
    }
}
