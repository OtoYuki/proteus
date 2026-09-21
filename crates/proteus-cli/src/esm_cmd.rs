//! `proteus esm …` subcommands and the ESM-2 scorer used by `proteus screen --scorer esm2`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};
use proteus_core::models::Sequence;
use proteus_esm::{parse_mutation, Device, Esm2, Mutation, AMINO_ACIDS};
use uuid::Uuid;

pub const DEFAULT_MODEL: &str = "facebook/esm2_t6_8M_UR50D";

/// Which signal ranks the screening leaderboard.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug, Default)]
pub enum Scorer {
    /// Structure-only composite fitness (default).
    #[default]
    Structure,
    /// ESM-2 zero-shot mutation score, summed over a variant's substitutions.
    Esm2,
    /// 0.7 · structure fitness + 0.3 · 100·sigmoid(ESM-2 score).
    Hybrid,
}

#[derive(Args, Debug, Clone)]
pub struct EsmOptions {
    /// Hub id or local directory (config.json + model.safetensors) of the ESM-2 checkpoint
    #[arg(long = "esm-model", default_value = DEFAULT_MODEL, global = true)]
    pub model: String,
    /// Masked marginals (one forward pass per mutated position) instead of wild-type marginals
    #[arg(long = "esm-masked", global = true)]
    pub masked: bool,
}

#[derive(Subcommand, Debug)]
pub enum EsmCommand {
    /// Score point mutations of a wild-type sequence (FASTA path, '-' for stdin, or raw residues)
    Score {
        wildtype: String,
        /// Comma-separated substitutions in P19A notation
        #[arg(long, value_delimiter = ',')]
        mutations: Vec<String>,
    },
    /// Full deep mutational scan (20 × L) of a wild-type sequence
    Scan {
        wildtype: String,
        /// Write the matrix as CSV (rows: position, wt, then one column per amino acid)
        #[arg(long)]
        export: Option<PathBuf>,
        /// How many best/worst substitutions to list
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
}

pub fn load_model(opts: &EsmOptions) -> Result<Esm2> {
    let path = std::path::Path::new(&opts.model);
    let model = if path.is_dir() {
        Esm2::from_files(
            &path.join("config.json"),
            &path.join("model.safetensors"),
            &Device::Cpu,
        )
    } else {
        Esm2::from_hub(&opts.model, &Device::Cpu)
    }
    .with_context(|| format!("loading ESM-2 model '{}'", opts.model))?;
    let c = model.config();
    eprintln!(
        "ESM-2 {}: {} layers, hidden {}, {} heads ({} marginals)",
        opts.model,
        c.num_hidden_layers,
        c.hidden_size,
        c.num_attention_heads,
        if opts.masked { "masked" } else { "wild-type" }
    );
    Ok(model)
}

/// Accept a FASTA file, '-' (stdin) or a bare residue string.
async fn read_wildtype(arg: &str) -> Result<(String, String)> {
    let text = if arg == "-" {
        use tokio::io::AsyncReadExt;
        let mut s = String::new();
        tokio::io::stdin().read_to_string(&mut s).await?;
        s
    } else if std::path::Path::new(arg).exists() {
        tokio::fs::read_to_string(arg).await?
    } else {
        return Ok(("wildtype".into(), arg.trim().to_ascii_uppercase()));
    };
    let seqs =
        proteus_core::sequence::validate_and_parse_multi_fasta(&text).context("parsing FASTA")?;
    let first = seqs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no sequence in input"))?;
    Ok((first.header, first.fasta))
}

fn score_mutations(model: &Esm2, wt: &str, muts: &[Mutation], masked: bool) -> Result<Vec<f32>> {
    Ok(if masked {
        proteus_esm::score_masked_marginal(model, wt, muts)?
    } else {
        proteus_esm::score_wt_marginal(model, wt, muts)?
    })
}

pub async fn run(cmd: EsmCommand, opts: EsmOptions) -> Result<()> {
    match cmd {
        EsmCommand::Score {
            wildtype,
            mutations,
        } => {
            let (header, wt) = read_wildtype(&wildtype).await?;
            let muts: Vec<Mutation> = mutations
                .iter()
                .filter(|m| !m.trim().is_empty())
                .map(|m| parse_mutation(m).map_err(|e| anyhow::anyhow!("{e}")))
                .collect::<Result<_>>()?;
            if muts.is_empty() {
                bail!("--mutations is required, e.g. --mutations P19A,C4S");
            }
            let model = load_model(&opts)?;
            let scores = score_mutations(&model, &wt, &muts, opts.masked)?;
            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
            table.set_header(vec!["Mutation", "ESM-2 score (log p_mt − log p_wt)"]);
            for (m, s) in muts.iter().zip(&scores) {
                table.add_row(vec![
                    Cell::new(m.to_string()),
                    Cell::new(format!("{s:+.3}")),
                ]);
            }
            println!("{header} ({} residues)\n{table}", wt.len());
        }
        EsmCommand::Scan {
            wildtype,
            export,
            top,
        } => {
            let (header, wt) = read_wildtype(&wildtype).await?;
            let model = load_model(&opts)?;
            let rows = proteus_esm::scan(&model, &wt, opts.masked)?;
            if let Some(path) = &export {
                let mut out = String::from("position,wt");
                for aa in AMINO_ACIDS {
                    out.push(',');
                    out.push(aa);
                }
                out.push('\n');
                for r in &rows {
                    out.push_str(&format!("{},{}", r.pos, r.wt));
                    for s in r.scores {
                        out.push_str(&format!(",{s:.4}"));
                    }
                    out.push('\n');
                }
                tokio::fs::write(path, out).await?;
                eprintln!("wrote {} × 20 scan -> {}", rows.len(), path.display());
            }
            let mut all: Vec<(Mutation, f32)> = rows
                .iter()
                .flat_map(|r| {
                    AMINO_ACIDS.iter().zip(r.scores).filter_map(move |(aa, s)| {
                        (*aa != r.wt).then_some((
                            Mutation {
                                wt: r.wt,
                                pos: r.pos,
                                mt: *aa,
                            },
                            s,
                        ))
                    })
                })
                .collect();
            all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let n = all.len();
            println!(
                "{header} ({} residues, {} substitutions scored)",
                wt.len(),
                n
            );
            print_heatmap(&rows);
            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
            table.set_header(vec!["Most tolerated", "score", "Least tolerated", "score"]);
            for i in 0..top.min(n) {
                let (best, bs) = &all[i];
                let (worst, ws) = &all[n - 1 - i];
                table.add_row(vec![
                    Cell::new(best.to_string()),
                    Cell::new(format!("{bs:+.2}")),
                    Cell::new(worst.to_string()),
                    Cell::new(format!("{ws:+.2}")),
                ]);
            }
            println!("{table}");
        }
    }
    Ok(())
}

/// 20-row text heat map: rows are amino acids, columns are positions (one cell per residue,
/// truecolor from red (deleterious) through grey (neutral) to blue (tolerated)).
fn print_heatmap(rows: &[proteus_esm::ScanRow]) {
    let width = rows.len();
    let max_cols = 160;
    let step = width.div_ceil(max_cols).max(1);
    let scale = |s: f32| -> (u8, u8, u8) {
        let t = (s / 6.0).clamp(-1.0, 1.0);
        if t < 0.0 {
            let k = -t;
            (
                (60.0 + 190.0 * k) as u8,
                (60.0 - 40.0 * k) as u8,
                (60.0 - 30.0 * k) as u8,
            )
        } else {
            (
                (60.0 - 30.0 * t) as u8,
                (60.0 + 70.0 * t) as u8,
                (60.0 + 190.0 * t) as u8,
            )
        }
    };
    for (k, aa) in AMINO_ACIDS.iter().enumerate() {
        let mut line = format!(" {aa} ");
        for chunk in rows.chunks(step) {
            let s = chunk.iter().map(|r| r.scores[k]).sum::<f32>() / chunk.len() as f32;
            let (r, g, b) = scale(s);
            let ch = if chunk.iter().any(|r| r.wt == *aa) {
                '·'
            } else {
                '█'
            };
            line.push_str(&format!("\x1b[38;2;{r};{g};{b}m{ch}\x1b[0m"));
        }
        println!("{line}");
    }
    println!(
        "   1{:>w$}  (red = deleterious, blue = tolerated, · = wild type){}",
        width,
        if step > 1 {
            format!("; {step} positions per column")
        } else {
            String::new()
        },
        w = width.div_ceil(step).saturating_sub(1)
    );
}

/// ESM-2 score per library entry for `proteus screen`.
///
/// The wild type is the entry whose header contains `[wildtype]` (as `proteus mutate` writes),
/// else the first entry. Each variant's substitutions come from `[mutation=P19A]` in its header
/// or, failing that, from a position-wise diff against the wild type (same length only).
/// Returns `None` for entries that cannot be related to the wild type.
pub fn score_library(
    sequences: &[Sequence],
    opts: &EsmOptions,
) -> Result<HashMap<Uuid, Option<f32>>> {
    let model = load_model(opts)?;
    let wt = sequences
        .iter()
        .find(|s| s.header.contains("[wildtype]"))
        .or_else(|| sequences.first())
        .ok_or_else(|| anyhow::anyhow!("empty library"))?;
    eprintln!("ESM-2 wild type: {}", wt.header);
    let mut out = HashMap::with_capacity(sequences.len());
    let mut skipped = 0usize;
    for s in sequences {
        let muts: Vec<Mutation> = if let Some(m) = s
            .header
            .split("[mutation=")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
        {
            m.split(|c| c == ',' || c == '/' || c == ';')
                .filter(|x| !x.trim().is_empty())
                .map(|x| parse_mutation(x).map_err(|e| anyhow::anyhow!("{e}")))
                .collect::<Result<_>>()?
        } else if s.fasta.len() == wt.fasta.len() {
            wt.fasta
                .chars()
                .zip(s.fasta.chars())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| Mutation {
                    wt: a,
                    pos: i + 1,
                    mt: b,
                })
                .collect()
        } else {
            skipped += 1;
            out.insert(s.id, None);
            continue;
        };
        let score = if muts.is_empty() {
            0.0
        } else {
            score_mutations(&model, &wt.fasta, &muts, opts.masked)?
                .iter()
                .sum()
        };
        out.insert(s.id, Some(score));
    }
    if skipped > 0 {
        eprintln!(
            "ESM-2: {skipped} entries skipped (length differs from wild type, no [mutation=] tag)"
        );
    }
    Ok(out)
}

/// Hybrid ranking: structure fitness and a sigmoid-squashed ESM-2 score on a 0–100 scale.
pub fn hybrid(fitness: f64, esm: f32) -> f64 {
    let squashed = 100.0 / (1.0 + (-(esm as f64)).exp());
    0.7 * fitness + 0.3 * squashed
}
