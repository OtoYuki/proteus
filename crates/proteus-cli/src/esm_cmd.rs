//! `proteus esm …` subcommands and the ESM-2 scorer used by `proteus screen --scorer esm2`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};
use proteus_core::models::Sequence;
use proteus_esm::{parse_mutation, Device, Esm2, MarginalScorer, Mutation, AMINO_ACIDS};
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

/// Where `--esm-model` points: a local checkpoint directory or a Hub id.
#[derive(Debug, PartialEq, Eq)]
enum ModelSource<'a> {
    Dir(&'a std::path::Path),
    Hub(&'a str),
}

/// A Hub id is `<owner>/<name>`, which is also a valid relative path; anything that can only be
/// a path and is not a directory is reported as such, not sent to the Hub as an id.
fn model_source(model: &str) -> Result<ModelSource<'_>> {
    let path = std::path::Path::new(model);
    if path.is_dir() {
        return Ok(ModelSource::Dir(path));
    }
    if path.exists() {
        bail!(
            "'{model}' is a file; --esm-model takes a directory holding config.json and \
             model.safetensors, or a Hub id such as {DEFAULT_MODEL}"
        );
    }
    let path_like = model.starts_with(['.', '/', '~'])
        || model.contains('\\')
        || model.matches('/').count() != 1;
    if path_like {
        bail!(
            "no such directory: '{model}' (--esm-model takes a directory holding config.json \
             and model.safetensors, or a Hub id such as {DEFAULT_MODEL})"
        );
    }
    Ok(ModelSource::Hub(model))
}

pub fn load_model(opts: &EsmOptions) -> Result<Esm2> {
    let model = match model_source(&opts.model)? {
        ModelSource::Dir(path) => Esm2::from_files(
            &path.join("config.json"),
            &path.join("model.safetensors"),
            &Device::Cpu,
        ),
        ModelSource::Hub(id) => Esm2::from_hub(id, &Device::Cpu),
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
        tokio::fs::read_to_string(arg)
            .await
            .with_context(|| format!("cannot read {arg} as a FASTA file"))?
    } else {
        return Ok(("wildtype".into(), raw_residues(arg)?));
    };
    let seqs =
        proteus_core::sequence::validate_and_parse_multi_fasta(&text).context("parsing FASTA")?;
    let first = seqs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no sequence in input"))?;
    Ok((first.header, first.fasta))
}

/// A wild-type argument that is not a file: residues only if every character is an amino-acid
/// letter. A mistyped file name (`wt.fast`) used to be upper-cased and scored as a protein.
fn raw_residues(arg: &str) -> Result<String> {
    let seq = proteus_esm::Tokenizer::normalize(arg)
        .map_err(|_| anyhow::anyhow!("'{arg}': no such file, and not a protein sequence"))?;
    // A short all-letter word may still be a file name without its extension.
    if seq.len() < 30 {
        eprintln!(
            "note: '{arg}' is not a file; scoring it as a {}-residue sequence",
            seq.len()
        );
    }
    Ok(seq)
}

fn score_mutations(model: &Esm2, wt: &str, muts: &[Mutation], masked: bool) -> Result<Vec<f32>> {
    Ok(if masked {
        proteus_esm::score_masked_marginal(model, wt, muts)?
    } else {
        proteus_esm::score_wt_marginal(model, wt, muts)?
    })
}

/// Say where a zero-shot ESM-2 score is known to be weak, once, at the point of use.
///
/// ProteinGym's per-taxon breakdown puts ESM-2 650M at Spearman ρ ≈ 0.46 on human assays and
/// ≈ 0.26 on viral ones; taxon cannot be told from a sequence, so the README carries that. What
/// can be told is length: on a long sequence the scores come from one context over several
/// domains, and scoring each known domain on its own is the conservative choice.
pub fn warn_if_outside_known_good(wt: &str) {
    const LONG_MULTI_DOMAIN: usize = 400;
    if wt.len() > LONG_MULTI_DOMAIN {
        eprintln!(
            "note: {} residues. Zero-shot ESM-2 scores are a triage signal; for a multi-domain \
             protein, scoring each known domain separately is the conservative choice.",
            wt.len()
        );
    }
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
            warn_if_outside_known_good(&wt);
            let scores = score_mutations(&model, &wt, &muts, opts.masked)?;
            let mut table = Table::new();
            table.load_style(UTF8_FULL);
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
            // Check where the scan will go before spending the forward passes on it.
            if let Some(path) = &export {
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("cannot create {}", parent.display()))?;
                }
                if path.is_dir() {
                    anyhow::bail!(
                        "--export {} is a directory; give a file name",
                        path.display()
                    );
                }
            }
            let model = load_model(&opts)?;
            warn_if_outside_known_good(&wt);
            let rows = proteus_esm::scan(&model, &wt, opts.masked)?;
            let unscanned = wt.len() - rows.len();
            if unscanned > 0 {
                eprintln!(
                    "note: {unscanned} positions with a non-standard wild-type residue \
                     (X/B/Z/U/O) have no substitution scores"
                );
            }
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
                tokio::fs::write(path, out)
                    .await
                    .with_context(|| format!("cannot write {}", path.display()))?;
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
            table.load_style(UTF8_FULL);
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
    // Rows keep their true positions; a non-standard wild-type residue has none.
    let first = rows.first().map_or(1, |r| r.pos).to_string();
    println!(
        "   {first}{:>w$}  (red = deleterious, blue = tolerated, · = wild type){}",
        rows.last().map_or(0, |r| r.pos),
        if step > 1 {
            format!("; {step} positions per column")
        } else {
            String::new()
        },
        w = width.div_ceil(step).saturating_sub(first.len())
    );
}

/// ESM-2 score per library entry for `proteus screen`.
///
/// The wild type is the entry whose header contains `[wildtype]` (as `proteus mutate` writes),
/// else the first entry. Each variant's substitutions come from `[mutation=P19A]` in its header
/// or, failing that, from a position-wise diff against the wild type (same length only).
/// Returns `None` for entries that cannot be related to the wild type.
/// Substitutions declared in a header's `[mutation=P19A,C4S]` tag, if any. `:` (ProteinGym's
/// multi-mutant separator), `,`, `/` and `;` all separate substitutions; one variant may not
/// mutate a position twice.
fn tagged_mutations(header: &str) -> Result<Option<Vec<Mutation>>> {
    let Some(m) = header
        .split("[mutation=")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
    else {
        return Ok(None);
    };
    let muts = m
        .split([',', '/', ';', ':'])
        .filter(|x| !x.trim().is_empty())
        .map(|x| parse_mutation(x).map_err(|e| anyhow::anyhow!("'{header}': {e}")))
        .collect::<Result<Vec<_>>>()?;
    let mut seen = std::collections::BTreeSet::new();
    if let Some(dup) = muts.iter().find(|m| !seen.insert(m.pos)) {
        bail!(
            "'{header}': position {} is mutated more than once in one variant",
            dup.pos
        );
    }
    Ok(Some(muts))
}

/// The wild-type sequence a library is scored against: the `[wildtype]` entry when there is
/// one; otherwise the first entry, with its own `[mutation=…]` tag reverted (a `mutate --no-wt`
/// library never carries the scaffold itself).
pub fn wild_type_of(sequences: &[Sequence]) -> Result<String> {
    if let Some(wt) = sequences.iter().find(|s| s.header.contains("[wildtype]")) {
        return Ok(wt.fasta.clone());
    }
    let first = sequences
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty library"))?;
    let Some(muts) = tagged_mutations(&first.header)? else {
        return Ok(first.fasta.clone());
    };
    let mut chars: Vec<char> = first.fasta.chars().collect();
    for m in &muts {
        let slot = chars.get_mut(m.pos - 1).ok_or_else(|| {
            anyhow::anyhow!(
                "'{}': position {} is beyond the sequence length {}",
                first.header,
                m.pos,
                first.fasta.len()
            )
        })?;
        if *slot != m.mt {
            bail!(
                "'{}': header says {} but residue {} is {}, not {}",
                first.header,
                m,
                m.pos,
                *slot,
                m.mt
            );
        }
        *slot = m.wt;
    }
    Ok(chars.into_iter().collect())
}

pub fn score_library(
    sequences: &[Sequence],
    opts: &EsmOptions,
) -> Result<HashMap<Uuid, Option<f32>>> {
    let model = load_model(opts)?;
    let wt_fasta = wild_type_of(sequences)?;
    // One forward pass for the whole library (wild-type marginals) or one per mutated
    // position (masked), instead of one per variant.
    warn_if_outside_known_good(&wt_fasta);
    let mut scorer = MarginalScorer::new(&model, &wt_fasta, opts.masked)?;
    let wt_header = sequences
        .iter()
        .find(|s| s.header.contains("[wildtype]"))
        .map(|s| s.header.as_str())
        .unwrap_or("(reconstructed from the first entry's mutation tag)");
    eprintln!("ESM-2 wild type: {wt_header}");
    let mut out = HashMap::with_capacity(sequences.len());
    let mut skipped = 0usize;
    for s in sequences {
        let muts: Vec<Mutation> = if let Some(tagged) = tagged_mutations(&s.header)? {
            tagged
        } else if s.fasta.len() == wt_fasta.len() {
            wt_fasta
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
            scorer
                .score(&muts)
                .with_context(|| format!("scoring '{}'", s.header))?
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

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(header: &str, fasta: &str) -> Sequence {
        Sequence {
            id: Uuid::new_v4(),
            header: header.into(),
            fasta: fasta.into(),
            length: fasta.len(),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn wild_type_is_the_tagged_entry_when_present() {
        let lib = vec![
            seq("x_T1A [mutation=T1A]", "ACDE"),
            seq("x_WT [wildtype]", "TCDE"),
        ];
        assert_eq!(wild_type_of(&lib).unwrap(), "TCDE");
    }

    #[test]
    fn wild_type_is_reconstructed_from_mutation_tags_when_no_entry_is_tagged() {
        // `proteus mutate --no-wt` output: every entry is a single substitution of the same scaffold.
        let lib = vec![
            seq("v_T1A [mutation=T1A]", "ACDE"),
            seq("v_C2A [mutation=C2A]", "TADE"),
        ];
        assert_eq!(wild_type_of(&lib).unwrap(), "TCDE");
    }

    #[test]
    fn wild_type_falls_back_to_the_first_untagged_entry() {
        let lib = vec![seq("scaffold", "TCDE"), seq("variant", "TADE")];
        assert_eq!(wild_type_of(&lib).unwrap(), "TCDE");
    }

    #[test]
    fn a_missing_file_is_not_scored_as_a_protein() {
        // `wt.fast` (a typo of wt.fasta) used to become the 7-residue protein "WT.FAST".
        for bad in ["wt.fast", "./wildtype", "data/wt.fasta", "P12345_1"] {
            let err = raw_residues(bad).unwrap_err().to_string();
            assert!(err.contains("no such file"), "{bad}: {err}");
        }
        assert_eq!(raw_residues("mktay iakqr*").unwrap(), "MKTAYIAKQR");
    }

    #[test]
    fn a_mistyped_model_directory_is_not_sent_to_the_hub() {
        for bad in [
            "./models/esm2",
            "../esm2",
            "/no/such/esm2",
            "models/esm2/t6",
            "~/esm2",
        ] {
            let err = model_source(bad).unwrap_err().to_string();
            assert!(err.contains("no such directory"), "{bad}: {err}");
        }
        let manifest = env!("CARGO_MANIFEST_DIR");
        let file = format!("{manifest}/Cargo.toml");
        assert!(model_source(&file)
            .unwrap_err()
            .to_string()
            .contains("is a file"));
        assert_eq!(
            model_source(manifest).unwrap(),
            ModelSource::Dir(std::path::Path::new(manifest))
        );
        assert_eq!(
            model_source(DEFAULT_MODEL).unwrap(),
            ModelSource::Hub(DEFAULT_MODEL)
        );
    }

    #[test]
    fn proteingym_multi_mutant_tags_parse() {
        // ProteinGym writes multi-mutants as A10G:C4S; the ':' used to abort the whole screen.
        let muts = tagged_mutations("v [mutation=A10G:C4S]").unwrap().unwrap();
        assert_eq!(muts.len(), 2);
        assert_eq!((muts[0].pos, muts[1].pos), (10, 4));
        assert_eq!(
            tagged_mutations("v [mutation=A10G,C4S;D5E/F6G]")
                .unwrap()
                .unwrap()
                .len(),
            4
        );
        assert!(tagged_mutations("no tag").unwrap().is_none());
    }

    #[test]
    fn a_variant_tag_is_strict() {
        // Both substitutions of A10G:A10C were scored and summed.
        let err = tagged_mutations("v [mutation=A10G:A10C]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("more than once"), "{err}");
        assert!(tagged_mutations("v [mutation=A+10G]").is_err());
        assert!(tagged_mutations("v [mutation=A10J]").is_err());
    }

    #[test]
    fn inconsistent_mutation_tag_is_an_error() {
        let lib = vec![seq("v_T1A [mutation=T1A]", "GCDE")];
        assert!(wild_type_of(&lib).is_err());
    }
}
