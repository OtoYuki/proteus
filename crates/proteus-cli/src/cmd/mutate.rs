//! `proteus mutate` — Generate an in-silico variant library from a scaffold.

use super::prelude::*;

/// Arguments of `proteus mutate`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Path to scaffold FASTA file (or '-' for stdin)
    scaffold: String,

    /// Mutagenesis mode
    #[arg(short, long, value_enum, default_value_t = CliMutagenesisMode::Alanine)]
    mode: CliMutagenesisMode,

    /// 1-indexed window start position (inclusive)
    #[arg(long)]
    start: Option<usize>,

    /// 1-indexed window end position (inclusive)
    #[arg(long)]
    end: Option<usize>,

    /// Maximum number of mutant variants to generate
    #[arg(long)]
    max_variants: Option<usize>,

    /// Exclude the unmutated wildtype scaffold from output library
    #[arg(long)]
    no_wt: bool,

    /// Custom prefix for variant sequence headers
    #[arg(long)]
    prefix: Option<String>,

    /// Optional output file path for multi-FASTA library (writes to stdout if omitted)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

pub async fn run(args: Args) -> Result<()> {
    let Args {
        scaffold,
        mode,
        start,
        end,
        max_variants,
        no_wt,
        prefix,
        output,
    } = args;
    let content: String = if scaffold == "-" {
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        tokio::io::stdin()
            .read_to_string(&mut buf)
            .await
            .context("Failed to read scaffold FASTA from stdin")?;
        buf
    } else {
        let p = Path::new(&scaffold);
        tokio::fs::read_to_string(p)
            .await
            .with_context(|| format!("Failed to read scaffold file at {:?}", p))?
    };

    let seq = proteus_core::sequence::validate_and_parse_fasta(&content)
        .context("Scaffold sequence validation failed")?;

    let config = proteus_core::mutagenesis::MutagenesisConfig {
        mode: mode.into(),
        window_start: start,
        window_end: end,
        max_variants,
        include_wildtype: !no_wt,
        prefix,
    };

    let library = proteus_core::mutagenesis::generate_mutant_library(&seq, &config)
        .context("Mutant variant generation failed")?;

    let formatted = proteus_core::sequence::format_multi_fasta(&library);

    if let Some(out_path) = output {
        tokio::fs::write(&out_path, &formatted)
            .await
            .with_context(|| format!("Failed to write mutant library to {:?}", out_path))?;
        eprintln!(
            "Generated {} variant sequences (scaffold len: {}) -> {:?}",
            library.len(),
            seq.length,
            out_path
        );
    } else {
        print!("{formatted}");
    }
    Ok(())
}
