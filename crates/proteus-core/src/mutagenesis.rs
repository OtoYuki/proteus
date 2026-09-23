use crate::error::CoreError;
use crate::models::Sequence;
use crate::sequence::STANDARD_AMINO_ACIDS;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Point mutation descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mutation {
    /// 1-indexed position along the sequence
    pub pos: usize,
    /// Wildtype amino acid single-letter code
    pub wt: char,
    /// Mutant amino acid single-letter code
    pub mutant: char,
}

impl Mutation {
    pub fn new(pos: usize, wt: char, mutant: char) -> Self {
        Self { pos, wt, mutant }
    }

    /// Standard biochemical notation: e.g., "K48A"
    pub fn notation(&self) -> String {
        format!("{}{}{}", self.wt, self.pos, self.mutant)
    }
}

/// Mutagenesis strategy for variant generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutagenesisMode {
    /// Systematic alanine scanning across residues.
    /// Non-alanine residues mutate to 'A'; alanine residues mutate to 'G'.
    AlanineScanning,
    /// Full site-saturation mutagenesis (19 non-wildtype substitutions per position).
    Saturation,
}

/// Configuration parameters for in-silico variant library generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutagenesisConfig {
    pub mode: MutagenesisMode,
    /// Optional 1-indexed start position (inclusive). Defaults to 1.
    pub window_start: Option<usize>,
    /// Optional 1-indexed end position (inclusive). Defaults to sequence length.
    pub window_end: Option<usize>,
    /// Optional upper bound on the number of mutants generated. The wild type, when
    /// included, does not count against it.
    pub max_variants: Option<usize>,
    /// Whether to include the unmutated wildtype scaffold as candidate #0.
    pub include_wildtype: bool,
    /// Optional header prefix (defaults to the scaffold sequence's header).
    pub prefix: Option<String>,
}

impl Default for MutagenesisConfig {
    fn default() -> Self {
        Self {
            mode: MutagenesisMode::AlanineScanning,
            window_start: None,
            window_end: None,
            max_variants: None,
            include_wildtype: true,
            prefix: None,
        }
    }
}

/// In-silico Deep Mutational Scanning (DMS) engine.
/// Generates a structured multi-variant sequence library from a wildtype scaffold.
pub fn generate_mutant_library(
    scaffold: &Sequence,
    config: &MutagenesisConfig,
) -> Result<Vec<Sequence>, CoreError> {
    let seq_chars: Vec<char> = scaffold.fasta.chars().collect();
    let len = seq_chars.len();

    if len == 0 {
        return Err(CoreError::InvalidFasta(
            "Cannot generate mutants from an empty scaffold sequence".into(),
        ));
    }

    let start = config.window_start.unwrap_or(1);
    let end = config.window_end.unwrap_or(len);

    if start == 0 {
        return Err(CoreError::InvalidWindow(
            "positions are 1-indexed; start must be >= 1".into(),
        ));
    }
    if start > len {
        return Err(CoreError::InvalidWindow(format!(
            "start ({start}) is beyond the scaffold, which has {len} residues"
        )));
    }
    if end > len {
        return Err(CoreError::InvalidWindow(format!(
            "end ({end}) exceeds scaffold length ({len})"
        )));
    }
    if start > end {
        return Err(CoreError::InvalidWindow(format!(
            "start ({start}) exceeds end ({end})"
        )));
    }

    let prefix = config
        .prefix
        .clone()
        .unwrap_or_else(|| scaffold.header.clone());

    let mut library = Vec::new();

    // Optionally prepend wildtype scaffold
    if config.include_wildtype {
        library.push(Sequence {
            id: Uuid::new_v4(),
            header: format!("{prefix}_WT [wildtype]"),
            fasta: scaffold.fasta.clone(),
            length: len,
            created_at: Utc::now(),
        });
    }

    // `max_variants` counts mutants only; the wild type sits in front of them.
    let cap = config
        .max_variants
        .map(|limit| limit + usize::from(config.include_wildtype));

    // Zero-indexed bounds [start - 1, end - 1]
    let zero_start = start - 1;
    let zero_end = end - 1;

    for i in zero_start..=zero_end {
        let wt_char = seq_chars[i];
        let pos_1indexed = i + 1;

        match config.mode {
            MutagenesisMode::AlanineScanning => {
                let target_aa = if wt_char == 'A' { 'G' } else { 'A' };
                let mut mutated_seq = seq_chars.clone();
                mutated_seq[i] = target_aa;
                let fasta: String = mutated_seq.into_iter().collect();

                let mutation = Mutation::new(pos_1indexed, wt_char, target_aa);
                let notation = mutation.notation();

                library.push(Sequence {
                    id: Uuid::new_v4(),
                    header: format!("{prefix}_{notation} [mutation={notation}]"),
                    fasta,
                    length: len,
                    created_at: Utc::now(),
                });

                if cap.is_some_and(|c| library.len() >= c) {
                    break;
                }
            }
            MutagenesisMode::Saturation => {
                for &target_aa in STANDARD_AMINO_ACIDS {
                    if target_aa == wt_char {
                        continue;
                    }

                    let mut mutated_seq = seq_chars.clone();
                    mutated_seq[i] = target_aa;
                    let fasta: String = mutated_seq.into_iter().collect();

                    let mutation = Mutation::new(pos_1indexed, wt_char, target_aa);
                    let notation = mutation.notation();

                    library.push(Sequence {
                        id: Uuid::new_v4(),
                        header: format!("{prefix}_{notation} [mutation={notation}]"),
                        fasta,
                        length: len,
                        created_at: Utc::now(),
                    });

                    if cap.is_some_and(|c| library.len() >= c) {
                        break;
                    }
                }
            }
        }

        if cap.is_some_and(|c| library.len() >= c) {
            break;
        }
    }

    // The checks above run after a push; a cap of 0 (or one hit mid-position) is enforced
    // here.
    if let Some(c) = cap {
        library.truncate(c);
    }
    Ok(library)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scaffold() -> Sequence {
        Sequence {
            id: Uuid::new_v4(),
            header: "test_protein".into(),
            fasta: "ACDEFGHIKL".into(),
            length: 10,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_alanine_scanning_generation() {
        let scaffold = sample_scaffold();
        let config = MutagenesisConfig {
            mode: MutagenesisMode::AlanineScanning,
            window_start: None,
            window_end: None,
            max_variants: None,
            include_wildtype: true,
            prefix: None,
        };

        let library = generate_mutant_library(&scaffold, &config).unwrap();
        // 1 WT + 10 positions = 11 sequences
        assert_eq!(library.len(), 11);
        assert_eq!(library[0].header, "test_protein_WT [wildtype]");
        assert_eq!(library[0].fasta, "ACDEFGHIKL");

        // Position 1 was 'A', so alanine scan mutates to 'G' -> A1G
        assert_eq!(library[1].header, "test_protein_A1G [mutation=A1G]");
        assert_eq!(library[1].fasta, "GCDEFGHIKL");

        // Position 2 was 'C', mutates to 'A' -> C2A
        assert_eq!(library[2].header, "test_protein_C2A [mutation=C2A]");
        assert_eq!(library[2].fasta, "ADEFGHIKL".replacen('D', "AD", 1));
        assert_eq!(library[2].fasta, "AADEFGHIKL");
    }

    #[test]
    fn test_site_saturation_window() {
        let scaffold = sample_scaffold();
        let config = MutagenesisConfig {
            mode: MutagenesisMode::Saturation,
            window_start: Some(2), // Residue 'C'
            window_end: Some(2),
            max_variants: None,
            include_wildtype: false,
            prefix: Some("cysteine_mutants".into()),
        };

        let library = generate_mutant_library(&scaffold, &config).unwrap();
        // 19 canonical amino acid substitutions (excluding 'C')
        assert_eq!(library.len(), 19);
        for seq in &library {
            assert_eq!(seq.length, 10);
            assert!(seq.header.starts_with("cysteine_mutants_C2"));
        }
    }

    #[test]
    fn test_window_out_of_bounds_error() {
        let scaffold = sample_scaffold();
        let config = MutagenesisConfig {
            mode: MutagenesisMode::AlanineScanning,
            window_start: Some(5),
            window_end: Some(20), // Exceeds length 10
            max_variants: None,
            include_wildtype: false,
            prefix: None,
        };

        let err = generate_mutant_library(&scaffold, &config).unwrap_err();
        assert!(err.to_string().contains("exceeds scaffold length"));
    }

    #[test]
    fn test_max_variants_limit() {
        let scaffold = sample_scaffold();
        let config = MutagenesisConfig {
            mode: MutagenesisMode::Saturation,
            window_start: None,
            window_end: None,
            max_variants: Some(5),
            include_wildtype: false,
            prefix: None,
        };

        let library = generate_mutant_library(&scaffold, &config).unwrap();
        assert_eq!(library.len(), 5);
    }
    #[test]
    fn max_variants_counts_mutants_and_is_exact() {
        // Reported: `max_variants = 0` still returned 1-2 sequences, and 1 with the wild type
        // included returned 2.
        let scaffold = sample_scaffold();
        for (limit, wt, expected) in [(0, false, 0), (0, true, 1), (1, true, 2), (3, false, 3)] {
            for mode in [
                MutagenesisMode::AlanineScanning,
                MutagenesisMode::Saturation,
            ] {
                let config = MutagenesisConfig {
                    mode,
                    max_variants: Some(limit),
                    include_wildtype: wt,
                    ..Default::default()
                };
                let lib = generate_mutant_library(&scaffold, &config).unwrap();
                assert_eq!(lib.len(), expected, "{mode:?} limit {limit} wt {wt}");
            }
        }
    }
}
