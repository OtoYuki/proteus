//! Zero-shot mutation scoring (Meier et al. 2021) and deep mutational scans.

use std::collections::BTreeSet;

use crate::model::Esm2;
use crate::tokenizer::{Tokenizer, AMINO_ACIDS, MASK_ID};
use crate::{EsmError, Result};

/// A single substitution, e.g. `P19A`: wild-type `P` at 1-based position 19 → `A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mutation {
    pub wt: char,
    /// 1-based residue position.
    pub pos: usize,
    pub mt: char,
}

impl std::fmt::Display for Mutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.wt, self.pos, self.mt)
    }
}

/// Parse `P19A`-style notation.
pub fn parse_mutation(s: &str) -> Result<Mutation> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if s.len() < 3 || !bytes[0].is_ascii_alphabetic() || !bytes[s.len() - 1].is_ascii_alphabetic() {
        return Err(EsmError::Mutation(format!(
            "'{s}' is not in <wt><position><mt> form (e.g. P19A)"
        )));
    }
    let pos: usize = s[1..s.len() - 1]
        .parse()
        .map_err(|_| EsmError::Mutation(format!("'{s}': position is not a number")))?;
    if pos == 0 {
        return Err(EsmError::Mutation(format!("'{s}': positions are 1-based")));
    }
    Ok(Mutation {
        wt: (bytes[0] as char).to_ascii_uppercase(),
        pos,
        mt: (bytes[s.len() - 1] as char).to_ascii_uppercase(),
    })
}

fn check(seq: &[u8], m: &Mutation) -> Result<()> {
    let Some(&actual) = seq.get(m.pos - 1) else {
        return Err(EsmError::Mutation(format!(
            "{m}: position {} is beyond the sequence length {}",
            m.pos,
            seq.len()
        )));
    };
    if (actual as char).to_ascii_uppercase() != m.wt {
        return Err(EsmError::Mutation(format!(
            "{m}: wild-type residue at {} is {}, not {}",
            m.pos, actual as char, m.wt
        )));
    }
    Ok(())
}

/// Wild-type marginals: one forward pass on the unmasked sequence;
/// `score = log p(mt | wt seq) − log p(wt | wt seq)` at the mutated position.
pub fn score_wt_marginal(model: &Esm2, wt_seq: &str, mutations: &[Mutation]) -> Result<Vec<f32>> {
    let seq = wt_seq.trim().as_bytes();
    for m in mutations {
        check(seq, m)?;
    }
    let (tokens, _) = Tokenizer::encode(wt_seq)?;
    let lp = model.log_probs(&tokens)?;
    Ok(mutations
        .iter()
        .map(|m| {
            let row = &lp[m.pos]; // +1 for <cls>
            row[Tokenizer::residue_id(m.mt) as usize] - row[Tokenizer::residue_id(m.wt) as usize]
        })
        .collect())
}

/// Masked marginals: mask each distinct mutated position (one forward pass per position);
/// `score = log p(mt | masked) − log p(wt | masked)`. The ProteinGym-standard variant.
pub fn score_masked_marginal(
    model: &Esm2,
    wt_seq: &str,
    mutations: &[Mutation],
) -> Result<Vec<f32>> {
    let seq = wt_seq.trim().as_bytes();
    for m in mutations {
        check(seq, m)?;
    }
    let (tokens, _) = Tokenizer::encode(wt_seq)?;
    let positions: BTreeSet<usize> = mutations.iter().map(|m| m.pos).collect();
    let mut rows = std::collections::HashMap::with_capacity(positions.len());
    for pos in positions {
        let mut masked = tokens.clone();
        masked[pos] = MASK_ID;
        let lp = model.log_probs(&masked)?;
        rows.insert(pos, lp[pos].clone());
    }
    Ok(mutations
        .iter()
        .map(|m| {
            let row = &rows[&m.pos];
            row[Tokenizer::residue_id(m.mt) as usize] - row[Tokenizer::residue_id(m.wt) as usize]
        })
        .collect())
}

/// One row of a deep mutational scan: scores for the twenty amino acids at a position.
#[derive(Debug, Clone)]
pub struct ScanRow {
    pub pos: usize,
    pub wt: char,
    /// Indexed like [`AMINO_ACIDS`]; the wild-type entry is 0 by construction.
    pub scores: [f32; 20],
}

/// Full L × 20 scan. `masked = false` uses wild-type marginals (one pass);
/// `masked = true` masks each position in turn (L passes).
pub fn scan(model: &Esm2, wt_seq: &str, masked: bool) -> Result<Vec<ScanRow>> {
    let seq: Vec<char> = wt_seq
        .trim()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let (tokens, _) = Tokenizer::encode(wt_seq)?;
    let wt_rows: Option<Vec<Vec<f32>>> = if masked {
        None
    } else {
        Some(model.log_probs(&tokens)?)
    };
    let mut out = Vec::with_capacity(seq.len());
    for (i, &wt) in seq.iter().enumerate() {
        let pos = i + 1;
        let row: Vec<f32> = match &wt_rows {
            Some(rows) => rows[pos].clone(),
            None => {
                let mut m = tokens.clone();
                m[pos] = MASK_ID;
                model.log_probs(&m)?[pos].clone()
            }
        };
        let wt_lp = row[Tokenizer::residue_id(wt) as usize];
        let mut scores = [0f32; 20];
        for (k, aa) in AMINO_ACIDS.iter().enumerate() {
            scores[k] = row[Tokenizer::residue_id(*aa) as usize] - wt_lp;
        }
        out.push(ScanRow {
            pos,
            wt: wt.to_ascii_uppercase(),
            scores,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_mutations() {
        let m = parse_mutation("p19a").unwrap();
        assert_eq!(
            m,
            Mutation {
                wt: 'P',
                pos: 19,
                mt: 'A'
            }
        );
        assert_eq!(m.to_string(), "P19A");
        assert!(parse_mutation("19A").is_err());
        assert!(parse_mutation("P0A").is_err());
        assert!(parse_mutation("PxA").is_err());
        let seq = b"TTCCPSIVARSNFNVCRLPGT";
        assert!(check(
            seq,
            &Mutation {
                wt: 'P',
                pos: 19,
                mt: 'A'
            }
        )
        .is_ok());
        assert!(check(
            seq,
            &Mutation {
                wt: 'A',
                pos: 19,
                mt: 'P'
            }
        )
        .is_err());
        assert!(check(
            seq,
            &Mutation {
                wt: 'A',
                pos: 99,
                mt: 'P'
            }
        )
        .is_err());
    }
}
