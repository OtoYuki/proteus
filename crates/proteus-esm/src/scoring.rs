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

/// Parse `P19A`-style notation. Both residues must be one of the twenty standard amino acids
/// (a score against `X` or `<unk>` means nothing); the position is 1-based decimal digits.
pub fn parse_mutation(s: &str) -> Result<Mutation> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if s.len() < 3 || !bytes[0].is_ascii_alphabetic() || !bytes[s.len() - 1].is_ascii_alphabetic() {
        return Err(EsmError::Mutation(format!(
            "'{s}' is not in <wt><position><mt> form (e.g. P19A)"
        )));
    }
    let digits = &s[1..s.len() - 1];
    // `usize::from_str` also takes a leading '+'.
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(EsmError::Mutation(format!(
            "'{s}': position is not a number"
        )));
    }
    let pos: usize = digits
        .parse()
        .map_err(|_| EsmError::Mutation(format!("'{s}': position is not a number")))?;
    let m = Mutation {
        wt: (bytes[0] as char).to_ascii_uppercase(),
        pos,
        mt: (bytes[s.len() - 1] as char).to_ascii_uppercase(),
    };
    check_residues(&m)?;
    Ok(m)
}

fn check_residues(m: &Mutation) -> Result<()> {
    if m.pos == 0 {
        return Err(EsmError::Mutation(format!("{m}: positions are 1-based")));
    }
    for c in [m.wt, m.mt] {
        if !AMINO_ACIDS.contains(&c) {
            return Err(EsmError::Mutation(format!(
                "{m}: '{c}' is not one of the 20 standard amino acids"
            )));
        }
    }
    Ok(())
}

/// `seq` is the [`Tokenizer::normalize`]d wild type.
fn check(seq: &[u8], m: &Mutation) -> Result<()> {
    check_residues(m)?;
    let Some(&actual) = seq.get(m.pos - 1) else {
        return Err(EsmError::Mutation(format!(
            "{m}: position {} is beyond the sequence length {}",
            m.pos,
            seq.len()
        )));
    };
    if actual as char != m.wt {
        return Err(EsmError::Mutation(format!(
            "{m}: wild-type residue at {} is {}, not {}",
            m.pos, actual as char, m.wt
        )));
    }
    Ok(())
}

/// The substitutions of one variant: each position at most once.
fn check_variant(mutations: &[Mutation]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for m in mutations {
        if !seen.insert(m.pos) {
            return Err(EsmError::Mutation(format!(
                "{m}: position {} is mutated more than once in one variant",
                m.pos
            )));
        }
    }
    Ok(())
}

/// Normalised wild type and its tokens. Positions in mutations count residues of the
/// normalised sequence, which is also what the tokens are built from, so the two cannot drift
/// apart (whitespace, for one, used to be counted by one and skipped by the other).
fn prepare(wt_seq: &str) -> Result<(String, Vec<u32>)> {
    let seq = Tokenizer::normalize(wt_seq)?;
    let (tokens, _) = Tokenizer::encode(&seq)?;
    Ok((seq, tokens))
}

/// Wild-type marginals: one forward pass on the unmasked sequence;
/// `score = log p(mt | wt seq) − log p(wt | wt seq)` at the mutated position.
///
/// `mutations` are scored independently (alternatives at one position are allowed); the wild
/// type goes through [`Tokenizer::normalize`].
pub fn score_wt_marginal(model: &Esm2, wt_seq: &str, mutations: &[Mutation]) -> Result<Vec<f32>> {
    let (seq, tokens) = prepare(wt_seq)?;
    for m in mutations {
        check(seq.as_bytes(), m)?;
    }
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
/// Same input rules as [`score_wt_marginal`].
pub fn score_masked_marginal(
    model: &Esm2,
    wt_seq: &str,
    mutations: &[Mutation],
) -> Result<Vec<f32>> {
    let (seq, tokens) = prepare(wt_seq)?;
    for m in mutations {
        check(seq.as_bytes(), m)?;
    }
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

/// Scores many variants of one wild type without repeating forward passes: wild-type
/// marginals need a single pass for the whole library, masked marginals one pass per distinct
/// mutated position. Produces exactly the numbers of [`score_wt_marginal`] /
/// [`score_masked_marginal`].
pub struct MarginalScorer<'m> {
    model: &'m Esm2,
    seq: Vec<u8>,
    tokens: Vec<u32>,
    masked: bool,
    /// Wild-type marginals: the one log-prob table. Masked: per mutated position.
    wt_rows: Option<Vec<Vec<f32>>>,
    masked_rows: std::collections::HashMap<usize, Vec<f32>>,
}

impl<'m> MarginalScorer<'m> {
    /// `wt_seq` goes through [`Tokenizer::normalize`].
    pub fn new(model: &'m Esm2, wt_seq: &str, masked: bool) -> Result<Self> {
        let (seq, tokens) = prepare(wt_seq)?;
        Ok(Self {
            model,
            seq: seq.into_bytes(),
            tokens,
            masked,
            wt_rows: None,
            masked_rows: std::collections::HashMap::new(),
        })
    }

    /// Score one variant's substitutions (same order as `mutations`). A variant mutates each
    /// position at most once.
    pub fn score(&mut self, mutations: &[Mutation]) -> Result<Vec<f32>> {
        check_variant(mutations)?;
        for m in mutations {
            check(&self.seq, m)?;
        }
        let mut out = Vec::with_capacity(mutations.len());
        for m in mutations {
            let row: &Vec<f32> = if self.masked {
                if !self.masked_rows.contains_key(&m.pos) {
                    let mut t = self.tokens.clone();
                    t[m.pos] = MASK_ID;
                    let lp = self.model.log_probs(&t)?;
                    self.masked_rows.insert(m.pos, lp[m.pos].clone());
                }
                &self.masked_rows[&m.pos]
            } else {
                if self.wt_rows.is_none() {
                    self.wt_rows = Some(self.model.log_probs(&self.tokens)?);
                }
                &self.wt_rows.as_ref().expect("filled above")[m.pos]
            };
            out.push(
                row[Tokenizer::residue_id(m.mt) as usize]
                    - row[Tokenizer::residue_id(m.wt) as usize],
            );
        }
        Ok(out)
    }
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
///
/// Positions whose wild type is not one of the twenty (`X`, `B`, `Z`, `U`, `O`) get no row: a
/// substitution score is relative to the wild-type residue, and there is none to compare with.
/// Rows keep their true `pos`, so the result can be shorter than the sequence.
pub fn scan(model: &Esm2, wt_seq: &str, masked: bool) -> Result<Vec<ScanRow>> {
    let (seq, tokens) = prepare(wt_seq)?;
    let seq: Vec<char> = seq.chars().collect();
    let wt_rows: Option<Vec<Vec<f32>>> = if masked {
        None
    } else {
        Some(model.log_probs(&tokens)?)
    };
    let mut out = Vec::with_capacity(seq.len());
    for (i, &wt) in seq.iter().enumerate() {
        if !AMINO_ACIDS.contains(&wt) {
            continue;
        }
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
        out.push(ScanRow { pos, wt, scores });
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

    fn m(s: &str) -> Mutation {
        parse_mutation(s).unwrap()
    }

    #[test]
    fn whitespace_in_the_wild_type_does_not_shift_positions() {
        // Residue 11 is the Q after the space. Counting the space made Q12A pass the wild-type
        // check and score residue 12 (I); Q23A scored the <eos> row.
        let model = crate::model::tests::tiny_model();
        let spaced = "MKTAYIAKQR QISFVKSHFSRQ";
        let plain = "MKTAYIAKQRQISFVKSHFSRQ";
        let q11 = [m("Q11A")];
        let want = score_wt_marginal(&model, plain, &q11).unwrap();
        assert_eq!(score_wt_marginal(&model, spaced, &q11).unwrap(), want);
        assert_eq!(
            score_masked_marginal(&model, "MKTAYIAKQR\nQISF VKSHFSRQ", &q11).unwrap(),
            score_masked_marginal(&model, plain, &q11).unwrap()
        );
        let mut scorer = MarginalScorer::new(&model, spaced, false).unwrap();
        assert_eq!(scorer.score(&q11).unwrap(), want);
        for bad in ["Q12A", "Q23A"] {
            assert!(
                score_wt_marginal(&model, spaced, &[m(bad)]).is_err(),
                "{bad}"
            );
            assert!(
                score_masked_marginal(&model, spaced, &[m(bad)]).is_err(),
                "{bad}"
            );
            assert!(scorer.score(&[m(bad)]).is_err(), "{bad}");
        }
        assert_eq!(scan(&model, spaced, false).unwrap().len(), 22);
    }

    #[test]
    fn non_amino_acid_characters_are_refused() {
        let model = crate::model::tests::tiny_model();
        // Mutation targets must be one of the twenty: J/X/B/Z/U/O were scored against <unk>
        // or ambiguity tokens.
        for bad in [
            "A1J", "A1X", "A1B", "A1Z", "A1U", "A1O", "X1A", "A1*", "A1-",
        ] {
            assert!(parse_mutation(bad).is_err(), "{bad}");
        }
        // Wild type: digits, gaps, stops and unknown letters used to become tokens.
        for bad in ["ACD1E", "AC-DE", "AC.DE", "AC*DE", "ACJDE", "AC_DE", ""] {
            assert!(
                score_wt_marginal(&model, bad, &[m("A1C")]).is_err(),
                "{bad:?}"
            );
            assert!(MarginalScorer::new(&model, bad, false).is_err(), "{bad:?}");
            assert!(scan(&model, bad, false).is_err(), "{bad:?}");
        }
        let err = score_wt_marginal(&model, "ACJDE", &[m("A1C")]).unwrap_err();
        assert!(err.to_string().contains("'J' at position 3"), "{err}");
        // A trailing stop is dropped; lower case is accepted.
        assert_eq!(
            score_wt_marginal(&model, "acdek*", &[m("A1C")]).unwrap(),
            score_wt_marginal(&model, "ACDEK", &[m("A1C")]).unwrap()
        );
        // ESM's own ambiguity tokens are part of the vocabulary the model was trained on.
        assert!(score_wt_marginal(&model, "ACXDE", &[m("A1C")]).is_ok());
    }

    #[test]
    fn scan_skips_non_canonical_wild_type_positions() {
        // A substitution score is relative to the wild-type residue, which X is not.
        let model = crate::model::tests::tiny_model();
        let rows = scan(&model, "ACXDE", false).unwrap();
        assert_eq!(rows.iter().map(|r| r.pos).collect::<Vec<_>>(), [1, 2, 4, 5]);
        for r in &rows {
            let k = AMINO_ACIDS.iter().position(|&a| a == r.wt).unwrap();
            assert_eq!(r.scores[k], 0.0);
        }
        assert_eq!(scan(&model, "ACXDE", true).unwrap().len(), 4);
        // Substitutions at an ambiguous wild-type position are refused, not scored against X.
        assert!(score_wt_marginal(&model, "ACXDE", &[m("C2A")]).is_ok());
        let x3 = Mutation {
            wt: 'X',
            pos: 3,
            mt: 'A',
        };
        assert!(score_wt_marginal(&model, "ACXDE", &[x3]).is_err());
    }

    #[test]
    fn position_zero_is_an_error_not_an_underflow() {
        let model = crate::model::tests::tiny_model();
        let zero = Mutation {
            wt: 'A',
            pos: 0,
            mt: 'C',
        };
        let err = score_wt_marginal(&model, "ACDE", &[zero]).unwrap_err();
        assert!(err.to_string().contains("1-based"), "{err}");
        assert!(score_masked_marginal(&model, "ACDE", &[zero]).is_err());
        let mut scorer = MarginalScorer::new(&model, "ACDE", true).unwrap();
        assert!(scorer.score(&[zero]).is_err());
    }

    #[test]
    fn mutation_syntax_is_strict() {
        // `usize::from_str` takes a leading '+'.
        assert!(parse_mutation("A+10G").is_err());
        assert!(parse_mutation("A 10G").is_err());
        assert!(parse_mutation("A1_0G").is_err());
        assert_eq!(parse_mutation(" a10g ").unwrap(), m("A10G"));
    }

    #[test]
    fn one_variant_cannot_mutate_a_position_twice() {
        // A10G,A10C is not a variant; summing both scores was meaningless.
        let model = crate::model::tests::tiny_model();
        let mut scorer = MarginalScorer::new(&model, "ACDE", false).unwrap();
        let err = scorer.score(&[m("A1G"), m("A1C")]).unwrap_err();
        assert!(err.to_string().contains("more than once"), "{err}");
        assert!(scorer.score(&[m("A1G"), m("C2A")]).is_ok());
        // The one-shot APIs score a list of independent substitutions, where alternatives at
        // one position are the point.
        assert_eq!(
            score_wt_marginal(&model, "ACDE", &[m("A1G"), m("A1C")])
                .unwrap()
                .len(),
            2
        );
    }
}
