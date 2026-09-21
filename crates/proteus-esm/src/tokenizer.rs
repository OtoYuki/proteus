//! The fixed 33-token ESM vocabulary (`vocab.txt` of every ESM-2 checkpoint).

use crate::{EsmError, Result};

/// Token strings in id order.
pub const VOCAB: [&str; 33] = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>",
    "<mask>",
];
pub const CLS_ID: u32 = 0;
pub const PAD_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const MASK_ID: u32 = 32;

/// The twenty canonical amino acids in the order used for scan matrices.
pub const AMINO_ACIDS: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

/// Character-level tokenizer: `<cls>` + one token per residue + `<eos>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tokenizer;

impl Tokenizer {
    /// Token id of a single residue letter (case-insensitive), `<unk>` for anything not in the vocab.
    pub fn residue_id(c: char) -> u32 {
        let up = c.to_ascii_uppercase();
        VOCAB
            .iter()
            .position(|t| t.len() == 1 && t.starts_with(up))
            .map(|i| i as u32)
            .unwrap_or(UNK_ID)
    }

    /// Residue letter for a token id (`None` for special tokens).
    pub fn id_residue(id: u32) -> Option<char> {
        VOCAB
            .get(id as usize)
            .filter(|t| t.len() == 1)
            .and_then(|t| t.chars().next())
    }

    /// Encode a protein sequence. Errors on empty input; counts `<unk>` substitutions.
    pub fn encode(seq: &str) -> Result<(Vec<u32>, usize)> {
        let seq = seq.trim();
        if seq.is_empty() {
            return Err(EsmError::Sequence("empty sequence".into()));
        }
        let mut ids = Vec::with_capacity(seq.len() + 2);
        let mut unknown = 0usize;
        ids.push(CLS_ID);
        for c in seq.chars() {
            if c.is_whitespace() {
                continue;
            }
            let id = Self::residue_id(c);
            if id == UNK_ID {
                unknown += 1;
            }
            ids.push(id);
        }
        ids.push(EOS_ID);
        Ok((ids, unknown))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hf_ids_for_crambin() {
        let (ids, unk) =
            Tokenizer::encode("TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN").unwrap();
        assert_eq!(unk, 0);
        assert_eq!(
            ids,
            vec![
                0, 11, 11, 23, 23, 14, 8, 12, 7, 5, 10, 8, 17, 18, 17, 7, 23, 10, 4, 14, 6, 11, 14,
                9, 5, 12, 23, 5, 11, 19, 11, 6, 23, 12, 12, 12, 14, 6, 5, 11, 23, 14, 6, 13, 19, 5,
                17, 2
            ]
        );
    }

    #[test]
    fn unknown_letters_become_unk() {
        let (ids, unk) = Tokenizer::encode("AJ").unwrap();
        assert_eq!(ids, vec![0, 5, 3, 2]);
        assert_eq!(unk, 1);
        assert_eq!(Tokenizer::id_residue(5), Some('A'));
        assert_eq!(Tokenizer::id_residue(32), None);
    }
}
