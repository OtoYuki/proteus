use crate::error::CoreError;
use crate::models::Sequence;
use chrono::Utc;
use uuid::Uuid;

pub const STANDARD_AMINO_ACIDS: &[char] = &[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

pub fn validate_and_parse_fasta(content: &str) -> Result<Sequence, CoreError> {
    let normalized = content.replace("\\n", "\n");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return Err(CoreError::InvalidFasta("Empty FASTA content".into()));
    }

    let mut lines = trimmed.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| CoreError::InvalidFasta("Missing header line".into()))?;

    if !header_line.starts_with('>') {
        return Err(CoreError::InvalidFasta(
            "FASTA header must begin with '>'".into(),
        ));
    }

    let header = header_line[1..].trim().to_string();
    if header.is_empty() {
        return Err(CoreError::InvalidFasta("Empty header string".into()));
    }

    let mut sequence_str = String::new();
    for line in lines {
        let l = line.trim();
        if l.starts_with('>') {
            return Err(CoreError::InvalidFasta(
                "Multi-sequence FASTA not supported in single target mode".into(),
            ));
        }
        sequence_str.push_str(l);
    }

    if sequence_str.is_empty() {
        return Err(CoreError::InvalidFasta("Empty sequence".into()));
    }

    let upper = sequence_str.to_uppercase();
    for c in upper.chars() {
        if !STANDARD_AMINO_ACIDS.contains(&c) {
            return Err(CoreError::InvalidFasta(format!(
                "Invalid amino acid character: '{c}'"
            )));
        }
    }

    Ok(Sequence {
        id: Uuid::new_v4(),
        header,
        fasta: upper.clone(),
        length: upper.len(),
        created_at: Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_fasta_parsing() {
        let input =
            ">sp|P01308|INS_HUMAN Insulin\nGIVEQCCTSICSLYQLENYCN\nFVNQHLCGSHLVEALYLVCGERGFFYTPKT";
        let seq = validate_and_parse_fasta(input).expect("Should parse valid FASTA");
        assert_eq!(seq.header, "sp|P01308|INS_HUMAN Insulin");
        assert_eq!(seq.length, 51);
        assert!(seq.fasta.starts_with("GIVEQCCTS"));
    }

    #[test]
    fn test_invalid_amino_acids() {
        let input = ">test\nACDEX123";
        let err = validate_and_parse_fasta(input).unwrap_err();
        assert!(err.to_string().contains("Invalid amino acid character"));
    }

    #[test]
    fn test_empty_fasta() {
        assert!(validate_and_parse_fasta("").is_err());
        assert!(validate_and_parse_fasta("   \n  ").is_err());
    }

    #[test]
    fn test_missing_header() {
        let input = "MKTLLILTCLVAVALARPK";
        assert!(validate_and_parse_fasta(input).is_err());
    }
}
