import re


def parse_fasta(fasta_content):
    """
    Parse a FASTA format string into a dictionary containing id and sequence.

    Args:
        fasta_content (str): FASTA format string (e.g., ">seq1\nMKT")

    Returns:
        dict: Dictionary with 'id' and 'sequence' keys

    Raises:
        Exception: If the FASTA format is invalid
    """
    # Basic FASTA format validation
    if not fasta_content or not isinstance(fasta_content, str):
        raise Exception("Invalid FASTA: Empty or non-string input")

    lines = fasta_content.strip().split("\n")
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise Exception(
            "Invalid FASTA: Must have header starting with '>' and sequence"
        )

    # Extract ID and sequence
    seq_id = lines[0][1:].strip()  # Remove '>' and whitespace
    sequence = "".join(lines[1:]).strip()

    # Validate sequence contains only valid amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not sequence or not all(aa in valid_aa for aa in sequence.upper()):
        raise Exception("Invalid FASTA: Sequence contains invalid amino acids")

    return {"id": seq_id, "sequence": sequence}
