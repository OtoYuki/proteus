//! Multi-chain and ligand inputs, and per-job prediction options, for the Boltz tier.
//!
//! A job's input is stored in [`crate::models::Sequence::fasta`]. A monomer is a bare string of
//! residues; anything richer — several chains, ligands, an MSA choice, several samples — is a
//! canonical multi-record FASTA in Boltz's own header syntax, `>ID|ENTITY|MSA`, preceded by
//! `#proteus key=value` option lines. A monomer never starts with `>` or `#`, so the two cannot be
//! confused, and no schema change is needed.
//!
//! ```text
//! #proteus samples=5
//! >A|protein|server
//! MQIFVKTLTG…
//! >B|protein|empty
//! MKTAYIAKQR…
//! >L|ccd
//! ATP
//! >M|smiles
//! CC(=O)Oc1ccccc1C(=O)O
//! ```

use crate::error::CoreError;
use crate::sequence::STANDARD_AMINO_ACIDS;

/// Where a protein chain's multiple sequence alignment comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MsaSource {
    /// Single-sequence mode: no alignment, nothing leaves the machine.
    Empty,
    /// The public ColabFold MMseqs2 server (Boltz `--use_msa_server`): the sequence is sent to it.
    Server,
    /// An alignment file (.a3m or Boltz .csv) given by the user, by absolute path.
    File(String),
}

impl MsaSource {
    fn field(&self) -> &str {
        match self {
            MsaSource::Empty => "empty",
            MsaSource::Server => "server",
            MsaSource::File(p) => p,
        }
    }
    fn parse(s: &str) -> Self {
        match s.trim() {
            "" | "empty" => MsaSource::Empty,
            "server" => MsaSource::Server,
            p => MsaSource::File(p.to_string()),
        }
    }
}

/// One entity of a complex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entity {
    Protein {
        sequence: String,
        msa: MsaSource,
    },
    /// A ligand from the PDB Chemical Component Dictionary, by its code (`ATP`, `HEM`).
    Ccd(String),
    /// A ligand as a SMILES string.
    Smiles(String),
}

/// A chain of a complex: its ID and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub id: String,
    pub entity: Entity,
}

/// A prediction input: chains and ligands, plus job options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ComplexSpec {
    pub chains: Vec<Chain>,
    /// Structures to sample (Boltz `--diffusion_samples`), ranked by confidence; 1 by default.
    pub samples: usize,
}

fn err(msg: impl Into<String>) -> CoreError {
    CoreError::InvalidFasta(msg.into())
}

/// Chain IDs handed out to records that do not name one.
const AUTO_IDS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// The most structures one job may sample: each one is a full diffusion run.
pub const MAX_SAMPLES: usize = 25;

impl ComplexSpec {
    /// A single protein chain.
    pub fn monomer(sequence: &str) -> Self {
        Self {
            chains: vec![Chain {
                id: "A".into(),
                entity: Entity::Protein {
                    sequence: sequence.to_string(),
                    msa: MsaSource::Empty,
                },
            }],
            samples: 1,
        }
    }

    /// True when a stored input is a spec rather than a bare monomer sequence.
    pub fn is_spec(stored: &str) -> bool {
        matches!(stored.trim_start().chars().next(), Some('>') | Some('#'))
    }

    /// Read what a job stored: a spec, or a bare monomer.
    pub fn from_stored(stored: &str) -> Result<Self, CoreError> {
        if Self::is_spec(stored) {
            Self::parse(stored)
        } else {
            Ok(Self::monomer(stored.trim()))
        }
    }

    /// Parse a FASTA of one or more records. Headers may be plain (`>my protein`, taken as a
    /// protein chain with the next free ID) or Boltz-style `>ID|ENTITY|MSA` with ENTITY one of
    /// `protein`, `ccd`, `smiles`. `#proteus samples=N` lines set options.
    pub fn parse(text: &str) -> Result<Self, CoreError> {
        let mut spec = ComplexSpec {
            chains: Vec::new(),
            samples: 1,
        };
        let mut records: Vec<(String, String)> = Vec::new();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            if let Some(opts) = l.strip_prefix("#proteus") {
                for kv in opts.split_whitespace() {
                    match kv.split_once('=') {
                        Some(("samples", v)) => {
                            let n: usize = v
                                .parse()
                                .map_err(|_| err(format!("samples={v} is not a number")))?;
                            if n == 0 || n > MAX_SAMPLES {
                                return Err(err(format!("samples must be 1–{MAX_SAMPLES}")));
                            }
                            spec.samples = n;
                        }
                        _ => return Err(err(format!("unknown #proteus option '{kv}'"))),
                    }
                }
            } else if l.starts_with('#') {
                continue;
            } else if let Some(h) = l.strip_prefix('>') {
                records.push((h.trim().to_string(), String::new()));
            } else {
                let Some(last) = records.last_mut() else {
                    return Err(err("sequence text before the first '>' header"));
                };
                last.1.push_str(l);
            }
        }
        if records.is_empty() {
            return Err(err("no records: a complex needs at least one '>' header"));
        }
        let mut used: Vec<String> = Vec::new();
        for (header, body) in records {
            let parts: Vec<&str> = header.split('|').map(str::trim).collect();
            let kind = parts.get(1).map(|k| k.to_ascii_lowercase());
            let boltz_style = matches!(kind.as_deref(), Some("protein" | "ccd" | "smiles"))
                || matches!(kind.as_deref(), Some("dna" | "rna"));
            let (id, entity) = if boltz_style {
                let id = parts[0].to_string();
                let entity = match kind.as_deref() {
                    Some("protein") => protein(&body, parts.get(2).copied().unwrap_or(""))?,
                    Some("ccd") => {
                        let code = body.trim().to_ascii_uppercase();
                        if code.is_empty()
                            || code.len() > 5
                            || !code.chars().all(|c| c.is_ascii_alphanumeric())
                        {
                            return Err(err(format!(
                                "'{}' is not a CCD code (1–5 letters or digits, e.g. ATP)",
                                body.trim()
                            )));
                        }
                        Entity::Ccd(code)
                    }
                    Some("smiles") => {
                        let s = body.trim();
                        if s.is_empty() || s.chars().any(|c| c.is_whitespace()) {
                            return Err(err("a SMILES record needs one SMILES string"));
                        }
                        Entity::Smiles(s.to_string())
                    }
                    _ => {
                        return Err(err(format!(
                            "chain {id}: DNA and RNA are not supported by this tier yet"
                        )))
                    }
                };
                (id, entity)
            } else {
                let id = AUTO_IDS
                    .chars()
                    .map(|c| c.to_string())
                    .find(|c| !used.contains(c))
                    .ok_or_else(|| err("more than 26 chains"))?;
                (id, protein(&body, "")?)
            };
            if id.is_empty() || id.len() > 4 || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(err(format!(
                    "'{id}' is not a chain ID (1–4 letters or digits)"
                )));
            }
            if used.contains(&id) {
                return Err(err(format!("chain ID '{id}' is used twice")));
            }
            used.push(id.clone());
            spec.chains.push(Chain { id, entity });
        }
        if !spec
            .chains
            .iter()
            .any(|c| matches!(c.entity, Entity::Protein { .. }))
        {
            return Err(err("a complex needs at least one protein chain"));
        }
        Ok(spec)
    }

    /// The canonical stored form (see the module docs). A plain monomer with default options is
    /// stored as its bare sequence, as before.
    pub fn to_stored(&self) -> String {
        if let [Chain {
            entity:
                Entity::Protein {
                    sequence,
                    msa: MsaSource::Empty,
                },
            ..
        }] = self.chains.as_slice()
        {
            if self.samples <= 1 {
                return sequence.clone();
            }
        }
        let mut out = String::new();
        if self.samples > 1 {
            out.push_str(&format!("#proteus samples={}\n", self.samples));
        }
        for c in &self.chains {
            match &c.entity {
                Entity::Protein { sequence, msa } => out.push_str(&format!(
                    ">{}|protein|{}\n{}\n",
                    c.id,
                    msa.field(),
                    sequence
                )),
                Entity::Ccd(code) => out.push_str(&format!(">{}|ccd\n{}\n", c.id, code)),
                Entity::Smiles(s) => out.push_str(&format!(">{}|smiles\n{}\n", c.id, s)),
            }
        }
        out
    }

    /// Total protein residues.
    pub fn residues(&self) -> usize {
        self.chains
            .iter()
            .map(|c| match &c.entity {
                Entity::Protein { sequence, .. } => sequence.len(),
                _ => 0,
            })
            .sum()
    }

    pub fn is_monomer(&self) -> bool {
        self.chains.len() == 1
    }

    /// True when any chain's alignment comes from the public MSA server.
    pub fn uses_msa_server(&self) -> bool {
        self.chains.iter().any(|c| {
            matches!(
                c.entity,
                Entity::Protein {
                    msa: MsaSource::Server,
                    ..
                }
            )
        })
    }

    /// Set every protein chain's MSA source.
    pub fn set_msa(&mut self, msa: MsaSource) {
        for c in &mut self.chains {
            if let Entity::Protein { msa: m, .. } = &mut c.entity {
                *m = msa.clone();
            }
        }
    }

    /// The protein chains' sequences joined with `:` (ColabFold's complex notation), for
    /// display and for anything that only needs the residues.
    pub fn protein_sequence(&self) -> String {
        self.chains
            .iter()
            .filter_map(|c| match &c.entity {
                Entity::Protein { sequence, .. } => Some(sequence.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(":")
    }
}

fn protein(body: &str, msa: &str) -> Result<Entity, CoreError> {
    let seq = body.trim().to_ascii_uppercase();
    if seq.is_empty() {
        return Err(err("empty protein chain"));
    }
    if let Some(c) = seq.chars().find(|c| !STANDARD_AMINO_ACIDS.contains(c)) {
        return Err(err(format!("Invalid amino acid character: '{c}'")));
    }
    Ok(Entity::Protein {
        sequence: seq,
        msa: MsaSource::parse(msa),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_records_become_chains_and_boltz_headers_are_kept() {
        let s = ComplexSpec::parse(">heavy\nEVQLV\n>light\nDIQMT\n>L|ccd\natp\n>M|smiles\nCCO\n")
            .unwrap();
        let ids: Vec<&str> = s.chains.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["A", "B", "L", "M"]);
        assert_eq!(s.chains[2].entity, Entity::Ccd("ATP".into()));
        assert_eq!(s.residues(), 10);
        assert_eq!(s.protein_sequence(), "EVQLV:DIQMT");
    }

    #[test]
    fn the_stored_form_round_trips_and_a_monomer_stays_bare() {
        let mut s = ComplexSpec::parse(">A|protein|server\nMQIF\n>L|ccd\nHEM\n").unwrap();
        s.samples = 5;
        let stored = s.to_stored();
        assert!(stored.starts_with("#proteus samples=5\n"));
        assert_eq!(ComplexSpec::from_stored(&stored).unwrap(), s);
        assert!(s.uses_msa_server());
        let m = ComplexSpec::monomer("MQIF");
        assert_eq!(m.to_stored(), "MQIF");
        assert_eq!(ComplexSpec::from_stored("MQIF").unwrap(), m);
    }

    #[test]
    fn bad_inputs_are_refused() {
        for bad in [
            "MQIF",                             // no header
            ">A|protein\nMQXZ\n",               // not amino acids
            ">A|protein\nMQ\n>A|protein\nKK\n", // duplicate ID
            ">L|ccd\nATP\n",                    // no protein
            ">L|ccd\nNOT A CODE\n>A|protein\nMQ\n",
            "#proteus samples=0\n>A|protein\nMQ\n",
            "#proteus bogus=1\n>A|protein\nMQ\n",
        ] {
            assert!(ComplexSpec::parse(bad).is_err(), "accepted: {bad:?}");
        }
    }
}
