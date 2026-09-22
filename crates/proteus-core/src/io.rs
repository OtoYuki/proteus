//! Single entry point for reading structures: PDB / mmCIF, optionally gzip-compressed.
//!
//! `pdbtbx::open_raw` only recognises PDB text when the first line is `HEADER`, which
//! predictor outputs (ESMFold, ColabFold, Boltz) never have. This module sniffs mmCIF by
//! its `data_` block and treats everything else as PDB.

use std::io::Read;
use std::path::Path;

use pdbtbx::StrictnessLevel;

use crate::error::CoreError;

/// Bytes of raw header text kept alongside a parsed structure for provenance detection
/// (`EXPDTA`, `TITLE`, mmCIF `_exptl`/`_ma_qa_metric` live near the top of the file).
pub const HEADER_PREVIEW_BYTES: usize = 16 * 1024;

/// A parsed structure plus the raw head of the file it came from.
pub struct LoadedStructure {
    pub pdb: pdbtbx::PDB,
    /// First `HEADER_PREVIEW_BYTES` (16 KiB) of the decompressed text, for `confidence` detection.
    pub header_preview: String,
    pub format: StructureFormat,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StructureFormat {
    Pdb,
    MmCif,
}

fn parse_err(what: &str, e: impl std::fmt::Debug) -> CoreError {
    CoreError::StructureParseError(format!("{what}: {e:?}"))
}

fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

fn looks_like_cif(text: &str, hint: Option<&str>) -> bool {
    if let Some(h) = hint {
        let h = h.to_ascii_lowercase();
        if h.ends_with(".cif")
            || h.ends_with(".mmcif")
            || h.ends_with(".cif.gz")
            || h.ends_with(".mmcif.gz")
        {
            return true;
        }
        if h.ends_with(".pdb")
            || h.ends_with(".ent")
            || h.ends_with(".pdb.gz")
            || h.ends_with(".ent.gz")
        {
            return false;
        }
    }
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.starts_with("data_"))
        .unwrap_or(false)
}

/// Which format a structure text is in, from the file-name hint when it is conclusive, else
/// from the content (mmCIF starts with a `data_` block). Used by the Mol* exports, which must
/// tell the viewer the format.
pub fn sniff_format(text: &str, hint: Option<&str>) -> StructureFormat {
    if looks_like_cif(text, hint) {
        StructureFormat::MmCif
    } else {
        StructureFormat::Pdb
    }
}

/// PDB record types that coordinate-based analysis needs. Sequence and annotation records
/// (SEQRES, SEQADV, DBREF, HELIX, SHEET, SITE, LINK, …) are dropped before parsing: they carry
/// nothing Proteus uses, and pdbtbx's lexer rejects legitimate deposited files on malformed
/// ones (e.g. blank sequence numbers in the SEQADV deletion records of 1TIM).
const COORDINATE_RECORDS: &[&str] = &[
    "HEADER", "REMARK", "CRYST1", "SCALE", "ORIGX", "MTRIX", "MODEL", "ATOM", "HETATM", "ANISOU",
    "TER", "ENDMDL", "END", "SSBOND", "CONECT", "MASTER",
];

fn coordinate_records_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if COORDINATE_RECORDS.iter().any(|r| line.starts_with(r)) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// pdbtbx panics (rather than erroring) on a coordinate that does not parse as a finite
/// number, and predictors do emit `nan` for failed atoms. Reject such rows up front with a
/// proper error. PDB: columns 31–54 of ATOM/HETATM records; mmCIF: every whitespace token of
/// ATOM/HETATM rows (a column layout is not fixed there, and `nan`/`inf` are never legitimate).
fn reject_non_finite_coordinates(text: &str, is_cif: bool) -> Result<(), CoreError> {
    let non_finite = |tok: &str| tok.parse::<f64>().is_ok_and(|v| !v.is_finite());
    for (n, line) in text.lines().enumerate() {
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let bad = if is_cif {
            line.split_whitespace().any(non_finite)
        } else {
            line.get(30..54)
                .map(|cols| cols.split_whitespace().any(non_finite))
                .unwrap_or(false)
        };
        if bad {
            return Err(CoreError::StructureParseError(format!(
                "line {}: non-finite coordinate in atom record",
                n + 1
            )));
        }
    }
    Ok(())
}

/// Open a structure file by path. Extension decides the format when recognised
/// (`.pdb`, `.ent`, `.cif`, `.mmcif`, each optionally `.gz`); otherwise the content is sniffed.
pub fn open_structure(path: &Path) -> Result<pdbtbx::PDB, CoreError> {
    load_structure(path).map(|l| l.pdb)
}

/// As [`open_structure`], additionally returning the raw header preview and the format.
pub fn load_structure(path: &Path) -> Result<LoadedStructure, CoreError> {
    let bytes = std::fs::read(path)
        .map_err(|e| CoreError::StructureParseError(format!("{}: {e}", path.display())))?;
    let hint = path.file_name().and_then(|n| n.to_str());
    load_structure_bytes(&bytes, hint)
}

/// Read a structure file as text, transparently decompressing gzip. For callers that need
/// the raw text (the renderer, the web export) rather than a parsed `PDB`.
pub fn read_structure_text(path: &Path) -> Result<String, CoreError> {
    let bytes = std::fs::read(path)
        .map_err(|e| CoreError::StructureParseError(format!("{}: {e}", path.display())))?;
    let bytes = if is_gzip(&bytes) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| parse_err("gzip", e))?;
        out
    } else {
        bytes
    };
    String::from_utf8(bytes).map_err(|e| parse_err("utf-8", e))
}

/// Parse a structure from memory. `hint` is an optional file name used only for its extension.
pub fn open_structure_bytes(bytes: &[u8], hint: Option<&str>) -> Result<pdbtbx::PDB, CoreError> {
    load_structure_bytes(bytes, hint).map(|l| l.pdb)
}

/// As [`open_structure_bytes`], additionally returning the raw header preview and the format.
pub fn load_structure_bytes(
    bytes: &[u8],
    hint: Option<&str>,
) -> Result<LoadedStructure, CoreError> {
    let decompressed;
    let bytes = if is_gzip(bytes) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut out)
            .map_err(|e| parse_err("gzip", e))?;
        decompressed = out;
        &decompressed[..]
    } else {
        bytes
    };
    let text = std::str::from_utf8(bytes).map_err(|e| parse_err("utf-8", e))?;
    let header_preview = text
        .char_indices()
        .take_while(|(i, _)| *i < HEADER_PREVIEW_BYTES)
        .map(|(_, c)| c)
        .collect::<String>();
    let is_cif = looks_like_cif(text, hint);
    reject_non_finite_coordinates(text, is_cif)?;
    let result = if is_cif {
        pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Mmcif)
            .set_level(StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(text.as_bytes()))
    } else {
        let coordinates = coordinate_records_only(text);
        pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Pdb)
            .set_level(StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(coordinates.as_bytes()))
    };
    let (mut pdb, _warnings) =
        result.map_err(|e| parse_err(if is_cif { "mmCIF" } else { "PDB" }, e))?;
    // NMR ensembles and multi-model files: analyse the first model only, as DSSP/MolProbity do.
    if pdb.model_count() > 1 {
        pdb.remove_models_except(&[0]);
    }
    Ok(LoadedStructure {
        pdb,
        header_preview,
        format: if is_cif {
            StructureFormat::MmCif
        } else {
            StructureFormat::Pdb
        },
    })
}

/// Element symbol of an atom, falling back to the first letter of its name.
pub fn element_symbol(atom: &pdbtbx::Atom) -> String {
    atom.element()
        .map(|e| e.symbol().to_string())
        .unwrap_or_else(|| atom.name().trim().chars().next().unwrap_or('C').to_string())
}

/// True when the residue looks like an amino acid: it has a `CA` atom whose element is carbon
/// (calcium ions are also named `CA`, with element Ca). C-alpha-only traces therefore count;
/// waters, ions and ligands do not. Modified residues (MSE, SEP, …) pass automatically.
pub fn is_protein_residue(residue: &pdbtbx::Residue) -> bool {
    residue
        .atoms()
        .any(|a| a.name().trim() == "CA" && element_symbol(a).eq_ignore_ascii_case("C"))
}

/// A copy of the structure restricted to protein residues, heavy atoms and the first
/// alternate conformation: what every biophysical metric (SASA, DSSP, Ramachandran,
/// overlaps, interactions) is defined on. Solvent, ions, ligands and hydrogens are removed.
pub fn protein_heavy_atoms(pdb: &pdbtbx::PDB) -> pdbtbx::PDB {
    let mut out = pdb.clone();
    out.remove_residues_by(|r| !is_protein_residue(r));
    out.remove_atoms_by(|a| element_symbol(a).eq_ignore_ascii_case("H"));
    // Alternate conformations: keep the first (highest-occupancy by PDB convention), as
    // mdtraj, DSSP and MolProbity do. Duplicated altloc atoms would otherwise inflate SASA.
    for residue in out.residues_mut() {
        while residue.conformer_count() > 1 {
            residue.remove_conformer(1);
        }
    }
    out.remove_empty();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name)
    }

    #[test]
    fn cif_and_pdb_give_same_metrics() {
        let a = crate::metrics::analyze_pdb_file(&data("1crn.pdb"), None).unwrap();
        let b = crate::metrics::analyze_pdb_file(&data("1crn.cif"), None).unwrap();
        assert!((a.radius_of_gyration - b.radius_of_gyration).abs() < 1e-3);
        assert_eq!(
            a.ramachandran_stats.as_ref().unwrap().outlier_count,
            b.ramachandran_stats.as_ref().unwrap().outlier_count
        );
        assert_eq!(
            a.secondary_structure_summary.as_ref().unwrap().dssp,
            b.secondary_structure_summary.as_ref().unwrap().dssp
        );
        assert_eq!(a.confidence_source, b.confidence_source);
        assert_eq!(
            a.confidence_source,
            crate::confidence::ConfidenceSource::ExperimentalBFactor
        );
    }

    #[test]
    fn headerless_pdb_text_parses() {
        let text =
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 88.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 88.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 88.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 88.00           O\n\
END\n";
        let loaded = load_structure_bytes(text.as_bytes(), None).unwrap();
        assert_eq!(loaded.format, StructureFormat::Pdb);
        assert_eq!(loaded.pdb.atom_count(), 4);
    }

    #[test]
    fn non_finite_coordinates_are_a_parse_error_not_a_panic() {
        // pdbtbx panics on `nan` in a coordinate field; predictors do emit such files.
        let pdb =
            "ATOM      1  CA  ALA A   1         nan     nan     nan  1.00 10.00           C\n\
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 10.00           C\n\
END\n";
        let err = match load_structure_bytes(pdb.as_bytes(), Some("x.pdb")) {
            Err(e) => e,
            Ok(_) => panic!("nan coordinates were accepted"),
        };
        assert!(err.to_string().contains("non-finite"), "{err}");
        let cif = "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n\
_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n\
_atom_site.label_entity_id\n_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n\
_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n\
_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.pdbx_PDB_model_num\n\
ATOM 1 C CA . ALA A 1 1 ? nan nan nan 1.00 10.00 1 A 1\n\
ATOM 2 C CA . ALA A 1 2 ? 3.8 0.0 0.0 1.00 10.00 2 A 1\n";
        let r = std::panic::catch_unwind(|| load_structure_bytes(cif.as_bytes(), Some("x.cif")));
        assert!(r.is_ok(), "mmCIF with nan must not panic");
        assert!(r.unwrap().is_err(), "mmCIF with nan must be rejected");
    }

    #[test]
    fn format_is_sniffed_from_content_when_the_name_says_nothing() {
        assert_eq!(
            sniff_format("data_1UBQ\n#\nloop_\n", None),
            StructureFormat::MmCif
        );
        assert_eq!(
            sniff_format("ATOM      1  N   MET A   1", None),
            StructureFormat::Pdb
        );
        assert_eq!(
            sniff_format("data_x", Some("weird.pdb")),
            StructureFormat::Pdb
        );
        assert_eq!(
            sniff_format("ATOM", Some("x.cif.gz")),
            StructureFormat::MmCif
        );
    }

    #[test]
    fn gz_roundtrip() {
        let raw = std::fs::read(data("1crn.pdb")).unwrap();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut enc, &raw).unwrap();
        let gz = enc.finish().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1crn.pdb.gz");
        std::fs::write(&path, gz).unwrap();
        let pdb = open_structure(&path).unwrap();
        assert_eq!(pdb.residue_count(), 46);
    }

    #[test]
    fn multi_model_file_keeps_first_model_only() {
        let one = std::fs::read_to_string(data("1crn.pdb")).unwrap();
        let atoms: String = one
            .lines()
            .filter(|l| l.starts_with("ATOM") || l.starts_with("HETATM"))
            .map(|l| format!("{l}\n"))
            .collect();
        let text = format!("MODEL        1\n{atoms}ENDMDL\nMODEL        2\n{atoms}ENDMDL\nEND\n");
        let pdb = open_structure_bytes(text.as_bytes(), Some("x.pdb")).unwrap();
        assert_eq!(pdb.model_count(), 1);
        assert_eq!(pdb.residue_count(), 46);
    }

    #[test]
    fn protein_only_drops_waters_ions_and_hydrogens() {
        let text =
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 10.00           O\n\
ATOM      5  H   ALA A   1      -0.500   0.800   0.000  1.00 10.00           H\n\
HETATM    6 CA    CA A 101      10.000  10.000  10.000  1.00 10.00          CA\n\
HETATM    7  O   HOH A 201      12.000  12.000  12.000  1.00 10.00           O\n\
END\n";
        let pdb = open_structure_bytes(text.as_bytes(), Some("x.pdb")).unwrap();
        assert_eq!(pdb.atom_count(), 7);
        let p = protein_heavy_atoms(&pdb);
        assert_eq!(p.residue_count(), 1);
        assert_eq!(p.atom_count(), 4);
        assert!(crate::backbone::extract_backbone(&p).len() == 1);
    }
}
