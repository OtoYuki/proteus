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

/// A parse failure the caller can act on, for the commonest mistake of all: handing a
/// structure command something that is not a structure. pdbtbx's own message for this is
/// "No Atoms in the given PDB struct while validating", which tells a first-time user nothing
/// about what they did or what to do instead.
fn no_coordinates_err(text: &str, what: &str) -> Option<CoreError> {
    let looks_like_fasta = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .is_some_and(|l| l.starts_with('>'));
    let hint = if looks_like_fasta {
        " — this looks like a FASTA file. Structure commands take a PDB or mmCIF file; to go \
         from a sequence to a structure use `proteus submit` or `proteus screen`."
    } else {
        " — no ATOM or HETATM records were found. Structure commands take a PDB or mmCIF file \
         (optionally gzipped)."
    };
    Some(CoreError::StructureParseError(format!(
        "{what} contains no atomic coordinates{hint}"
    )))
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
/// from the content (mmCIF starts with a `data_` block). Used by the HTML viewer exports, which must
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

/// The machine-readable provenance records wherever they sit in the file: `EXPDTA` and
/// `CRYST1` (PDB), `_exptl.method` with the rows of its loop, and the first ModelCIF
/// `_ma_qa_metric` line (mmCIF). In deposited mmCIF `_exptl` usually comes 70–210 KB in, far
/// past [`HEADER_PREVIEW_BYTES`], so confidence detection needs them collected explicitly.
fn provenance_lines(text: &str) -> String {
    let mut out = String::new();
    let mut take_next = 0;
    let mut seen_ma = false;
    for line in text.lines() {
        if take_next > 0 {
            take_next -= 1;
            out.push_str(line);
            out.push('\n');
            if line.starts_with('#') || line.starts_with("loop_") {
                take_next = 0;
            }
            continue;
        }
        if line.starts_with("EXPDTA") || line.starts_with("CRYST1") {
            out.push_str(line);
            out.push('\n');
        } else if line.starts_with("_exptl.method") {
            out.push_str(line);
            out.push('\n');
            // In a loop the value is on a later row; also covers a value wrapped onto the
            // next line.
            take_next = 4;
        } else if !seen_ma && line.starts_with("_ma_qa_metric") {
            seen_ma = true;
            out.push_str(line);
            out.push('\n');
        } else if line.starts_with("ATOM") || line.starts_with("_atom_site.") {
            // Provenance precedes the coordinates in both formats; stop scanning there.
            break;
        }
    }
    if out.is_empty() {
        out
    } else {
        format!("\n{out}")
    }
}

/// PDB is a fixed-column ASCII format, and pdbtbx slices record lines by byte offset: a
/// multi-byte UTF-8 character in a record it reads panics inside the lexer ("byte index is
/// not a char boundary"). In an atom record that shifts every later column, so the file is
/// refused with the line named; in free-text records (REMARK, HEADER, …) each such character
/// becomes `?`, which keeps the columns and loses nothing the analysis reads.
fn ascii_records(coordinates: String) -> Result<String, CoreError> {
    if coordinates.is_ascii() {
        return Ok(coordinates);
    }
    let mut out = String::with_capacity(coordinates.len());
    for (n, line) in coordinates.lines().enumerate() {
        if !line.is_ascii() {
            if ["ATOM", "HETATM", "ANISOU"]
                .iter()
                .any(|r| line.starts_with(r))
            {
                return Err(CoreError::StructureParseError(format!(
                    "PDB atom record {} contains non-ASCII characters, so its columns cannot \
                     be read; PDB is a fixed-column ASCII format",
                    n + 1
                )));
            }
            out.extend(line.chars().map(|c| if c.is_ascii() { c } else { '?' }));
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    Ok(out)
}

thread_local! {
    /// Set while this thread runs pdbtbx under [`parse_guarded`]; the panic hook stays quiet.
    static PARSING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run a pdbtbx parse, turning a panic inside it into an ordinary parse error. pdbtbx asserts
/// rather than errors on several malformed inputs (an mmCIF `?` where a coordinate, element
/// or atom id is required, among others); a structure file from anywhere must not be able to
/// take down a batch run or the daemon. The default panic message is suppressed for these
/// parses only — the hook is installed once and defers to the previous hook otherwise.
fn parse_guarded<T>(kind: &str, parse: impl FnOnce() -> T) -> Result<T, CoreError> {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !PARSING.with(|p| p.get()) {
                previous(info);
            }
        }));
    });
    PARSING.with(|p| p.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(parse));
    PARSING.with(|p| p.set(false));
    result.map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "malformed input".into());
        CoreError::StructureParseError(format!("{kind}: {msg}"))
    })
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
    let mut header_preview = text
        .char_indices()
        .take_while(|(i, _)| *i < HEADER_PREVIEW_BYTES)
        .map(|(_, c)| c)
        .collect::<String>();
    header_preview.push_str(&provenance_lines(text));
    let is_cif = looks_like_cif(text, hint);
    reject_non_finite_coordinates(text, is_cif)?;
    let kind = if is_cif { "mmCIF" } else { "PDB" };
    let result = if is_cif {
        parse_guarded(kind, || {
            pdbtbx::ReadOptions::default()
                .set_format(pdbtbx::Format::Mmcif)
                .set_level(StrictnessLevel::Loose)
                .read_raw(std::io::BufReader::new(text.as_bytes()))
        })?
    } else {
        let coordinates = ascii_records(coordinate_records_only(text))?;
        parse_guarded(kind, || {
            pdbtbx::ReadOptions::default()
                .set_format(pdbtbx::Format::Pdb)
                .set_level(StrictnessLevel::Loose)
                .read_raw(std::io::BufReader::new(coordinates.as_bytes()))
        })?
    };
    let (mut pdb, _warnings) = result.map_err(|e| {
        // A file with no coordinates at all is a user mistake, not a malformed structure, and
        // deserves a message that names the mistake.
        let has_coordinates = text
            .lines()
            .any(|l| l.starts_with("ATOM") || l.starts_with("HETATM") || l.contains("_atom_site."));
        if has_coordinates {
            parse_err(kind, e)
        } else {
            no_coordinates_err(text, kind).unwrap_or_else(|| parse_err(kind, e))
        }
    })?;
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

    /// The commonest first-use mistake in both directions. The message has to name what was
    /// handed over and what to run instead; "No Atoms in the given PDB struct while validating"
    /// (pdbtbx's own) tells a new user neither.
    #[test]
    fn a_fasta_passed_to_a_structure_command_says_so() {
        let err = open_structure_bytes(b">1crn\nTTCCPSIVARSNFNVCRLPG\n", Some("x.pdb"))
            .expect_err("a FASTA is not a structure");
        let msg = err.to_string();
        assert!(msg.contains("looks like a FASTA"), "{msg}");
        assert!(msg.contains("proteus submit"), "{msg}");
        assert!(
            !msg.contains("while validating"),
            "internal parser text leaked: {msg}"
        );
    }

    #[test]
    fn a_file_with_no_coordinates_says_what_is_missing() {
        let err = open_structure_bytes(b"hello world\n", Some("x.pdb")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no ATOM or HETATM records"), "{msg}");
    }
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
    #[test]
    fn malformed_files_are_errors_not_panics() {
        // Reported: an mmCIF with `?` in a required field, and PDB lines with multi-byte
        // characters, panicked inside pdbtbx and took the whole process down.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/edge");
        for name in [
            "q_x.cif",
            "q_type.cif",
            "q_id.cif",
            "q_comp.cif",
            "nonascii.pdb",
            "nonascii2.pdb",
            "nonascii3.pdb",
            "nonascii4.pdb",
        ] {
            let path = dir.join(name);
            let outcome = std::panic::catch_unwind(|| load_structure(&path));
            assert!(outcome.is_ok(), "{name} panicked");
        }
        let err = load_structure(&dir.join("nonascii4.pdb"))
            .err()
            .expect("non-ASCII atom");
        assert!(err.to_string().contains("non-ASCII"), "{err}");
        // Non-ASCII text in REMARKs is not a reason to refuse the coordinates.
        assert_eq!(
            load_structure(&dir.join("nonascii3.pdb"))
                .unwrap()
                .pdb
                .atom_count(),
            1
        );
        let err = load_structure(&dir.join("q_x.cif"))
            .err()
            .expect("missing coordinate");
        assert!(
            err.to_string().starts_with("Structure parse error: mmCIF"),
            "{err}"
        );
    }
}
