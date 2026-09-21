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
const HEADER_PREVIEW_BYTES: usize = 16 * 1024;

/// A parsed structure plus the raw head of the file it came from.
pub struct LoadedStructure {
    pub pdb: pdbtbx::PDB,
    /// First [`HEADER_PREVIEW_BYTES`] of the decompressed text, for `confidence` detection.
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
    let result = if is_cif {
        pdbtbx::open_mmcif_raw(text, StrictnessLevel::Loose)
    } else {
        pdbtbx::open_pdb_raw(
            std::io::BufReader::new(bytes),
            pdbtbx::Context::None,
            StrictnessLevel::Loose,
        )
    };
    let (pdb, _warnings) =
        result.map_err(|e| parse_err(if is_cif { "mmCIF" } else { "PDB" }, e))?;
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
}
