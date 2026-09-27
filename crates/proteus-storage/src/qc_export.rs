//! Export of [`StructureQc`] rows (`proteus analyze` over many files) to Parquet, CSV and JSON.

use crate::error::StorageError;
use crate::export::check_export_path;
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use proteus_core::qc::StructureQc;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Version of the structure-QC table (CSV/JSON columns, Parquet fields), written to the
/// Parquet key-value metadata as `proteus.qc_schema_version`.
pub const QC_SCHEMA_VERSION: u32 = 3;

/// Column names, in order. CSV header, Arrow schema and JSON field names all agree.
pub const QC_COLUMNS: &[&str] = &[
    "file",
    "model",
    "n_chains",
    "n_residues",
    "sequence",
    "confidence_source",
    "plddt_mean",
    "plddt_median",
    "plddt_ge70_pct",
    "plddt_ge90_pct",
    "rg",
    "rg_expected",
    "rg_ratio",
    "helix_pct",
    "strand_pct",
    "coil_pct",
    "dssp",
    "rama_favored_pct",
    "rama_allowed_pct",
    "rama_outliers",
    "sasa_total",
    "hydrophobic_burial_pct",
    "heavy_atom_overlap_score",
    "overlap_count",
    "bond_rmsz",
    "bond_outliers",
    "angle_rmsz",
    "angle_outliers",
    "chirality_outliers",
    "handedness_swaps",
    "planarity_outliers",
    "cbeta_outliers",
    "cis_nonpro",
    "twisted_peptides",
    "rotamer_outlier_pct",
    "hbond_count",
    "salt_bridge_count",
    "pi_stacking_count",
    "cation_pi_count",
    "rmsd_to_reference",
    "fitness",
    "interface_binder",
    "interface_target",
    "interface_binder_residues",
    "interface_target_residues",
    "interface_dsasa",
    "interface_sc",
    "interface_hbonds",
    "interface_salt_bridges",
    "iptm",
    "ipae",
    "ipsae_min",
    "ipsae_max",
    "lis",
];

enum Col {
    Str(fn(&StructureQc) -> &str),
    Int(fn(&StructureQc) -> usize),
    F64(fn(&StructureQc) -> f64),
    OptF64(fn(&StructureQc) -> Option<f64>),
    OptInt(fn(&StructureQc) -> Option<usize>),
    OptStr(fn(&StructureQc) -> Option<&str>),
}

fn columns() -> Vec<Col> {
    use Col::*;
    vec![
        Str(|r| &r.file),
        Str(|r| &r.model),
        Int(|r| r.n_chains),
        Int(|r| r.n_residues),
        Str(|r| &r.sequence),
        Str(|r| &r.confidence_source),
        OptF64(|r| r.plddt_mean),
        OptF64(|r| r.plddt_median),
        OptF64(|r| r.plddt_ge70_pct),
        OptF64(|r| r.plddt_ge90_pct),
        F64(|r| r.rg),
        F64(|r| r.rg_expected),
        F64(|r| r.rg_ratio),
        F64(|r| r.helix_pct),
        F64(|r| r.strand_pct),
        F64(|r| r.coil_pct),
        Str(|r| &r.dssp),
        F64(|r| r.rama_favored_pct),
        F64(|r| r.rama_allowed_pct),
        Int(|r| r.rama_outliers),
        F64(|r| r.sasa_total),
        F64(|r| r.hydrophobic_burial_pct),
        F64(|r| r.heavy_atom_overlap_score),
        Int(|r| r.overlap_count),
        OptF64(|r| r.bond_rmsz),
        OptInt(|r| r.bond_outliers),
        OptF64(|r| r.angle_rmsz),
        OptInt(|r| r.angle_outliers),
        OptInt(|r| r.chirality_outliers),
        OptInt(|r| r.handedness_swaps),
        OptInt(|r| r.planarity_outliers),
        OptInt(|r| r.cbeta_outliers),
        OptInt(|r| r.cis_nonpro),
        OptInt(|r| r.twisted_peptides),
        OptF64(|r| r.rotamer_outlier_pct),
        Int(|r| r.hbond_count),
        Int(|r| r.salt_bridge_count),
        Int(|r| r.pi_stacking_count),
        Int(|r| r.cation_pi_count),
        OptF64(|r| r.rmsd_to_reference),
        F64(|r| r.fitness),
        OptStr(|r| r.interface.interface_binder.as_deref()),
        OptStr(|r| r.interface.interface_target.as_deref()),
        OptInt(|r| r.interface.interface_binder_residues),
        OptInt(|r| r.interface.interface_target_residues),
        OptF64(|r| r.interface.interface_dsasa),
        OptF64(|r| r.interface.interface_sc),
        OptInt(|r| r.interface.interface_hbonds),
        OptInt(|r| r.interface.interface_salt_bridges),
        OptF64(|r| r.interface.iptm),
        OptF64(|r| r.interface.ipae),
        OptF64(|r| r.interface.ipsae_min),
        OptF64(|r| r.interface.ipsae_max),
        OptF64(|r| r.interface.lis),
    ]
}

/// The Arrow schema of the QC table.
pub fn qc_schema() -> Schema {
    let fields = QC_COLUMNS
        .iter()
        .zip(columns())
        .map(|(name, col)| match col {
            Col::Str(_) => Field::new(*name, DataType::Utf8, false),
            Col::Int(_) => Field::new(*name, DataType::Int64, false),
            Col::F64(_) => Field::new(*name, DataType::Float64, false),
            Col::OptF64(_) => Field::new(*name, DataType::Float64, true),
            Col::OptInt(_) => Field::new(*name, DataType::Int64, true),
            Col::OptStr(_) => Field::new(*name, DataType::Utf8, true),
        })
        .collect::<Vec<_>>();
    Schema::new(fields)
}

/// Converts QC rows into one Arrow `RecordBatch`.
pub fn qc_to_record_batch(rows: &[StructureQc]) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let arrays: Vec<ArrayRef> = columns()
        .into_iter()
        .map(|col| -> ArrayRef {
            match col {
                Col::Str(f) => Arc::new(rows.iter().map(|r| Some(f(r))).collect::<StringArray>()),
                Col::Int(f) => Arc::new(rows.iter().map(|r| f(r) as i64).collect::<Int64Array>()),
                Col::F64(f) => Arc::new(rows.iter().map(|r| Some(f(r))).collect::<Float64Array>()),
                Col::OptF64(f) => Arc::new(rows.iter().map(f).collect::<Float64Array>()),
                Col::OptInt(f) => Arc::new(
                    rows.iter()
                        .map(|r| f(r).map(|v| v as i64))
                        .collect::<Int64Array>(),
                ),
                Col::OptStr(f) => Arc::new(rows.iter().map(f).collect::<StringArray>()),
            }
        })
        .collect();
    RecordBatch::try_new(Arc::new(qc_schema()), arrays)
}

/// Writes QC rows as ZSTD-compressed Parquet.
pub fn export_qc_to_parquet<W: Write + Send>(
    rows: &[StructureQc],
    writer: W,
) -> Result<(), StorageError> {
    let batch = qc_to_record_batch(rows)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "proteus.qc_schema_version".to_string(),
            QC_SCHEMA_VERSION.to_string(),
        )]))
        .build();
    let mut w = ArrowWriter::try_new(writer, batch.schema(), Some(props))?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// RFC 4180 CSV with a header row. Floats keep full precision; missing values are empty.
pub fn export_qc_to_csv(rows: &[StructureQc]) -> String {
    let cols = columns();
    let mut out = QC_COLUMNS.join(",");
    out.push('\n');
    for r in rows {
        let fields: Vec<String> = cols
            .iter()
            .map(|c| match c {
                Col::Str(f) => csv_field(f(r)),
                Col::Int(f) => f(r).to_string(),
                Col::F64(f) => f(r).to_string(),
                Col::OptF64(f) => f(r).map(|v| v.to_string()).unwrap_or_default(),
                Col::OptInt(f) => f(r).map(|v| v.to_string()).unwrap_or_default(),
                Col::OptStr(f) => f(r).map(csv_field).unwrap_or_default(),
            })
            .collect();
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

/// Writes QC rows to `path`, choosing the format from its extension
/// (`.parquet`, `.csv` or `.json`).
pub fn save_qc_table(rows: &[StructureQc], path: &Path) -> Result<(), StorageError> {
    check_export_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    let bytes = match ext.as_deref() {
        Some("parquet") => {
            let mut buf = Vec::new();
            export_qc_to_parquet(rows, &mut buf)?;
            buf
        }
        Some("csv") => export_qc_to_csv(rows).into_bytes(),
        _ => serde_json::to_vec_pretty(rows)?,
    };
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn row(model: &str, plddt: Option<f64>) -> StructureQc {
        StructureQc {
            file: format!("models/{model}.pdb"),
            model: model.into(),
            n_chains: 1,
            n_residues: 3,
            sequence: "ACD".into(),
            confidence_source: if plddt.is_some() {
                "predicted"
            } else {
                "experimental"
            }
            .into(),
            plddt_mean: plddt,
            plddt_median: plddt,
            plddt_ge70_pct: plddt.map(|_| 100.0),
            plddt_ge90_pct: plddt.map(|_| 0.0),
            rg: 4.5,
            rg_expected: 3.4,
            rg_ratio: 1.32,
            helix_pct: 0.0,
            strand_pct: 0.0,
            coil_pct: 100.0,
            dssp: "---".into(),
            rama_favored_pct: 100.0,
            rama_allowed_pct: 0.0,
            rama_outliers: 0,
            sasa_total: 512.25,
            hydrophobic_burial_pct: 10.0,
            heavy_atom_overlap_score: 0.0,
            overlap_count: 0,
            bond_rmsz: Some(0.8),
            bond_outliers: Some(0),
            angle_rmsz: Some(1.1),
            angle_outliers: Some(1),
            chirality_outliers: Some(0),
            handedness_swaps: Some(0),
            planarity_outliers: Some(0),
            cbeta_outliers: Some(0),
            cis_nonpro: Some(0),
            twisted_peptides: Some(0),
            rotamer_outlier_pct: None,
            hbond_count: 1,
            salt_bridge_count: 0,
            pi_stacking_count: 0,
            cation_pi_count: 0,
            rmsd_to_reference: None,
            fitness: 55.5,
            interface: Default::default(),
        }
    }

    #[test]
    fn schema_json_and_csv_agree_on_column_names() {
        let schema = qc_schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, QC_COLUMNS);

        let json = serde_json::to_value(row("a", Some(80.0))).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        let mut expected = QC_COLUMNS.to_vec();
        keys.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            keys, expected,
            "struct fields and QC_COLUMNS must stay in step"
        );

        let csv = export_qc_to_csv(&[row("a", None)]);
        assert_eq!(csv.lines().next().unwrap(), QC_COLUMNS.join(","));
    }

    #[test]
    fn csv_quotes_awkward_file_names_and_leaves_missing_values_empty() {
        let mut r = row("a", None);
        r.file = "odd, \"name\".pdb".into();
        let csv = export_qc_to_csv(&[r]);
        let line = csv.lines().nth(1).unwrap();
        assert!(line.starts_with("\"odd, \"\"name\"\".pdb\",a,1,3,ACD,experimental,,,,,4.5,"));
    }

    #[test]
    fn parquet_round_trips_values_nulls_and_the_schema_version() {
        let rows = vec![row("a", Some(81.25)), row("b", None)];
        let mut buf = Vec::new();
        export_qc_to_parquet(&rows, &mut buf).unwrap();

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(buf)).unwrap();
        let kv = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .clone();
        assert!(kv.iter().any(|k| k.key == "proteus.qc_schema_version"
            && k.value.as_deref() == Some(&QC_SCHEMA_VERSION.to_string())));
        let batch = builder.build().unwrap().next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        let plddt = batch
            .column_by_name("plddt_mean")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(plddt.value(0), 81.25);
        assert!(plddt.is_null(1));
        let model = batch
            .column_by_name("model")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(model.value(1), "b");
    }

    #[test]
    fn unknown_extensions_are_refused_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qc.xlsx");
        assert!(save_qc_table(&[row("a", None)], &path).is_err());
        assert!(!path.exists());
    }
}
