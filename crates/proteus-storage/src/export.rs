use crate::error::StorageError;
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// High-density biophysical record for a screening candidate variant.
/// Version of the screening export schema (CSV/JSON columns, Parquet fields).
/// 2: `clashscore` renamed to `heavy_atom_overlap_score`.
/// 3: nullable `esm2_score` column (ESM-2 zero-shot mutation score, `--scorer esm2|hybrid`).
/// 4: `engine` column — which runner produced the structure (`esmfold-api`, `oci`,
///    `simulated`). Rows from the simulator are synthetic helices, not predictions.
pub const EXPORT_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScreeningRecord {
    pub rank: usize,
    pub job_id: Uuid,
    pub header: String,
    pub length: usize,
    pub plddt: f64,
    pub rg: f64,
    pub hydrophobic_burial_pct: f64,
    pub helix_pct: f64,
    pub strand_pct: f64,
    pub coil_pct: f64,
    pub favored_ramachandran_pct: f64,
    pub rama_outliers: usize,
    pub heavy_atom_overlap_score: f64,
    pub hbond_count: usize,
    pub salt_bridge_count: usize,
    pub pi_stacking_count: usize,
    pub cation_pi_count: usize,
    pub fitness: f64,
    /// ESM-2 zero-shot score summed over the variant's substitutions; `None` when not computed.
    #[serde(default)]
    pub esm2_score: Option<f64>,
    /// Runner that produced the structure: `esmfold-api`, `oci`, `simulated`, or `unknown`.
    #[serde(default)]
    pub engine: String,
}

/// Serializes candidate screening records to standard RFC-4180 CSV format.
pub fn export_records_to_csv(records: &[ScreeningRecord]) -> String {
    let mut out = String::new();
    out.push_str("rank,job_id,header,length,plddt,rg,hydrophobic_burial_pct,helix_pct,strand_pct,coil_pct,favored_ramachandran_pct,rama_outliers,heavy_atom_overlap_score,hbond_count,salt_bridge_count,pi_stacking_count,cation_pi_count,fitness,esm2_score,engine\n");

    for r in records {
        // Escape quotes in header if needed
        let safe_header = if r.header.contains(',') || r.header.contains('"') {
            format!("\"{}\"", r.header.replace('"', "\"\""))
        } else {
            r.header.clone()
        };

        out.push_str(&format!(
            "{},{},{},{},{:.2},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{},{:.2},{},{},{},{},{:.2},{},{}\n",
            r.rank,
            r.job_id,
            safe_header,
            r.length,
            r.plddt,
            r.rg,
            r.hydrophobic_burial_pct,
            r.helix_pct,
            r.strand_pct,
            r.coil_pct,
            r.favored_ramachandran_pct,
            r.rama_outliers,
            r.heavy_atom_overlap_score,
            r.hbond_count,
            r.salt_bridge_count,
            r.pi_stacking_count,
            r.cation_pi_count,
            r.fitness,
            r.esm2_score.map(|e| format!("{e:.4}")).unwrap_or_default(),
            r.engine
        ));
    }

    out
}

/// Serializes candidate screening records to pretty-printed JSON.
pub fn export_records_to_json(records: &[ScreeningRecord]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(records)
}

/// Defines the canonical Apache Arrow schema for Proteus screening records.
pub fn screening_record_schema() -> Schema {
    Schema::new(vec![
        Field::new("rank", DataType::Int64, false),
        Field::new("job_id", DataType::Utf8, false),
        Field::new("header", DataType::Utf8, false),
        Field::new("length", DataType::Int64, false),
        Field::new("plddt", DataType::Float64, false),
        Field::new("rg", DataType::Float64, false),
        Field::new("hydrophobic_burial_pct", DataType::Float64, false),
        Field::new("helix_pct", DataType::Float64, false),
        Field::new("strand_pct", DataType::Float64, false),
        Field::new("coil_pct", DataType::Float64, false),
        Field::new("favored_ramachandran_pct", DataType::Float64, false),
        Field::new("rama_outliers", DataType::Int64, false),
        Field::new("heavy_atom_overlap_score", DataType::Float64, false),
        Field::new("hbond_count", DataType::Int64, false),
        Field::new("salt_bridge_count", DataType::Int64, false),
        Field::new("pi_stacking_count", DataType::Int64, false),
        Field::new("cation_pi_count", DataType::Int64, false),
        Field::new("fitness", DataType::Float64, false),
        Field::new("esm2_score", DataType::Float64, true),
        Field::new("engine", DataType::Utf8, false),
    ])
}

/// Converts a batch of `ScreeningRecord` into an Arrow `RecordBatch`.
pub fn records_to_record_batch(
    records: &[ScreeningRecord],
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(screening_record_schema());

    let ranks: Int64Array = records.iter().map(|r| r.rank as i64).collect();
    let job_ids: StringArray = records.iter().map(|r| Some(r.job_id.to_string())).collect();
    let headers: StringArray = records.iter().map(|r| Some(r.header.as_str())).collect();
    let lengths: Int64Array = records.iter().map(|r| r.length as i64).collect();
    let plddts: Float64Array = records.iter().map(|r| Some(r.plddt)).collect();
    let rgs: Float64Array = records.iter().map(|r| Some(r.rg)).collect();
    let hydrophobic_burials: Float64Array = records
        .iter()
        .map(|r| Some(r.hydrophobic_burial_pct))
        .collect();
    let helix_pcts: Float64Array = records.iter().map(|r| Some(r.helix_pct)).collect();
    let strand_pcts: Float64Array = records.iter().map(|r| Some(r.strand_pct)).collect();
    let coil_pcts: Float64Array = records.iter().map(|r| Some(r.coil_pct)).collect();
    let favored_ramas: Float64Array = records
        .iter()
        .map(|r| Some(r.favored_ramachandran_pct))
        .collect();
    let rama_outliers: Int64Array = records.iter().map(|r| r.rama_outliers as i64).collect();
    let overlap_scores: Float64Array = records
        .iter()
        .map(|r| Some(r.heavy_atom_overlap_score))
        .collect();
    let hbonds: Int64Array = records.iter().map(|r| r.hbond_count as i64).collect();
    let salt_bridges: Int64Array = records.iter().map(|r| r.salt_bridge_count as i64).collect();
    let pi_stacks: Int64Array = records.iter().map(|r| r.pi_stacking_count as i64).collect();
    let cation_pis: Int64Array = records.iter().map(|r| r.cation_pi_count as i64).collect();
    let fitnesses: Float64Array = records.iter().map(|r| Some(r.fitness)).collect();
    let esm2_scores: Float64Array = records.iter().map(|r| r.esm2_score).collect();
    let engines: StringArray = records.iter().map(|r| Some(r.engine.as_str())).collect();

    let columns: Vec<ArrayRef> = vec![
        Arc::new(ranks),
        Arc::new(job_ids),
        Arc::new(headers),
        Arc::new(lengths),
        Arc::new(plddts),
        Arc::new(rgs),
        Arc::new(hydrophobic_burials),
        Arc::new(helix_pcts),
        Arc::new(strand_pcts),
        Arc::new(coil_pcts),
        Arc::new(favored_ramas),
        Arc::new(rama_outliers),
        Arc::new(overlap_scores),
        Arc::new(hbonds),
        Arc::new(salt_bridges),
        Arc::new(pi_stacks),
        Arc::new(cation_pis),
        Arc::new(fitnesses),
        Arc::new(esm2_scores),
        Arc::new(engines),
    ];

    RecordBatch::try_new(schema, columns)
}

/// Serializes candidate screening records to Apache Parquet format with ZSTD compression.
pub fn export_records_to_parquet<W: Write + Send>(
    records: &[ScreeningRecord],
    writer: W,
) -> Result<(), StorageError> {
    let batch = records_to_record_batch(records)?;
    let schema = batch.schema();

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "proteus.schema_version".to_string(),
            EXPORT_SCHEMA_VERSION.to_string(),
        )]))
        .build();

    let mut arrow_writer = ArrowWriter::try_new(writer, schema, Some(props))?;
    arrow_writer.write(&batch)?;
    arrow_writer.close()?;

    Ok(())
}

/// Writes screening records to a target path, inferring format from file extension (.parquet, .json, .csv).
pub async fn save_screening_dataset(
    records: &[ScreeningRecord],
    path: &Path,
) -> Result<(), StorageError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(StorageError::IoError)?;
        }
    }

    match path.extension().and_then(|s| s.to_str()) {
        Some("json") => {
            let content =
                export_records_to_json(records).map_err(StorageError::SerializationError)?;
            tokio::fs::write(path, content)
                .await
                .map_err(StorageError::IoError)?;
        }
        Some("parquet") => {
            let mut buffer = Vec::new();
            export_records_to_parquet(records, &mut buffer)?;
            tokio::fs::write(path, buffer)
                .await
                .map_err(StorageError::IoError)?;
        }
        _ => {
            let content = export_records_to_csv(records);
            tokio::fs::write(path, content)
                .await
                .map_err(StorageError::IoError)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn sample_record(rank: usize, header: &str) -> ScreeningRecord {
        ScreeningRecord {
            rank,
            job_id: Uuid::new_v4(),
            header: header.into(),
            length: 46,
            plddt: 85.5,
            rg: 20.12,
            hydrophobic_burial_pct: 54.3,
            helix_pct: 42.0,
            strand_pct: 18.0,
            coil_pct: 40.0,
            favored_ramachandran_pct: 95.5,
            rama_outliers: 0,
            heavy_atom_overlap_score: 1.25,
            hbond_count: 28,
            salt_bridge_count: 1,
            pi_stacking_count: 2,
            cation_pi_count: 1,
            fitness: 78.4,
            esm2_score: Some(-1.25),
            engine: "esmfold-api".into(),
        }
    }

    #[test]
    fn every_format_carries_the_engine_column() {
        let records = vec![sample_record(1, "crambin_WT")];
        let csv = export_records_to_csv(&records);
        assert!(csv.lines().next().unwrap().ends_with(",engine"), "{csv}");
        assert!(
            csv.lines().nth(1).unwrap().ends_with(",esmfold-api"),
            "{csv}"
        );
        let json = export_records_to_json(&records).unwrap();
        assert!(json.contains("\"engine\": \"esmfold-api\""), "{json}");
        let batch = records_to_record_batch(&records).unwrap();
        let engine_col = batch
            .column_by_name("engine")
            .expect("engine field")
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(engine_col.value(0), "esmfold-api");
    }

    #[test]
    fn test_csv_export() {
        let records = vec![
            sample_record(1, "crambin_WT"),
            sample_record(2, "crambin, mutated"),
        ];

        let csv = export_records_to_csv(&records);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // Header + 2 rows
        assert!(lines[0].starts_with("rank,job_id,header"));
        assert!(lines[0].contains("heavy_atom_overlap_score"));
        assert!(!lines[0].contains("clashscore"));
        assert!(lines[1].starts_with("1,"));
        assert!(lines[2].contains("\"crambin, mutated\""));
    }

    #[test]
    fn test_json_export() {
        let records = vec![sample_record(1, "crambin_WT")];
        let json = export_records_to_json(&records).unwrap();
        assert!(json.contains("crambin_WT"));
        assert!(json.contains("hydrophobic_burial_pct"));
        assert!(json.contains("heavy_atom_overlap_score"));
    }

    #[tokio::test]
    async fn test_save_dataset_csv_and_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("results.csv");
        let parquet_path = dir.path().join("results.parquet");

        let records = vec![sample_record(1, "variant_1"), sample_record(2, "variant_2")];

        // Test CSV
        save_screening_dataset(&records, &csv_path).await.unwrap();
        let read_csv = tokio::fs::read_to_string(&csv_path).await.unwrap();
        assert!(read_csv.contains("variant_1"));
        assert!(read_csv.contains("variant_2"));

        // Test Parquet
        save_screening_dataset(&records, &parquet_path)
            .await
            .unwrap();
        let parquet_bytes = tokio::fs::read(&parquet_path).await.unwrap();
        assert!(!parquet_bytes.is_empty());

        // Read Parquet back using ParquetRecordBatchReaderBuilder
        let file = std::fs::File::open(&parquet_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.schema().fields().len(), 20);

        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);

        let rank_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(rank_col.value(0), 1);
        assert_eq!(rank_col.value(1), 2);

        let clash_col = batch
            .column(12)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((clash_col.value(0) - 1.25).abs() < 1e-5);

        let hbond_col = batch
            .column(13)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(hbond_col.value(0), 28);

        let salt_col = batch
            .column(14)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(salt_col.value(0), 1);

        // Test nested directory automatic creation
        let nested_path = dir.path().join("sub/nested/dir/results.parquet");
        save_screening_dataset(&records, &nested_path)
            .await
            .unwrap();
        assert!(nested_path.exists());
    }

    #[test]
    fn parquet_carries_schema_version_metadata() {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let records = vec![sample_record(1, "crambin_WT")];
        let mut buf: Vec<u8> = Vec::new();
        export_records_to_parquet(&records, &mut buf).unwrap();
        let reader = SerializedFileReader::new(bytes::Bytes::from(buf)).unwrap();
        let kv = reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default();
        let version = kv
            .iter()
            .find(|k| k.key == "proteus.schema_version")
            .and_then(|k| k.value.clone());
        assert_eq!(version.as_deref(), Some("4"));
        let names: Vec<String> = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        assert!(names.iter().any(|n| n == "heavy_atom_overlap_score"));
        assert!(!names.iter().any(|n| n == "clashscore"));
    }
}
