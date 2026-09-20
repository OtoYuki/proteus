use crate::error::StorageError;
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// High-density biophysical record for a screening candidate variant.
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
    pub fitness: f64,
}

/// Serializes candidate screening records to standard RFC-4180 CSV format.
pub fn export_records_to_csv(records: &[ScreeningRecord]) -> String {
    let mut out = String::new();
    out.push_str("rank,job_id,header,length,plddt,rg,hydrophobic_burial_pct,helix_pct,strand_pct,coil_pct,favored_ramachandran_pct,rama_outliers,fitness\n");

    for r in records {
        // Escape quotes in header if needed
        let safe_header = if r.header.contains(',') || r.header.contains('"') {
            format!("\"{}\"", r.header.replace('"', "\"\""))
        } else {
            r.header.clone()
        };

        out.push_str(&format!(
            "{},{},{},{},{:.2},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{},{:.2}\n",
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
            r.fitness
        ));
    }

    out
}

/// Serializes candidate screening records to pretty-printed JSON.
pub fn export_records_to_json(records: &[ScreeningRecord]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(records)
}

/// Writes screening records to a target path, inferring format from file extension.
pub async fn save_screening_dataset(
    records: &[ScreeningRecord],
    path: &Path,
) -> Result<(), StorageError> {
    let content = match path.extension().and_then(|s| s.to_str()) {
        Some("json") => {
            export_records_to_json(records).map_err(StorageError::SerializationError)?
        }
        _ => export_records_to_csv(records),
    };

    tokio::fs::write(path, content)
        .await
        .map_err(StorageError::IoError)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            fitness: 78.4,
        }
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
        assert!(lines[1].starts_with("1,"));
        assert!(lines[2].contains("\"crambin, mutated\""));
    }

    #[test]
    fn test_json_export() {
        let records = vec![sample_record(1, "crambin_WT")];
        let json = export_records_to_json(&records).unwrap();
        assert!(json.contains("crambin_WT"));
        assert!(json.contains("hydrophobic_burial_pct"));
    }

    #[tokio::test]
    async fn test_save_dataset_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.csv");
        let records = vec![sample_record(1, "variant_1")];

        save_screening_dataset(&records, &path).await.unwrap();
        let read_back = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(read_back.contains("variant_1"));
    }
}
