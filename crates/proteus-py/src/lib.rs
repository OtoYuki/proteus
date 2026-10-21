//! The native half of the `proteus_bio` Python package: [`proteus_core::qc`] over a list of
//! paths, returned as one Arrow record batch. The Python half (`python/proteus_bio`) turns the
//! batch into a `pyarrow.Table` and the failures into warnings.

use arrow_pyarrow::ToPyArrow;
use proteus_core::confidence::ConfidenceSource;
use proteus_core::interface::InterfaceSpec;
use proteus_core::qc::{analyze_many, collect_structure_files, QcOptions};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// `(path, reason)` for each input that could not be analysed.
type Failures = Vec<(String, String)>;

/// Analyse every structure under `paths`. Returns `(record_batch, failures)`.
#[pyfunction]
#[pyo3(signature = (paths, interface=None, reference=None, confidence="auto", jobs=None))]
fn analyze_batch<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    interface: Option<&str>,
    reference: Option<PathBuf>,
    confidence: &str,
    jobs: Option<usize>,
) -> PyResult<(Bound<'py, PyAny>, Failures)> {
    let interface = interface
        .map(InterfaceSpec::parse)
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("interface: {e}")))?;
    let confidence = match confidence {
        "auto" => None,
        "predicted" => Some(ConfidenceSource::Predicted),
        "experimental" => Some(ConfidenceSource::ExperimentalBFactor),
        other => {
            return Err(PyValueError::new_err(format!(
                "confidence must be 'auto', 'predicted' or 'experimental', not {other:?}"
            )))
        }
    };
    let reference = reference
        .as_deref()
        .map(proteus_core::io::open_structure)
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("reference: {e}")))?;
    let jobs = jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    // The analysis holds no Python objects, so other Python threads run meanwhile.
    let (rows, failures) = py.detach(|| {
        let (inputs, mut failures) = collect_structure_files(&paths);
        let opts = QcOptions {
            reference: reference.as_ref(),
            confidence,
            interface: interface.as_ref(),
        };
        let mut rows = Vec::with_capacity(inputs.len());
        for (path, r) in inputs
            .iter()
            .zip(analyze_many(&inputs, &opts, jobs, &|| {}))
        {
            match r {
                Ok(row) => rows.push(row),
                Err(e) => failures.push((path.clone(), e)),
            }
        }
        (rows, failures)
    });

    let batch = proteus_storage::qc_export::qc_to_record_batch(&rows)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let failures = failures
        .into_iter()
        .map(|(p, why)| (p.display().to_string(), why))
        .collect();
    Ok((batch.to_pyarrow(py)?, failures))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_batch, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
