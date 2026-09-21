use crate::error::StorageError;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, trace};
use uuid::Uuid;

/// Represents the result of storing an object in the Content-Addressable Store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasEntry {
    /// The lowercase 64-character hex BLAKE3 hash of the object content.
    pub hash: String,
    /// Size of the stored object in bytes.
    pub size_bytes: u64,
    /// Whether the object already existed in CAS prior to this store operation.
    pub is_duplicate: bool,
    /// Absolute filesystem path to the immutable CAS object.
    pub path: PathBuf,
}

/// Content-Addressable Storage (CAS) backed by BLAKE3 cryptographic hashing.
///
/// Features:
/// - Fast 256-bit SIMD tree hashing via BLAKE3 (>10 GB/s on modern multi-core).
/// - Two-level directory prefix fan-out (`objects/ab/cd/<hash>`) preventing Linux inode saturation.
/// - Atomic write staging via temporary files (`tmp/<uuid>.tmp`) to ensure zero corrupted partial reads.
/// - In-place O(1) deduplication across screening campaigns.
/// - Strict cryptographic integrity validation on read.
#[derive(Debug, Clone)]
pub struct CasStore {
    root: PathBuf,
    objects_dir: PathBuf,
    tmp_dir: PathBuf,
}

impl CasStore {
    /// Initializes a new CAS store at the specified root directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root = root.into();
        let objects_dir = root.join("objects");
        let tmp_dir = root.join("tmp");

        fs::create_dir_all(&objects_dir)?;
        fs::create_dir_all(&tmp_dir)?;

        Ok(Self {
            root,
            objects_dir,
            tmp_dir,
        })
    }

    /// Root directory of this CAS store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Computes the 64-character hex BLAKE3 hash of raw bytes.
    pub fn compute_hash(data: &[u8]) -> String {
        blake3::hash(data).to_hex().to_string()
    }

    /// Resolves the canonical filesystem path for a given 64-character BLAKE3 hash.
    pub fn object_path(&self, hash: &str) -> Result<PathBuf, StorageError> {
        if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(StorageError::CasError(format!(
                "Invalid BLAKE3 hash format (expected 64 hex chars): '{hash}'"
            )));
        }

        let prefix1 = &hash[0..2];
        let prefix2 = &hash[2..4];

        Ok(self.objects_dir.join(prefix1).join(prefix2).join(hash))
    }

    /// Returns true if an object with the specified hash exists in the store.
    pub fn has_object(&self, hash: &str) -> bool {
        match self.object_path(hash) {
            Ok(p) => p.is_file(),
            Err(_) => false,
        }
    }

    /// Stores raw bytes into the CAS.
    ///
    /// If an identical object already exists, returns the existing entry with `is_duplicate = true`
    /// without performing redundant disk writes.
    pub fn store_bytes(&self, data: &[u8]) -> Result<CasEntry, StorageError> {
        let hash = Self::compute_hash(data);
        let target_path = self.object_path(&hash)?;
        let size_bytes = data.len() as u64;

        if target_path.is_file() {
            trace!(hash = %hash, "CAS deduplication hit: object already exists");
            return Ok(CasEntry {
                hash,
                size_bytes,
                is_duplicate: true,
                path: target_path,
            });
        }

        // Parent directory fan-out
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Atomic write staging in tmp/
        let staging_filename = format!("{}.tmp", Uuid::new_v4());
        let staging_path = self.tmp_dir.join(staging_filename);

        {
            let mut file = File::create(&staging_path)?;
            file.write_all(data)?;
            file.sync_data()?;
        }

        // Atomic rename to final CAS path
        if let Err(e) = fs::rename(&staging_path, &target_path) {
            // Clean up staging file on failure
            let _ = fs::remove_file(&staging_path);
            // In high concurrency, another thread might have just completed the rename
            if target_path.is_file() {
                return Ok(CasEntry {
                    hash,
                    size_bytes,
                    is_duplicate: true,
                    path: target_path,
                });
            }
            return Err(StorageError::IoError(e));
        }

        debug!(hash = %hash, size = size_bytes, "Stored new object in CAS");
        Ok(CasEntry {
            hash,
            size_bytes,
            is_duplicate: false,
            path: target_path,
        })
    }

    /// Stores a UTF-8 string into the CAS.
    pub fn store_str(&self, text: &str) -> Result<CasEntry, StorageError> {
        self.store_bytes(text.as_bytes())
    }

    /// Stores an existing file into the CAS, verifying hash and deduplicating.
    pub fn store_file(&self, source_path: &Path) -> Result<CasEntry, StorageError> {
        let mut file = File::open(source_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        self.store_bytes(&buffer)
    }

    /// Reads and verifies an object from the CAS.
    ///
    /// Validates the read bytes against the expected BLAKE3 hash.
    /// Returns `StorageError::IntegrityViolation` if the file has been tampered with or corrupted.
    pub fn read_bytes(&self, hash: &str) -> Result<Vec<u8>, StorageError> {
        let path = self.object_path(hash)?;
        if !path.is_file() {
            return Err(StorageError::NotFound(format!(
                "CAS object not found: {hash}"
            )));
        }

        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let computed = Self::compute_hash(&buffer);
        if computed != hash {
            return Err(StorageError::IntegrityViolation(format!(
                "CAS integrity check failed for {hash}: computed hash was {computed}"
            )));
        }

        Ok(buffer)
    }

    /// Reads an object and parses it as a UTF-8 string.
    pub fn read_to_string(&self, hash: &str) -> Result<String, StorageError> {
        let bytes = self.read_bytes(hash)?;
        String::from_utf8(bytes)
            .map_err(|e| StorageError::CasError(format!("Object {hash} is not valid UTF-8: {e}")))
    }

    /// Returns the total number of unique objects currently stored in the CAS.
    pub fn count_objects(&self) -> Result<usize, StorageError> {
        let mut count = 0;
        if !self.objects_dir.exists() {
            return Ok(0);
        }

        for prefix1_entry in fs::read_dir(&self.objects_dir)? {
            let prefix1_path = prefix1_entry?.path();
            if prefix1_path.is_dir() {
                for prefix2_entry in fs::read_dir(&prefix1_path)? {
                    let prefix2_path = prefix2_entry?.path();
                    if prefix2_path.is_dir() {
                        for object_entry in fs::read_dir(&prefix2_path)? {
                            let obj_path = object_entry?.path();
                            if obj_path.is_file() {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cas_store_and_read_roundtrip() {
        let tmp = tempdir().unwrap();
        let cas = CasStore::new(tmp.path()).unwrap();

        let payload = b"HEADER    CRAMBIN (1CRN)\nATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N\n";
        let entry = cas.store_bytes(payload).unwrap();

        assert!(!entry.is_duplicate);
        assert_eq!(entry.size_bytes, payload.len() as u64);
        assert_eq!(entry.hash.len(), 64);
        assert!(cas.has_object(&entry.hash));

        // Path should follow two-level fanout
        let expected_subpath = format!(
            "objects/{}/{}/{}",
            &entry.hash[0..2],
            &entry.hash[2..4],
            entry.hash
        );
        assert!(entry.path.ends_with(expected_subpath));

        // Read and verify
        let retrieved = cas.read_bytes(&entry.hash).unwrap();
        assert_eq!(retrieved, payload);
    }

    #[test]
    fn test_cas_deduplication_zero_writes() {
        let tmp = tempdir().unwrap();
        let cas = CasStore::new(tmp.path()).unwrap();

        let payload = b">sp|P01308|INS_HUMAN Insulin\nMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN";

        // First store
        let entry1 = cas.store_bytes(payload).unwrap();
        assert!(!entry1.is_duplicate);

        // Second store of identical payload
        let entry2 = cas.store_bytes(payload).unwrap();
        assert!(entry2.is_duplicate);
        assert_eq!(entry1.hash, entry2.hash);
        assert_eq!(entry1.path, entry2.path);

        // Count objects should remain exactly 1
        assert_eq!(cas.count_objects().unwrap(), 1);
    }

    #[test]
    fn test_cas_integrity_violation_detection() {
        let tmp = tempdir().unwrap();
        let cas = CasStore::new(tmp.path()).unwrap();

        let payload = b"Sensitive protein structure file data";
        let entry = cas.store_bytes(payload).unwrap();

        // Tamper with file on disk directly
        fs::write(&entry.path, b"Tampered data here").unwrap();

        // Read must return IntegrityViolation
        let result = cas.read_bytes(&entry.hash);
        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::IntegrityViolation(msg) => {
                assert!(msg.contains("CAS integrity check failed"));
            }
            other => panic!("Expected IntegrityViolation, got {:?}", other),
        }
    }

    #[test]
    fn test_cas_invalid_hash_rejection() {
        let tmp = tempdir().unwrap();
        let cas = CasStore::new(tmp.path()).unwrap();

        assert!(cas.read_bytes("short-hash").is_err());
        assert!(cas
            .read_bytes("1234567890abcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/")
            .is_err());
    }

    #[test]
    fn test_cas_store_str() {
        let tmp = tempdir().unwrap();
        let cas = CasStore::new(tmp.path()).unwrap();

        let text = ">sequence1\nACDEFGHIKLMNPQRSTVWY\n";
        let entry = cas.store_str(text).unwrap();
        let retrieved = cas.read_to_string(&entry.hash).unwrap();
        assert_eq!(retrieved, text);
    }
}
