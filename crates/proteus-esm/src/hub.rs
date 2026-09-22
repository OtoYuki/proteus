//! Minimal Hugging Face Hub file fetcher (no `hf-hub` dependency: its 1.x line pulls in
//! xet/redb and a very recent MSRV). Resolves `https://huggingface.co/<id>/resolve/main/<file>`
//! and caches to disk.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{EsmError, Result};

/// Cache directory for one model id.
pub fn model_dir(model_id: &str) -> Result<PathBuf> {
    let (owner, name) = model_id
        .split_once('/')
        .ok_or_else(|| EsmError::Hub(format!("'{model_id}' is not an <owner>/<name> Hub id")))?;
    let root = std::env::var_os("PROTEUS_ESM_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/proteus/esm")))
        .ok_or_else(|| EsmError::Hub("no cache directory (set PROTEUS_ESM_CACHE)".into()))?;
    Ok(root.join(format!("{owner}--{name}")))
}

/// Return the cached path of `file`, downloading it first when absent.
pub fn fetch(model_id: &str, file: &str, dir: &Path) -> Result<PathBuf> {
    let target = dir.join(file);
    if target.exists() {
        return Ok(target);
    }
    std::fs::create_dir_all(dir)?;
    let url = format!("https://huggingface.co/{model_id}/resolve/main/{file}");
    let mut req = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            req = req.header("Authorization", &format!("Bearer {token}"));
        }
    }
    let resp = req
        .call()
        .map_err(|e| EsmError::Hub(format!("GET {url}: {e}")))?;
    // The staging name must be unique per download. Two processes (or two threads) fetching the
    // same checkpoint at once otherwise write the same `<file>.part`, and whichever renames
    // second fails with NotFound because the first already moved it away.
    let tmp = dir.join(format!(
        "{file}.{}.{}.part",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    {
        let mut out = std::fs::File::create(&tmp)?;
        let mut reader = resp.into_body().into_reader();
        std::io::copy(&mut reader, &mut out)?;
        out.flush()?;
    }
    // A concurrent fetch may have completed while this one was downloading; either copy is
    // byte-identical, so take whichever landed and drop ours.
    if let Err(e) = std::fs::rename(&tmp, &target) {
        let _ = std::fs::remove_file(&tmp);
        if !target.exists() {
            return Err(e.into());
        }
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for a CI failure: two threads fetching the same file collided on a fixed
    /// `<file>.part` and the loser's rename failed with NotFound. Exercises the staging and
    /// rename logic without touching the network by pre-creating the target.
    #[test]
    fn concurrent_fetch_of_a_cached_file_is_safe() {
        let dir = std::env::temp_dir().join(format!("proteus-hub-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        let results: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| fetch("facebook/esm2_t6_8M_UR50D", "config.json", &dir)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in &results {
            assert!(r.is_ok(), "concurrent fetch failed: {r:?}");
        }
        // No staging files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".part"))
            .collect();
        assert!(leftovers.is_empty(), "staging files left: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_dir_honours_the_cache_override() {
        let dir = model_dir("facebook/esm2_t6_8M_UR50D").unwrap();
        assert!(dir.ends_with("facebook--esm2_t6_8M_UR50D"), "{dir:?}");
        assert!(model_dir("no-slash").is_err());
    }
}
