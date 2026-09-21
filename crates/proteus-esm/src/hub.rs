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
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/proteus/esm"))
        })
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
    let tmp = dir.join(format!("{file}.part"));
    {
        let mut out = std::fs::File::create(&tmp)?;
        let mut reader = resp.into_body().into_reader();
        std::io::copy(&mut reader, &mut out)?;
        out.flush()?;
    }
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}
