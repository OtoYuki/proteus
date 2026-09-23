//! Minimal Hugging Face Hub file fetcher (no `hf-hub` dependency: its 1.x line pulls in
//! xet/redb and a very recent MSRV). Resolves `https://huggingface.co/<id>/resolve/main/<file>`
//! and caches to disk.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{io_at, EsmError, Result};

/// Cache directory for one model id.
pub fn model_dir(model_id: &str) -> Result<PathBuf> {
    // Hub ids are `<owner>/<name>` over [A-Za-z0-9._-]. Anything else (a mistyped local path, a
    // stray shell word) would otherwise become a request that fails with a bare HTTP status and
    // an empty cache directory named after it.
    let valid = |s: &str| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    let (owner, name) = model_id
        .split_once('/')
        .filter(|(o, n)| valid(o) && valid(n))
        .ok_or_else(|| {
            EsmError::Hub(format!(
                "'{model_id}' is neither an existing directory nor an <owner>/<name> Hub id"
            ))
        })?;
    let root = std::env::var_os("PROTEUS_ESM_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/proteus/esm")))
        .ok_or_else(|| EsmError::Hub("no cache directory (set PROTEUS_ESM_CACHE)".into()))?;
    Ok(root.join(format!("{owner}--{name}")))
}

/// Return the cached path of `file`, downloading it first when absent.
pub fn fetch(model_id: &str, file: &str, dir: &Path) -> Result<PathBuf> {
    fetch_from(HUB, model_id, file, dir)
}

const HUB: &str = "https://huggingface.co";

/// Removes the staging file when dropped, so that no error path leaves a `.part` behind.
struct Staging(PathBuf);

impl Drop for Staging {
    fn drop(&mut self) {
        // After a successful rename the file is gone and this fails harmlessly.
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Turn a failed request into something the caller can act on.
fn request_error(model_id: &str, file: &str, url: &str, e: ureq::Error) -> EsmError {
    match e {
        // config.json is fetched first, so a 404 here means the repository exists without
        // safetensors weights: the 3B and 15B ESM-2 repositories.
        ureq::Error::StatusCode(404) if file == "model.safetensors" => EsmError::Hub(format!(
            "{model_id} has no model.safetensors on the Hub (the ESM-2 3B and 15B repositories \
             publish only PyTorch .bin files). Convert it once with `python -c \"from \
             transformers import EsmForMaskedLM; EsmForMaskedLM.from_pretrained('{model_id}')\
             .save_pretrained('<dir>', max_shard_size='100GB')\"` and load that directory \
             (Esm2::from_files with <dir>/config.json and <dir>/model.safetensors)"
        )),
        // The Hub answers 401 for repositories that do not exist as well as for gated ones.
        ureq::Error::StatusCode(code @ (401 | 403 | 404)) => EsmError::Hub(format!(
            "{model_id}: HTTP {code} for {file}. No such model on the Hub, or it is gated or \
             private (set HF_TOKEN); if '{model_id}' was meant as a local directory, it does not \
             exist"
        )),
        e => EsmError::Hub(format!("GET {url}: {e}")),
    }
}

fn fetch_from(base: &str, model_id: &str, file: &str, dir: &Path) -> Result<PathBuf> {
    let target = dir.join(file);
    if target.exists() {
        return Ok(target);
    }
    let url = format!("{base}/{model_id}/resolve/main/{file}");
    let mut req = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            req = req.header("Authorization", &format!("Bearer {token}"));
        }
    }
    let resp = req
        .call()
        .map_err(|e| request_error(model_id, file, &url, e))?;
    // Only now: a failed request should not leave an empty cache directory behind.
    std::fs::create_dir_all(dir).map_err(|e| io_at(dir, e))?;
    // The staging name must be unique per download. Two processes (or two threads) fetching the
    // same checkpoint at once otherwise write the same `<file>.part`, and whichever renames
    // second fails with NotFound because the first already moved it away.
    let tmp = Staging(dir.join(format!(
        "{file}.{}.{}.part",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )));
    {
        let mut out = std::fs::File::create(&tmp.0).map_err(|e| io_at(&tmp.0, e))?;
        let mut reader = resp.into_body().into_reader();
        std::io::copy(&mut reader, &mut out)
            .map_err(|e| EsmError::Hub(format!("GET {url}: download failed: {e}")))?;
        out.flush().map_err(|e| io_at(&tmp.0, e))?;
    }
    // A concurrent fetch may have completed while this one was downloading; either copy is
    // byte-identical, so take whichever landed and drop ours.
    if let Err(e) = std::fs::rename(&tmp.0, &target) {
        if !target.exists() {
            return Err(io_at(&target, e));
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

    /// A one-shot HTTP server on 127.0.0.1 that answers every request with `response`.
    fn serve(response: &'static [u8]) -> String {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { return };
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);
                let _ = s.write_all(response);
            }
        });
        format!("http://{addr}")
    }

    fn part_files(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.contains(".part"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn interrupted_download_leaves_no_staging_file() {
        // Content-Length promises 1 MB, the connection closes after 1000 bytes: the copy fails
        // after the staging file was created, and `?` used to return before removing it.
        let base = serve(
            b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\nxxxxxxxxxx",
        );
        let dir = tempfile::tempdir().unwrap();
        let err = fetch_from(
            &base,
            "facebook/esm2_t6_8M_UR50D",
            "model.safetensors",
            dir.path(),
        )
        .unwrap_err();
        assert!(!dir.path().join("model.safetensors").exists(), "{err}");
        assert_eq!(part_files(dir.path()), Vec::<String>::new(), "{err}");
    }

    #[test]
    fn missing_safetensors_says_how_to_convert() {
        // What the 3B and 15B repositories answer; used to surface as "http status: 404".
        let base =
            serve(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let dir = tempfile::tempdir().unwrap();
        let id = "facebook/esm2_t36_3B_UR50D";
        let err = fetch_from(&base, id, "model.safetensors", dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no model.safetensors"), "{err}");
        assert!(err.contains("save_pretrained"), "{err}");
        assert!(err.contains("from_files"), "{err}");
    }

    #[test]
    fn unknown_repository_is_explained_and_leaves_no_cache_directory() {
        // The Hub answers 401 for a repository that does not exist.
        let base =
            serve(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("models--esm2");
        let err = fetch_from(&base, "models/esm2", "config.json", &dir)
            .unwrap_err()
            .to_string();
        assert!(err.contains("HF_TOKEN"), "{err}");
        assert!(err.contains("local directory"), "{err}");
        assert!(!dir.exists(), "an empty cache directory was created");
    }

    #[test]
    fn unwritable_cache_is_reported_with_its_path() {
        let base = serve(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
        let root = tempfile::tempdir().unwrap();
        // A regular file where the cache directory should be: create_dir_all fails.
        let blocker = root.path().join("cache");
        std::fs::write(&blocker, b"").unwrap();
        let dir = blocker.join("facebook--esm2_t6_8M_UR50D");
        let err = fetch_from(&base, "facebook/esm2_t6_8M_UR50D", "config.json", &dir)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&dir.display().to_string()), "{err}");
    }

    #[test]
    fn things_that_are_not_hub_ids_are_refused() {
        for bad in [
            "./models/esm2",
            "../esm2",
            "/abs/esm2",
            "a/b/c",
            "35M masked facebook/esm2_t12_35M_UR50D",
            "facebook/",
            "/esm2",
            "facebook/..",
        ] {
            let err = model_dir(bad).unwrap_err().to_string();
            assert!(
                err.contains("neither an existing directory"),
                "{bad}: {err}"
            );
        }
        assert!(model_dir("facebook/esm2_t6_8M_UR50D").is_ok());
        assert!(model_dir("some-user/esm2.v1_ft").is_ok());
    }

    #[test]
    fn model_dir_honours_the_cache_override() {
        let dir = model_dir("facebook/esm2_t6_8M_UR50D").unwrap();
        assert!(dir.ends_with("facebook--esm2_t6_8M_UR50D"), "{dir:?}");
        assert!(model_dir("no-slash").is_err());
    }
}
