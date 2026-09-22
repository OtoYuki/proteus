//! Execution backends for GA4GH TES executors.
//!
//! [`ContainerExecutor`] runs each executor inside its declared `image` through the
//! Podman/Docker socket, with the task's `resources` enforced as container limits. This is the
//! backend TES requires. [`HostExecutor`] runs the command as a host process, ignores `image`,
//! and exists only for loopback-bound local development.

use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path};
use std::time::Duration;

use async_trait::async_trait;
use bollard::container::LogOutput;
use bollard::models::{ContainerCreateBody, HostConfig};
use bollard::query_parameters::{
    AttachContainerOptions, CreateContainerOptions, CreateImageOptions, KillContainerOptions,
    LogsOptions, RemoveContainerOptions, StartContainerOptions, WaitContainerOptions,
};
use bollard::Docker;
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::error::EngineError;

/// Everything one TES executor needs to run.
pub struct ExecutorRequest<'a> {
    pub image: &'a str,
    pub command: &'a [String],
    pub workdir: Option<&'a str>,
    pub env: &'a HashMap<String, String>,
    pub stdin: Option<&'a str>,
    /// Host directory that mirrors the container's root for every declared path
    /// (`<work_dir>/data/in.pdb` ↔ `/data/in.pdb`).
    pub work_dir: &'a Path,
    /// Absolute container paths declared by the task (inputs, outputs, volumes, workdirs); the
    /// first component of each is bind-mounted from `work_dir`.
    pub mount_roots: &'a BTreeSet<String>,
    pub cpu_cores: Option<u32>,
    pub ram_gb: Option<f64>,
    /// Allow outbound network from the container (off by default).
    pub network: bool,
    pub timeout: Duration,
    pub cancel: CancellationToken,
}

#[derive(Debug, Default, Clone)]
pub struct ExecutorResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i64,
    /// Backend diagnostics for `TesTaskLog.system_logs` (pulls, limits, kills).
    pub system_logs: Vec<String>,
}

#[async_trait]
pub trait TesExecutor: Send + Sync {
    async fn run(&self, req: ExecutorRequest<'_>) -> Result<ExecutorResult, EngineError>;
    /// `"container"` or `"host"`, reported in `service-info` tags.
    fn kind(&self) -> &'static str;
}

/// Container paths that must never be shadowed by a bind mount from the work directory.
const FORBIDDEN_MOUNT_ROOTS: &[&str] = &[
    "bin", "boot", "dev", "etc", "lib", "lib32", "lib64", "libx32", "proc", "root", "run", "sbin",
    "sys", "usr", "var",
];

/// First path component of an absolute container path (`/data/x/y` → `data`), or an error for
/// relative paths and system directories.
pub fn mount_root(path: &str) -> Result<String, EngineError> {
    let p = Path::new(path);
    if !p.is_absolute() {
        return Err(EngineError::Tes(format!(
            "TES paths must be absolute container paths, got '{path}'"
        )));
    }
    // Container paths are joined onto the host work dir; a `..` would climb out of it.
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(EngineError::Tes(format!(
            "TES paths may not contain '..', got '{path}'"
        )));
    }
    let root = p
        .components()
        .find_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().to_string()),
            _ => None,
        })
        .ok_or_else(|| EngineError::Tes(format!("'{path}' has no directory component")))?;
    if FORBIDDEN_MOUNT_ROOTS.contains(&root.as_str()) {
        return Err(EngineError::Tes(format!(
            "'{path}' would mount over the container's /{root}; use a dedicated directory such as /data or /work"
        )));
    }
    Ok(root)
}

/// Runs executors as host processes. **Ignores `image`.** Loopback-only development aid.
///
/// The working directory is `<work_dir>/<executor.workdir>`, so commands that use paths
/// relative to their `workdir` see the staged inputs; absolute container paths are not
/// rewritten and will not resolve on the host.
pub struct HostExecutor;

#[async_trait]
impl TesExecutor for HostExecutor {
    async fn run(&self, req: ExecutorRequest<'_>) -> Result<ExecutorResult, EngineError> {
        let Some(prog) = req.command.first() else {
            return Ok(ExecutorResult::default());
        };
        let exec_dir = req
            .workdir
            .map(|w| req.work_dir.join(w.trim_start_matches('/')))
            .unwrap_or_else(|| req.work_dir.to_path_buf());
        tokio::fs::create_dir_all(&exec_dir).await?;
        let mut cmd = tokio::process::Command::new(prog);
        cmd.args(&req.command[1..]).current_dir(&exec_dir);
        for (k, v) in req.env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        cmd.kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ExecutorResult {
                    stderr: format!("Failed to spawn command '{prog}': {e}"),
                    exit_code: -1,
                    system_logs: vec![format!("host executor: spawn failed: {e}")],
                    ..Default::default()
                })
            }
        };
        if let (Some(mut stdin), Some(text)) = (child.stdin.take(), req.stdin) {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(text.as_bytes()).await;
        } else {
            drop(child.stdin.take());
        }
        let waited = tokio::select! {
            r = tokio::time::timeout(req.timeout, child.wait_with_output()) => r,
            _ = req.cancel.cancelled() => {
                return Ok(ExecutorResult {
                    exit_code: -1,
                    system_logs: vec!["host executor: canceled".into()],
                    ..Default::default()
                });
            }
        };
        match waited {
            Ok(Ok(output)) => Ok(ExecutorResult {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1) as i64,
                system_logs: vec![format!(
                    "host executor: ran '{prog}' on the daemon host; image '{}' ignored",
                    req.image
                )],
            }),
            Ok(Err(e)) => Ok(ExecutorResult {
                stderr: e.to_string(),
                exit_code: -1,
                ..Default::default()
            }),
            Err(_) => Ok(ExecutorResult {
                exit_code: -1,
                system_logs: vec![format!(
                    "host executor: timed out after {}s",
                    req.timeout.as_secs()
                )],
                ..Default::default()
            }),
        }
    }

    fn kind(&self) -> &'static str {
        "host"
    }
}

/// Runs executors inside their declared image via bollard.
pub struct ContainerExecutor {
    docker: Docker,
    pull_missing: bool,
}

impl ContainerExecutor {
    /// Connect to the first reachable socket (rootless Podman, Docker, system Podman).
    pub fn connect(pull_missing: bool) -> Result<Self, EngineError> {
        let uid = current_uid();
        let podman_user = format!("/run/user/{uid}/podman/podman.sock");
        let candidates = [
            podman_user.as_str(),
            "/var/run/docker.sock",
            "/run/podman/podman.sock",
        ];
        for sock in candidates {
            if Path::new(sock).exists() {
                match Docker::connect_with_unix(sock, 120, bollard::API_DEFAULT_VERSION) {
                    Ok(docker) => {
                        info!("TES container executor using {sock}");
                        return Ok(Self {
                            docker,
                            pull_missing,
                        });
                    }
                    Err(e) => warn!("socket {sock} unusable: {e}"),
                }
            }
        }
        let docker = Docker::connect_with_socket_defaults().map_err(|e| {
            EngineError::Container(format!(
                "no container runtime reachable (tried {}): {e}. Start a Podman socket \
                 (`systemctl --user enable --now podman.socket`) or Docker, or use \
                 `--executor host` for loopback-only development.",
                candidates.join(", ")
            ))
        })?;
        Ok(Self {
            docker,
            pull_missing,
        })
    }

    async fn ensure_image(&self, image: &str, logs: &mut Vec<String>) -> Result<(), EngineError> {
        if self.docker.inspect_image(image).await.is_ok() {
            return Ok(());
        }
        // `alpine` → `alpine:latest`; without a tag the engine API pulls every tag.
        let image = if image
            .rsplit('/')
            .next()
            .is_some_and(|last| !last.contains(':') && !last.contains('@'))
        {
            format!("{image}:latest")
        } else {
            image.to_string()
        };
        let image = image.as_str();
        if self.docker.inspect_image(image).await.is_ok() {
            return Ok(());
        }
        if !self.pull_missing {
            return Err(EngineError::Container(format!(
                "image '{image}' not present locally and pulling is disabled (--no-pull)"
            )));
        }
        logs.push(format!("pulling image {image}"));
        let mut stream = self.docker.create_image(
            Some(CreateImageOptions {
                from_image: Some(image.to_string()),
                ..Default::default()
            }),
            None,
            None,
        );
        while let Some(item) = stream.next().await {
            item.map_err(|e| EngineError::Container(format!("pull of '{image}' failed: {e}")))?;
        }
        Ok(())
    }
}

/// Real uid without pulling in `libc`/`nix`: read it from /proc, else 1000.
pub(crate) fn current_uid() -> u32 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Uid:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(1000)
}

#[async_trait]
impl TesExecutor for ContainerExecutor {
    async fn run(&self, req: ExecutorRequest<'_>) -> Result<ExecutorResult, EngineError> {
        let mut system_logs = Vec::new();
        if req.command.is_empty() {
            return Ok(ExecutorResult::default());
        }
        self.ensure_image(req.image, &mut system_logs).await?;

        let mut binds = Vec::with_capacity(req.mount_roots.len());
        for root in req.mount_roots {
            let host_dir = req.work_dir.join(root);
            tokio::fs::create_dir_all(&host_dir).await?;
            binds.push(format!("{}:/{root}:Z", host_dir.display()));
        }
        let host_config = HostConfig {
            binds: Some(binds),
            nano_cpus: req.cpu_cores.map(|c| c as i64 * 1_000_000_000),
            memory: req.ram_gb.map(|g| (g * 1024.0 * 1024.0 * 1024.0) as i64),
            network_mode: Some(if req.network { "bridge" } else { "none" }.to_string()),
            ..Default::default()
        };
        if let Some(c) = req.cpu_cores {
            system_logs.push(format!("cpu limit {c} cores"));
        }
        if let Some(g) = req.ram_gb {
            system_logs.push(format!("memory limit {g} GiB"));
        }
        let env: Vec<String> = req.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
        // TES `command` is argv: override the image entrypoint so images that ship an
        // ENTRYPOINT (like Proteus's own) do not prepend it.
        let config = ContainerCreateBody {
            image: Some(req.image.to_string()),
            entrypoint: Some(vec![req.command[0].clone()]),
            cmd: Some(req.command[1..].to_vec()),
            working_dir: req.workdir.map(|w| w.to_string()),
            env: Some(env),
            host_config: Some(host_config),
            open_stdin: Some(req.stdin.is_some()),
            stdin_once: Some(req.stdin.is_some()),
            attach_stdin: Some(req.stdin.is_some()),
            ..Default::default()
        };
        let name = format!("proteus-tes-{}", uuid::Uuid::new_v4());
        self.docker
            .create_container(
                Some(CreateContainerOptions {
                    name: Some(name.clone()),
                    ..Default::default()
                }),
                config,
            )
            .await
            .map_err(|e| EngineError::Container(format!("create container: {e}")))?;

        let result = self.drive(&name, &req).await;
        let _ = self
            .docker
            .remove_container(
                &name,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await;
        let mut result = result?;
        result.system_logs.splice(0..0, system_logs);
        Ok(result)
    }

    fn kind(&self) -> &'static str {
        "container"
    }
}

impl ContainerExecutor {
    async fn drive(
        &self,
        name: &str,
        req: &ExecutorRequest<'_>,
    ) -> Result<ExecutorResult, EngineError> {
        if let Some(text) = req.stdin {
            // Attach before start so the stdin write is not lost.
            let attach = self
                .docker
                .attach_container(
                    name,
                    Some(AttachContainerOptions {
                        stdin: true,
                        stream: true,
                        ..Default::default()
                    }),
                )
                .await
                .map_err(|e| EngineError::Container(format!("attach: {e}")))?;
            let mut input = attach.input;
            let text = text.to_string();
            tokio::spawn(async move {
                use tokio::io::AsyncWriteExt;
                let _ = input.write_all(text.as_bytes()).await;
                let _ = input.shutdown().await;
            });
        }
        // A start failure is almost always "executable not found" inside the image: that is the
        // executor's error (exit 127 by shell convention), so ignore_error semantics apply.
        if let Err(e) = self
            .docker
            .start_container(name, None::<StartContainerOptions>)
            .await
        {
            return Ok(ExecutorResult {
                stderr: format!("failed to start '{}' in {}: {e}", req.command[0], req.image),
                exit_code: 127,
                system_logs: vec![format!("container start failed: {e}")],
                ..Default::default()
            });
        }
        debug!("started {name} ({})", req.image);

        let mut wait = self
            .docker
            .wait_container(name, None::<WaitContainerOptions>);
        let outcome = tokio::select! {
            w = tokio::time::timeout(req.timeout, wait.next()) => w,
            _ = req.cancel.cancelled() => {
                let _ = self.docker.kill_container(name, None::<KillContainerOptions>).await;
                return Ok(ExecutorResult { exit_code: -1, system_logs: vec!["container killed: task canceled".into()], ..Default::default() });
            }
        };
        let (exit_code, mut system_logs) = match outcome {
            Ok(Some(Ok(w))) => (w.status_code, Vec::new()),
            // bollard yields Err for non-zero exits on some API versions; the code is in the message.
            Ok(Some(Err(bollard::errors::Error::DockerContainerWaitError { code, .. }))) => {
                (code, Vec::new())
            }
            Ok(Some(Err(e))) => {
                return Err(EngineError::Container(format!("wait: {e}")));
            }
            Ok(None) => (-1, vec!["wait stream ended without a status".into()]),
            Err(_) => {
                let _ = self
                    .docker
                    .kill_container(name, None::<KillContainerOptions>)
                    .await;
                (
                    -1,
                    vec![format!(
                        "container killed: timed out after {}s",
                        req.timeout.as_secs()
                    )],
                )
            }
        };

        let (mut stdout, mut stderr) = (String::new(), String::new());
        let mut logs = self.docker.logs(
            name,
            Some(LogsOptions {
                stdout: true,
                stderr: true,
                ..Default::default()
            }),
        );
        while let Some(chunk) = logs.next().await {
            match chunk {
                Ok(LogOutput::StdOut { message }) => {
                    stdout.push_str(&String::from_utf8_lossy(&message))
                }
                Ok(LogOutput::StdErr { message }) => {
                    stderr.push_str(&String::from_utf8_lossy(&message))
                }
                Ok(_) => {}
                Err(e) => {
                    system_logs.push(format!("log stream error: {e}"));
                    break;
                }
            }
        }
        Ok(ExecutorResult {
            stdout,
            stderr,
            exit_code,
            system_logs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_roots_are_first_components_and_reject_system_dirs() {
        assert_eq!(mount_root("/data/in.pdb").unwrap(), "data");
        assert_eq!(mount_root("/work/sub/out.txt").unwrap(), "work");
        assert!(mount_root("relative/path").is_err());
        assert!(mount_root("/etc/passwd").is_err());
        assert!(mount_root("/usr/bin/x").is_err());
    }

    #[test]
    fn mount_root_rejects_parent_dir_components() {
        // `/data/../..` would resolve above the task work dir once joined on the host.
        assert!(mount_root("/data/../etc/passwd").is_err());
        assert!(mount_root("/data/sub/../../../home/x").is_err());
        assert!(mount_root("/data/./in.pdb").is_ok());
    }

    #[tokio::test]
    async fn host_executor_runs_and_captures_output() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = vec![
            "sh".to_string(),
            "-c".to_string(),
            "echo out; echo err >&2; exit 3".to_string(),
        ];
        let env = HashMap::new();
        let roots = BTreeSet::new();
        let r = HostExecutor
            .run(ExecutorRequest {
                image: "ignored",
                command: &cmd,
                workdir: None,
                env: &env,
                stdin: None,
                work_dir: dir.path(),
                mount_roots: &roots,
                cpu_cores: None,
                ram_gb: None,
                network: false,
                timeout: Duration::from_secs(10),
                cancel: CancellationToken::new(),
            })
            .await
            .unwrap();
        assert_eq!(r.stdout.trim(), "out");
        assert_eq!(r.stderr.trim(), "err");
        assert_eq!(r.exit_code, 3);
    }

    #[tokio::test]
    async fn host_executor_honours_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = vec!["sleep".to_string(), "30".to_string()];
        let env = HashMap::new();
        let roots = BTreeSet::new();
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            c2.cancel();
        });
        let started = std::time::Instant::now();
        let r = HostExecutor
            .run(ExecutorRequest {
                image: "ignored",
                command: &cmd,
                workdir: None,
                env: &env,
                stdin: None,
                work_dir: dir.path(),
                mount_roots: &roots,
                cpu_cores: None,
                ram_gb: None,
                network: false,
                timeout: Duration::from_secs(60),
                cancel,
            })
            .await
            .unwrap();
        assert_eq!(r.exit_code, -1);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Needs a reachable Podman/Docker socket: `PROTEUS_TEST_OCI=1 cargo test -p proteus-engine -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn container_executor_runs_alpine_with_bind_mount() {
        if std::env::var("PROTEUS_TEST_OCI").is_err() {
            eprintln!("PROTEUS_TEST_OCI not set; skipping");
            return;
        }
        let ex = ContainerExecutor::connect(true).unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("data")).unwrap();
        std::fs::write(dir.path().join("data/in.txt"), "hello").unwrap();
        let cmd = vec![
            "sh".to_string(),
            "-c".to_string(),
            "cat /data/in.txt > /data/out.txt; echo done; exit 0".to_string(),
        ];
        let env = HashMap::new();
        let roots: BTreeSet<String> = ["data".to_string()].into_iter().collect();
        let r = ex
            .run(ExecutorRequest {
                image: "docker.io/library/alpine:3.20",
                command: &cmd,
                workdir: Some("/data"),
                env: &env,
                stdin: None,
                work_dir: dir.path(),
                mount_roots: &roots,
                cpu_cores: Some(1),
                ram_gb: Some(0.25),
                network: false,
                timeout: Duration::from_secs(120),
                cancel: CancellationToken::new(),
            })
            .await
            .unwrap();
        assert_eq!(r.exit_code, 0, "{r:?}");
        assert_eq!(r.stdout.trim(), "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("data/out.txt")).unwrap(),
            "hello"
        );
    }

    /// Same gate. Covers the paths the first test does not: TES `stdin` delivered through the
    /// attach API, a non-zero exit reported as the exit code (not an error), and the
    /// wall-clock timeout killing the container.
    #[tokio::test]
    #[ignore]
    async fn container_executor_stdin_exit_code_and_timeout() {
        if std::env::var("PROTEUS_TEST_OCI").is_err() {
            eprintln!("PROTEUS_TEST_OCI not set; skipping");
            return;
        }
        let ex = ContainerExecutor::connect(true).unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("data")).unwrap();
        let env = HashMap::new();
        let roots: BTreeSet<String> = ["data".to_string()].into_iter().collect();
        let request = |cmd: &'static str, stdin: Option<&'static str>, timeout: u64| {
            let command = vec!["sh".to_string(), "-c".to_string(), cmd.to_string()];
            (command, stdin, Duration::from_secs(timeout))
        };

        let (cmd, stdin, timeout) = request("cat; echo; exit 3", Some("from stdin"), 120);
        let r = ex
            .run(ExecutorRequest {
                image: "docker.io/library/alpine:3.20",
                command: &cmd,
                workdir: Some("/data"),
                env: &env,
                stdin,
                work_dir: dir.path(),
                mount_roots: &roots,
                cpu_cores: None,
                ram_gb: None,
                network: false,
                timeout,
                cancel: CancellationToken::new(),
            })
            .await
            .unwrap();
        assert_eq!(r.exit_code, 3, "{r:?}");
        assert_eq!(r.stdout.trim(), "from stdin", "{r:?}");

        let (cmd, stdin, timeout) = request("sleep 30", None, 2);
        let started = std::time::Instant::now();
        let r = ex
            .run(ExecutorRequest {
                image: "docker.io/library/alpine:3.20",
                command: &cmd,
                workdir: Some("/data"),
                env: &env,
                stdin,
                work_dir: dir.path(),
                mount_roots: &roots,
                cpu_cores: None,
                ram_gb: None,
                network: false,
                timeout,
                cancel: CancellationToken::new(),
            })
            .await
            .unwrap();
        assert_eq!(r.exit_code, -1, "{r:?}");
        assert!(
            r.system_logs.iter().any(|l| l.contains("timed out")),
            "{r:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(15));
    }
}
