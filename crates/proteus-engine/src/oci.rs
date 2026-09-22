use crate::error::EngineError;
use crate::runner::{ComputeRunner, RunResult};
use bollard::container::{
    Config, CreateContainerOptions, LogsOptions, StartContainerOptions, WaitContainerOptions,
};
use bollard::models::HostConfig;
use bollard::Docker;
use futures_util::stream::StreamExt;
use proteus_core::models::{PipelineJob, PipelineTier, Sequence};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

pub struct OciRunner {
    docker: Docker,
    socket_path: String,
}

impl OciRunner {
    /// Detect and connect to Podman rootless socket or Docker UNIX socket.
    pub fn new() -> Result<Self, EngineError> {
        let podman_user_sock = format!("/run/user/{}/podman/podman.sock", users_uid());

        let socket_candidates = vec![
            podman_user_sock.as_str(),
            "/var/run/docker.sock",
            "/run/podman/podman.sock",
        ];

        for sock in socket_candidates {
            if std::path::Path::new(sock).exists() {
                info!("Found container engine socket at: {}", sock);
                match Docker::connect_with_unix(sock, 120, bollard::API_DEFAULT_VERSION) {
                    Ok(docker) => {
                        return Ok(Self {
                            docker,
                            socket_path: sock.to_string(),
                        });
                    }
                    Err(e) => {
                        warn!("Failed connecting to socket {}: {}", sock, e);
                    }
                }
            }
        }

        // Fallback to default bollard discovery
        let docker = Docker::connect_with_socket_defaults().map_err(|e| {
            EngineError::Container(format!("Cannot connect to container runtime: {e}"))
        })?;

        Ok(Self {
            docker,
            socket_path: "default".to_string(),
        })
    }

    pub fn socket_path(&self) -> &str {
        &self.socket_path
    }

    pub async fn has_image(&self, image: &str) -> bool {
        self.docker.inspect_image(image).await.is_ok()
    }
}

fn users_uid() -> u32 {
    // Get current effective UID via standard posix or default to 1000
    std::env::var("UID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
}

#[async_trait::async_trait]
impl ComputeRunner for OciRunner {
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError> {
        let image = match job.tier {
            PipelineTier::FastScreening => "ghcr.io/proteus/esmfold:latest",
            PipelineTier::HighFidelity => "ghcr.io/jwohlwend/boltz:latest",
            PipelineTier::FullValidation => "ghcr.io/proteus/openmm:latest",
        };

        if !self.has_image(image).await {
            return Err(EngineError::Container(format!(
                "Required SOTA container image '{image}' is not available locally. Pull it with 'podman pull {image}', or run with '--runner esm-api' for live ESMFold folding or '--runner simulated'."
            )));
        }

        let container_name = format!("proteus-{}-{}", job.tier_slug(), job.id);
        tokio::fs::create_dir_all(work_dir).await?;

        let fasta_path = work_dir.join("input.fasta");
        let fasta_content = format!(">{}\n{}\n", sequence.header, sequence.fasta);
        tokio::fs::write(&fasta_path, fasta_content).await?;

        let abs_work_dir = std::fs::canonicalize(work_dir)
            .map_err(|e| EngineError::Container(format!("Failed to canonicalize work_dir: {e}")))?;

        let bind_mount = format!("{}:/workspace:Z", abs_work_dir.to_string_lossy());

        let cmd = match job.tier {
            PipelineTier::FastScreening => vec![
                "python",
                "-m",
                "esmfold.inference",
                "--fasta",
                "/workspace/input.fasta",
                "--output-dir",
                "/workspace",
            ],
            PipelineTier::HighFidelity => vec![
                "boltz",
                "predict",
                "/workspace/input.fasta",
                "--out_dir",
                "/workspace",
                "--output_format",
                "pdb",
                "--override",
            ],
            PipelineTier::FullValidation => vec![
                "python",
                "-m",
                "proteus.relax",
                "--input",
                "/workspace/input.pdb",
                "--output",
                "/workspace/relaxed.pdb",
                "--forcefield",
                "amber14sb",
            ],
        };

        let host_config = HostConfig {
            binds: Some(vec![bind_mount]),
            auto_remove: Some(true),
            ..Default::default()
        };

        let config = Config {
            image: Some(image),
            cmd: Some(cmd),
            host_config: Some(host_config),
            working_dir: Some("/workspace"),
            ..Default::default()
        };

        info!("Creating OCI container: {}", container_name);
        self.docker
            .create_container(
                Some(CreateContainerOptions {
                    name: container_name.as_str(),
                    platform: None,
                }),
                config,
            )
            .await
            .map_err(|e| EngineError::Container(format!("Failed to create container: {e}")))?;

        info!("Starting OCI container: {}", container_name);
        self.docker
            .start_container(&container_name, None::<StartContainerOptions<String>>)
            .await
            .map_err(|e| EngineError::Container(format!("Failed to start container: {e}")))?;

        // Stream logs in background
        let mut logs = self.docker.logs(
            &container_name,
            Some(LogsOptions::<String> {
                stdout: true,
                stderr: true,
                follow: true,
                ..Default::default()
            }),
        );

        while let Some(log_item) = logs.next().await {
            match log_item {
                Ok(chunk) => {
                    info!("[{}] {}", container_name, chunk.to_string().trim());
                }
                Err(e) => {
                    warn!("Log stream error for {}: {}", container_name, e);
                    break;
                }
            }
        }

        // Wait for container completion
        let mut wait_stream = self
            .docker
            .wait_container(&container_name, None::<WaitContainerOptions<String>>);

        if let Some(wait_res) = wait_stream.next().await {
            let res = wait_res.map_err(|e| EngineError::Container(format!("Wait failed: {e}")))?;
            if res.status_code != 0 {
                return Err(EngineError::Container(format!(
                    "Container exited with non-zero status code: {}",
                    res.status_code
                )));
            }
        }

        // Locate predicted PDB in work_dir (recursively scanning in case engine creates subdirectories)
        let mut found_pdb: Option<PathBuf> = None;
        let mut dirs = vec![work_dir.to_path_buf()];
        while let Some(current_dir) = dirs.pop() {
            let mut entries = tokio::fs::read_dir(&current_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    dirs.push(path);
                } else if let Some(ext) = path.extension() {
                    if ext == "pdb" || ext == "cif" {
                        found_pdb = Some(path);
                        break;
                    }
                }
            }
            if found_pdb.is_some() {
                break;
            }
        }

        let pdb_path = found_pdb.ok_or_else(|| {
            EngineError::Container(format!(
                "Container completed but no PDB found in {:?}",
                work_dir
            ))
        })?;

        Ok(RunResult {
            pdb_path,
            plddt: None,
            metadata: Some(serde_json::json!({
                "engine": crate::runner::ENGINE_OCI,
                "runner": "oci",
                "image": image,
                "socket": self.socket_path
            })),
        })
    }
}
