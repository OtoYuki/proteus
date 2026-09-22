use crate::error::EngineError;
use crate::runner::{ComputeRunner, RunResult};
use bollard::models::{ContainerCreateBody, HostConfig};
use bollard::query_parameters::{
    CreateContainerOptions, LogsOptions, RemoveContainerOptions, StartContainerOptions,
    WaitContainerOptions,
};
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
        let podman_user_sock = format!(
            "/run/user/{}/podman/podman.sock",
            crate::tes_exec::current_uid()
        );

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

/// Container image for a prediction tier. None of the defaults is published: build or pull
/// an image yourself and either tag it with the default name or point the matching
/// `PROTEUS_IMAGE_{FAST,SOTA,RELAX}` variable at it.
pub fn tier_image(tier: &PipelineTier) -> String {
    let (var, default) = match tier {
        PipelineTier::FastScreening => ("PROTEUS_IMAGE_FAST", "ghcr.io/proteus/esmfold:latest"),
        PipelineTier::HighFidelity => ("PROTEUS_IMAGE_SOTA", "ghcr.io/jwohlwend/boltz:latest"),
        PipelineTier::FullValidation => ("PROTEUS_IMAGE_RELAX", "ghcr.io/proteus/openmm:latest"),
    };
    std::env::var(var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The FASTA handed to the tier's container. Boltz requires `>CHAIN|ENTITY|MSA` headers
/// (`empty` = single-sequence mode); ESMFold-style tools take a plain header.
pub fn input_fasta_for(tier: &PipelineTier, header: &str, sequence: &str) -> String {
    match tier {
        PipelineTier::HighFidelity => format!(">A|protein|empty\n{sequence}\n"),
        _ => format!(">{header}\n{sequence}\n"),
    }
}

/// Tiers the OCI runner can actually execute. Relaxation needs a structure, and a pipeline
/// job carries only a sequence.
pub fn tier_supported(tier: &PipelineTier) -> Result<(), EngineError> {
    match tier {
        PipelineTier::FullValidation => Err(EngineError::Container(
            "the relax tier (FullValidation) needs an input structure, which a sequence job \
             does not carry; it is not implemented in this release"
                .into(),
        )),
        _ => Ok(()),
    }
}

#[async_trait::async_trait]
impl ComputeRunner for OciRunner {
    async fn execute_job(
        &self,
        job: &PipelineJob,
        sequence: &Sequence,
        work_dir: &Path,
    ) -> Result<RunResult, EngineError> {
        tier_supported(&job.tier)?;
        let image = tier_image(&job.tier);
        let image = image.as_str();

        if !self.has_image(image).await {
            return Err(EngineError::Container(format!(
                "container image '{image}' is not available locally. Build or pull one and tag it \
                 with that name (or set PROTEUS_IMAGE_FAST/SOTA/RELAX), or run with \
                 '--runner esm-api' for live ESMFold folding or '--runner simulated'."
            )));
        }

        let container_name = format!("proteus-{}-{}", job.tier_slug(), job.id);
        tokio::fs::create_dir_all(work_dir).await?;

        let fasta_path = work_dir.join("input.fasta");
        tokio::fs::write(
            &fasta_path,
            input_fasta_for(&job.tier, &sequence.header, &sequence.fasta),
        )
        .await?;

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

        // No `auto_remove`: an auto-removed container can vanish before `wait_container` is
        // polled, which surfaces as a 404 on Docker. The container is removed explicitly below.
        let host_config = HostConfig {
            binds: Some(vec![bind_mount]),
            ..Default::default()
        };

        let config = ContainerCreateBody {
            image: Some(image.to_string()),
            cmd: Some(cmd.iter().map(|s| s.to_string()).collect()),
            host_config: Some(host_config),
            working_dir: Some("/workspace".to_string()),
            ..Default::default()
        };

        info!("Creating OCI container: {}", container_name);
        self.docker
            .create_container(
                Some(CreateContainerOptions {
                    name: Some(container_name.clone()),
                    ..Default::default()
                }),
                config,
            )
            .await
            .map_err(|e| EngineError::Container(format!("Failed to create container: {e}")))?;

        info!("Starting OCI container: {}", container_name);
        self.docker
            .start_container(&container_name, None::<StartContainerOptions>)
            .await
            .map_err(|e| EngineError::Container(format!("Failed to start container: {e}")))?;

        // Stream logs in background
        let mut logs = self.docker.logs(
            &container_name,
            Some(LogsOptions {
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
            .wait_container(&container_name, None::<WaitContainerOptions>);

        let waited = wait_stream.next().await;
        let _ = self
            .docker
            .remove_container(
                &container_name,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await;
        if let Some(wait_res) = waited {
            let status_code = match wait_res {
                Ok(w) => w.status_code,
                // bollard reports non-zero exits as an error on some API versions.
                Err(bollard::errors::Error::DockerContainerWaitError { code, .. }) => code,
                Err(e) => return Err(EngineError::Container(format!("Wait failed: {e}"))),
            };
            if status_code != 0 {
                return Err(EngineError::Container(format!(
                    "Container exited with non-zero status code: {status_code}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::models::PipelineTier;

    #[test]
    fn tier_images_default_to_documented_names_and_honour_env_overrides() {
        assert_eq!(
            tier_image(&PipelineTier::FastScreening),
            "ghcr.io/proteus/esmfold:latest"
        );
        std::env::set_var("PROTEUS_IMAGE_SOTA", "localhost/my-boltz:2");
        assert_eq!(
            tier_image(&PipelineTier::HighFidelity),
            "localhost/my-boltz:2"
        );
        std::env::remove_var("PROTEUS_IMAGE_SOTA");
        assert_eq!(
            tier_image(&PipelineTier::HighFidelity),
            "ghcr.io/jwohlwend/boltz:latest"
        );
    }

    #[test]
    fn boltz_input_uses_the_chain_entity_header_format() {
        // Boltz rejects plain `>name` headers; it needs `>CHAIN|protein|<msa>`.
        assert_eq!(
            input_fasta_for(&PipelineTier::HighFidelity, "wt", "ACDE"),
            ">A|protein|empty\nACDE\n"
        );
        assert_eq!(
            input_fasta_for(&PipelineTier::FastScreening, "wt", "ACDE"),
            ">wt\nACDE\n"
        );
    }

    #[test]
    fn relax_tier_is_refused_up_front() {
        let err = tier_supported(&PipelineTier::FullValidation).unwrap_err();
        assert!(err.to_string().contains("structure"), "{err}");
        assert!(tier_supported(&PipelineTier::FastScreening).is_ok());
    }
}
