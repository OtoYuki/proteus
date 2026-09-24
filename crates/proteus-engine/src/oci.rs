use crate::error::EngineError;
use crate::runner::{ComputeRunner, RunResult};
use bollard::models::{ContainerCreateBody, DeviceRequest, HostConfig};
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

/// How long one tier container may run: `PROTEUS_OCI_TIMEOUT_SECS`, else 3 hours (a large
/// complex with several samples on a laptop GPU is well under that).
fn run_timeout() -> std::time::Duration {
    let secs = std::env::var("PROTEUS_OCI_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(3 * 3600);
    std::time::Duration::from_secs(secs)
}

/// /dev/shm for tier containers: 2 GiB.
const TIER_SHM_BYTES: i64 = 2 << 30;

/// Where the NVIDIA Container Toolkit writes its CDI spec (`nvidia-ctk cdi generate`).
const NVIDIA_CDI_SPECS: [&str; 2] = ["/etc/cdi/nvidia.yaml", "/var/run/cdi/nvidia.yaml"];

/// The CDI device a tier container is given, from `PROTEUS_GPU`: `off` runs on CPU, any
/// other value is a CDI device name (`nvidia.com/gpu=0`), and unset means
/// `nvidia.com/gpu=all` when an NVIDIA CDI spec is installed, else CPU.
pub fn gpu_device() -> Option<String> {
    let installed = NVIDIA_CDI_SPECS.iter().any(|p| Path::new(p).exists());
    gpu_device_for(std::env::var("PROTEUS_GPU").ok().as_deref(), installed)
}

fn gpu_device_for(setting: Option<&str>, nvidia_cdi_installed: bool) -> Option<String> {
    match setting.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) if s.eq_ignore_ascii_case("off") => None,
        Some(s) => Some(s.to_string()),
        None => nvidia_cdi_installed.then(|| "nvidia.com/gpu=all".to_string()),
    }
}

/// The FASTA handed to the tier's container. Boltz requires `>CHAIN|ENTITY|MSA` headers
/// (`empty` = single-sequence mode); ESMFold-style tools take a plain header.
pub fn input_fasta_for(tier: &PipelineTier, header: &str, sequence: &str) -> String {
    match tier {
        PipelineTier::HighFidelity => format!(">A|protein|empty\n{sequence}\n"),
        _ => format!(">{header}\n{sequence}\n"),
    }
}

/// A job's input as Boltz reads it, and what the container needs besides the FASTA.
#[derive(Debug, Clone, PartialEq)]
pub struct BoltzInput {
    pub fasta: String,
    /// `(host path, file name under /workspace/msa/)` for alignments the user supplied.
    pub msa_files: Vec<(std::path::PathBuf, String)>,
    pub use_msa_server: bool,
    pub samples: usize,
}

/// Build Boltz's input from a stored job input (a bare monomer or a
/// [`proteus_core::complex::ComplexSpec`]). An alignment file is copied into the work
/// directory and referred to by its container path; the server option drops the MSA field,
/// which is how Boltz asks `--use_msa_server` for one.
pub fn boltz_input(stored: &str) -> Result<BoltzInput, EngineError> {
    use proteus_core::complex::{ComplexSpec, Entity, MsaSource};
    let spec = ComplexSpec::from_stored(stored)?;
    let mut fasta = String::new();
    let mut msa_files = Vec::new();
    for c in &spec.chains {
        match &c.entity {
            Entity::Protein { sequence, msa } => {
                let field = match msa {
                    MsaSource::Empty => "|empty".to_string(),
                    MsaSource::Server => String::new(),
                    MsaSource::File(p) => {
                        let host = std::path::PathBuf::from(p);
                        let ext = host
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("a3m")
                            .to_string();
                        let name = format!("{}.{ext}", c.id);
                        msa_files.push((host, name.clone()));
                        format!("|/workspace/msa/{name}")
                    }
                };
                fasta.push_str(&format!(">{}|protein{field}\n{sequence}\n", c.id));
            }
            Entity::Ccd(code) => fasta.push_str(&format!(">{}|ccd\n{code}\n", c.id)),
            Entity::Smiles(smiles) => fasta.push_str(&format!(">{}|smiles\n{smiles}\n", c.id)),
        }
    }
    Ok(BoltzInput {
        fasta,
        msa_files,
        use_msa_server: spec.uses_msa_server(),
        samples: spec.samples.max(1),
    })
}

/// The structure file Boltz ranks first: `*_model_0.*` (models are sorted by confidence), else
/// the first structure file found. Walks subdirectories, in name order so the choice is stable.
fn best_model(work_dir: &std::path::Path) -> std::io::Result<Option<PathBuf>> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut dirs = vec![work_dir.to_path_buf()];
    while let Some(d) = dirs.pop() {
        for entry in std::fs::read_dir(&d)? {
            let p = entry?.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|e| e == "pdb" || e == "cif") {
                found.push(p);
            }
        }
    }
    found.sort();
    let zero = found.iter().find(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.ends_with("_model_0"))
    });
    Ok(zero.or(found.first()).cloned())
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
        let is_spec = proteus_core::complex::ComplexSpec::is_spec(&sequence.fasta);
        let boltz = if job.tier == PipelineTier::HighFidelity {
            Some(boltz_input(&sequence.fasta)?)
        } else if is_spec {
            return Err(EngineError::Container(
                "complexes, ligands, MSA options and several samples need the sota tier \
                 (Boltz); this tier folds one chain"
                    .into(),
            ));
        } else {
            None
        };
        match &boltz {
            Some(b) => {
                tokio::fs::write(&fasta_path, &b.fasta).await?;
                if !b.msa_files.is_empty() {
                    let msa_dir = work_dir.join("msa");
                    tokio::fs::create_dir_all(&msa_dir).await?;
                    for (host, name) in &b.msa_files {
                        tokio::fs::copy(host, msa_dir.join(name))
                            .await
                            .map_err(|e| {
                                EngineError::Container(format!(
                                    "cannot copy the alignment {}: {e}",
                                    host.display()
                                ))
                            })?;
                    }
                }
            }
            None => {
                tokio::fs::write(
                    &fasta_path,
                    input_fasta_for(&job.tier, &sequence.header, &sequence.fasta),
                )
                .await?
            }
        }

        let abs_work_dir = std::fs::canonicalize(work_dir)
            .map_err(|e| EngineError::Container(format!("Failed to canonicalize work_dir: {e}")))?;

        let bind_mount = format!("{}:/workspace:Z", abs_work_dir.to_string_lossy());

        let samples = boltz.as_ref().map_or(1, |b| b.samples).to_string();
        // Samples one at a time by default: Boltz's default of 5 in parallel runs a laptop GPU
        // out of memory on a 250-token complex (it then skips the input and exits 0).
        let parallel = std::env::var("PROTEUS_BOLTZ_PARALLEL_SAMPLES")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1)
            .to_string();
        // MSA depth: Boltz's default of 8 192 sequences needs a 660 MB tensor for a 243-token
        // complex, more than a 6 GB GPU has left; 1 024 (the subsample it already uses for the
        // trunk) fits, measured on HIV protease + indinavir.
        let max_msa = std::env::var("PROTEUS_BOLTZ_MAX_MSA_SEQS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1024)
            .to_string();
        let mut cmd = match job.tier {
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
                "--diffusion_samples",
                samples.as_str(),
                "--max_parallel_samples",
                parallel.as_str(),
                "--max_msa_seqs",
                max_msa.as_str(),
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

        if boltz.as_ref().is_some_and(|b| b.use_msa_server) {
            // Sends the sequences to the public ColabFold MMseqs2 server: opted into per job.
            cmd.push("--use_msa_server");
        }
        // No `auto_remove`: an auto-removed container can vanish before `wait_container` is
        // polled, which surfaces as a 404 on Docker. The container is removed explicitly below.
        // Podman and Docker 25+ both take a CDI device through the `cdi` device-request driver.
        let gpu = gpu_device();
        let host_config = HostConfig {
            binds: Some(vec![bind_mount]),
            // PyTorch's data-loader workers pass tensors through /dev/shm; the runtimes'
            // 64 MB default fails a Boltz complex with "unable to allocate shared memory".
            shm_size: Some(TIER_SHM_BYTES),
            device_requests: gpu.clone().map(|id| {
                vec![DeviceRequest {
                    driver: Some("cdi".to_string()),
                    device_ids: Some(vec![id]),
                    ..Default::default()
                }]
            }),
            ..Default::default()
        };

        let config = ContainerCreateBody {
            image: Some(image.to_string()),
            cmd: Some(cmd.iter().map(|s| s.to_string()).collect()),
            host_config: Some(host_config),
            working_dir: Some("/workspace".to_string()),
            ..Default::default()
        };

        info!(
            "Creating OCI container: {} (gpu: {})",
            container_name,
            gpu.as_deref().unwrap_or("none")
        );
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
        if let Err(e) = self
            .docker
            .start_container(&container_name, None::<StartContainerOptions>)
            .await
        {
            // The container exists but never ran (typically: the command is missing from the
            // image). Remove it here, or it is left behind in `Created` state.
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
            return Err(EngineError::Container(format!(
                "Failed to start container: {e}"
            )));
        }

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

        // The last lines of output, for the error when the run fails or hangs.
        let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        let mut out_of_memory = false;
        let limit = run_timeout();
        let streamed = tokio::time::timeout(limit, async {
            while let Some(log_item) = logs.next().await {
                match log_item {
                    Ok(chunk) => {
                        let line = chunk.to_string();
                        info!("[{}] {}", container_name, line.trim());
                        if line.contains("ran out of memory") || line.contains("OutOfMemoryError") {
                            out_of_memory = true;
                        }
                        tail.push_back(line.trim().to_string());
                        if tail.len() > 6 {
                            tail.pop_front();
                        }
                    }
                    Err(e) => {
                        warn!("Log stream error for {}: {}", container_name, e);
                        break;
                    }
                }
            }
            let mut wait_stream = self
                .docker
                .wait_container(&container_name, None::<WaitContainerOptions>);
            wait_stream.next().await
        })
        .await;
        let waited = match streamed {
            Ok(w) => w,
            Err(_) => {
                // A crashed data-loader worker can leave the predictor waiting forever.
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
                return Err(EngineError::Container(format!(
                    "the container ran past {} s and was stopped (PROTEUS_OCI_TIMEOUT_SECS \
                     raises the limit); its last output: {}",
                    limit.as_secs(),
                    tail.iter().cloned().collect::<Vec<_>>().join(" | ")
                )));
            }
        };
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
                    "Container exited with non-zero status code: {status_code}; its last \
                     output: {}",
                    tail.iter().cloned().collect::<Vec<_>>().join(" | ")
                )));
            }
        }

        let found_pdb = best_model(work_dir)?;
        if found_pdb.is_none() && out_of_memory {
            return Err(EngineError::Container(
                "the GPU ran out of memory and the predictor skipped the input: fewer --samples, \
                 a smaller complex, or a larger GPU"
                    .into(),
            ));
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
                "samples": boltz.as_ref().map_or(1, |b| b.samples),
                "msa": boltz.as_ref().map(|b| if b.use_msa_server {
                    "server"
                } else if b.msa_files.is_empty() {
                    "none"
                } else {
                    "file"
                }),
                "socket": self.socket_path,
                "gpu": gpu
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
    fn boltz_input_from_a_complex_spec() {
        let stored = "#proteus samples=3\n>A|protein|server\nMQIF\n>B|protein|/data/b.a3m\nKKLL\n>L|ccd\nATP\n";
        let b = boltz_input(stored).unwrap();
        assert_eq!(
            b.fasta,
            ">A|protein\nMQIF\n>B|protein|/workspace/msa/B.a3m\nKKLL\n>L|ccd\nATP\n"
        );
        assert!(b.use_msa_server);
        assert_eq!(b.samples, 3);
        assert_eq!(
            b.msa_files,
            vec![(std::path::PathBuf::from("/data/b.a3m"), "B.a3m".to_string())]
        );
        // A bare monomer is the single-sequence input it always was.
        let m = boltz_input("MQIF").unwrap();
        assert_eq!(m.fasta, ">A|protein|empty\nMQIF\n");
        assert!(!m.use_msa_server);
    }

    #[test]
    fn the_top_ranked_boltz_model_is_picked() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("boltz_results_input/predictions/input");
        std::fs::create_dir_all(&p).unwrap();
        for f in [
            "input_model_2.pdb",
            "input_model_0.pdb",
            "input_model_1.pdb",
        ] {
            std::fs::write(p.join(f), "").unwrap();
        }
        let best = best_model(dir.path()).unwrap().unwrap();
        assert!(best.ends_with("input_model_0.pdb"), "{best:?}");
    }

    #[test]
    fn gpu_follows_the_cdi_spec_unless_proteus_gpu_says_otherwise() {
        assert_eq!(
            gpu_device_for(None, true).as_deref(),
            Some("nvidia.com/gpu=all")
        );
        assert_eq!(gpu_device_for(None, false), None);
        assert_eq!(gpu_device_for(Some(" "), false), None);
        assert_eq!(gpu_device_for(Some("off"), true), None);
        assert_eq!(gpu_device_for(Some("OFF"), true), None);
        assert_eq!(
            gpu_device_for(Some("nvidia.com/gpu=0"), false).as_deref(),
            Some("nvidia.com/gpu=0")
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
