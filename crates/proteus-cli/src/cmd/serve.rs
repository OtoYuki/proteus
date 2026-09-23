//! `proteus serve` — Run the headless daemon (GA4GH TES 1.1 + the native API).

use super::prelude::*;

/// Arguments of `proteus serve`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Compute runner mode
    #[arg(long, value_enum, default_value_t = RunnerMode::Auto)]
    runner: RunnerMode,

    /// How TES executors run: inside their container image (default) or as host
    /// processes (development only; refused unless bound to loopback)
    #[arg(long, value_enum, default_value_t = ExecutorMode::Container)]
    executor: ExecutorMode,

    /// Glob pattern an executor image must match (repeatable). No patterns = allow all.
    #[arg(long = "allow-image", value_name = "GLOB")]
    allow_images: Vec<String>,

    /// Host directory that task `file://` input and output URLs may reference
    /// (repeatable). Default: the daemon's own artifacts directory.
    #[arg(long = "allow-dir", value_name = "DIR")]
    allow_dirs: Vec<PathBuf>,

    /// Require `Authorization: Bearer <token>` on the TES and native APIs
    #[arg(long, env = "PROTEUS_AUTH_TOKEN", hide_env_values = true)]
    auth_token: Option<String>,

    /// Do not pull executor images that are missing locally
    #[arg(long)]
    no_pull: bool,

    /// Give executor containers outbound network access
    #[arg(long)]
    executor_network: bool,

    /// Wall-clock limit per executor, seconds
    #[arg(long, default_value_t = 3600)]
    executor_timeout: u64,
}

/// `--host` as users write it: an IPv4 or IPv6 literal (with or without brackets) or a name
/// such as `localhost`, which resolves to its first address.
fn resolve_listen_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return Ok(SocketAddr::new(ip, port));
    }
    use std::net::ToSocketAddrs;
    (host, port)
        .to_socket_addrs()
        .with_context(|| format!("--host {host}: not an IP address or a resolvable name"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("--host {host} resolves to no address"))
}

pub async fn run(
    args: Args,
    db_path: &std::path::Path,
    artifacts_dir: &std::path::Path,
) -> Result<()> {
    let db_path = db_path.to_path_buf();
    let artifacts_dir = artifacts_dir.to_path_buf();
    let Args {
        port,
        host,
        runner,
        executor,
        allow_images,
        allow_dirs,
        auth_token,
        no_pull,
        executor_network,
        executor_timeout,
    } = args;
    let pool = create_sqlite_pool(&db_path).await?;
    let repo = ProteusRepository::new(pool);
    let compute_runner = resolve_runner(runner)?;
    let addr = resolve_listen_addr(&host, port)?;

    let tes_executor: Arc<dyn TesExecutor> = match executor {
        ExecutorMode::Container => Arc::new(
            ContainerExecutor::connect(!no_pull).context("TES container executor unavailable")?,
        ),
        ExecutorMode::Host => {
            if !addr.ip().is_loopback() {
                anyhow::bail!(
                    "--executor host runs TES commands directly on this machine and is only \
                     allowed with a loopback --host (got {})",
                    addr.ip()
                );
            }
            tracing::warn!("TES executors run as HOST PROCESSES; container images are ignored");
            Arc::new(HostExecutor)
        }
    };
    let mut patterns = Vec::with_capacity(allow_images.len());
    for g in &allow_images {
        patterns.push(
            glob::Pattern::new(g)
                .with_context(|| format!("invalid --allow-image pattern '{g}'"))?,
        );
    }
    let tes = TesExecutionConfig {
        executor: tes_executor,
        allow_images: patterns,
        executor_timeout: std::time::Duration::from_secs(executor_timeout),
        network: executor_network,
        allow_dirs: if allow_dirs.is_empty() {
            vec![artifacts_dir.clone()]
        } else {
            allow_dirs
        },
    };
    let scheduler =
        PipelineScheduler::new(repo, compute_runner, artifacts_dir).with_tes_config(tes);
    run_server_with_options(addr, scheduler, ServerOptions { auth_token }).await?;
    Ok(())
}

#[cfg(test)]
mod host_tests {
    use super::resolve_listen_addr;

    #[test]
    fn host_accepts_names_and_bare_ipv6() {
        // Reported: `--host localhost` and `--host ::1` were refused as bad socket addresses.
        assert!(resolve_listen_addr("localhost", 8080)
            .unwrap()
            .ip()
            .is_loopback());
        assert_eq!(
            resolve_listen_addr("::1", 8080).unwrap().to_string(),
            "[::1]:8080"
        );
        assert_eq!(
            resolve_listen_addr("[::1]", 8080).unwrap().to_string(),
            "[::1]:8080"
        );
        assert_eq!(
            resolve_listen_addr("0.0.0.0", 9).unwrap().to_string(),
            "0.0.0.0:9"
        );
    }
}
