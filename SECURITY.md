# Security

## Reporting

Report vulnerabilities privately through GitHub's *Report a vulnerability* form on this
repository (Security → Advisories). Please do not open a public issue for exploitable problems.

## Scope and known limitations

`proteus serve` exposes a GA4GH TES v1.1 endpoint. Since 0.4.0 executors run **inside the
task's container image** (Podman/Docker socket) with `resources.cpu_cores`/`ram_gb` enforced,
no network unless the operator passes `--executor-network` (a task cannot turn it on), a
per-executor timeout and cancellation. Releases before 0.4.0
ran executors as host processes without authentication; do not expose them. Hardening knobs:

- `--auth-token` / `PROTEUS_AUTH_TOKEN` — bearer token required on `/v1`, `/ga4gh`, `/api`
  (compared as fixed-length digests in constant time; `/health`, `/metrics`, `/swagger-ui`
  stay open).
- No CORS headers are sent, so a web page in the operator's browser cannot drive a daemon on
  localhost. TES clients are not browsers.
- `--allow-image GLOB` (repeatable) — executor images must match; rejected tasks get 400. The
  patterns are advertised in `service-info.tags`. Without patterns any image is accepted.
- `--allow-dir DIR` (repeatable) — `file://` input and output URLs, and bare host paths, must
  resolve (symlinks followed) inside one of these directories; the default is the daemon's own
  artifacts directory. Outside it the task is rejected with 400, and a task record that never
  went through validation is re-checked when inputs are staged and outputs delivered. Without
  this, any client could read and write the host filesystem as the daemon's user.
- The task id is always assigned by the server (it names the task's work dir on the host);
  a client-supplied `id`, `state`, `logs` or `creation_time` is ignored.
- Task paths must be absolute, may not contain `..`, and may not mount over system directories
  (`/etc`, `/usr`, …). An executor can write inside the work dir, so everything the daemon
  does there afterwards assumes it may have planted symlinks: outputs that resolve outside the
  work dir are not delivered (`SYSTEM_ERROR`), a symlink anywhere inside a copied directory
  fails the copy (inputs and outputs), `stdout`/`stderr` files are created fresh rather than
  written through an existing link, and `stdin` is read only from inside the work dir.
- Captured stdout/stderr is limited to 8 MiB per stream per executor; the rest is dropped
  and the drop recorded in `system_logs`.
- A task left unfinished by a daemon that stopped is marked `SYSTEM_ERROR` when the daemon
  starts again, and its labelled container removed. One daemon per data directory is assumed.
  Host-executor processes from the old daemon are not tracked and are not stopped.
- `--executor host` runs commands directly on the daemon host, ignores `image`, and is refused
  unless `--host` is loopback. Development only: workflow engines write container paths
  (`/work/…`, `/mnt/task/…`) into their command scripts, which do not exist on the host, so
  Nextflow and Sprocket tasks need the container executor.

What is **not** covered: TLS (terminate it in a reverse proxy), per-user authorization or GA4GH
Passports, rate limiting, disk quotas (`disk_gb` is accepted but not enforced), and output
delivery to object stores (`file://` only). Inputs may be fetched from `http(s)://` URLs the
server can reach — restrict egress if the daemon runs inside a private network.

Bind to localhost or put it behind an authenticating reverse proxy on shared networks.

Dependencies are checked with `cargo deny` and `cargo audit` in CI.
