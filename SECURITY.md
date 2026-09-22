# Security

## Reporting

Report vulnerabilities privately through GitHub's *Report a vulnerability* form on this
repository (Security → Advisories). Please do not open a public issue for exploitable problems.

## Scope and known limitations

`proteus serve` exposes a GA4GH TES v1.1 endpoint. Since 0.4.0 executors run **inside the
task's container image** (Podman/Docker socket) with `resources.cpu_cores`/`ram_gb` enforced,
no network unless requested, a per-executor timeout and cancellation. Releases before 0.4.0
ran executors as host processes without authentication; do not expose them. Hardening knobs:

- `--auth-token` / `PROTEUS_AUTH_TOKEN` — bearer token required on `/v1`, `/ga4gh`, `/api`
  (constant-time comparison; `/health`, `/metrics`, `/swagger-ui` stay open).
- `--allow-image GLOB` (repeatable) — executor images must match; rejected tasks get 400. The
  patterns are advertised in `service-info.tags`. Without patterns any image is accepted.
- `--allow-dir DIR` (repeatable) — `file://` input and output URLs, and bare host paths, must
  resolve (symlinks followed) inside one of these directories; the default is the daemon's own
  artifacts directory. Outside it the task is rejected with 400, and a task record that never
  went through validation is re-checked when inputs are staged and outputs delivered. Without
  this, any client could read and write the host filesystem as the daemon's user.
- Task paths must be absolute, may not contain `..`, and may not mount over system directories
  (`/etc`, `/usr`, …). Outputs that resolve outside the task work dir (a planted symlink) are
  not delivered and the task ends in `SYSTEM_ERROR`.
- `--executor host` runs commands directly on the daemon host, ignores `image`, and is refused
  unless `--host` is loopback. Development only.

What is **not** covered: TLS (terminate it in a reverse proxy), per-user authorization or GA4GH
Passports, rate limiting, disk quotas (`disk_gb` is accepted but not enforced), and output
delivery to object stores (`file://` only). Inputs may be fetched from `http(s)://` URLs the
server can reach — restrict egress if the daemon runs inside a private network.

Bind to localhost or put it behind an authenticating reverse proxy on shared networks.

Dependencies are checked with `cargo deny` and `cargo audit` in CI.
