# SP3 — TES execution, hardening and ecosystem hooks

**Status:** approved under the 2026-09-21 goal · **Scope:** `proteus-engine`, `proteus-server`, `proteus-cli`, `examples/`, CI
**Depends on:** SP1 (correct science), SP2 (release infrastructure, v0.3.0)

## 1. Problem

At v0.3.0 `proteusd` advertises GA4GH TES v1.1 but (`crates/proteus-engine/src/scheduler.rs:407-470`):

1. **Executors run as host processes.** `executors[].image` is ignored; `tokio::process::Command::new(executor.command[0])` runs on the daemon host. Not TES-compliant, and remote code execution for anyone who can reach the port.
2. `resources` (`cpu_cores`, `ram_gb`, `disk_gb`, `preemptible`, `zones`) are parsed and ignored.
3. No authentication, no image allow-list, no request limits.
4. Conformance has never been checked against the official suite; the audience for this project (St. Jude Rust Labs / Seqera-style infra teams) will run one.
5. No worked example against a real workflow engine's TES backend other than the Nextflow config.

## 2. Goals

1. TES executors run **inside the task's `image`** via the OCI runner (bollard, Podman/Docker socket), with inputs staged into the work dir, `resources.cpu_cores`/`ram_gb` enforced as container limits, stdout/stderr captured per executor, outputs harvested. Host-process execution survives only behind an explicit `--executor host` flag for local development and is refused when `--host` is not loopback.
2. `--auth-token <secret>` (or `PROTEUS_AUTH_TOKEN`): when set, every `/v1/*`, `/ga4gh/*`, `/api/*` request needs `Authorization: Bearer <secret>`; `/health` and `/metrics` stay open (metrics can be restricted with `--metrics-token`). Refused with 401 otherwise.
3. `--allow-image <glob>` (repeatable): when any is given, task creation with an executor image not matching any glob is rejected with 400 and a message naming the image. Default (no flags): allow all, but log a warning at startup when bound to a non-loopback address.
4. `openapi-test-runner` (ELIXIR TES compliance suite) runs against `proteusd` in CI on every push; the JSON report is uploaded; a badge in the README.
5. `examples/wdl/`: a WDL workflow that runs `proteus analyze` on an input structure, executed with **Sprocket** (`sprocket run` with its TES backend pointed at `proteusd`), verified in CI.
6. `examples/nextflow/` stays and gets a CI smoke test with `nextflow run … -with-tes` when Java is available on the runner.

Non-goals: Boltz-2/OpenFold3 runner images (SP5), multi-node scheduling, TLS termination (reverse proxy), OAuth/OIDC (GA4GH Passport) — bearer token is enough for this milestone.

## 3. Design

### 3.1 Executor backend (`proteus-engine`)
```rust
pub enum ExecutorBackend { Container(Arc<OciRunner>), Host }
pub struct TesExecutionConfig { pub backend: ExecutorBackend, pub allow_images: Vec<glob::Pattern>, pub default_cpu: f64, pub default_ram_gb: f64 }
```
`PipelineScheduler::execute_tes_task` dispatches on the backend. Container path, per executor:
- `create_container` with `Image`, `Cmd = command`, `WorkingDir = executor.workdir.unwrap_or("/")`, `Env`, `HostConfig { binds: [work_dir:/proteus/work] , nano_cpus: cpu_cores*1e9, memory: ram_gb*2^30, network_mode: "none" unless task tags[\"proteus.network\"]="true" }`; inputs are staged into `work_dir/<input.path>` before the first executor, outputs harvested from `work_dir/<output.path>` after the last (the existing staging/harvest code stays).
- `start_container`, `wait_container`, `logs` (stdout/stderr split) → `TesExecutorLog { exit_code, stdout, stderr, start/end }`; container removed with `force: true` in all paths (also on cancel).
- Image missing locally → `pull` once (`create_image`) unless `--no-pull`; pull failure → task `SYSTEM_ERROR` with the message in `system_logs`.
- Cancellation: a `CancellationToken` per task; cancel → `kill_container` + `CANCELED`.
- `executor.stdin` (TES field) is passed as container stdin when present.

Host path: unchanged code, but only constructed when `--executor host` is passed; `run_server` refuses `--executor host` with a non-loopback bind unless `--i-know-what-i-am-doing`. TES `service-info` reports `"proteus.executor": "container"|"host"` under `tags` so clients can tell.

### 3.2 Auth middleware (`proteus-server`)
`tower_http::validate_request::ValidateRequestHeaderLayer::bearer(token)` on the API routers; `/health`, `/metrics`, `/swagger-ui` excluded (metrics optionally behind `--metrics-token`). Token compared in constant time (`subtle`). Unit test: 401 without header, 200 with.

### 3.3 Image allow-list
Checked in `create_task` before enqueue; error body `{ "error": "image 'x' not allowed; allowed: [...]" }` with 400. Also enforced at execution time (defence in depth). `glob` crate patterns, e.g. `--allow-image 'ghcr.io/otoyuki/*' --allow-image 'docker.io/library/alpine:*'`.

### 3.4 Conformance in CI
`.github/workflows/tes-conformance.yml`: build `proteus`, start `proteus serve --executor container` on `127.0.0.1:8080` with Docker available on the runner, `pip install git+https://github.com/elixir-cloud-aai/openapi-test-runner`, `openapi-test-runner report --server http://127.0.0.1:8080/ --version 1.1.0 --output_path report.json`, fail the job if any test failed (parse the JSON), upload `report.json`. Fixes to `tes_api.rs`/`tes.rs` as the suite demands (expected: `ListTasks` pagination fields `page_size`/`page_token`/`next_page_token`, `view=MINIMAL|BASIC|FULL` projections, `state` filter, 404 body shape, `service-info` required fields `id/name/type/organization/version`). Badge from the workflow.

### 3.5 Sprocket example
`examples/wdl/analyze.wdl` (task `analyze { input File structure; command { proteus analyze --pdb ~{structure} > metrics.txt }; runtime { container: "ghcr.io/otoyuki/proteus:latest" } output { File metrics = "metrics.txt" } }`), `examples/wdl/inputs.json`, `examples/wdl/sprocket.toml` (TES backend → `http://127.0.0.1:8080/v1`), `examples/wdl/README.md` with the two commands. CI job installs the Sprocket release binary, runs it against `proteusd` with the container executor, asserts `metrics.txt` contains `Radius of Gyration`.

### 3.6 Docs
README "Run it from a workflow engine" section (Nextflow and Sprocket snippets), SECURITY.md updated to the new model, CHANGELOG Unreleased → 0.4.0 items.

## 4. Error handling
Container runtime unavailable at startup with `--executor container` (default) → startup fails with a message that names the sockets tried and suggests `--executor host` for loopback-only dev. Pull failure, non-zero exit, timeout (`--task-timeout`, default 1 h) → `EXECUTOR_ERROR`/`SYSTEM_ERROR` with `system_logs`. Allow-list rejection is 400 at submit time, never a silent skip.

## 5. Testing
- `proteus-engine`: container executor integration test gated on a reachable socket (`#[ignore]` + `PROTEUS_TEST_OCI=1`), using `docker.io/library/alpine` with `sh -c 'echo hi > /proteus/work/out.txt'` and an output declaration; asserts harvest + exit code + logs; cancel test.
- `proteus-server`: auth 401/200, allow-list 400, existing `tes_e2e.rs` extended with `view=` projections and pagination.
- CI: conformance job + Sprocket job (both need Docker on `ubuntu-latest`, which has it).
