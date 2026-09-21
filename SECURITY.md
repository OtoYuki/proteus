# Security

## Reporting

Report vulnerabilities privately through GitHub's *Report a vulnerability* form on this
repository (Security → Advisories). Please do not open a public issue for exploitable problems.

## Scope and known limitations

`proteus serve` exposes a GA4GH TES v1.1 endpoint. **In 0.3.x, TES executors run as plain host
processes: the task's `executors[].image` field is accepted but not used** (the OCI/Podman runner
is only used for the prediction tiers, not for TES executors), and there is no authentication, no
command allow-list and no resource limiting. Anyone who can reach the port can run arbitrary
commands as the daemon's user. Bind it to localhost only (`--host 127.0.0.1`, the default) or put it
behind an authenticating reverse proxy; never expose it on a shared network as is.

Containerised executors honouring `image` and `resources`, a bearer-token option and an image
allow-list are the first items of the next release (see `CHANGELOG.md` → Unreleased).

Dependencies are checked with `cargo deny` and `cargo audit` in CI.
