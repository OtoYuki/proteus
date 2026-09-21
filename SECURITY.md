# Security

## Reporting

Report vulnerabilities privately through GitHub's *Report a vulnerability* form on this
repository (Security → Advisories). Please do not open a public issue for exploitable problems.

## Scope and known limitations

`proteus serve` exposes a GA4GH TES v1.1 endpoint and, when a Podman/Docker socket is available,
runs the container image named in each task. **It has no authentication and no image allow-list**;
anyone who can reach the port can run arbitrary containers on the host. Run it only on localhost or
behind an authenticating reverse proxy, and do not mount a container socket on an untrusted network.
An allow-list and resource limits are tracked for the next release.

Dependencies are checked with `cargo deny` and `cargo audit` in CI.
