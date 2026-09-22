# proteus-server

The `proteusd` HTTP daemon of [Proteus](https://github.com/OtoYuki/proteus), on axum: a GA4GH
TES 1.1 server (`/v1/tasks`, `/ga4gh/tes/v1/…`; passes the ELIXIR compliance suite), the native
job API (`/api/v1/…`) with Server-Sent Events per job, a Mol* page per prediction (`/view/{job}`),
Prometheus `/metrics`, `/health`, and Swagger UI at `/swagger-ui`. Optional bearer-token auth
covers everything except `/health`, `/metrics` and the Swagger UI.

```rust
use proteus_server::{build_router_with_options, AppState, ServerOptions};
let router = build_router_with_options(AppState::new(scheduler), ServerOptions::default());
axum::serve(listener, router).await?;
```
