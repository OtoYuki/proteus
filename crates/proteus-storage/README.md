# proteus-storage

Persistence for [Proteus](https://github.com/OtoYuki/proteus): an embedded SQLite repository
(SQLx, WAL mode, embedded migrations) for sequences, jobs, predictions, metrics and TES tasks; a
BLAKE3 content-addressable store for structure artifacts; and screening exports to Apache
Parquet (ZSTD), CSV and JSON with a versioned column schema (`proteus.schema_version = 4`).

```rust
use proteus_storage::{create_sqlite_pool, repository::ProteusRepository};
let pool = create_sqlite_pool("proteus.db").await?;
let repo = ProteusRepository::new(pool);
let page = repo.list_tes_tasks_page(Some("COMPLETE"), None, 50, 0).await?;
```
