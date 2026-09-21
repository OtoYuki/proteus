# Contributing

## Ground rules

- **Every scientific number needs a reference.** A change to `proteus-core` biophysics or
  `proteus-dssp` must keep `make validate` green (43 structures vs mdtraj / FreeSASA / cctbx). If a
  tolerance in `validate/tolerances.toml` has to move, say why in `validate/README.md`.
- **Label approximations.** If a metric is not the thing it is named after (see the heavy-atom
  overlap score vs MolProbity clashscore), the name, the doc comment and the README must say so.
- **No unmeasured performance claims.** Numbers in `bench/README.md` come from `bench/run.sh`.

## Workflow

```bash
cargo nextest run --workspace          # or: cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
cargo deny check                       # licences, advisories, duplicates
make validate                          # reference harness (downloads ~50 MB corpus once)
```

CI runs the same commands on Linux and macOS, plus the MSRV (`rust-version` in `Cargo.toml`).

Commit messages: `type(scope): summary` (`feat`, `fix`, `refactor`, `test`, `docs`, `build`,
`chore`), imperative, with a body that says *what changed in the output* when it did.

## Adding a structure to the validation corpus

1. Append a `[[structure]]` block to `validate/corpus.toml` (RCSB or AlphaFold-DB URL).
2. `make fetch` — paste the printed sha256 into the block.
3. `make reference` — commits a new `validate/reference/<id>_<fmt>.json`.
4. `make validate`.

## Licence

By contributing you agree that your contributions are licensed under MIT OR Apache-2.0, at the
user's option, like the rest of the project.
