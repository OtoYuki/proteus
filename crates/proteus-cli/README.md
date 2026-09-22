# proteus-cli

The `proteus` command-line binary of [Proteus](https://github.com/OtoYuki/proteus).

```
proteus analyze --pdb 1crn.pdb                 # all-atom biophysics of a PDB/mmCIF file
proteus mutate wt.fasta --mode alanine         # in-silico variant library (stdout or --output)
proteus screen lib.fasta --export out.parquet  # fold, score, rank; Parquet/CSV/JSON export
proteus esm scan wt.fasta --export scan.csv    # ESM-2 deep mutational scan, pure Rust
proteus view 1crn.pdb --interactive --dashboard
proteus submit --file wt.fasta && proteus inspect <job>
proteus serve --port 8080 --allow-image 'ghcr.io/otoyuki/*' --allow-dir /srv/tes
```

Install with `cargo install proteus-cli`, from the GitHub releases, or as the
`ghcr.io/otoyuki/proteus` image. The repository README documents every command and flag.
