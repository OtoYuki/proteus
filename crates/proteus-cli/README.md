<img src="https://raw.githubusercontent.com/OtoYuki/proteus/main/docs/brand/proteus-mark.png" width="48" align="right" alt="Proteus">

# proteus-cli

The `proteus` command-line binary of [Proteus](https://github.com/OtoYuki/proteus).

<img src="https://raw.githubusercontent.com/OtoYuki/proteus/main/docs/media/home.gif" alt="the home screen that bare proteus opens: jobs, a folder of structures measured as you move, and the fold form">

```
proteus analyze 1crn.pdb                       # all-atom biophysics of a PDB/mmCIF file
proteus analyze models/ --export qc.parquet    # one row per structure, over a whole folder
proteus mutate wt.fasta --mode alanine         # in-silico variant library (stdout or --output)
proteus screen lib.fasta --export out.parquet  # fold, score, rank; Parquet/CSV/JSON export
proteus esm scan wt.fasta --export scan.csv    # ESM-2 deep mutational scan, pure Rust
proteus                                        # the home screen: jobs, files, forms
proteus view 1crn.pdb --interactive --dashboard
proteus view 1crn.pdb --web                    # the same structure in a browser, offline
proteus submit --file wt.fasta && proteus inspect <job>
proteus serve --port 8080 --allow-image 'ghcr.io/otoyuki/*' --allow-dir /srv/tes
```

Install from the [GitHub releases](https://github.com/OtoYuki/proteus/releases) (Linux and
macOS, x86_64 and arm64), as the `ghcr.io/otoyuki/proteus` image, or from source:

```bash
cargo install --git https://github.com/OtoYuki/proteus proteus-cli
```

This crate is not published to crates.io — the name belongs to an unrelated project — so the
git URL is the source install. The repository README documents every command and flag.
