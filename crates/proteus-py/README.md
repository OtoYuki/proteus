# proteus-bio (Python)

The measurements of [`proteus analyze`](https://github.com/OtoYuki/proteus), from Python: one row
per predicted model, as a `pyarrow.Table`. The analysis runs in Rust, in parallel, with the GIL
released; nothing else is installed besides `pyarrow`.

```python
import proteus_bio as pb

table = pb.analyze("boltz_results/", interface="A")       # chain A against the rest
df = table.to_pandas()
df.sort_values("ipsae_min", ascending=False).head(20)
```

`analyze(paths, *, interface=None, reference=None, confidence="auto", jobs=None, strict=False)`

- `paths`: a file or directory, or a list of them. Directories are searched recursively for
  `.pdb`, `.ent`, `.cif` and `.mmcif` files, optionally gzipped.
- `interface`: `"A"`, `"A:B"`, `"H,L:A"`, or `""` for the first chain against the rest. When
  the predictor's PAE and score files sit beside each model (AlphaFold 3, Boltz, Chai-1,
  OpenFold3, Protenix, ColabFold), adds ipTM, ipAE, ipSAE and LIS.
- `reference`: adds each model's Kabsch Cα RMSD against this structure.
- `strict`: raise `AnalysisError` when any input fails; by default each failure is an
  `AnalysisWarning` and the rest are still returned.

The columns are those of `proteus analyze --export`; the repository's README defines them.

## Building from source

```bash
cd crates/proteus-py
uv venv && uv pip install maturin pyarrow pytest pandas
maturin develop --release
pytest
```
