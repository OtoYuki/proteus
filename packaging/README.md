# Packaging

Proteus users mostly run their pipelines on a cluster, through Nextflow, Snakemake or a shell
script. This directory holds what those channels need.

## Bioconda — `bioconda/proteus/`

`meta.yaml` and `build.sh` are the recipe to submit to
[bioconda-recipes](https://github.com/bioconda/bioconda-recipes). It builds the `proteus` binary
from the release tarball with conda-forge's Rust and C toolchains, bundles third-party licences
with `cargo-bundle-licenses`, and tests `analyze` on crambin and on trypsin–BPTI with
`--interface`.

It has not been submitted yet. Submission waits for a release tag that contains
`--interface`. At that release:

```bash
git tag v0.9.0 && git push origin v0.9.0     # the release workflow builds binaries and the image
packaging/bioconda/update.sh 0.9.0          # fills in the version and the tarball's sha256
# then copy bioconda/proteus/ into recipes/proteus/ of a bioconda-recipes fork and open a PR
```

Once the package exists, BioContainers publishes `quay.io/biocontainers/proteus`
automatically. The nf-core module's `container` line can then point there instead of at
`ghcr.io/otoyuki/proteus`.

**Local check** (how the recipe was verified, without Bioconda's CI): the same recipe with
`source: git_url` pointing at the repository, built with conda-build and conda-forge's pinning
in a `condaforge/miniforge3` container:

```bash
conda build -m $CONDA_PREFIX/conda_build_config.yaml -c conda-forge -c bioconda --override-channels recipe/
```

## nf-core module — `nf-core/modules/proteus/analyze/`

`PROTEUS_ANALYZE` follows the nf-core module layout: `main.nf`, `meta.yml`, `environment.yml`
and `tests/main.nf.test`. Its input is a set of model files, staged together with the PAE and
scores files the predictor wrote beside them. Its output is one Parquet table, and it reports
its version on the `versions` topic. Tool options go through `ext.args`; for binder triage:

```groovy
process {
    withName: 'PROTEUS_ANALYZE' { ext.args = '--interface A:B' }
}
```

```bash
cd packaging/nf-core && nf-test test modules/proteus/analyze/tests/main.nf.test   # runs in CI
nextflow run examples/nextflow/binder-triage --campaigns 'boltz_results/*' --interface A:B
```

To offer it to nf-core/modules, copy the directory to `modules/nf-core/proteus/analyze/` in a
fork. Then switch the test inputs to nf-core's test-datasets and the container to the
BioContainers image. Until the Bioconda package exists, `environment.yml` names a version that
does not resolve, and the module runs from a `proteus` on `PATH` or from the ghcr image
(`-with-docker`).
