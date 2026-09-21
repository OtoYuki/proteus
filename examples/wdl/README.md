# WDL on Proteus via GA4GH TES (Sprocket)

`analyze.wdl` runs `proteus analyze` on a structure inside the `ghcr.io/otoyuki/proteus`
container. Any TES 1.1 client can drive it; this example uses
[Sprocket](https://sprocket.bio), St. Jude Rust Labs' WDL engine, whose TES backend submits each
task to `proteusd`, which runs it in a container through the Podman/Docker socket.

```bash
# 1. TES server (container executor; allow only the images you expect)
proteus serve --port 8080 --allow-image 'ghcr.io/otoyuki/*'

# 2. Sprocket's file-backed storage roots (it does not create them itself)
mkdir -p /tmp/tes-store/in/file /tmp/tes-store/out

# 3. Run the workflow
sprocket run -c examples/wdl/sprocket.toml -s examples/wdl/analyze.wdl @examples/wdl/inputs.json
# → { "analyze_structure.metrics": "file:///tmp/tes-store/out/<task>/work/metrics.txt" }
```

What crosses the wire: Sprocket uploads `1crn.pdb` to `inputs`, POSTs a task with the input
URL, a `command` file (bash), `workdir: /mnt/task/work`, three outputs (`work` directory,
`stdout`, `stderr`) with `file://` destination URLs, and `resources` (cpu, RAM). `proteusd`
stages the inputs, runs `bash /mnt/task/command` in the image with those limits, copies the
outputs to their URLs and reports `size_bytes` as strings, as the TES int64 mapping requires.

The same server also runs `examples/nextflow/` (`nextflow run … -with-tes`).
`.github/workflows/tes-conformance.yml` runs the ELIXIR/GA4GH compliance suite and this
workflow in CI.
