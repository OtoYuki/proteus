# proteus-engine

Job execution for [Proteus](https://github.com/OtoYuki/proteus): the `PipelineScheduler`
(persisted jobs, worker pool, event broadcast), prediction runners behind one `ComputeRunner`
trait (`OciRunner` for your own ESMFold/Boltz images over the Podman/Docker socket,
`EsmApiRunner` for the ESM Atlas fold API, `SimulatedRunner` for offline placeholders, and
`AutoRunner` which tries them in that order and records any tier downgrade in the prediction
metadata), and GA4GH TES 1.1 task execution (`tes_exec`) that stages inputs, runs each executor
in its container image with cpu/memory limits, harvests outputs and delivers them to `file://`
URLs under an allow-list.

Every runner writes `metadata.engine` (`oci`, `esmfold-api`, `simulated`) so downstream tables
can say where a structure came from.
