#!/usr/bin/env bash
# Proteus as a GA4GH TES backend, driven by Nextflow. Recorded by record.sh tes.
# Needs the image (`podman build -t localhost/proteus:qa .`), a container socket, and nextflow
# on PATH (or NF=/path/to/nextflow).
source "$(dirname "$0")/_lib.sh"
NF="${NF:-$(command -v nextflow)}"
export RUST_LOG=warn
demo_home tes
export NXF_HOME="$HOME/.nextflow" PROTEUS_TES_ENDPOINT=http://127.0.0.1:18400
mkdir -p "$HOME/work"
clear
type_out "proteus serve --port 18400 --allow-image 'localhost/*' --allow-dir ~/work &"
proteus serve --port 18400 --allow-image 'localhost/*' --allow-dir ~/work --allow-dir "$REPO" --no-pull >"$HOME/serve.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:18400/v1/service-info >/dev/null && break; sleep 0.5; done
type_out 'curl -s localhost:18400/v1/service-info | jq -c "{name, type, version}"'
curl -s http://127.0.0.1:18400/v1/service-info | jq -c '{name, type: .type.artifact, version: .type.version}'
sleep 2
printf '\n'
type_out "nextflow run examples/nextflow/screening.nf --image localhost/proteus:qa -w ~/work"
# -ansi-log false: Nextflow's live-updating display is cursor control, and a filtered pipe
# turns it into noise. The sed puts back the newline the nf-ga4gh plugin's log line omits,
# which otherwise glues each [TES] line to the [PROCESS] line after it.
( cd "$HOME/work" && "$NF" run "$REPO/examples/nextflow/screening.nf" \
    -ansi-log false --image localhost/proteus:qa --outdir "$HOME/work/results" -w "$HOME/work" 2>&1 \
    | command sed 's/\[PIPELINE\]/\n[PIPELINE]/g; s/\[PROCESS /\n[PROCESS /g' \
    | rg --color never 'TES|PROCESS|SUCCESS' | head -12 )

printf '\n'
# view=BASIC: ListTasks defaults to MINIMAL, which is id and state only, per TES 1.1.
type_out 'curl -s "localhost:18400/v1/tasks?view=BASIC" | jq -c ".tasks[] | {name, state}"'
curl -s 'http://127.0.0.1:18400/v1/tasks?view=BASIC' | jq -c '.tasks[] | {name, state}'
sleep 5
