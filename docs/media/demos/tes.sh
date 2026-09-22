#!/usr/bin/env bash
# Proteus as a GA4GH TES backend, driven by Nextflow. Recorded by record.sh tes.
source "$(dirname "$0")/_lib.sh"
export PATH="$PWD/target/release:$PATH" RUST_LOG=warn
export PROTEUS_DATA_DIR=/tmp/proteus-demo-tes NXF_HOME=/tmp/proteus-demo-tes/nf
export PROTEUS_TES_ENDPOINT=http://127.0.0.1:18400
WORK=/tmp/proteus-demo-tes/work
rm -rf /tmp/proteus-demo-tes; mkdir -p "$WORK"
clear
type_out "proteus serve --port 18400 --allow-image 'localhost/*' --allow-dir $WORK &"
proteus serve --port 18400 --allow-image 'localhost/*' --allow-dir "$WORK" --allow-dir "$PWD" --no-pull >/tmp/proteus-demo-tes/srv.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:18400/v1/service-info >/dev/null && break; sleep 0.5; done
type_out 'curl -s localhost:18400/v1/service-info | jq -c "{name, type, version}"' 
curl -s http://127.0.0.1:18400/v1/service-info | jq -c '{name, type: .type.artifact, version: .type.version}'
sleep 2
printf '\n'
type_out "nextflow run examples/nextflow/screening.nf --image localhost/proteus:qa"
# -ansi-log false: Nextflow's live-updating display is cursor control, and a filtered pipe
# turns it into noise. The sed puts back the newline the nf-ga4gh plugin's log line omits,
# which otherwise glues each [TES] line to the [PROCESS] line after it.
( cd "$WORK" && "${NF:-nextflow}" run "$OLDPWD/examples/nextflow/screening.nf" \
    -ansi-log false --image localhost/proteus:qa --outdir "$WORK/results" -w "$WORK" 2>&1 \
    | command sed 's/\[PIPELINE\]/\n[PIPELINE]/g; s/\[PROCESS /\n[PROCESS /g' \
    | rg --color never 'TES|PROCESS|SUCCESS' | head -12 )

printf '\n'
# view=BASIC: ListTasks defaults to MINIMAL, which is id and state only, per TES 1.1.
type_out 'curl -s "localhost:18400/v1/tasks?view=BASIC" | jq -c ".tasks[] | {name, state}"'
curl -s 'http://127.0.0.1:18400/v1/tasks?view=BASIC' | jq -c '.tasks[] | {name, state}'
sleep 5
