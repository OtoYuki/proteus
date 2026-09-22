#!/usr/bin/env bash
# The design loop: a scaffold goes in, a ranked library comes out.
# Uses the ESM Atlas fold API so the leaderboard shows real, differing structures rather than
# the offline simulator's placeholders.
source "$(dirname "$0")/_lib.sh"
export PATH="$PWD/target/release:$PATH" RUST_LOG=off PROTEUS_DATA_DIR=/tmp/proteus-demo-loop
rm -rf "$PROTEUS_DATA_DIR"
WT=docs/media/demos/protein_g.fasta
clear
type_out "proteus mutate $WT --mode alanine --start 24 --end 29 | proteus screen - --runner esm-api --top 5"
proteus mutate $WT --mode alanine --start 24 --end 29 \
  | proteus screen - --runner esm-api --top 5 --export /tmp/proteus-demo-loop/lib.parquet
sleep 4
