#!/usr/bin/env bash
# The design loop: a scaffold goes in, a ranked library comes out.
# Uses the ESM Atlas fold API so the leaderboard shows real, differing structures rather than
# the offline simulator's placeholders.
source "$(dirname "$0")/_lib.sh"
export RUST_LOG=off
demo_home loop
cp "$REPO/docs/media/demos/protein_g.fasta" "$HOME/wt.fasta"
cd "$HOME"
clear
type_out "proteus mutate wt.fasta --mode alanine --start 24 --end 29 | proteus screen - --runner esm-api --top 5 --export library.parquet"
proteus mutate wt.fasta --mode alanine --start 24 --end 29 \
  | proteus screen - --runner esm-api --top 5 --export library.parquet
sleep 4
