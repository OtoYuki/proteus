#!/usr/bin/env bash
# Every number checked against an implementation someone else wrote.
source "$(dirname "$0")/_lib.sh"
export PATH="$PWD/target/release:$PATH" RUST_LOG=off
clear
type_out "make validate     # 53 structures vs mdtraj, FreeSASA, cctbx and PLIP"
cargo test -p proteus-core --release --test validation -- --ignored --nocapture 2>/dev/null \
  | rg --color never --line-buffered '^\| (id|---|1crn|1ubq|1igt|2rh1|4hhb|6vxx|af-p69905)|^test result'
printf '\n'
sleep 1
cargo test -p proteus-core --release --test plip_validation -- --ignored --nocapture 2>/dev/null \
  | rg --color never --line-buffered 'corpus totals|salt bridges:|pi-pi:|cation-pi:'
printf '\n'
sleep 1
cargo test -p proteus-core --release --test fitness_discrimination -- --ignored --nocapture 2>/dev/null \
  | rg --color never --line-buffered 'smallest native'
sleep 4
