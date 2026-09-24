#!/usr/bin/env bash
# Every number checked against an implementation someone else wrote. The rows are the
# validation tests' own output for a few structures, aligned into columns; nothing is edited.
source "$(dirname "$0")/_lib.sh"
export RUST_LOG=off
DIM=$'\033[2m' ACC=$'\033[38;2;153;146;11m' OFF=$'\033[0m'
clear
type_out "make validate     # 53 structures vs mdtraj, FreeSASA, cctbx and PLIP"
printf '%s(per structure)%s %s9 of 53 files; kind and DSSP-3 columns left out%s\n' "$ACC" "$OFF" "$DIM" "$OFF"
VALIDATION=$(cargo test -p proteus-core --release --test validation -- --ignored --nocapture 2>/dev/null)
rg --color never '^\| (id|1crn|1ubq|1igt|2rh1|4hhb|6vxx|af-p69905) ' <<<"$VALIDATION" \
  | command sed 's/^| //; s/ |$//' \
  | awk -F' [|] ' -v OFS='|' '{ print $1, $2, $4, $5, $6, $7, $8, $9, $11, $12, $13 }' \
  | column -t -s '|' -o "${DIM} │ ${OFF}" \
  | command sed "1s/^/${DIM}/; 1s/\$/${OFF}/"
printf '\n%s(interactions vs PLIP, whole corpus)%s\n' "$ACC" "$OFF"
cargo test -p proteus-core --release --test plip_validation -- --ignored --nocapture 2>/dev/null \
  | rg --color never 'salt bridges:|pi-pi:|cation-pi:' | command sed 's/^ *//' | column -t -s ':' -o ':'
printf '\n%s(triage score)%s\n' "$ACC" "$OFF"
cargo test -p proteus-core --release --test fitness_discrimination -- --ignored --nocapture 2>/dev/null \
  | rg --color never 'smallest native'
rg --color never '^test result' <<<"$VALIDATION" \
  | command sed "s/ok\./${ACC}ok${OFF}./"
sleep 4
