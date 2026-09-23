#!/usr/bin/env bash
# Re-record the README demos. One argument selects which, no argument records all.
#
#   docs/media/record.sh            # all of them
#   docs/media/record.sh loop       # just docs/media/loop.gif
#
# Needs: cargo build --release, asciinema (uv tool install asciinema), agg
# (cargo install --git https://github.com/asciinema/agg), and for the tes demo a container
# runtime plus nextflow on PATH (or NF=/path/to/nextflow).
#
# The interactive viewer is sent 'q' through asciinema's stdin after its recording window. A
# demo with a docs/media/demos/<name>.keys file is driven by it instead, inside tmux: one
# `sleep SECONDS` or `send TEXT` (`send \r` is Enter) per line.
set -euo pipefail
cd "$(dirname "$0")/../.."

# name  cols rows  seconds-before-q  idle-limit  font-size  fps-cap
DEMOS=(
  "loop      138 26  0   2  14  12"
  "view      140 38  9   2  12   8"
  "validate  144 21  0   2  13  12"
  "tes       124 24  0   3  14  12"
  "home      110 30  0   3  14  12"
)

record() {
  local name=$1 cols=$2 rows=$3 quit_after=$4 idle=$5 font=$6 fps=$7
  local cast="docs/media/${name}.cast" gif="docs/media/${name}.gif"
  echo "recording $name (${cols}x${rows})"
  local keys="docs/media/demos/${name}.keys"
  if [[ -f "$keys" ]]; then
    # A real terminal on both ends (the home screen checks), so asciinema runs inside a
    # detached tmux session and the keys go in with `tmux send-keys`.
    local session="proteus-rec-$$"
    tmux new-session -d -s "$session" -x "$cols" -y "$rows" \
      "asciinema rec --overwrite -c 'bash docs/media/demos/${name}.sh' '$cast'"
    while read -r cmd arg; do
      case "$cmd" in
        sleep) sleep "$arg" ;;
        send) if [[ "$arg" == '\r' ]]; then tmux send-keys -t "$session" Enter
              else tmux send-keys -t "$session" -l "$arg"; fi ;;
      esac
    done <"$keys"
    while tmux has-session -t "$session" 2>/dev/null; do sleep 0.5; done
  elif [[ "$quit_after" -gt 0 ]]; then
    ( sleep "$quit_after"; printf 'q'; sleep 30 ) \
      | asciinema rec --overwrite --cols "$cols" --rows "$rows" \
          -c "bash docs/media/demos/${name}.sh" "$cast"
  else
    asciinema rec --overwrite --cols "$cols" --rows "$rows" \
      -c "bash docs/media/demos/${name}.sh" "$cast"
  fi
  agg --font-size "$font" --theme monokai --fps-cap "$fps" --idle-time-limit "$idle" \
      --last-frame-duration 3 "$cast" "$gif"
  echo "  -> $gif ($(du -h "$gif" | cut -f1))"
}

for spec in "${DEMOS[@]}"; do
  read -r name cols rows quit idle font fps <<<"$spec"
  [[ $# -gt 0 && "$1" != "$name" ]] && continue
  record "$name" "$cols" "$rows" "$quit" "$idle" "$font" "$fps"
done
