#!/usr/bin/env bash
# Re-record the README demos. One argument selects which, no argument records all.
#
#   docs/media/record.sh            # all of them
#   docs/media/record.sh loop       # just docs/media/loop.gif
#
# Needs: cargo build --release -p proteus-cli, asciinema (uv tool install asciinema), agg 1.9+
# (cargo install --git https://github.com/asciinema/agg; or AGG=/path/to/agg), tmux, and the
# Geist Mono font (falls back to JetBrains Mono). The tes demo also needs a container runtime,
# the image (`podman build -t localhost/proteus:qa .`) and nextflow (or NF=/path/to/nextflow).
#
# Frames are rendered in the identity's own terminal colours, docs/brand/proteus-dark.agg,
# which proteus_render::brand::assets generates. Recordings (.cast) go to CAST_DIR, default
# ~/.cache/proteus-demo/casts; only the GIFs are committed.
#
# The interactive viewer is sent 'q' through asciinema's stdin after its recording window. A
# demo with a docs/media/demos/<name>.keys file is driven by it instead, inside tmux: one
# `sleep SECONDS`, `send TEXT` (`send \r` is Enter) or `wait TEXT` (until the screen shows TEXT,
# up to 2 min) per line.
set -euo pipefail
cd "$(dirname "$0")/../.."
AGG="${AGG:-$(command -v agg || echo "$HOME/.cargo/bin/agg")}"
CAST_DIR="${CAST_DIR:-$HOME/.cache/proteus-demo/casts}"
mkdir -p "$CAST_DIR"

# name  cols rows  seconds-before-q  idle-limit  font-size  fps-cap
DEMOS=(
  "loop      138 26  0   2  14  12"
  "view      140 38  9   2  12   8"
  "validate  144 21  0   2  13  12"
  "tes       124 24  0   3  14  12"
  "home      120 30  0   3  14  12"
)

record() {
  local name=$1 cols=$2 rows=$3 quit_after=$4 idle=$5 font=$6 fps=$7
  local cast="$CAST_DIR/${name}.cast" gif="docs/media/${name}.gif"
  echo "recording $name (${cols}x${rows})"
  local keys="docs/media/demos/${name}.keys"
  if [[ -f "$keys" ]]; then
    # A real terminal on both ends (the home screen checks), so asciinema runs inside a
    # detached tmux session and the keys go in with `tmux send-keys`.
    local session="proteus-rec-$$"
    tmux -L proteus-rec new-session -d -s "$session" -x "$cols" -y "$rows" \
      "asciinema rec --overwrite -c 'bash docs/media/demos/${name}.sh' '$cast'"
    while read -r cmd arg; do
      case "$cmd" in
        sleep) sleep "$arg" ;;
        wait) for _ in $(seq 1 240); do
                tmux -L proteus-rec capture-pane -t "$session" -p | grep -qF -- "$arg" && break
                sleep 0.5
              done ;;
        send) if [[ "$arg" == '\r' ]]; then tmux -L proteus-rec send-keys -t "$session" Enter
              else tmux -L proteus-rec send-keys -t "$session" -l "$arg"; fi ;;
      esac
    done <"$keys"
    while tmux -L proteus-rec has-session -t "$session" 2>/dev/null; do sleep 0.5; done
    # End on the full-screen app, not on the shell prompt it returns to: drop everything from
    # the event that leaves the alternate screen onwards.
    local cut
    cut=$(grep -n '\\u001b\[?1049l' "$cast" | tail -1 | cut -d: -f1)
    if [[ -n "$cut" ]]; then head -n $((cut - 1)) "$cast" >"$cast.tmp" && mv "$cast.tmp" "$cast"; fi
  elif [[ "$quit_after" -gt 0 ]]; then
    ( sleep "$quit_after"; printf 'q'; sleep 30 ) \
      | asciinema rec --overwrite --cols "$cols" --rows "$rows" \
          -c "bash docs/media/demos/${name}.sh" "$cast"
  else
    asciinema rec --overwrite --cols "$cols" --rows "$rows" \
      -c "bash docs/media/demos/${name}.sh" "$cast"
  fi
  "$AGG" --font-size "$font" --theme "$(<docs/brand/proteus-dark.agg)" \
      --text-font-family "Geist Mono,JetBrains Mono,DejaVu Sans Mono" \
      --fps-cap "$fps" --idle-time-limit "$idle" --last-frame-duration 3 "$cast" "$gif"
  echo "  -> $gif ($(du -h "$gif" | cut -f1))"
}

for spec in "${DEMOS[@]}"; do
  read -r name cols rows quit idle font fps <<<"$spec"
  [[ $# -gt 0 && "$1" != "$name" ]] && continue
  record "$name" "$cols" "$rows" "$quit" "$idle" "$font" "$fps"
done
