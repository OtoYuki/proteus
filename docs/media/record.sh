#!/usr/bin/env bash
# Re-record docs/media/demo.gif. Needs: cargo build --release, asciinema (uv tool install asciinema),
# agg (cargo install --git https://github.com/asciinema/agg). The interactive viewer receives 'q'
# after 15 s through asciinema's stdin.
set -euo pipefail
cd "$(dirname "$0")/../.."
( sleep 15; printf 'q'; sleep 45 ) | asciinema rec --overwrite --cols 140 --rows 38 -c "bash docs/media/demo.sh" docs/media/demo.cast
agg --font-size 14 --theme monokai --fps-cap 15 --idle-time-limit 3 --last-frame-duration 4 docs/media/demo.cast docs/media/demo.gif
