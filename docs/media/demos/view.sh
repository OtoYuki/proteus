#!/usr/bin/env bash
# The terminal viewer with the live dashboard. Recorded by docs/media/record.sh view.
source "$(dirname "$0")/_lib.sh"
export RUST_LOG=off
demo_home view
cp "$REPO/validate/corpus/1pgb.pdb" "$HOME/"
cd "$HOME"
clear
type_out "proteus view 1pgb.pdb --interactive --dashboard"
proteus view 1pgb.pdb --interactive --dashboard
# The TUI restores the terminal on quit, so leave a still behind: otherwise the GIF's last
# frame — the one a reader sees longest — is an empty screen.
clear
type_out "proteus view 1pgb.pdb --backend halfblock"
proteus view 1pgb.pdb --backend halfblock --width 132 --height 32
sleep 4
