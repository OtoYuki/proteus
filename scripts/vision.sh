#!/usr/bin/env bash
# See the terminal UI as a user sees it: a real kitty (fonts, Nerd Font icons, kitty-graphics
# pictures) on an invisible Hyprland output, driven by keys and captured with grim. The PNGs are
# what a reviewer, human or model, looks at while changing the UI.
#
#   scripts/vision.sh up [WxH]          # headless output + kitty in this repo (default 1920x1080)
#   scripts/vision.sh run 'proteus'     # type a command line and press enter
#   scripts/vision.sh keys j j enter 2  # kitty key names: j, enter, esc, ctrl+e, alt+1, …
#   scripts/vision.sh type 'MKT…'       # literal text
#   scripts/vision.sh shot NAME [WAIT]  # wait (default 1 s), capture to $VISION_DIR/NAME.png
#   scripts/vision.sh down              # close kitty, remove the output
#
# Needs Hyprland 0.56+ (Lua config: `hyprctl eval`), kitty and grim. Nothing appears on the
# user's screens: the output sits far off at 20000x5000 on the workspace named `pvision`.
set -euo pipefail

OUT=PVISION
SOCK=/tmp/pvision.sock
WS=name:pvision
VISION_DIR="${VISION_DIR:-${TMPDIR:-/tmp}/proteus-vision}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$VISION_DIR"

k() { kitty @ --to "unix:$SOCK" "$@"; }

case "${1:-}" in
up)
    size="${2:-1920x1080}"
    # Rules first: the output then opens on the workspace, and kitty lands on it without
    # taking focus from the user's screens.
    hyprctl eval "hl.workspace_rule({ workspace = \"$WS\", monitor = \"$OUT\", default = true })" >/dev/null
    hyprctl eval "hl.window_rule({ name = \"pvision\", match = { class = \"^pvision\$\" }, workspace = \"$WS silent\" })" >/dev/null
    hyprctl monitors | grep -q "Monitor $OUT " || hyprctl output create headless "$OUT" >/dev/null
    hyprctl eval "hl.monitor({ output = \"$OUT\", mode = \"$size@30\", position = \"20000x5000\", scale = 1 })" >/dev/null
    if ! [ -S "$SOCK" ]; then
        hyprctl eval "hl.exec_cmd(\"kitty --class pvision --directory $ROOT -o allow_remote_control=yes --listen-on unix:$SOCK\")" >/dev/null
        for _ in $(seq 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
        sleep 0.5
    fi
    echo "vision: $OUT ($size), shots in $VISION_DIR"
    ;;
run)  k send-text "$2"$'\r' ;;
type) k send-text "$2" ;;
keys) shift; for key in "$@"; do k send-key "$key"; sleep 0.08; done ;;
shot)
    sleep "${3:-1}"
    grim -o "$OUT" "$VISION_DIR/$2.png"
    echo "$VISION_DIR/$2.png"
    ;;
down)
    k close-window 2>/dev/null || true
    rm -f "$SOCK"
    hyprctl output remove "$OUT" >/dev/null 2>&1 || true
    ;;
*) sed -n '2,15p' "$0"; exit 2 ;;
esac
