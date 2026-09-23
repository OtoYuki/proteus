# Shared helpers for the recorded demos.
# `type_out` writes a command a character at a time so the recording has motion; the real
# commands then run unpiped, so progress bars and spinners animate as they do in a terminal.
type_out() {
    # The s1re.sh prompt: a dim `~ $`.
    printf '\033[2m~ $\033[0m '
    local i
    for ((i = 0; i < ${#1}; i++)); do
        printf '%s' "${1:i:1}"
        sleep 0.012
    done
    printf '\n'
    sleep 0.4
}

# The binary under test: the release build in CARGO_TARGET_DIR (or ./target).
REPO="$PWD"
export PATH="${CARGO_TARGET_DIR:-$REPO/target}/release:$PATH"
proteus --version >/dev/null || { echo "build first: cargo build --release -p proteus-cli" >&2; exit 1; }

# A fresh home for the demo, so paths print as `~/…` and no real jobs show up.
# PROTEUS_DEMO_ROOT defaults to ~/.cache/proteus-demo.
demo_home() {
    export HOME="${PROTEUS_DEMO_ROOT:-$HOME/.cache/proteus-demo}/$1"
    rm -rf "$HOME"
    mkdir -p "$HOME"
    unset PROTEUS_DATA_DIR
}
