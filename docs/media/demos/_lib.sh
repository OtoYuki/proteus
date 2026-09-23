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
