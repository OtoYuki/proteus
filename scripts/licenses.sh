#!/usr/bin/env bash
# The licence texts and notices that must travel with a proteus binary: ours, and those of the
# third-party data and fonts compiled into it. One list for every channel that ships a binary:
#
#   scripts/licenses.sh --list   print the files, one per line (CI runs this; Bioconda's
#                                about.license_file must name the same files)
#   scripts/licenses.sh DEST     copy them into DEST under their paths in this repository (the
#                                release archives' licenses/ and the container's
#                                /usr/share/doc/proteus)
#
# A NOTICE, *-OFL.txt or LICENSE-*.txt under crates/ that the list leaves out fails both forms,
# so an embedded asset cannot ship without its notice. Rust dependencies are not covered here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

files=(
    LICENSE-MIT
    LICENSE-APACHE
    crates/proteus-core/data/LICENSE-cctbx.txt
    crates/proteus-core/data/cdl/NOTICE
    crates/proteus-core/data/rama8000/NOTICE
    crates/proteus-core/data/rota8000/NOTICE
    crates/proteus-core/data/sc/NOTICE
    crates/proteus-render/assets/film/fonts/Geist-OFL.txt
    crates/proteus-render/assets/web/fonts/Figtree-OFL.txt
    crates/proteus-render/assets/web/fonts/Geist-OFL.txt
)

# Test fixtures and build output never reach a binary, so they are not searched.
unlisted=$(find crates \( -name tests -o -name target \) -prune -o -type f \
                \( -name NOTICE -o -name '*-OFL.txt' -o -name 'LICENSE-*.txt' \) -print |
           sort | while IFS= read -r f; do
               printf '%s\n' "${files[@]}" | grep -qxF -- "$f" || printf '%s\n' "$f"
           done)
if [[ -n "$unlisted" ]]; then
    printf 'scripts/licenses.sh: these notices are not in the list, so no binary would ship them:\n%s\n' \
        "$unlisted" >&2
    exit 1
fi

case "${1:-}" in
    --list)
        printf '%s\n' "${files[@]}"
        ;;
    "" | -*)
        echo "usage: scripts/licenses.sh --list | DEST" >&2
        exit 2
        ;;
    *)
        for f in "${files[@]}"; do
            mkdir -p "$1/$(dirname "$f")"
            cp "$f" "$1/$f"
        done
        ;;
esac
