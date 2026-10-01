#!/usr/bin/env bash
# Run DunbrackLab's ipsae.py at two commits over every AlphaFold 3 model of the binder dataset
# (validate/binders/fetch.sh first): 3480750 (2026-01-02, the last before the d0 floor moved from
# 27 to 26) and 6174cf9 (2026-01-03, current). Each design runs in its own directory under
# $PROTEUS_BINDERS/ipsae_versions/<version>/<design>/, because ipsae.py writes beside its input.
# Then validate/binders/ipsae_versions.py compares the two and the dataset.
set -euo pipefail
DEST="${PROTEUS_BINDERS:-$HOME/.cache/proteus-validate/binders}"
PY="${PY:-$(pwd)/validate/.venv/bin/python}"
OUT="$DEST/ipsae_versions"
mkdir -p "$OUT"
if [[ ! -d "$OUT/IPSAE" ]]; then
    git clone -q https://github.com/DunbrackLab/IPSAE "$OUT/IPSAE"
fi
git -C "$OUT/IPSAE" show 3480750:ipsae.py > "$OUT/ipsae_3480750.py"
git -C "$OUT/IPSAE" show 6174cf9:ipsae.py > "$OUT/ipsae_6174cf9.py"
rm -f "$OUT/failed.txt"

export OUT PY
one() {
    d="$1"; n=$(basename "$d")
    # AF3 lower-cases the files but not the directory: glob, never build names from $n.
    cif=$(ls "$d"/*_model.cif 2>/dev/null | head -1)
    js=$(ls "$d"/*_confidences.json 2>/dev/null | grep -v summary | head -1)
    if [[ -z "$cif" || -z "$js" ]]; then echo "missing $n" >> "$OUT/failed.txt"; return; fi
    for v in 3480750 6174cf9; do
        w="$OUT/$v/$n"; mkdir -p "$w"
        ln -sf "$cif" "$w/m_model.cif"; ln -sf "$js" "$w/m_confidences.json"
        (cd "$w" && "$PY" "$OUT/ipsae_$v.py" m_confidences.json m_model.cif 10 10 > /dev/null 2> err.txt) \
            || echo "failed $v $n" >> "$OUT/failed.txt"
        [[ -s "$w/m_model_10_10.txt" ]] || echo "no output $v $n" >> "$OUT/failed.txt"
    done
}
export -f one
# One numpy thread per process, or 12 processes oversubscribe every core.
find "$DEST/af3/AF3_outputs" -mindepth 1 -maxdepth 1 -type d -print0 |
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 xargs -0 -P "${JOBS:-12}" -I{} bash -c 'one "$1"' _ {}
if [[ -s "$OUT/failed.txt" ]]; then
    echo "$(wc -l < "$OUT/failed.txt") failures, see $OUT/failed.txt" >&2
    exit 1
fi
