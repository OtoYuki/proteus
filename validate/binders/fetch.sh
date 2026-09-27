#!/usr/bin/env bash
# Fetch the binder meta-analysis dataset (Overath et al. 2025, bioRxiv 2025.08.14.670059;
# Zenodo 10.5281/zenodo.15722219, CC-BY-4.0) into ~/.cache/proteus-validate/binders:
#   final_dataset.csv     one row per design: wet-lab `binder` label, AF3/Boltz/ColabFold
#                         ipSAE/LIS/ipAE, Rosetta interface metrics
#   af3/…/<name>_model.cif, <name>_confidences.json
#                         the AlphaFold 3 top model of every design and its PAE (~3.2 GB)
# Downloads are checked against Zenodo's md5. Nothing is written into the repository.
set -euo pipefail
DEST="${PROTEUS_BINDERS:-$HOME/.cache/proteus-validate/binders}"
mkdir -p "$DEST"
cd "$DEST"
fetch() { # fetch <file> <md5>
    if [[ -s "$1" ]] && [[ "$(md5sum "$1" | cut -d' ' -f1)" == "$2" ]]; then return; fi
    echo "downloading $1"
    curl -fsSL -o "$1.part" "https://zenodo.org/api/records/15722219/files/$1/content"
    [[ "$(md5sum "$1.part" | cut -d' ' -f1)" == "$2" ]] || { echo "md5 mismatch: $1" >&2; exit 1; }
    mv "$1.part" "$1"
}
fetch final_dataset.csv 3a69ee9b0fecf53924a8c6479bac146e
fetch AF3_outputs.tar.zstd 6430cbacd4ea54191e71583db5faa90d
if [[ ! -d af3/AF3_outputs ]]; then
    echo "extracting the top-ranked AF3 model and PAE of each design"
    mkdir -p af3
    tar --zstd -xf AF3_outputs.tar.zstd -C af3 --wildcards '*_model.cif' '*_confidences.json'
fi
echo "binders dataset in $DEST"
