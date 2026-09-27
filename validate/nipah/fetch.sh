#!/usr/bin/env bash
# Fetch the Adaptyv Nipah binder competition results (ProteinBase collection
# `nipah-binder-competition-results`, ODC-By) into ~/.cache/proteus-validate/nipah:
#   collection.csv        one row per design; `evaluations` holds the lab results (`binding`,
#                         one entry per measurement) and the Boltz-2 model and PAE file locations
#   models/<id>/<id>-model_v1.cif, <id>-predicted_aligned_error_v1.json
#                         the Boltz-2 complex and its full PAE, named the way AlphaFold DB names
#                         a model and its PAE so `proteus analyze` pairs them (~11 GB)
# ProteinBase publishes no checksums; compare.py records the table's sha256 in its report.
# Nothing is written into the repository.
set -euo pipefail
DEST="${PROTEUS_NIPAH:-$HOME/.cache/proteus-validate/nipah}"
mkdir -p "$DEST/models"
cd "$DEST"
if [[ ! -s collection.csv ]]; then
    echo "downloading the collection table"
    curl -fsSL -o collection.csv.part \
        'https://proteinbase.com/api/proteins/download?collectionId=019be357-ae36-ec95-4bc6-9db0046b0600&slug=nipah-binder-competition-results'
    mv collection.csv.part collection.csv
fi
python3 - <<'EOF' > files.txt
import csv, json
for r in csv.DictReader(open("collection.csv", encoding="utf-8-sig")):
    ev = {e["metric"]: e["value"] for e in json.loads(r["evaluations"])}
    cif = (ev.get("boltz2_structure_prediction") or {}).get("url")
    pae = ((ev.get("pae_file") or {}).get("file") or {}).get("url")
    if not cif or not pae:
        continue
    pae = pae.replace("s3://proteinbase-pub/", "https://proteinbase-pub.t3.storage.dev/")
    d = f"models/{r['id']}"
    print(f"{cif} {d}/{r['id']}-model_v1.cif")
    print(f"{pae} {d}/{r['id']}-predicted_aligned_error_v1.json")
EOF
echo "$(($(wc -l < files.txt) / 2)) designs with a Boltz-2 model and PAE"
# shellcheck disable=SC2016
xargs -P 8 -n 2 sh -c '[ -s "$2" ] || { mkdir -p "$(dirname "$2")" && curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }' _ < files.txt
echo "nipah dataset in $DEST"
