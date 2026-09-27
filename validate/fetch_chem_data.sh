#!/usr/bin/env bash
# Build the cctbx chemical data that validate/geometry_ref.py needs and the pip cctbx-base
# wheel does not ship: Phenix geostd (monomer restraints, BSD-3) and the Top8000 rotamer and
# Ramachandran contour grids (CC-BY-4.0). Only the machine that regenerates
# validate/reference/geometry/ needs it; CI compares against the committed references.
#
# Usage: validate/fetch_chem_data.sh [dir]        (default: ~/.cache/proteus-validate/chem_data)
# Then:  PROTEUS_CHEM_DATA=<dir> make geometry-reference
set -euo pipefail

DEST="${1:-$HOME/.cache/proteus-validate/chem_data}"
PY="$(dirname "$0")/.venv/bin/python"
# Pinned so that regenerated references are reproducible.
GEOSTD_COMMIT=4e1c1c0444d9ab9c4a1e01626011ea0f62fddb16
REFDATA_COMMIT=5ee6875fc29eccc3c9dc7cbe705ff9fd1b505d7d

mkdir -p "$DEST"
fetch_commit() { # repo-url commit dir [sparse-path...]
  local url=$1 commit=$2 dir=$3
  shift 3
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ]; then
    return
  fi
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  if [ $# -gt 0 ]; then
    git -C "$dir" sparse-checkout set --no-cone "$@"
  fi
  git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
}

echo "geostd @ ${GEOSTD_COMMIT:0:12}"
fetch_commit https://github.com/phenix-project/geostd "$GEOSTD_COMMIT" "$DEST/geostd"

echo "Top8000 grids @ ${REFDATA_COMMIT:0:12}"
fetch_commit https://github.com/rlabduke/reference_data "$REFDATA_COMMIT" "$DEST/refdata" \
  '/Top8000/Top8000_rotamer_pct_contour_grids/' '/Top8000/Top8000_ramachandran_pct_contour_grids/'
mkdir -p "$DEST/rotarama_data"
cp "$DEST"/refdata/Top8000/Top8000_rotamer_pct_contour_grids/*.data "$DEST/rotarama_data/"
cp "$DEST"/refdata/Top8000/Top8000_ramachandran_pct_contour_grids/*.data "$DEST/rotarama_data/"

echo "building rotarama pickle caches"
PROTEUS_CHEM_DATA="$DEST" "$PY" - <<'PYEOF'
import os
import libtbx.load_env  # noqa: F401
import libtbx
from libtbx.path import absolute_path, relocatable_path
libtbx.env.repository_paths.append(
    relocatable_path(absolute_path(os.environ["PROTEUS_CHEM_DATA"]), "."))
from mmtbx.command_line import rebuild_rotarama_cache
rebuild_rotarama_cache.run()
PYEOF
echo "done: export PROTEUS_CHEM_DATA=$DEST"
