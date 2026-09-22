#!/usr/bin/env bash
# Re-vendor 3Dmol.js: scripts/vendor_3dmol.sh 2.5.5
set -euo pipefail
V="${1:?usage: vendor_3dmol.sh <version>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/crates/proteus-core/assets"
URL="https://cdn.jsdelivr.net/npm/3dmol@$V/build/3Dmol-min.js"
curl -fsSL "$URL" -o "$DEST/3Dmol-min.js"
curl -fsSL "https://raw.githubusercontent.com/3dmol/3Dmol.js/master/LICENSE" -o "$DEST/3Dmol-LICENSE.txt"
SHA=$(sha256sum "$DEST/3Dmol-min.js" | cut -d' ' -f1)
SIZE=$(( $(stat -c%s "$DEST/3Dmol-min.js") / 1024 ))
python3 - "$DEST/README.md" "$V" "$SHA" "$SIZE" <<'PY'
import re, sys
p, v, sha, size = sys.argv[1:5]
s = open(p).read()
s = re.sub(r'https://cdn\.jsdelivr\.net/npm/3dmol@[^/]+/', f'https://cdn.jsdelivr.net/npm/3dmol@{v}/', s)
s = re.sub(r'\| version \| [^|]+\|', f'| version | {v} |', s)
s = re.sub(r'\| sha256 \| `[^`]+` \|', f'| sha256 | `{sha}` |', s)
s = re.sub(r'\| size \| \d+ KB \|', f'| size | {size} KB |', s)
open(p, 'w').write(s)
PY
echo "vendored 3Dmol.js $V ($SIZE KB, sha256 $SHA)"
