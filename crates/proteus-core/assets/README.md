# Vendored browser assets

`3Dmol-min.js` is vendored so that `proteus view --html` and the daemon's `/view` page are
**self-contained**: they open with no network, which is the point on an HPC login node, an
airgapped cluster or a plane. A CDN `<script src>` would make the page useless exactly where
Proteus is meant to be used, and would pin a scientific artifact to a third party's uptime.

| | |
|---|---|
| source | https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js |
| version | 2.5.5 |
| sha256 | `f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016` |
| licence | BSD-3-Clause (`3Dmol-LICENSE.txt`) |
| size | 525 KB |

Refresh with `scripts/vendor_3dmol.sh <version>`, which rewrites this table and the checksum.
Verify what is checked in matches upstream:

```bash
curl -sL https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js | sha256sum
```

## Why 3Dmol.js and not Mol\*

Mol\* (molstar) is the field standard and what RCSB, PDBe and AlphaFold DB run — for interactive
analysis it is the better tool and Proteus does not try to replace it. But Proteus's web page
shows **one** structure with **one** representation and **one** colouring; it uses none of Mol\*'s
sequence panel, measurements, selections, volume maps or plugin system. Mol\* 5.11 is 4.9 MB
(1.4 MB gzipped), 3Dmol.js 2.5.5 is 525 KB (153 KB gzipped) — small enough to embed in the
binary, which is what buys the offline guarantee. The structure file itself is always on disk,
so anyone wanting Mol\*, PyMOL or ChimeraX opens it there.
