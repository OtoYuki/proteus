#!/usr/bin/env python3
"""Convert the Top8000 rotamer contour grids (rota8000-*.data) into compact binary tables.

Usage: python scripts/convert_rota8000.py <rotarama_data dir> [<mmtbx/rotamer dir>]

Reads the 17 grids cctbx's `mmtbx/rotamer/rotamer_eval.py` actually loads (`aminoAcids`: Phe and
Tyr share `phetyr`; `leu` is the pruned table, not `leu-raw`; `pro` is the 1-D chi1 table, not
`pro3d`), fills each dense n-dimensional grid exactly as `NDimTable.createFromText` does
(`mmtbx/rotamer/n_dim_table.py`: bin = floor((x - min) / width) capped at nBins - 1, value stored
as C float, unlisted bins 0), and writes `crates/proteus-core/data/rota8000/<table>.bin`:

    zlib( b"ROT8" u8 version=1 u8 n_dim
          n_dim x (f64 min, f64 max, u32 n_bins, u8 wrap)
          u32 n_values, n_values x f32 (distinct grid values, ascending)
          N x u16 index into the values, row-major (last dimension fastest),
                stored as all low bytes then all high bytes )

All integers and floats little-endian. The value dictionary is lossless: every grid cell decodes
to the identical f32 cctbx holds in its `flex.float` lookup table.

With the optional second argument, `rotamer_names.props` (cctbx, BSD-3) is copied next to the
tables verbatim; it defines the rotamer-name boxes used by `RotamerID.identify`.

Source: https://github.com/rlabduke/reference_data, Top8000/Top8000_rotamer_pct_contour_grids
(CC-BY-4.0, Richardson laboratory, Duke University).
"""
import math
import pathlib
import re
import shutil
import struct
import sys
import zlib

# rotamer_eval.aminoAcids values (file stems), deduplicated.
TABLES = [
    "arg", "asn", "asp", "cys", "gln", "glu", "his", "ile", "leu", "lys", "met", "phetyr",
    "pro", "ser", "thr", "trp", "val",
]


def read_table(path):
    with open(path) as f:
        name = re.search(r': +"(.+)"$', f.readline().rstrip("\n")).group(1)
        n_dim = int(re.search(r": +(\d+)$", f.readline().rstrip("\n")).group(1))
        f.readline()
        dims = []
        data_re = re.compile(r": +([^ ]+) +([^ ]+) +([^ ]+) +([^ ]+)$")
        for _ in range(n_dim):
            m = data_re.search(f.readline().rstrip("\n"))
            lo, hi, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
            wrap = m.group(4).lower().strip() in ("true", "yes", "on", "1")
            dims.append((lo, hi, n, wrap))
        value_first = "first" in f.readline()
        size = math.prod(d[2] for d in dims)
        grid = [0.0] * size
        for line in f:
            fields = line.split()
            if len(fields) <= n_dim:
                continue
            if value_first:
                val, coords = float(fields[0]), [float(x) for x in fields[1 : n_dim + 1]]
            else:
                val, coords = float(fields[n_dim]), [float(x) for x in fields[:n_dim]]
            idx = 0
            for x, (lo, hi, n, _) in zip(coords, dims):
                b = int(min(math.floor((x - lo) / ((hi - lo) / n)), n - 1))
                assert 0 <= b < n, (path, line)
                idx = idx * n + b
            # flex.float: the double is narrowed to a C float.
            grid[idx] = struct.unpack("<f", struct.pack("<f", val))[0]
    return name, dims, grid


def encode(dims, grid):
    values = sorted(set(grid))
    assert len(values) <= 0xFFFF
    index = {v: i for i, v in enumerate(values)}
    out = bytearray(b"ROT8")
    out += struct.pack("<BB", 1, len(dims))
    for lo, hi, n, wrap in dims:
        out += struct.pack("<ddIB", lo, hi, n, wrap)
    out += struct.pack("<I", len(values))
    out += struct.pack("<%df" % len(values), *values)
    idx = [index[v] for v in grid]
    out += bytes(i & 0xFF for i in idx)
    out += bytes(i >> 8 for i in idx)
    return zlib.compress(bytes(out), 9)


def main():
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(__file__).resolve().parents[1] / "crates/proteus-core/data/rota8000"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem in TABLES:
        name, dims, grid = read_table(src / f"rota8000-{stem}.data")
        blob = encode(dims, grid)
        (out / f"{stem}.bin").write_bytes(blob)
        total += len(blob)
        nonzero = sum(1 for v in grid if v != 0.0)
        print(f"{stem:7} {name!r:40} dims={[d[2] for d in dims]} nonzero={nonzero} bytes={len(blob)}")
    if len(sys.argv) > 2:
        shutil.copyfile(pathlib.Path(sys.argv[2]) / "rotamer_names.props", out / "rotamer_names.props")
    print("total", total)


if __name__ == "__main__":
    main()
