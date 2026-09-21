#!/usr/bin/env python3
"""Convert cctbx mmtbx/validation/ramachandran/rama8000_tables.h into six 180x180 LE-f32 grids.

Usage: python scripts/convert_rama8000.py path/to/rama8000_tables.h
Source: https://github.com/cctbx/cctbx_project (BSD-3, LBNL). Data: Top8000 (Richardson lab, Duke).
"""
import pathlib
import re
import struct
import sys

NAMES = {
    "general": "general",
    "glycine": "glycine",
    "cis_pro": "cispro",
    "trans_pro": "transpro",
    "pre_pro": "prepro",
    "ile_val": "ileval",
}
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(__file__).resolve().parents[1] / "crates/proteus-core/data/rama8000"
out.mkdir(parents=True, exist_ok=True)
for key, fname in NAMES.items():
    m = re.search(r"linear_table_%s\[\] = \{([^}]*)\}" % key, src, re.S)
    vals = [float(v) for v in m.group(1).replace("\n", "").split(",") if v.strip()]
    assert len(vals) == 180 * 180, (key, len(vals))
    (out / f"{fname}.f32").write_bytes(struct.pack("<%df" % len(vals), *vals))
    print(fname, len(vals), "min", min(vals), "max", max(vals))
