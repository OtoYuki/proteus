#!/usr/bin/env python3
"""Download the validation corpus into validate/corpus/ (idempotent, sha256-verified when recorded)."""
import hashlib
import pathlib
import sys
import tomllib
import urllib.request

root = pathlib.Path(__file__).parent
cfg = tomllib.loads((root / "corpus.toml").read_text())
dst = root / "corpus"
dst.mkdir(exist_ok=True)
fail = 0
for s in cfg["structure"]:
    p = dst / f"{s['id']}.{s['format']}"
    if not p.exists():
        try:
            urllib.request.urlretrieve(s["url"], p)
        except Exception as e:  # noqa: BLE001
            print("FAIL", s["id"], e)
            fail += 1
            continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if "sha256" in s and s["sha256"] != h:
        print("SHA MISMATCH", s["id"], h)
        fail += 1
    print(f"{s['id']:12s} {s['format']:3s} {p.stat().st_size:9d} {h[:12]}")
sys.exit(1 if fail else 0)
