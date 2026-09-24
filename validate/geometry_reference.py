#!/usr/bin/env python3
"""Generate validate/reference/geometry/<id>.json with cctbx (see geometry_ref.py).

Covers every corpus entry plus the committed predicted models in validate/predicted/. Needs
PROTEUS_CHEM_DATA (built by fetch_chem_data.sh); runs one cctbx process per structure.

Usage: geometry_reference.py [--full] [id ...]
"""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import tomllib

root = pathlib.Path(__file__).parent
out = root / "reference" / "geometry"


def targets():
    cfg = tomllib.loads((root / "corpus.toml").read_text())
    for s in cfg["structure"]:
        yield f"{s['id']}_{s['format']}", root / "corpus" / f"{s['id']}.{s['format']}", s["kind"]
    for p in sorted((root / "predicted" / "esmfold").glob("*.pdb")):
        yield f"esmfold_{p.stem}", p, "esmfold"


def run(item, full):
    name, path, kind = item
    args = [sys.executable, str(root / "geometry_ref.py"), str(path)] + (["--full"] if full else [])
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        return name, None, proc.stderr.strip().splitlines()[-1:] or ["no output"]
    data = json.loads(proc.stdout)
    data["kind"] = kind
    (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return name, data, None


def main():
    if not os.environ.get("PROTEUS_CHEM_DATA"):
        sys.exit("set PROTEUS_CHEM_DATA (see fetch_chem_data.sh)")
    full = "--full" in sys.argv[1:]
    only = {a for a in sys.argv[1:] if not a.startswith("--")}
    out.mkdir(parents=True, exist_ok=True)
    items = [t for t in targets() if not only or t[0] in only]
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for name, data, err in pool.map(lambda t: run(t, full), items):
            if err:
                failed += 1
                print(f"FAIL {name}: {err[0]}")
                continue
            r = data["restraints"]
            print(f"{name:22s} bonds {r['bond']['n']:6d}/{r['bond']['n_outliers']:<4d} "
                  f"angles {r['angle']['n']:6d}/{r['angle']['n_outliers']:<4d} "
                  f"cbeta {sum(v['outlier'] for v in data['cbetadev'].values()):3d} "
                  f"rota {len(data['rotalyze']):5d}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
