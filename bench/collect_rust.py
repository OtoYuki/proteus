#!/usr/bin/env python3
"""Collect criterion medians from target/criterion into bench/results/rust_<host>.json."""
import json
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
crit = ROOT / "target" / "criterion"
out = {"host": platform.node(), "cpu": platform.processor() or platform.machine(),
       "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
       "results": {}}
for group in sorted(p for p in crit.iterdir() if p.is_dir() and p.name != "report"):
    for case in sorted(p for p in group.iterdir() if p.is_dir() and p.name != "report"):
        est = case / "new" / "estimates.json"
        if not est.exists():
            continue
        d = json.loads(est.read_text())
        out["results"].setdefault(case.name, {})[group.name] = d["median"]["point_estimate"] / 1e9  # seconds
dest = ROOT / "bench" / "results" / f"rust_{platform.node()}.json"
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
