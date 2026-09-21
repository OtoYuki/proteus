#!/usr/bin/env python3
"""Evaluate Proteus's ESM-2 scorer against ProteinGym DMS assays (Spearman rho).

Downloads the reference table and the substitution CSVs for a few small single-mutant assays,
runs `proteus esm scan`, and correlates the predicted scores with the measured fitness.

Usage: bench/proteingym.py [--assays N] [--model ID] [--masked]
Outputs bench/results/proteingym_<host>.json and prints a table.
"""
import argparse
import csv
import json
import pathlib
import platform
import subprocess
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "bench" / ".proteingym"
REF_URL = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
ZIP_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_ProteinGym_substitutions.zip"


def reference_rows():
    CACHE.mkdir(parents=True, exist_ok=True)
    ref = CACHE / "DMS_substitutions.csv"
    if not ref.exists():
        urllib.request.urlretrieve(REF_URL, ref)
    with ref.open() as fh:
        return list(csv.DictReader(fh))


def pick(rows, n, max_len):
    """Smallest single-mutant assays, so the benchmark runs on a laptop CPU."""
    single = [
        r for r in rows
        if int(r["seq_len"]) <= max_len
        and r["includes_multiple_mutants"] in ("False", "FALSE", "0")
    ]
    single.sort(key=lambda r: int(r["seq_len"]))
    return single[:n]


def assay_csv(dms_id, filename):
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / filename
    if local.exists():
        return local
    archive = CACHE / "substitutions.zip"
    if not archive.exists():
        print(f"downloading the ProteinGym substitution archive (~41 MB) -> {archive}")
        urllib.request.urlretrieve(ZIP_URL, archive)
    with zipfile.ZipFile(archive) as z:
        member = next((m for m in z.namelist() if m.endswith(filename)), None)
        if member is None:
            raise SystemExit(f"{filename} not found in the archive")
        with z.open(member) as src, local.open("wb") as dst:
            dst.write(src.read())
    return local


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def scan_scores(binary, seq, model, masked):
    out = CACHE / "scan.csv"
    cmd = [binary, "esm", "scan", seq, "--export", str(out), "--esm-model", model, "--top", "1"]
    if masked:
        cmd.append("--esm-masked")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    table = {}
    with out.open() as fh:
        for row in csv.DictReader(fh):
            pos = int(row["position"])
            for aa, v in row.items():
                if len(aa) == 1 and aa.isalpha():
                    table[(pos, aa)] = float(v)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--binary", default=str(ROOT / "target/release/proteus"))
    args = ap.parse_args()

    results = []
    for r in pick(reference_rows(), args.assays, args.max_len):
        path = assay_csv(r["DMS_id"], r["DMS_filename"])
        with path.open() as fh:
            rows = [x for x in csv.DictReader(fh) if ":" not in x["mutant"]]
        table = scan_scores(args.binary, r["target_seq"], args.model, args.masked)
        pred, meas = [], []
        for x in rows:
            m = x["mutant"]
            pos, mt = int(m[1:-1]), m[-1]
            if (pos, mt) in table:
                pred.append(table[(pos, mt)])
                meas.append(float(x["DMS_score"]))
        rho = spearman(pred, meas)
        results.append({
            "assay": r["DMS_id"],
            "seq_len": int(r["seq_len"]),
            "n_scored": len(pred),
            "spearman": rho,
        })
        print(f"{r['DMS_id']:42s} len={r['seq_len']:>4} n={len(pred):>5} rho={rho:+.3f}")

    if results:
        mean = sum(abs(x["spearman"]) for x in results) / len(results)
        print(f"{'mean |rho|':42s} {mean:+.3f}")
        dest = ROOT / "bench" / "results" / f"proteingym_{platform.node()}.json"
        dest.write_text(json.dumps({
            "model": args.model,
            "marginals": "masked" if args.masked else "wild-type",
            "host": platform.node(),
            "mean_abs_spearman": mean,
            "assays": results,
        }, indent=1))
        print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
