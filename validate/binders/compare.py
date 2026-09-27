#!/usr/bin/env python3
"""Compare `proteus analyze --interface` on the binder meta-analysis dataset with the dataset's
own values and with the wet-lab outcome.

    validate/binders/compare.py ANALYSIS.jsonl FINAL_DATASET.csv [--report validate/binders/last_run.md]

ANALYSIS.jsonl is `proteus analyze <af3 dir> --interface A --json` over the AlphaFold 3 top model
of every design (validate/binders/fetch.sh). Three sections:

1. Parity with the dataset's PAE metrics on single-chain targets, where the definitions are
   unambiguous. Gated by validate/binders/tolerances.toml.
2. Agreement with the dataset's Rosetta interface metrics (Sc, dSASA, interface H-bonds, interface
   residues). The authors adapted BindCraft's Rosetta scoring; the exact protocol (whether models
   were relaxed first) is not published, so this is correlation, gated only as a regression floor.
3. How well each metric separates the 394 designs that bound in the lab from the rest: average
   precision (AP) and AUROC. Reported, not gated: it is a property of the metric, not of the code.

Standard library only. Exit status 1 when a gate fails.
"""

import csv
import json
import math
import os
import statistics as st
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pearson(pairs):
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    ma, mb = st.fmean(a), st.fmean(b)
    sab = sum((x - ma) * (y - mb) for x, y in pairs)
    return sab / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(pairs):
    ra = ranks([x for x, _ in pairs])
    rb = ranks([y for _, y in pairs])
    return pearson(list(zip(ra, rb)))


def average_precision(scores, labels):
    """sklearn's average_precision_score: sum over thresholds of (R_n - R_{n-1}) P_n."""
    pos = sum(labels)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ap, tp, fp, prev_recall = 0.0, 0, 0, 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            tp += labels[order[j]]
            fp += 1 - labels[order[j]]
            j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j
    return ap


def auroc(scores, labels):
    r = ranks(scores)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    s = sum(ri for ri, l in zip(r, labels) if l)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def main():
    args = sys.argv[1:]
    report = None
    if "--report" in args:
        k = args.index("--report")
        report = Path(args[k + 1])
        del args[k : k + 2]
    analysis, dataset = Path(args[0]), Path(args[1])
    tol = tomllib.loads((HERE / "tolerances.toml").read_text())

    ref = {r["binder_id"]: r for r in csv.DictReader(open(dataset))}
    rows = [json.loads(l) for l in open(analysis)]
    # The design id is the directory AlphaFold 3 wrote the model into (file names are lower-cased).
    for o in rows:
        o["_id"] = os.path.basename(os.path.dirname(o["file"]))
    rows = [o for o in rows if o["_id"] in ref]
    single = [o for o in rows if ref[o["_id"]]["target_chains"] == '["B"]']

    out = []
    failed = []
    w = out.append
    w("# Binder triage vs the Overath et al. 2025 meta-analysis\n")
    w(
        "Dataset: Zenodo 10.5281/zenodo.15722219 (CC-BY-4.0), `final_dataset.csv` and the "
        "AlphaFold 3 top model of each design. Proteus: `proteus analyze <af3 dir> --interface A`, "
        "binder = chain A against every other chain. Written by `validate/binders/compare.py`.\n"
    )
    w(f"{len(rows)} designs matched ({len(single)} with a single-chain target).\n")

    # 1. Parity -------------------------------------------------------------------------------
    w("## 1. PAE metrics against the dataset's own values (single-chain targets)\n")
    w_later = []
    w("| ours | dataset | n | median \\|Δ\\| | p99 \\|Δ\\| | max \\|Δ\\| | gate: p99 ≤ | |")
    w("|---|---|---|---|---|---|---|---|")
    for name, spec in tol["parity"].items():
        pool = single
        if spec.get("only_positive"):
            pool = [o for o in single if (o.get(spec["ours"]) or 0) > 0]
            excluded = len(single) - len(pool)
            w_later.append(
                f"- `{spec['ours']}` parity leaves out the {excluded} designs where it is 0 "
                f"(see below); their largest `ipsae_max` is "
                f"{max((o['ipsae_max'] or 0) for o in single if not (o.get(spec['ours']) or 0) > 0):.3f}."
            )
        pairs = [
            (o[spec["ours"]], num(ref[o["_id"]][spec["dataset"]]))
            for o in pool
            if o.get(spec["ours"]) is not None and num(ref[o["_id"]][spec["dataset"]]) is not None
        ]
        d = sorted(abs(x - y) for x, y in pairs)
        med, p99, mx = d[len(d) // 2], d[min(len(d) - 1, int(len(d) * 0.99))], d[-1]
        ok = p99 <= spec["p99_abs"]
        if not ok:
            failed.append(f"parity {name}: p99 |Δ| {p99:.4f} > {spec['p99_abs']}")
        w(
            f"| `{spec['ours']}` | `{spec['dataset']}` | {len(pairs)} | {med:.4f} | {p99:.4f} | "
            f"{mx:.4f} | {spec['p99_abs']} | {'✓' if ok else '✗'} |"
        )
    w("")
    for note in w_later:
        w(note)
    for note in tol["notes"]["parity"]:
        w(f"- {note}")
    w("")

    # 2. Agreement with Rosetta ---------------------------------------------------------------
    w("## 2. Structure-based interface metrics against the dataset's Rosetta values (AF3 models)\n")
    w("| ours | dataset (Rosetta) | n | Pearson r | Spearman ρ | mean Δ | gate: r ≥ | |")
    w("|---|---|---|---|---|---|---|---|")
    for name, spec in tol["agreement"].items():
        pairs = [
            (o[spec["ours"]], num(ref[o["_id"]][spec["dataset"]]))
            for o in rows
            if o.get(spec["ours"]) is not None and num(ref[o["_id"]][spec["dataset"]]) is not None
        ]
        r = pearson(pairs)
        ok = r >= spec["min_pearson"]
        if not ok:
            failed.append(f"agreement {name}: r {r:.3f} < {spec['min_pearson']}")
        w(
            f"| `{spec['ours']}` | `{spec['dataset']}` | {len(pairs)} | {r:.3f} | "
            f"{spearman(pairs):.3f} | {st.fmean(x - y for x, y in pairs):+.2f} | "
            f"{spec['min_pearson']} | {'✓' if ok else '✗'} |"
        )
    w("")
    for note in tol["notes"]["agreement"]:
        w(f"- {note}")
    w("")

    # 3. Wet-lab outcome ----------------------------------------------------------------------
    labelled = [o for o in rows if ref[o["_id"]]["binder"] in ("True", "False")]
    labels = [1 if ref[o["_id"]]["binder"] == "True" else 0 for o in labelled]
    prevalence = sum(labels) / len(labels)
    w("## 3. Separating designs that bound in the lab from those that did not\n")
    w(
        f"{len(labelled)} designs with an outcome, {sum(labels)} binders "
        f"(prevalence {prevalence:.3f}, the AP of a random ranking). Higher AP and AUROC are "
        "better; each metric is oriented so that larger means more likely to bind. Missing "
        "values rank last. Pooled over all 15 targets.\n"
    )
    w("| metric | source | AP | AUROC |")
    w("|---|---|---|---|")
    metrics = [
        ("ipSAE_min", "proteus", lambda o, x: o["ipsae_min"]),
        ("ipSAE_min", "dataset (AF3)", lambda o, x: num(x["af3_ipSAE_min"])),
        ("ipSAE_max", "proteus", lambda o, x: o["ipsae_max"]),
        ("ipTM", "proteus (from AF3's file)", lambda o, x: o["iptm"]),
        ("−ipAE", "proteus", lambda o, x: None if o["ipae"] is None else -o["ipae"]),
        ("LIS", "proteus", lambda o, x: o["lis"]),
        ("pLDDT (mean)", "proteus", lambda o, x: o["plddt_mean"]),
        ("Sc", "proteus", lambda o, x: o["interface_sc"]),
        ("Sc", "dataset (Rosetta)", lambda o, x: num(x["af3_rosetta_interface_sc"])),
        ("dSASA", "proteus", lambda o, x: o["interface_dsasa"]),
        ("interface H-bonds", "proteus", lambda o, x: o["interface_hbonds"]),
        (
            "LIS × Sc",
            "proteus",
            lambda o, x: None
            if o["lis"] is None or o["interface_sc"] is None
            else o["lis"] * o["interface_sc"],
        ),
    ]
    for name, source, f in metrics:
        s = [f(o, ref[o["_id"]]) for o in labelled]
        s = [v if v is not None else -1e18 for v in s]
        w(f"| {name} | {source} | {average_precision(s, labels):.3f} | {auroc(s, labels):.3f} |")
    w("")
    thr = tol["outcome"]["ipsae_min_threshold"]
    sel = [l for o, l in zip(labelled, labels) if (o["ipsae_min"] or 0) > thr]
    w(
        f"Filter `ipsae_min > {thr}` (the paper's single-metric threshold): keeps {len(sel)} "
        f"designs, of which {sum(sel)} bound (precision {sum(sel) / max(1, len(sel)):.3f}, "
        f"recall {sum(sel) / sum(labels):.3f}).\n"
    )

    w("## Result\n")
    w("All gates pass." if not failed else "**Failed:**\n\n" + "\n".join(f"- {f}" for f in failed))
    text = "\n".join(out) + "\n"
    print(text)
    if report:
        report.write_text(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
