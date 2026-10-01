#!/usr/bin/env python3
"""Compare ipsae.py before and after its 2026-01-03 d0 change, and test the meta-analysis
paper's stated multi-chain rule, over the AlphaFold 3 models of the binder dataset.

    validate/binders/ipsae_versions.py BINDERS_DIR [--report validate/binders/ipsae_versions.md]

BINDERS_DIR holds final_dataset.csv and ipsae_versions/ from validate/binders/ipsae_versions.sh.
The binder is chain A in every design (the dataset's `binder_chain`). Needs numpy and scipy.
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

OLD, NEW = "3480750", "6174cf9"


def read(path):
    """ipsae.py's chain-pair file -> {(chain1, chain2): {...}} for the asym rows."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 24 or p[4] != "asym":
            continue
        out[(p[0], p[1])] = {"ipsae": float(p[5]), "n0res": int(p[15]), "dist1": int(p[21]), "dist2": int(p[22])}
    return out


def ap(lab, s):
    o = np.argsort(-s, kind="stable")
    lab = lab[o]
    tp = np.cumsum(lab)
    return float((tp / np.arange(1, len(lab) + 1))[lab == 1].mean())


def main():
    args = sys.argv[1:]
    report = None
    if "--report" in args:
        k = args.index("--report")
        report = Path(args[k + 1])
        del args[k : k + 2]
    root = Path(args[0])
    rows = {r["binder_id"]: r for r in csv.DictReader(open(root / "final_dataset.csv")) if r["binder"] in ("True", "False")}
    lower = {k.lower(): k for k in rows}
    res = defaultdict(dict)
    for v in (OLD, NEW):
        for f in glob.glob(str(root / "ipsae_versions" / v / "*" / "m_model_10_10.txt")):
            n = os.path.basename(os.path.dirname(f))
            k = n if n in rows else lower.get(n.lower())
            if k is not None:
                res[k][v] = read(f)
    both = [k for k in res if OLD in res[k] and NEW in res[k]]
    two = [k for k in both if {c for p in res[k][NEW] for c in p} == {"A", "B"}]
    multi = [k for k in both if k not in set(two)]

    out = []
    w = out.append
    w("# ipsae.py before and after its d0 change, on the binder dataset\n")
    w(
        f"`ipsae.py` (DunbrackLab/IPSAE, no tagged releases) at {OLD} (2026-01-02) and {NEW} "
        "(2026-01-03, current), PAE and distance cutoffs 10 Å, over the AlphaFold 3 model of each "
        "design in Overath et al.'s dataset. Written by `validate/binders/ipsae_versions.py`; run "
        f"with `make validate-ipsae-versions`. {len(both)} of {len(rows)} labelled designs scored "
        f"by both versions: {len(two)} with a single-chain target, {len(multi)} with several.\n"
    )
    w(
        f"{NEW} changed d0 for small interfaces: the vectorised `calc_d0_array` (ipSAE proper) now "
        "floors the residue count at 26 instead of 27, and the scalar `calc_d0` (d0chn, d0dom) "
        "returns 1.0 for counts up to 27. At a count of exactly 27 the two functions now disagree "
        "(1.0 against 1.0389).\n"
    )

    mn = lambda t: min(t[("A", "B")]["ipsae"], t[("B", "A")]["ipsae"])
    old = np.array([mn(res[k][OLD]) for k in two])
    new = np.array([mn(res[k][NEW]) for k in two])
    ds = np.array([float(rows[k]["af3_ipSAE_min"]) for k in two])
    lab = np.array([rows[k]["binder"] == "True" for k in two], float)
    by_t = defaultdict(list)
    for i, k in enumerate(two):
        by_t[rows[k]["target_id"]].append(i)
    per_target = lambda s: np.mean([ap(lab[ix], s[ix]) for ix in map(np.array, by_t.values()) if 0 < lab[ix].sum() < len(ix)])

    w("## 1. What the change does to ipSAE_min (single-chain targets)\n")
    moved = np.abs(new - old) > 1e-9
    dd = (new - old)[moved]
    w("| quantity | value |")
    w("|---|---|")
    w(f"| designs whose ipSAE_min changed | {moved.sum()} of {len(two)} ({moved.mean():.1%}) |")
    w(f"| raised / lowered | {(new > old + 1e-9).sum()} / {(new < old - 1e-9).sum()} |")
    w(f"| change on those, mean / largest | {dd.mean():.5f} / {dd.min():.5f} |")
    w(f"| Spearman ρ / Kendall τ, old vs new | {spearmanr(old, new).correlation:.6f} / {kendalltau(old, new).correlation:.6f} |")
    w(f"| AP per target, old / new / dataset | {per_target(old):.4f} / {per_target(new):.4f} / {per_target(ds):.4f} |")
    ko, kn = old > 0.61, new > 0.61
    w(f"| kept by `ipSAE_min > 0.61`, old / new | {ko.sum()} ({int(lab[ko].sum())} bound) / {kn.sum()} ({int(lab[kn].sum())} bound) |")
    w(f"| designs crossing 0.61 | {int((ko != kn).sum())} |")
    w("")
    w(
        "AP per target here is over single-chain targets only, so it differs from last_run.md, "
        "which also scores the pMHC designs.\n"
    )

    w("## 2. Which version made the dataset\n")
    nz = (old > 0) & (new > 0)
    w("| version | designs | median \\|Δ\\| | p99 | within the dataset's 3-decimal rounding (5e-4) |")
    w("|---|---|---|---|---|")
    for name, s in ((OLD, old), (NEW, new)):
        d = np.abs(s[nz] - ds[nz])
        w(f"| {name} | {nz.sum()} | {np.median(d):.5f} | {np.percentile(d, 99):.5f} | {(d <= 5e-4).mean():.1%} |")
    w("")
    zero = new == 0
    dz = ds[zero]
    w(
        f"Left out above: the {zero.sum()} designs where one direction has no PAE under 10 Å, which "
        f"both versions score 0 ({int((old[zero] == 0).sum())} of {zero.sum()} under {OLD} too). The dataset "
        f"holds 0 for {int((dz == 0).sum())}; the other {int((dz > 0).sum())} hold values up to "
        f"{dz.max():.3f} that neither direction of the model reproduces.\n"
    )

    w("## 3. The paper's multi-chain rule\n")
    w(
        "The paper's Methods: \"If the target had several subchains we took the average of the min "
        "and max values across both directions of binder → target comparisons, but only if there are "
        "interacting residues between the binder chain and a given target subchain.\" Read as: for "
        "each target subchain with residues within the distance cutoff of the binder (`ipsae.py`'s "
        "dist1 + dist2 > 0), take the min of its two directions; average those.\n"
    )
    w("| version | designs | median \\|Δ\\| vs dataset | within 5e-4 |")
    w("|---|---|---|---|")
    for v in (OLD, NEW):
        d = []
        for k in multi:
            t = res[k][v]
            vals = [
                min(t[("A", c)]["ipsae"], t[(c, "A")]["ipsae"])
                for c in sorted({c for p in t for c in p} - {"A"})
                if ("A", c) in t and (c, "A") in t and t[("A", c)]["dist1"] + t[("A", c)]["dist2"] > 0
            ]
            d.append(abs((np.mean(vals) if vals else 0.0) - float(rows[k]["af3_ipSAE_min"])))
        d = np.array(d)
        w(f"| {v} | {len(d)} | {np.median(d):.5f} | {(d <= 5e-4).sum()} of {len(d)} |")
    w("")
    text = "\n".join(out) + "\n"
    print(text)
    if report:
        report.write_text(text)


if __name__ == "__main__":
    main()
