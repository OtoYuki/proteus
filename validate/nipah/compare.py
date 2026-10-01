#!/usr/bin/env python3
"""Rank the Adaptyv Nipah binder competition designs with `proteus analyze --interface B` and
score the ranking against the lab outcome.

    validate/nipah/compare.py ANALYSIS.jsonl COLLECTION.csv [--report validate/nipah/last_run.md]

ANALYSIS.jsonl is `proteus analyze <models dir> --interface B --json` over the Boltz-2 complex
of every design (validate/nipah/fetch.sh); chain B is the binder, chain A the Nipah G head.
COLLECTION.csv is ProteinBase's `nipah-binder-competition-results` table (ODC-By). This is the
second, independent check of the triage metrics after validate/binders (Overath et al. 2025):
a different predictor (Boltz-2, not AlphaFold 3), a different lab and assay, one target.

Two sections:
1. Agreement with ProteinBase's own Boltz-2 ipSAE, gated as a regression floor (they used
   different cutoffs, so values differ; see the section's text).
2. Average precision and AUROC of each metric against the lab label, reported, not gated.

A design counts as a binder when any of its binding measurements was positive. Adaptyv's own
positive controls (author `adaptyv-bio`) are left out: they are not designs.

Standard library only. Exit status 1 when a gate fails.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "binders"))
from compare import auroc, average_precision, num, pearson, spearman  # noqa: E402

# A regression floor on reading the files (chain order, PAE orientation), not parity: ProteinBase
# scored with 15 Å cutoffs, Proteus with 10 Å, which reorders the near-zero designs (so Spearman
# is a poor gate) but barely moves confident ones.
MIN_PEARSON = 0.98


def main():
    args = sys.argv[1:]
    report = None
    if "--report" in args:
        k = args.index("--report")
        report = Path(args[k + 1])
        del args[k : k + 2]
    analysis, collection = Path(args[0]), Path(args[1])

    ref = {}
    controls = 0
    for r in csv.DictReader(open(collection, encoding="utf-8-sig")):
        if r["author"] == "adaptyv-bio":
            controls += 1
            continue
        ev = json.loads(r["evaluations"])
        binding = [e["value"] for e in ev if e["metric"] == "binding"]
        scores = {e["metric"]: e["value"] for e in ev if e["valueType"] == "numeric"}
        ref[r["id"]] = {"binding": binding, **scores, "method": r["designMethod"], "author": r["author"]}

    rows = [json.loads(l) for l in open(analysis) if l.startswith('{"file"')]
    for o in rows:
        o["_id"] = os.path.basename(os.path.dirname(o["file"]))
    rows = [o for o in rows if o["_id"] in ref and ref[o["_id"]]["binding"]]
    labels = [1 if any(ref[o["_id"]]["binding"]) else 0 for o in rows]
    prevalence = sum(labels) / len(labels)

    out, failed = [], []
    w = out.append
    w("# Binder triage vs the Adaptyv Nipah binder competition\n")
    w(
        "Dataset: ProteinBase collection `nipah-binder-competition-results` (ODC-By), table sha256 "
        f"`{hashlib.sha256(collection.read_bytes()).hexdigest()[:16]}`, and the Boltz-2 complex "
        "and full PAE of each design as ProteinBase publishes them. Proteus: `proteus analyze "
        "<models> --interface B`. Written by `validate/nipah/compare.py`.\n"
    )
    w(
        f"{len(rows)} designs with a model and a lab result ({controls} positive controls left "
        f"out), {sum(labels)} binders (prevalence {prevalence:.3f}, the AP of a random ranking). "
        "One target, so per-target and pooled are the same number.\n"
    )
    above = sum(1 for o in rows if (o["ipsae_min"] or 0) > 0.61)
    w(
        "**These designs were chosen by ipSAE before they were tested.** The competition sent the "
        "60 collections with the best average ipSAE (600 designs) to the lab (Adaptyv's "
        "`nipah_ipsae_pipeline` README), so the tested set is already filtered on the score being "
        f"evaluated: {above} of {len(rows)} ({above / len(rows):.0%}) have `ipsae_min > 0.61`. "
        "Section 3 splits the tested set by how each design got there.\n"
    )

    # 1. Agreement ---------------------------------------------------------------------------
    w("## 1. ipSAE against ProteinBase's own values\n")
    w(
        "ProteinBase's numbers come from Adaptyv's `nipah_ipsae_pipeline`, which runs `ipsae.py` "
        "with PAE and distance cutoffs of 15 Å (the standard, and Proteus, use 10 Å). Its "
        "`boltz2_ipsae` is the max over the two directions; its `boltz2_min_ipsae` is not a "
        "minimum but one direction alone, A→B (target-aligned, binder scored): the notebook takes "
        "the `asym` row whose chain order matches the `max` row, which `ipsae.py` always labels A,B. Checked once on "
        "45 designs by running that pipeline's `ipsae.py`: both reproduce ProteinBase to 1e-6, and "
        "Proteus's `ipsae_min` matches the same script at 10 Å to 0.0009 (the d0 floor moved "
        "since; see validate/binders). So the values below agree in rank, not in value.\n"
    )
    w("| ours | theirs | n | Pearson r | Spearman ρ | median \\|Δ\\| | gate: r ≥ | |")
    w("|---|---|---|---|---|---|---|---|")
    for ours, theirs, gate in (("ipsae_max", "boltz2_ipsae", MIN_PEARSON), ("ipsae_min", "boltz2_min_ipsae", MIN_PEARSON)):
        pairs = [
            (o[ours], num(ref[o["_id"]].get(theirs)))
            for o in rows
            if o.get(ours) is not None and num(ref[o["_id"]].get(theirs)) is not None
        ]
        d = sorted(abs(x - y) for x, y in pairs)
        r = pearson(pairs)
        ok = r >= gate
        if not ok:
            failed.append(f"agreement {ours}: Pearson {r:.3f} < {gate}")
        w(
            f"| `{ours}` | `{theirs}` | {len(pairs)} | {r:.3f} | {spearman(pairs):.3f} | "
            f"{d[len(d) // 2]:.4f} | {gate} | {'✓' if ok else '✗'} |"
        )
    w("")

    # 2. Wet-lab outcome ----------------------------------------------------------------------
    w("## 2. Separating designs that bound in the lab from those that did not\n")
    w(
        "Higher AP and AUROC are better; each metric is oriented so that larger means more likely "
        "to bind. Missing values rank last. `proteus` rows are computed here from the model and "
        "PAE; `ProteinBase` rows are the values published with the dataset.\n"
    )
    w("| metric | source | AP | AUROC |")
    w("|---|---|---|---|")
    neg = lambda v: None if v is None else -v
    theirs = lambda k: lambda o, x: num(x.get(k))
    metrics = [
        ("ipSAE_min", "proteus", lambda o, x: o["ipsae_min"]),
        ("ipSAE_min", "ProteinBase", theirs("boltz2_min_ipsae")),
        ("ipSAE_max", "proteus", lambda o, x: o["ipsae_max"]),
        ("ipSAE", "ProteinBase", theirs("boltz2_ipsae")),
        ("LIS", "proteus", lambda o, x: o["lis"]),
        ("LIS", "ProteinBase", theirs("boltz2_lis")),
        ("−ipAE", "proteus", lambda o, x: neg(o["ipae"])),
        ("ipTM", "ProteinBase", theirs("boltz2_iptm")),
        ("interface pLDDT", "ProteinBase", theirs("boltz2_complex_iplddt")),
        ("pLDDT (mean)", "proteus", lambda o, x: o["plddt_mean"]),
        ("Sc", "proteus", lambda o, x: o["interface_sc"]),
        ("Sc", "ProteinBase", theirs("shape_complimentarity_boltz2_binder_ss")),
        ("dSASA", "proteus", lambda o, x: o["interface_dsasa"]),
        ("interface H-bonds", "proteus", lambda o, x: o["interface_hbonds"]),
        ("ESMFold pLDDT (binder alone)", "ProteinBase", theirs("esmfold_plddt")),
    ]
    for name, source, f in metrics:
        s = [f(o, ref[o["_id"]]) for o in rows]
        s = [v if v is not None else -1e18 for v in s]
        w(f"| {name} | {source} | {average_precision(s, labels):.3f} | {auroc(s, labels):.3f} |")
    w("")
    w(
        "ProteinBase's `boltz2_pdockq` and `boltz2_pdockq2` are left out: each holds one value for "
        "every design (0.0183 and 0.0073), which is what `ipsae.py` returns when it finds no pLDDT "
        "file beside the PAE.\n"
    )
    w("Binder rate by Proteus's `ipsae_min`:\n")
    w("| ipsae_min | designs | binders | binder rate |")
    w("|---|---|---|---|")
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.61), (0.61, 0.7), (0.7, 0.8), (0.8, 1.01)):
        sel = [l for o, l in zip(rows, labels) if lo <= (o["ipsae_min"] or 0) < hi]
        w(
            f"| {lo:.2f}–{min(hi, 1):.2f} | {len(sel)} | {sum(sel)} | "
            f"{sum(sel) / max(1, len(sel)):.3f} |"
        )
    w("")
    for thr in (0.61,):
        sel = [l for o, l in zip(rows, labels) if (o["ipsae_min"] or 0) > thr]
        w(
            f"Filter `ipsae_min > {thr}` (the threshold from validate/binders): keeps {len(sel)} "
            f"designs, of which {sum(sel)} bound (precision {sum(sel) / max(1, len(sel)):.3f}, "
            f"recall {sum(sel) / sum(labels):.3f}).\n"
        )

    # 3. Selection --------------------------------------------------------------------------
    w("## 3. Designs ipSAE selected, against designs it did not\n")
    # A collection is one author's 10 designs. Adaptyv's rule: "the 60 best collections (600
    # designs) given the average ipSAE will be considered for wet-lab validation". Rank the full
    # collections by mean ipsae_min and take 60 (group A); everything else came by other routes.
    by_author = {}
    for i, o in enumerate(rows):
        by_author.setdefault(ref[o["_id"]]["author"], []).append(i)
    full = [ix for ix in by_author.values() if len(ix) == 10]
    full.sort(key=lambda ix: -sum(rows[i]["ipsae_min"] or 0 for i in ix) / len(ix))
    in_a = {i for ix in full[:60] for i in ix}
    w(
        f"{len(full)} authors had exactly 10 tested designs; ranking those collections by mean "
        f"`ipsae_min` and taking 60 gives {len(in_a)} designs (group A; the README says 600). "
        "Group B is every other tested design: community voting, curation, partial collections. "
        "B is less selected by ipSAE, not unselected. Enrichment is AP ÷ prevalence, comparable "
        "across groups whose binder rates differ.\n"
    )
    w("| group | designs | binders | prevalence | ipSAE_min AP (enrichment) | interface pLDDT AP (enrichment) |")
    w("|---|---|---|---|---|---|")
    ipl = theirs("boltz2_complex_iplddt")
    for name, keep in (("all", lambda i: True), ("A: ipSAE-selected", lambda i: i in in_a), ("B: other routes", lambda i: i not in in_a)):
        ix = [i for i in range(len(rows)) if keep(i)]
        lab = [labels[i] for i in ix]
        p = sum(lab) / len(lab)
        a1 = average_precision([rows[i]["ipsae_min"] if rows[i]["ipsae_min"] is not None else -1e18 for i in ix], lab)
        a2 = average_precision([v if (v := ipl(rows[i], ref[rows[i]["_id"]])) is not None else -1e18 for i in ix], lab)
        w(f"| {name} | {len(ix)} | {sum(lab)} | {p:.4f} | {a1:.3f} ({a1 / p:.2f}×) | {a2:.3f} ({a2 / p:.2f}×) |")
    w("")
    w(
        "Interface pLDDT is ProteinBase's `boltz2_complex_iplddt`, not computed by Proteus. "
        "Boltz-2 defines it as a weighted mean over all residues (interface 10, the rest 1); "
        "Boltz-1 as the mean over interface residues alone. On validate/binders the dataset's "
        "Boltz-1 interface pLDDT ranks below Boltz-1's ipSAE_min (see its table), so a lead for "
        "interface pLDDT here does not carry over to that set.\n"
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
