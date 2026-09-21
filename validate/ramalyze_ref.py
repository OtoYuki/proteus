#!/usr/bin/env python3
"""cctbx ramalyze reference for one structure, as JSON on stdout.

Runs in its own process: importing cctbx into the same interpreter as mdtraj corrupts
mdtraj's native SASA kernel (observed 2026-09-21: 47374 vs 2969 Å² on 1CRN) and can segfault.
"""
import json
import sys

import iotbx.pdb
from mmtbx.validation.ramalyze import ramalyze

path = sys.argv[1]
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
models = list(h.models())
for m in models[1:]:
    h.remove_model(m)
r = ramalyze(pdb_hierarchy=h, outliers_only=False)
labels = {}
for res in r.results:
    k = f"{res.chain_id.strip()}:{int(res.resseq)}:{(res.icode or '').strip()}"
    labels[k] = {"OUTLIER": "O", "Allowed": "A", "Favored": "F"}[res.ramalyze_type()]
json.dump({"favored": r.n_favored, "allowed": r.n_allowed, "outliers": r.n_outliers, "labels": labels}, sys.stdout)
