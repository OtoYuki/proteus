#!/usr/bin/env python3
"""Generate validate/reference/<id>_<format>.json from mdtraj, freesasa and cctbx ramalyze.

Every per-residue value is keyed by "chain:resseq:icode" so that Proteus can align on
residue identity rather than on index.
"""
import json
import pathlib
import sys
import tomllib

import subprocess

import freesasa
import mdtraj as md
import numpy as np

root = pathlib.Path(__file__).parent
cfg = tomllib.loads((root / "corpus.toml").read_text())
out = root / "reference"
out.mkdir(exist_ok=True)


def key(chain_id, resseq, icode):
    return f"{chain_id.strip()}:{int(resseq)}:{(icode or '').strip()}"


def one(s):
    p = root / "corpus" / f"{s['id']}.{s['format']}"
    t = md.load(str(p))
    if t.n_frames > 1:
        t = t[0]
    t = t.atom_slice(t.topology.select("protein and not element H"))
    ca = t.topology.select("name CA")
    rg_ca = float(md.compute_rg(t.atom_slice(ca))[0] * 10)
    sasa_sr = float(md.shrake_rupley(t, probe_radius=0.14, n_sphere_points=960).sum() * 100)
    sasa_lr = None
    if s["format"] == "pdb":
        opts = {"hetatm": False, "hydrogen": False}
        fs = freesasa.Structure(str(p), options=opts)
        sasa_lr = float(freesasa.calc(fs).totalArea())
    ss8 = md.compute_dssp(t, simplified=False)[0]
    ss3 = md.compute_dssp(t, simplified=True)[0]
    phi_idx, phi = md.compute_phi(t)
    psi_idx, psi = md.compute_psi(t)
    residues = list(t.topology.residues)
    # Two keys per residue: by chain id (PDB auth ids) and by chain ordinal. mmCIF readers
    # disagree on which asym id is "the" chain id (mdtraj: label_asym_id, pdbtbx: auth), so
    # the consumer falls back to ordinal keys when id keys do not match.
    chain_ids = list(dict.fromkeys(r.chain.index for r in residues))
    ordinal = {c: i for i, c in enumerate(chain_ids)}
    rkey = [key(r.chain.chain_id if hasattr(r.chain, "chain_id") else str(r.chain.index), r.resSeq, "") for r in residues]
    okey = [key(str(ordinal[r.chain.index]), r.resSeq, "") for r in residues]
    phis = [None] * len(residues)
    psis = [None] * len(residues)
    for k, quad in enumerate(phi_idx):
        r = t.topology.atom(int(quad[2])).residue
        phis[r.index] = float(np.degrees(phi[0][k]))
    for k, quad in enumerate(psi_idx):
        r = t.topology.atom(int(quad[1])).residue
        psis[r.index] = float(np.degrees(psi[0][k]))
    dssp8 = [(c if c != " " else "-") for c in ss8]
    dssp3 = list(ss3)

    # cctbx must not share a process with mdtraj (native-library conflict); see ramalyze_ref.py.
    proc = subprocess.run(
        [sys.executable, str(root / "ramalyze_ref.py"), str(p)],
        capture_output=True, text=True, check=True,
    )
    rama = json.loads(proc.stdout)
    # cctbx keys use auth chain ids (as pdbtbx does); add an ordinal-keyed copy for the fallback.
    cctbx_chain_order = list(dict.fromkeys(k.split(":")[0] for k in rama["labels"]))
    cctbx_ord = {c: i for i, c in enumerate(cctbx_chain_order)}
    rama["labels_ordinal"] = {
        key(str(cctbx_ord[k.split(":")[0]]), k.split(":")[1], k.split(":")[2]): v
        for k, v in rama["labels"].items()
    }
    return {
        "id": s["id"],
        "format": s["format"],
        "kind": s["kind"],
        "n_residues": len(residues),
        "residue_keys": rkey,
        "residue_keys_ordinal": okey,
        "rg_ca": rg_ca,
        "sasa_freesasa_lr": sasa_lr,
        "sasa_mdtraj_sr": sasa_sr,
        "dssp8": dssp8,
        "dssp3": dssp3,
        "phi": phis,
        "psi": psis,
        "rama": rama,
    }


ids = sys.argv[1:] or [s["id"] for s in cfg["structure"]]
failed = 0
for s in cfg["structure"]:
    if s["id"] not in ids:
        continue
    try:
        d = one(s)
    except Exception as e:  # noqa: BLE001
        print("FAIL", s["id"], s["format"], repr(e))
        failed += 1
        continue
    (out / f"{s['id']}_{s['format']}.json").write_text(json.dumps(d))
    print(f"{s['id']:12s} {s['format']:3s} res={d['n_residues']:5d} rg={d['rg_ca']:7.3f} "
          f"sasa_sr={d['sasa_mdtraj_sr']:9.1f} rama F/A/O={d['rama']['favored']}/{d['rama']['allowed']}/{d['rama']['outliers']}")
sys.exit(1 if failed else 0)
