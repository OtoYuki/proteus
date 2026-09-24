#!/usr/bin/env python3
"""cctbx covalent-geometry reference for one structure, as JSON on stdout.

Usage: geometry_ref.py <structure> [--full]

The structure is reduced exactly as Proteus reduces it before any metric
(`proteus_core::io::protein_heavy_atoms`): first model, protein residues, no hydrogens, first
alternate conformation. cctbx then builds its default restraints (geostd monomer library,
Conformation-Dependent Library v1.2, EH99 cis-Pro) and this script reports

  * every bond, angle, chirality and planarity restraint whose atoms all belong to the 20
    standard amino acids or selenomethionine (`--full`), or only the counts, RMSZ and > 4σ outliers (default);
  * cbetadev, omegalyze and rotalyze per residue.

cctbx needs its chemical data (`chem_data`: geostd, a monomer library and rotarama_data), which
the pip `cctbx-base` wheel does not ship. `validate/fetch_chem_data.sh` builds it; point
`PROTEUS_CHEM_DATA` at the directory. cctbx must not share a process with mdtraj, so
`geometry_reference.py` runs this as a subprocess.
"""
import contextlib
import io
import json
import math
import os
import sys

import libtbx.load_env  # noqa: F401
import libtbx
from libtbx.path import absolute_path, relocatable_path

CHEM_DATA = os.environ.get("PROTEUS_CHEM_DATA")
if not CHEM_DATA or not os.path.isdir(os.path.join(CHEM_DATA, "geostd")):
    sys.exit("PROTEUS_CHEM_DATA must point at a chem_data directory (see fetch_chem_data.sh)")
# cctbx finds geostd/, mon_lib/ and rotarama_data/ under any repository path.
libtbx.env.repository_paths.append(relocatable_path(absolute_path(CHEM_DATA), "."))

import cctbx.geometry_restraints as gr  # noqa: E402
import iotbx.pdb  # noqa: E402
import mmtbx.model  # noqa: E402
from mmtbx.validation.cbetadev import cbetadev  # noqa: E402
from mmtbx.validation.omegalyze import omegalyze  # noqa: E402
from mmtbx.validation.rotalyze import rotalyze  # noqa: E402

STANDARD = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
}
SIGMA_CUTOFF = 4.0


def reduce_like_proteus(path):
    inp = iotbx.pdb.input(file_name=path)
    h = inp.construct_hierarchy()
    for m in list(h.models())[1:]:
        h.remove_model(m)
    # Proteus keeps only residues with a CA (`io::is_protein_residue`).
    for chain in h.chains():
        for rg in list(chain.residue_groups()):
            if not any(a.name.strip() == "CA" for a in rg.atoms()):
                chain.remove_residue_group(residue_group=rg)
    # Microheterogeneity (two residue types at one position, e.g. 1EJG 22 PRO/SER) comes out
    # of iotbx as consecutive residue groups with the same number; pdbtbx makes them
    # conformers of one residue and keeps the first. Do the same.
    for chain in h.chains():
        prev = None
        for rg in list(chain.residue_groups()):
            key = (rg.resseq, rg.icode)
            if key == prev:
                chain.remove_residue_group(residue_group=rg)
            prev = key
    # Keep the first alternate conformation in file order, as pdbtbx does (conformer 0).
    # cctbx's own default keeps the one with the highest mean occupancy instead, so drop the
    # others first and let remove_alt_confs merge what is left.
    for rg in h.residue_groups():
        altlocs = [ag.altloc for ag in rg.atom_groups() if ag.altloc != ""]
        for ag in list(rg.atom_groups()):
            if altlocs and ag.altloc not in ("", altlocs[0]):
                rg.remove_atom_group(atom_group=ag)
    h.remove_alt_confs(always_keep_one_conformer=True)
    sel = h.atom_selection_cache().selection("protein and not (element H or element D)")
    h = h.select(sel, copy_atoms=True)
    h.atoms().reset_i_seq()
    return inp, h


def atom_label(atom):
    rg = atom.parent().parent()
    return {
        "chain": atom.parent().parent().parent().id.strip(),
        "resseq": rg.resseq_as_int(),
        "icode": rg.icode.strip(),
        "resname": atom.parent().resname.strip(),
        "name": atom.name.strip(),
    }


def label_str(a):
    return f"{a['chain']}:{a['resseq']}:{a['icode']}:{a['resname']}:{a['name']}"


def main():
    path = sys.argv[1]
    full = "--full" in sys.argv[2:]
    sink = io.StringIO()
    inp, h = reduce_like_proteus(path)

    p = mmtbx.model.manager.get_default_pdb_interpretation_params()
    pi = p.pdb_interpretation
    pi.allow_polymer_cross_special_position = True
    pi.clash_guard.nonbonded_distance_threshold = None
    pi.proceed_with_excessive_length_bonds = True
    pi.disable_uc_volume_vs_n_atoms_check = True
    with contextlib.redirect_stdout(sink):
        model = mmtbx.model.manager(
            model_input=None, pdb_hierarchy=h.deep_copy(),
            crystal_symmetry=inp.crystal_symmetry(), log=sink)
        model.process(make_restraints=True, pdb_interpretation_params=p)
    grm = model.get_restraints_manager().geometry
    sites = model.get_sites_cart()
    # Proxy i_seqs index the processed model's atoms, which can be ordered differently from
    # `h` (altloc groups are merged), so label from the model itself.
    atoms = model.get_hierarchy().atoms()
    labels = [atom_label(a) for a in atoms]
    standard = [lab["resname"] in STANDARD for lab in labels]

    out = {"file": os.path.basename(path), "n_atoms": atoms.size()}
    kinds = {}

    def keep(i_seqs):
        return all(standard[i] for i in i_seqs)

    def record(kind, i_seqs, ideal, sigma, model_value, score, extra=None):
        k = kinds.setdefault(kind, {"n": 0, "sum_z2": 0.0, "outliers": [], "all": []})
        k["n"] += 1
        k["sum_z2"] += score * score
        row = {
            "atoms": [label_str(labels[i]) for i in i_seqs],
            "ideal": round(ideal, 5),
            "sigma": round(sigma, 5),
            "model": round(model_value, 5),
            "z": round(score, 4),
        }
        if extra:
            row.update(extra)
        if abs(score) > SIGMA_CUTOFF:
            k["outliers"].append(row)
        if full:
            k["all"].append(row)

    for proxy in grm.pair_proxies(sites_cart=sites).bond_proxies.simple:
        i, j = proxy.i_seqs
        if not keep((i, j)):
            continue
        b = gr.bond(sites_cart=sites, proxy=proxy)
        sigma = gr.weight_as_sigma(proxy.weight)
        record("bond", (i, j), proxy.distance_ideal, sigma, b.distance_model, b.delta / sigma,
               {"origin": int(proxy.origin_id)})
    for proxy in grm.angle_proxies:
        if not keep(proxy.i_seqs):
            continue
        a = gr.angle(sites_cart=sites, proxy=proxy)
        sigma = gr.weight_as_sigma(proxy.weight)
        record("angle", tuple(proxy.i_seqs), proxy.angle_ideal, sigma, a.angle_model,
               a.delta / sigma, {"origin": int(proxy.origin_id)})
    for proxy in grm.chirality_proxies:
        if not keep(proxy.i_seqs):
            continue
        c = gr.chirality(sites_cart=sites, proxy=proxy)
        sigma = gr.weight_as_sigma(proxy.weight)
        record("chirality", tuple(proxy.i_seqs), proxy.volume_ideal, sigma, c.volume_model,
               c.delta / sigma, {"both_signs": bool(proxy.both_signs)})
    for proxy in grm.planarity_proxies:
        i_seqs = tuple(proxy.i_seqs)
        if not keep(i_seqs):
            continue
        pl = gr.planarity(sites_cart=sites, proxy=proxy)
        deltas = list(pl.deltas())
        sigmas = [gr.weight_as_sigma(w) for w in proxy.weights]
        devs = [abs(d) / s for d, s in zip(deltas, sigmas)]
        worst = max(devs)
        rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
        record("planarity", i_seqs, 0.0, sigmas[devs.index(worst)], rms, worst,
               {"delta_max": round(max(abs(d) for d in deltas), 5)})

    out["restraints"] = {}
    for kind, k in kinds.items():
        entry = {
            "n": k["n"],
            "n_outliers": len(k["outliers"]),
            "rmsz": round(math.sqrt(k["sum_z2"] / k["n"]), 5) if k["n"] else None,
            "outliers": k["outliers"],
        }
        if full:
            entry["all"] = k["all"]
        out["restraints"][kind] = entry

    def res_key(r):
        return f"{r.chain_id.strip()}:{int(r.resseq)}:{(r.icode or '').strip()}"

    # phenix.molprobity runs the per-residue validators on the processed model, whose
    # symmetric side-chain atoms pdb_interpretation has already renamed (flip_symmetric_amino_
    # acids); rotalyze in particular depends on it.
    h = model.get_hierarchy()
    cb = cbetadev(pdb_hierarchy=h, outliers_only=False, out=sink, quiet=True)
    out["cbetadev"] = {
        res_key(r): {"resname": r.resname.strip(), "deviation": round(r.deviation, 4),
                     "outlier": bool(r.is_outlier())}
        for r in cb.results
    }
    om = omegalyze(pdb_hierarchy=h, nontrans_only=False, out=sink, quiet=True)
    out["omegalyze"] = {
        res_key(r): {"resname": r.resname.strip(), "omega": round(r.omega, 3),
                     "type": ["trans", "cis", "twisted"][r.omega_type],
                     "pro": r.res_type == 1}
        for r in om.results
    }
    with contextlib.redirect_stdout(sink):
        ro = rotalyze(pdb_hierarchy=h, outliers_only=False, out=sink, quiet=True)
    out["rotalyze"] = {
        res_key(r): {"resname": r.resname.strip(), "score": round(r.score, 4),
                     "evaluation": r.evaluation, "rotamer": r.rotamer_name,
                     "chi": [None if c is None else round(c, 3) for c in (r.chi_angles or [])]}
        for r in ro.results
    }
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
