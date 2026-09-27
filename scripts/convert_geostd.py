#!/usr/bin/env python3
"""Generate Proteus's covalent-geometry restraint library from Phenix geostd and the cctbx CDL.

Writes
  crates/proteus-core/src/geometry/library.rs   heavy-atom restraints of the 20 amino acids,
                                                the peptide links and the C-terminal COO mod
  crates/proteus-core/data/cdl/cdl_v1_2.f32     Conformation-Dependent Library v1.2 grid

Usage (validate venv, chem_data as for validate/geometry_ref.py):
  PROTEUS_CHEM_DATA=... validate/.venv/bin/python scripts/convert_geostd.py

The monomer definitions are read through cctbx's own monomer-library server so that the
selection and parsing are the ones pdb_interpretation uses: geostd is searched before any
other monomer library, links and mods come from geostd/list/mon_lib_list.cif. Chirality ideal
volumes are computed by cctbx (`comp_comp_id.get_chir_volume_ideal`) from the same monomer's
ideal bonds and angles, which is what pdb_interpretation does.

Sources (BSD-3-style LBNL licence): https://github.com/phenix-project/geostd and
https://github.com/cctbx/cctbx_project (mmtbx/conformation_dependent_library/cdl_database.py).
"""
import os
import pathlib
import struct
import sys

import libtbx.load_env  # noqa: F401
import libtbx
from libtbx.path import absolute_path, relocatable_path

CHEM_DATA = os.environ.get("PROTEUS_CHEM_DATA")
if not CHEM_DATA:
    sys.exit("set PROTEUS_CHEM_DATA (see validate/fetch_chem_data.sh)")
libtbx.env.repository_paths.append(relocatable_path(absolute_path(CHEM_DATA), "."))

from mmtbx.conformation_dependent_library.cdl_database import cdl_database, version  # noqa: E402
from mmtbx.monomer_library import server  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Selenomethionine: cctbx counts it among the common amino acids (CDL, validation).
    "MSE",
]
# Bonds and angles of a peptide link always come from TRANS or PTRANS: pdb_interpretation
# switches the link id to CIS/PCIS only while adding the omega dihedral, after the bond and
# angle proxies exist, and the heavy-atom planes of the cis links are identical.
LINKS = ["TRANS", "PTRANS"]
# Order of the eight residue classes in cdl_v1_2.f32 (must match geometry::cdl::CdlGroup).
CDL_GROUPS = [
    "NonPGIV_nonxpro", "IleVal_nonxpro", "Gly_nonxpro", "Pro_nonxpro",
    "NonPGIV_xpro", "IleVal_xpro", "Gly_xpro", "Pro_xpro",
]

srv = server.server()


def heavy_atoms(comp):
    return {a.atom_id for a in comp.atom_list if a.type_symbol not in ("H", "D")}


def f(x):
    return repr(float(x))


def residue_block(name):
    comp = srv.get_comp_comp_id_direct(name)
    assert comp is not None, name
    source = comp.source_info or ""
    assert "geostd" in source, f"{name} did not come from geostd: {source}"
    heavy = heavy_atoms(comp)
    bonds = [b for b in comp.bond_list if {b.atom_id_1, b.atom_id_2} <= heavy]
    angles = [a for a in comp.angle_list if {a.atom_id_1, a.atom_id_2, a.atom_id_3} <= heavy]
    chirs = []
    for c in comp.chir_list:
        atoms = [c.atom_id_centre, c.atom_id_1, c.atom_id_2, c.atom_id_3]
        if not set(atoms) <= heavy:
            continue
        ideal = comp.get_chir_volume_ideal(c)
        assert ideal is not None, (name, atoms)
        chirs.append((atoms, c.volume_sign, ideal))
    planes = []
    for p in comp.get_planes():
        atoms = [(a.atom_id, a.dist_esd) for a in p.plane_atoms if a.atom_id in heavy]
        if len(atoms) >= 4:
            planes.append(atoms)
    lines = [f"    ResidueDef {{", f"        name: {name!r},".replace("'", '"')]
    lines.append("        bonds: &[")
    for b in bonds:
        lines.append(f'            BondDef {{ a: "{b.atom_id_1}", b: "{b.atom_id_2}", ideal: {f(b.value_dist)}, esd: {f(b.value_dist_esd)} }},')
    lines.append("        ],")
    lines.append("        angles: &[")
    for a in angles:
        lines.append(f'            AngleDef {{ a: "{a.atom_id_1}", b: "{a.atom_id_2}", c: "{a.atom_id_3}", ideal: {f(a.value_angle)}, esd: {f(a.value_angle_esd)} }},')
    lines.append("        ],")
    lines.append("        chiralities: &[")
    for atoms, sign, ideal in chirs:
        both = "true" if sign.startswith("both") else "false"
        q = ", ".join(f'"{x}"' for x in atoms)
        lines.append(f"            ChiralDef {{ atoms: [{q}], ideal: {f(ideal)}, both_signs: {both} }},")
    lines.append("        ],")
    lines.append("        planes: &[")
    for atoms in planes:
        q = ", ".join(f'("{x}", {f(e)})' for x, e in atoms)
        lines.append(f"            &[{q}],")
    lines.append("        ],")
    lines.append("    },")
    return "\n".join(lines)


def link_block(link_id):
    link = srv.link_link_id_dict[link_id]
    lines = [f"    LinkDef {{", f'        id: "{link_id}",', "        bonds: &["]
    for b in link.bond_list:
        lines.append(f'            LinkBondDef {{ a: ({b.atom_1_comp_id}, "{b.atom_id_1}"), b: ({b.atom_2_comp_id}, "{b.atom_id_2}"), ideal: {f(b.value_dist)}, esd: {f(b.value_dist_esd)} }},')
    lines.append("        ],")
    lines.append("        angles: &[")
    for a in link.angle_list:
        ids = [(a.atom_1_comp_id, a.atom_id_1), (a.atom_2_comp_id, a.atom_id_2), (a.atom_3_comp_id, a.atom_id_3)]
        if any(x[1].startswith("H") for x in ids):
            continue
        q = ", ".join(f'({c}, "{n}")' for c, n in ids)
        lines.append(f"            LinkAngleDef {{ atoms: [{q}], ideal: {f(a.value_angle)}, esd: {f(a.value_angle_esd)} }},")
    lines.append("        ],")
    lines.append("        planes: &[")
    planes = {}
    for pa in link.plane_list:
        planes.setdefault(pa.plane_id, []).append(pa)
    for pid, atoms in planes.items():
        heavy = [a for a in atoms if not a.atom_id.startswith("H")]
        if len(heavy) < 4:
            continue
        q = ", ".join(f'({a.atom_comp_id}, "{a.atom_id}", {f(a.dist_esd)})' for a in heavy)
        lines.append(f"            &[{q}],")
    lines.append("        ],")
    lines.append("    },")
    return "\n".join(lines)


def coo_block():
    mod = srv.mod_mod_id_dict["COO"]
    lines = ["pub(super) static COO: ModDef = ModDef {", "    bonds: &["]
    for b in mod.bond_list:
        lines.append(f'        ModBondDef {{ add: {str(b.function == "add").lower()}, bond: BondDef {{ a: "{b.atom_id_1}", b: "{b.atom_id_2}", ideal: {f(b.new_value_dist)}, esd: {f(b.new_value_dist_esd)} }} }},')
    lines.append("    ],")
    lines.append("    angles: &[")
    for a in mod.angle_list:
        lines.append(f'        ModAngleDef {{ add: {str(a.function == "add").lower()}, angle: AngleDef {{ a: "{a.atom_id_1}", b: "{a.atom_id_2}", c: "{a.atom_id_3}", ideal: {f(a.new_value_angle)}, esd: {f(a.new_value_angle_esd)} }} }},')
    lines.append("    ],")
    planes = {}
    for pa in mod.plane_atom_list:
        assert pa.function == "add"
        planes.setdefault(pa.plane_id, []).append(pa)
    lines.append("    planes: &[")
    for pid, atoms in planes.items():
        q = ", ".join(f'("{a.atom_id}", {f(a.new_dist_esd)})' for a in atoms)
        lines.append(f"        &[{q}],")
    lines.append("    ],")
    lines.append("};")
    return "\n".join(lines)


def main():
    out = []
    out.append("// @generated by scripts/convert_geostd.py from Phenix geostd; do not edit by hand.")
    out.append("// Source: https://github.com/phenix-project/geostd (BSD-3-style LBNL licence);")
    out.append("// see crates/proteus-core/data/cdl/NOTICE for provenance and licence text.")
    out.append("")
    out.append("use super::library_types::*;")
    out.append("")
    out.append("/// Heavy-atom restraints of the 20 standard amino acids and selenomethionine (geostd).")
    out.append("pub(super) static RESIDUES: &[ResidueDef] = &[")
    for name in RESIDUES:
        out.append(residue_block(name))
    out.append("];")
    out.append("")
    out.append("/// Peptide links: TRANS, and PTRANS to a proline; heavy atoms only.")
    out.append("pub(super) static LINKS: &[LinkDef] = &[")
    for link_id in LINKS:
        out.append(link_block(link_id))
    out.append("];")
    out.append("")
    out.append("/// The C-terminal carboxylate modification, applied when the residue carries OXT.")
    out.append(coo_block())
    target = ROOT / "crates/proteus-core/src/geometry/library.rs"
    target.write_text("\n".join(out) + "\n")
    print("wrote", target)

    vals = []
    for group in CDL_GROUPS:
        table = cdl_database[group]
        for phi in range(-180, 180, 10):
            for psi in range(-180, 180, 10):
                row = table[(phi, psi)]
                assert len(row) == 26, (group, phi, psi)
                vals.extend(float(v) for v in row[2:])
    cdl = ROOT / "crates/proteus-core/data/cdl/cdl_v1_2.f32"
    cdl.parent.mkdir(parents=True, exist_ok=True)
    cdl.write_bytes(struct.pack("<%df" % len(vals), *vals))
    print("wrote", cdl, len(vals), "values,", version)


if __name__ == "__main__":
    main()
