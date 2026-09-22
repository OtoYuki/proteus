#!/usr/bin/env python3
"""mdtraj baker_hubbard H-bond reference for structures that carry explicit hydrogens.

Proteus detects hydrogen bonds from heavy atoms only (donor/acceptor distance plus antecedent
angles); mdtraj uses the explicit H (D-H...A distance and angle). The two criteria are related
but not identical, so this reference exists to measure the overlap honestly, not to assert
equality. Writes validate/reference/hbonds/<id>.json.
"""
import json
import pathlib
import sys

import mdtraj as md

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "validate" / "reference" / "hbonds"


def donor_acceptor_pairs(path):
    t = md.load(str(path))
    if t.n_frames > 1:
        t = t[0]
    top = t.topology
    pairs = []
    for d_h, _h, a in md.baker_hubbard(t, freq=0.0, periodic=False):
        donor = top.atom(int(d_h)).residue
        acceptor = top.atom(int(a)).residue
        pairs.append(
            {
                "donor_resseq": int(donor.resSeq),
                "donor_resname": donor.name,
                "acceptor_resseq": int(acceptor.resSeq),
                "acceptor_resname": acceptor.name,
            }
        )
    return t.n_atoms, sum(1 for a in top.atoms if a.element.symbol == "H"), pairs


def main():
    # Every corpus PDB entry that carries explicit hydrogens (all six NMR depositions).
    ids = sys.argv[1:] or ["1d3z", "2kod", "1g6j", "2l3b", "1gb1", "1l2y"]
    OUT.mkdir(parents=True, exist_ok=True)
    for i in ids:
        path = ROOT / "validate" / "corpus" / f"{i}.pdb"
        if not path.exists():
            print("skip", i, "(run make fetch)")
            continue
        n_atoms, n_h, pairs = donor_acceptor_pairs(path)
        if n_h == 0:
            print("skip", i, "(no explicit hydrogens)")
            continue
        (OUT / f"{i}.json").write_text(
            json.dumps({"id": i, "n_atoms": n_atoms, "n_hydrogens": n_h, "hbonds": pairs})
        )
        print(f"{i}: {len(pairs)} baker_hubbard H-bonds ({n_h} hydrogens)")


if __name__ == "__main__":
    main()
