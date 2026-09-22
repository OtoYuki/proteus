#!/usr/bin/env python3
"""PLIP reference for the non-covalent interactions Proteus reports without one.

Salt bridges, pi-pi stacking and cation-pi were the rows in the README that said "no widely
used reference implementation with the same definitions". PLIP (Salentin et al. 2015,
https://github.com/pharmai/plip) is that reference, run here in **intra-chain** mode so it
profiles a protein against itself rather than against a ligand.

The criteria are *not* the same, deliberately, and that is why this measures overlap rather
than equality:

| interaction | PLIP                                   | Proteus                                  |
|-------------|----------------------------------------|------------------------------------------|
| salt bridge | <= 5.5 A between **centres of charge** | <= 4.0 A between **closest atoms**       |
| pi-pi       | <= 5.5 A centroids, <= 30 deg dev,     | <= 6.5 A centroids, <= 30 deg parallel   |
|             | ring offset <= 2.0 A                   | or 60-120 deg T-shaped, no offset test   |
| cation-pi   | <= 6.0 A, offset <= 2.0 A              | <= 6.0 A, angle to ring normal <= 45 deg |

So a disagreement is usually a threshold, not a bug, and the harness reports recall and
precision by residue pair instead of asserting a match. PLIP also reports each salt bridge
twice (once per direction); pairs are normalised and de-duplicated here.

Scope: intra-chain only. PLIP's INTRA mode profiles one chain against itself, so inter-chain
contacts are outside this reference — those are covered by Proteus's own chain-awareness test
in `crates/proteus-core/src/interactions.rs`.

Writes validate/reference/plip/<id>.json.
"""
import json
import logging
import pathlib
import sys

# PLIP warns once per structure that it is not assigning polar hydrogens. That is intentional
# here (predicted models never carry them) and the warning would drown the progress output.
logging.getLogger("plip").setLevel(logging.ERROR)

from plip.basic import config  # noqa: E402
from plip.structure.preparation import PDBComplex  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "validate" / "reference" / "plip"

# Single-model X-ray entries in PDB format, spanning sizes and compositions. PLIP is slow on
# very large structures and openbabel's perception is the bottleneck, so 6VXX and the big
# AlphaFold models are left out on purpose.
TARGETS = [
    "1crn",
    "1ubq",
    "2ci2",
    "1pgb",
    "1mbn",
    "3ptb",
    "2lzm",
    "1stn",
    "1aki",
    "1bni",
    "1lb5",
    "2ptc",
    "1hho",
    "3ssi",
    "1bpi",
]


def chains_of(path):
    seen = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and len(line) > 21:
                c = line[21]
                if c not in seen:
                    seen.append(c)
    return seen


def pair(a_chain, a_res, b_chain, b_res):
    """Residue pair as a sorted, orientation-free key."""
    lo, hi = sorted([(a_chain, a_res), (b_chain, b_res)])
    return [lo[0], lo[1], hi[0], hi[1]]


def profile(path, chain):
    """Every intra-chain interaction PLIP finds in `chain`, by residue pair."""
    config.INTRA = chain
    config.NOHYDRO = True  # deposited files here carry no polar hydrogens
    complex_ = PDBComplex()
    complex_.load_pdb(str(path))
    complex_.analyze()

    out = {"salt_bridges": [], "pi_stacking": [], "cation_pi": []}
    for site in complex_.interaction_sets.values():
        for s in site.saltbridge_lneg + site.saltbridge_pneg:
            out["salt_bridges"].append(pair(chain, int(s.resnr), chain, int(s.resnr_l)))
        for s in site.pistacking:
            out["pi_stacking"].append(pair(chain, int(s.resnr), chain, int(s.resnr_l)))
        for s in site.pication_laro + site.pication_paro:
            out["cation_pi"].append(pair(chain, int(s.resnr), chain, int(s.resnr_l)))

    # PLIP lists each interaction from both sides; collapse to unique residue pairs, and drop
    # self-pairs, which Proteus excludes by construction.
    for k, v in out.items():
        uniq = {tuple(p) for p in v if not (p[0] == p[2] and p[1] == p[3])}
        out[k] = sorted(list(p) for p in uniq)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:] or TARGETS
    for pdb_id in ids:
        path = ROOT / "validate" / "corpus" / f"{pdb_id}.pdb"
        if not path.exists():
            print(f"skip {pdb_id}: not fetched", file=sys.stderr)
            continue
        merged = {"id": pdb_id, "salt_bridges": [], "pi_stacking": [], "cation_pi": []}
        for chain in chains_of(path):
            try:
                found = profile(path, chain)
            except Exception as exc:  # noqa: BLE001 - a chain PLIP cannot parse is recorded, not fatal
                print(f"  {pdb_id} chain {chain}: PLIP failed ({exc})", file=sys.stderr)
                continue
            for key in merged:
                if key != "id":
                    merged[key].extend(found[key])
        for key in merged:
            if key != "id":
                merged[key] = sorted({tuple(p) for p in merged[key]})
                merged[key] = [list(p) for p in merged[key]]
        dest = OUT / f"{pdb_id}.json"
        dest.write_text(json.dumps(merged, indent=1))
        print(
            f"{pdb_id}: {len(merged['salt_bridges'])} salt bridges, "
            f"{len(merged['pi_stacking'])} pi-pi, {len(merged['cation_pi'])} cation-pi "
            f"-> {dest.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
