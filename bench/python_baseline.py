#!/usr/bin/env python3
"""Wall-clock baselines for the same metrics on the same structures as the criterion benches.

Each timing is the median of `--runs` calls, after one warm-up call, of the metric alone
(file parsing is excluded, as in the Rust benches). Output: bench/results/python_<host>.json.
"""
import argparse
import json
import pathlib
import platform
import statistics
import time

import freesasa
import mdtraj as md
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from Bio.PDB.SASA import ShrakeRupley

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRUCTURES = [
    ("1crn", "crates/proteus-core/tests/data/1crn.pdb"),
    ("1ubq", "validate/corpus/1ubq.pdb"),
    ("4hhb", "validate/corpus/4hhb.pdb"),
    ("6vxx", "validate/corpus/6vxx.cif"),
]


def timeit(fn, runs):
    fn()  # warm-up
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    out = {"host": platform.node(), "python": platform.python_version(), "runs": args.runs,
           "mdtraj": md.__version__, "freesasa": freesasa.__version__ if hasattr(freesasa, "__version__") else "?",
           "cpu": platform.processor() or platform.machine(), "results": {}}
    for label, rel in STRUCTURES:
        path = ROOT / rel
        if not path.exists():
            print("skip", label)
            continue
        t = md.load(str(path))
        t = t.atom_slice(t.topology.select("protein and not element H"))
        r = {"n_atoms": t.n_atoms, "n_residues": t.n_residues}
        r["mdtraj_sasa_960"] = timeit(lambda: md.shrake_rupley(t, n_sphere_points=960), args.runs)
        r["mdtraj_sasa_96"] = timeit(lambda: md.shrake_rupley(t, n_sphere_points=96), args.runs)
        r["mdtraj_dssp"] = timeit(lambda: md.compute_dssp(t, simplified=False), args.runs)
        r["mdtraj_phi_psi"] = timeit(lambda: (md.compute_phi(t), md.compute_psi(t)), args.runs)
        r["mdtraj_rg"] = timeit(lambda: md.compute_rg(t), args.runs)
        if path.suffix == ".pdb":
            fs = freesasa.Structure(str(path), options={"hetatm": False, "hydrogen": False})
            r["freesasa_lr"] = timeit(lambda: freesasa.calc(fs), args.runs)
            r["freesasa_sr_100"] = timeit(
                lambda: freesasa.calc(fs, freesasa.Parameters({"algorithm": freesasa.ShrakeRupley, "n-points": 100})),
                args.runs)
        parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
        structure = parser.get_structure(label, str(path))
        model = structure[0]
        for chain in model:
            for res in list(chain):
                if res.id[0] != " ":
                    chain.detach_child(res.id)
        r["biopython_sasa_sr_100"] = timeit(lambda: ShrakeRupley(n_points=100).compute(model, level="A"), max(1, args.runs // 2))
        r["biopython_phi_psi"] = timeit(lambda: [pp.get_phi_psi_list() for pp in PPBuilder().build_peptides(model)], args.runs)
        out["results"][label] = r
        print(label, {k: (f"{v*1e3:.2f} ms" if isinstance(v, float) else v) for k, v in r.items()})
    dest = ROOT / "bench" / "results" / f"python_{platform.node()}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
