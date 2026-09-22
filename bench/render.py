#!/usr/bin/env python3
"""Render bench/README.md from bench/results/{rust,python}_<host>.json (same host)."""
import json
import pathlib
import platform
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
host = sys.argv[1] if len(sys.argv) > 1 else platform.node()
rust = json.loads((ROOT / "bench/results" / f"rust_{host}.json").read_text())
py = json.loads((ROOT / "bench/results" / f"python_{host}.json").read_text())

# (row label, proteus criterion group, baseline key, baseline label)
ROWS = [
    ("SASA, Shrake–Rupley 960 pts/atom", "sasa_shrake_rupley_960", "mdtraj_sasa_960", "mdtraj (C++)"),
    ("SASA, Shrake–Rupley 96 vs 100 pts/atom", "sasa_shrake_rupley_96", "biopython_sasa_sr_100", "Biopython (pure Python)"),
    ("SASA, Shrake–Rupley 96 vs 100 pts/atom", "sasa_shrake_rupley_96", "freesasa_sr_100", "FreeSASA (C)"),
    ("SASA, Shrake–Rupley 960 vs Lee–Richards", "sasa_shrake_rupley_960", "freesasa_lr", "FreeSASA L&R (C)"),
    ("DSSP 8-state", "dssp", "mdtraj_dssp", "mdtraj (C++)"),
    ("φ/ψ + MolProbity Ramachandran vs φ/ψ only", "phi_psi_ramachandran", "mdtraj_phi_psi", "mdtraj `compute_phi`/`compute_psi` (Python API; rebuilds atom indices per call)"),
    ("φ/ψ + MolProbity Ramachandran vs φ/ψ only", "phi_psi_ramachandran", "biopython_phi_psi", "Biopython (pure Python)"),
]


def fmt(sec):
    if sec is None:
        return "–"
    if sec < 1e-3:
        return f"{sec*1e6:.0f} µs"
    if sec < 1:
        return f"{sec*1e3:.2f} ms"
    return f"{sec:.2f} s"


lines = ["# Benchmarks", "",
         f"Measured on `{rust['host']}` ({rust['cpu']}), `{rust['rustc']}`, Python {py['python']}, mdtraj {py['mdtraj']}.",
         "Rust: criterion median (`cargo bench -p proteus-core`). Python: median of 5 calls after a warm-up, metric only",
         "(parsing excluded on both sides), on protein heavy atoms. Same structure files, same metric definition.",
         "Regenerate with `bench/run.sh`; raw numbers in `bench/results/`.", ""]
structures = [s for s in ["1crn", "1ubq", "4hhb", "6vxx"] if s in rust["results"]]
NOTES = {
    "mdtraj_phi_psi": "The mdtraj row measures its public per-call API, most of which is Python-side index building at these sizes; it is not a comparison of dihedral kernels.",
}
for label, group, key, base_label in ROWS:
    lines += [f"### {label}", ""]
    if key in NOTES:
        lines += [NOTES[key], ""]
    lines += [f"| structure | atoms | residues | proteus | {base_label} | ratio |", "|---|---|---|---|---|---|"]
    for s in structures:
        r = rust["results"][s].get(group)
        p = py["results"].get(s, {})
        b = p.get(key)
        ratio = f"{b / r:.1f}×" if (r and b) else "–"
        lines.append(f"| {s} | {p.get('n_atoms','–')} | {p.get('n_residues','–')} | {fmt(r)} | {fmt(b)} | {ratio} |")
    lines.append("")
lines += ["### Proteus-only kernels (no like-for-like baseline)", "",
          "| structure | heavy-atom overlap | interaction network | Kabsch RMSD (Cα) | full profile |", "|---|---|---|---|---|"]
for s in structures:
    r = rust["results"][s]
    lines.append(f"| {s} | {fmt(r.get('steric_overlap'))} | {fmt(r.get('interaction_network'))} | {fmt(r.get('kabsch_rmsd'))} | {fmt(r.get('full_profile'))} |")
lines += ["", "Ratios > 1 mean Proteus is faster. Where the point counts differ (96 vs 100, 960 vs Lee–Richards) or",
          "Proteus does more work (φ/ψ **plus** Top8000 Ramachandran scoring vs φ/ψ only), the row label says so.",
          "The interaction network and overlap score have no drop-in equivalent in mdtraj/Biopython (mdtraj's",
          "`baker_hubbard` needs explicit hydrogens), so they are reported without a ratio.", ""]
# Everything from the first hand-written `## ` section onward (ProteinGym, storage, …) is kept
# verbatim; only the kernel tables above it are regenerated.
readme = ROOT / "bench" / "README.md"
tail = ""
if readme.exists():
    old = readme.read_text()
    idx = old.find("\n## ")
    if idx != -1:
        tail = old[idx:]
readme.write_text("\n".join(lines).rstrip("\n") + "\n" + tail)
print("wrote bench/README.md")
