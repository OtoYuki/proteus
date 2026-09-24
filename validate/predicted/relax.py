#!/usr/bin/env python3
"""AlphaFold2-style restrained energy minimisation of a predicted model.

Usage: relax.py <in.pdb> <out.pdb>

Mirrors the AF2 `AmberRelaxation` protocol (Jumper et al. 2021, supplementary 1.8.6;
alphafold/relax/amber_minimize.py): hydrogens added, Amber99SB in vacuum without cutoffs,
harmonic restraints of 10 kcal/mol/Å² on every heavy atom to its predicted position, and
L-BFGS minimisation with AF2's tolerance of 2.39 (kcal/mol; OpenMM 8 interprets the tolerance
as a force, so it is given here as 2.39 kcal/mol/Å). One pass, without AF2's outer loop that
repeats the minimisation while violations remain.

The minimised heavy-atom coordinates are written back into the input's own ATOM records, so
atom order, names and the B-factor (pLDDT) column are unchanged and only x, y, z move.

Needs OpenMM and PDBFixer (not part of validate/requirements.txt; any Python with
`pip install openmm pdbfixer` will do). This is how validate/predicted/esmfold_relaxed/ was
made.
"""
import io
import sys

import openmm
from openmm import app, unit
from pdbfixer import PDBFixer

STIFFNESS = 10.0 * unit.kilocalories_per_mole / unit.angstroms**2
TOLERANCE = 2.39 * unit.kilocalories_per_mole / unit.angstrom


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    fixer = PDBFixer(pdbfile=io.StringIO(text))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = app.ForceField("amber99sb.xml")
    system = forcefield.createSystem(
        fixer.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", STIFFNESS.value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2))
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
        if atom.element.symbol != "H":
            restraint.addParticle(atom.index, pos.value_in_unit(unit.nanometer))
    system.addForce(restraint)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(fixer.topology, system, integrator, platform)
    simulation.context.setPositions(fixer.positions)
    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(tolerance=TOLERANCE, maxIterations=0)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    new = {}
    for atom in fixer.topology.atoms():
        if atom.element.symbol == "H":
            continue
        res = atom.residue
        key = (res.chain.id, int(res.id), res.insertionCode.strip(), atom.name)
        new[key] = positions[atom.index]

    out = []
    moved = 0
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], int(line[22:26]), line[26].strip(), line[12:16].strip())
            if key in new:
                x, y, z = new[key]
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                moved += 1
        out.append(line)
    open(dst, "w").write("\n".join(out) + "\n")
    e1 = state.getPotentialEnergy()
    print(f"{src}: {moved} atoms, energy {e0.value_in_unit(unit.kilocalories_per_mole):.0f} -> "
          f"{e1.value_in_unit(unit.kilocalories_per_mole):.0f} kcal/mol")


if __name__ == "__main__":
    main()
