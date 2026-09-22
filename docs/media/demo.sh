#!/usr/bin/env bash
# Scripted terminal session recorded by docs/media/record.sh (asciinema → agg → demo.gif).
export PATH="$PWD/target/release:$PATH" RUST_LOG=off
PDB=crates/proteus-core/tests/data/1crn.pdb
FASTA=crates/proteus-core/tests/data/1crn.fasta
say() { printf '\033[1;35m$ \033[0m'; printf '%s' "$1" | pv -qL 60 2>/dev/null || printf '%s' "$1"; printf '\n'; }
clear
say "# 1. all-atom biophysics on any PDB/mmCIF — validated against mdtraj, FreeSASA, cctbx"
say "proteus analyze --pdb $PDB"
proteus analyze --pdb $PDB
sleep 4
clear
say "# 2. 3D cartoon + live Ramachandran/DSSP dashboard, in the terminal (no GPU, no X11)"
say "proteus view $PDB --interactive --dashboard --color ss"
proteus view $PDB --interactive --dashboard --color ss
clear
say "# 3. predicted models get pLDDT; experimental ones never do (provenance is detected)"
say "proteus analyze --pdb validate/corpus/af-p69905.cif | head -12"
proteus analyze --pdb validate/corpus/af-p69905.cif | head -12
sleep 3
clear
say "# 4. every number is checked against mdtraj / FreeSASA / cctbx on 43 structures, in CI"
say "make validate"
cargo test -p proteus-core --release --test validation -- --ignored --nocapture 2>/dev/null \
  | rg --color never '^\| (id|---|1crn|1ubq|4hhb|1tim|2kod|6vxx|af-p69905|af-p38398)|^test result'
sleep 4
