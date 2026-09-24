#!/usr/bin/env bash
# The home screen: bare `proteus` in a terminal. Recorded by docs/media/record.sh home, which
# drives it with home.keys. Jobs come from a throwaway home seeded here, folded by the ESM Atlas
# API (network), so the table shows real predictions.
source "$(dirname "$0")/_lib.sh"
export RUST_LOG=off
demo_home home
for seq in $'>ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG' \
           $'>trp-cage\nNLYIQWLKDGGPSSGRPPPS'; do
    proteus submit --fasta "$seq" --runner esm-api >/dev/null 2>&1
done
mkdir -p "$HOME/models"
for f in 1crn.pdb 1gb1.pdb 1l2y.pdb 1ubq.cif 1pgb.pdb 2ptc.cif 4hhb.cif 1aki.pdb 1mbn.pdb \
         1stn.pdb 2lzm.pdb 1tim.cif af-p42212.cif af-p69905.cif af-p04637.cif; do
    cp "$REPO/validate/corpus/$f" "$HOME/models/"
done
cd "$HOME/models"
clear
type_out "proteus"
proteus
