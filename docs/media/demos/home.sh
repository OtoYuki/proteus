#!/usr/bin/env bash
# The home screen: bare `proteus` in a terminal. Recorded by docs/media/record.sh home, which
# drives it with home.keys. Jobs come from a throwaway data directory seeded here.
source "$(dirname "$0")/_lib.sh"
export PATH="$PWD/target/release:$PATH" RUST_LOG=off
export PROTEUS_DATA_DIR="${TMPDIR:-/tmp}/proteus-home-demo"
rm -rf "$PROTEUS_DATA_DIR"
for seq in $'>ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG' \
           $'>trp-cage\nNLYIQWLKDGGPSSGRPPPS'; do
    proteus submit --fasta "$seq" --runner simulated >/dev/null 2>&1
done
cd validate/corpus
clear
type_out "proteus"
proteus
