#!/usr/bin/env bash
# End-to-end smoke test of the release binary: every user-facing path once, with assertions.
#
#   scripts/smoke.sh                 # host-only checks (no container runtime needed)
#   scripts/smoke.sh --tes IMAGE     # also submit a TES task that runs IMAGE through the daemon
#
# Exit status is non-zero on the first failed assertion. Nothing here is a benchmark or a claim;
# it is the checklist a release is not cut without.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${PROTEUS_BIN:-$ROOT/target/release/proteus}"
PDB="$ROOT/crates/proteus-core/tests/data/1crn.pdb"
CIF="$ROOT/crates/proteus-core/tests/data/1crn.cif"
FASTA="$ROOT/crates/proteus-core/tests/data/1crn.fasta"
WORK="$(mktemp -d)"
PORT="${SMOKE_PORT:-18999}"
TES_IMAGE=""
if [[ "${1:-}" == "--tes" ]]; then TES_IMAGE="${2:?--tes needs an image}"; fi

export PROTEUS_DATA_DIR="$WORK/data" RUST_LOG=warn
DAEMON_PID=""
cleanup() {
    [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null && wait "$DAEMON_PID" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

pass=0
check() { # check <label> <command...>  — the command's stdout is kept in $OUT
    local label="$1"; shift
    if OUT="$("$@" 2>&1)"; then pass=$((pass + 1)); printf 'ok   %s\n' "$label"
    else printf 'FAIL %s\n%s\n' "$label" "$OUT"; exit 1; fi
}
expect() { # expect <label> <pattern>  — $OUT must match the regex
    if grep -Eq -- "$2" <<<"$OUT"; then pass=$((pass + 1)); printf 'ok   %s\n' "$1"
    else printf 'FAIL %s: pattern %q not found in:\n%s\n' "$1" "$2" "$OUT"; exit 1; fi
}

[[ -x "$BIN" ]] || { echo "no binary at $BIN (cargo build --release)"; exit 1; }
check "binary runs" "$BIN" --version
expect "reports its version" '^proteus [0-9]+\.[0-9]+\.[0-9]+'

# --- analyze: PDB and mmCIF give the same numbers; experimental B-factors are not pLDDT
check "analyze pdb" "$BIN" analyze --pdb "$PDB"
expect "Rg of crambin" 'Radius of Gyration.*9\.676'
expect "no pLDDT for an X-ray structure" 'pLDDT.*n/a'
expect "salt bridge ARG17-GLU23" 'ARG17:NH2-GLU23:OE2'
check "analyze cif" "$BIN" analyze --pdb "$CIF"
expect "mmCIF gives the same Rg" 'Radius of Gyration.*9\.676'

# --- analyze, many structures: a directory becomes one row per file, in every export format
mkdir -p "$WORK/models/sub"
cp "$PDB" "$WORK/models/a.pdb"; cp "$CIF" "$WORK/models/sub/b.cif"; cp "$FASTA" "$WORK/models/notes.fasta"
gzip -c "$PDB" >"$WORK/models/c.pdb.gz"
check "analyze a directory → parquet" "$BIN" analyze "$WORK/models" --export "$WORK/qc.parquet"
expect "three structures, the FASTA skipped" '3 of 3 structures analysed'
check "qc parquet written" test -s "$WORK/qc.parquet"
check "analyze → csv" "$BIN" analyze "$WORK/models" --export "$WORK/qc.csv"
check "csv has a header and three rows" test "$(wc -l <"$WORK/qc.csv")" -eq 4
check "csv header" grep -q '^file,model,n_chains,n_residues,sequence,' "$WORK/qc.csv"
check "analyze --json" bash -c "'$BIN' analyze '$WORK/models/a.pdb' --json 2>/dev/null"
expect "json line carries the crambin sequence" '"sequence":"TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"'
expect "json: no pLDDT for an X-ray structure" '"plddt_mean":null'
check "a bad file fails the run but not the others" bash -c "! '$BIN' analyze '$WORK/models' '$FASTA' --export '$WORK/qc2.json' 2>'$WORK/err'"
check "…and the good rows are still written" grep -q '"model": "a"' "$WORK/qc2.json"
check "…and the failure is named" grep -q 'failed: .*1crn.fasta' "$WORK/err"
ln -s .. "$WORK/models/sub/loop"
check "a symlink loop is walked once" "$BIN" analyze "$WORK/models" --export "$WORK/qc3.CSV"
expect "…still three structures" '3 of 3 structures analysed'
check "…and an upper-case extension is honoured" grep -q '^file,model,' "$WORK/qc3.CSV"
rm "$WORK/models/sub/loop"
check "a closed pipe does not cost the export" bash -c "'$BIN' analyze '$WORK/models' --json --export '$WORK/piped.parquet' 2>/dev/null | head -1 >/dev/null; test -s '$WORK/piped.parquet'"

# --- mutate → screen pipeline, exports in every format, simulated runner is labelled
check "mutate alanine scan" "$BIN" mutate "$FASTA" --mode alanine --start 1 --end 5 --output "$WORK/lib.fasta"
check "library has WT + 5 variants" test "$(grep -c '^>' "$WORK/lib.fasta")" -eq 6
check "screen (simulated) → parquet" "$BIN" screen "$WORK/lib.fasta" --runner simulated --export "$WORK/screen.parquet"
expect "leaderboard is labelled simulated" 'SIMULATED'
check "parquet written" test -s "$WORK/screen.parquet"
check "mutate | screen - → csv" bash -c "'$BIN' mutate '$FASTA' --mode alanine --start 1 --end 3 | '$BIN' screen - --runner simulated --export '$WORK/screen.csv'"
check "csv has an engine column" grep -q ',engine' "$WORK/screen.csv"
check "unknown export extension is refused" bash -c "! '$BIN' screen '$WORK/lib.fasta' --runner simulated --export '$WORK/out.xlsx' 2>/dev/null"

# --- job lifecycle through the CLI
check "submit (simulated, wait)" "$BIN" submit --file "$FASTA" --runner simulated
JOB="$(grep -Eo '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' <<<"$OUT" | head -1)"
check "status" "$BIN" status "$JOB"
expect "job completed" 'Completed'
check "inspect" "$BIN" inspect "$JOB"
expect "inspect names the engine" 'simulated'
expect "inspect shows the tier row" 'Tier'
# Short IDs: the leaderboard prints 8 characters, so every command that takes a job must
# accept them, and must refuse a prefix that matches nothing rather than picking something.
check "status by short id" "$BIN" status "${JOB:0:8}"
expect "short id resolves to the same job" "$JOB"
check "inspect by short id" "$BIN" inspect "${JOB:0:8}"
check "unknown short id is refused" bash -c "! '$BIN' status deadbeef 2>/dev/null"
check "too-short prefix is refused" bash -c "! '$BIN' status ab 2>/dev/null"
# Bare `proteus` opens the home screen only at a terminal; in a script it is still usage + exit 2.
check "bare proteus outside a terminal prints usage" bash -c "'$BIN' </dev/null 2>&1 | grep -q 'Usage: proteus <COMMAND>'"
check "  and exits 2" bash -c "'$BIN' </dev/null >/dev/null 2>&1; test \$? -eq 2"

# --- viewers: every backend draws something; HTML export is our self-contained WebGL2 page
for backend in halfblock braille sixel kitty; do
    check "view --backend $backend" "$BIN" view "$PDB" --backend "$backend" --width 80 --height 24
    check "  frame is not blank ($backend)" test "$(tr -d ' \n' <<<"$OUT" | wc -c)" -gt 0
done
# Sixel must be decodable by libsixel, not merely non-empty, where libsixel is installed.
if command -v sixel2png >/dev/null; then
    "$BIN" view "$PDB" --backend sixel --width 60 --height 20 > "$WORK/out.six" 2>/dev/null
    check "libsixel decodes our sixel output" sixel2png -i "$WORK/out.six" -o "$WORK/out.png"
    check "  decoded image is non-empty" test -s "$WORK/out.png"
fi
check "view --compare" "$BIN" view "$PDB" --compare "$CIF" --backend halfblock --width 80 --height 24
check "view --html" "$BIN" view "$PDB" --html "$WORK/view.html"
check "html carries our WebGL2 viewer" grep -q "getContext('webgl2'" "$WORK/view.html"
check "  and the mesh" grep -q 'id="proteus-mesh"' "$WORK/view.html"
check "  and nothing from the network" bash -c "! grep -qE 'https?://|src=' '$WORK/view.html'"
check "view a job (C-alpha-only simulated model)" "$BIN" view "$JOB" --backend halfblock --width 80 --height 24
check "  frame is not blank" test "$(tr -d ' \n' <<<"$OUT" | wc -c)" -gt 0

# --- daemon: health, TES service-info, native API, auth
"$BIN" serve --port "$PORT" --executor host --runner simulated --auth-token smoke >"$WORK/serve.log" 2>&1 &
DAEMON_PID=$!
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.2; done
check "health is public" curl -sf "http://127.0.0.1:$PORT/health"
check "TES needs the token" bash -c "test \"\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/service-info)\" = 401"
check "service-info with token" curl -sf -H 'Authorization: Bearer smoke' "http://127.0.0.1:$PORT/v1/service-info"
expect "advertises TES 1.1" '"version":"1\.1\.0"'
check "metrics is public" curl -sf "http://127.0.0.1:$PORT/metrics"
expect "prometheus counters present" 'proteus_http_requests_total'
check "list tasks pages" curl -sf -H 'Authorization: Bearer smoke' "http://127.0.0.1:$PORT/v1/tasks?page_size=1"
expect "empty listing" '"tasks":\[\]'

if [[ -n "$TES_IMAGE" ]]; then
    kill "$DAEMON_PID"; wait "$DAEMON_PID" 2>/dev/null || true; DAEMON_PID=""
    "$BIN" serve --port "$PORT" --allow-image "$TES_IMAGE" --allow-dir "$WORK" >"$WORK/serve.log" 2>&1 &
    DAEMON_PID=$!
    for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.2; done
    cp "$PDB" "$WORK/in.pdb"
    TASK=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/tasks" -H 'content-type: application/json' -d "{
      \"name\": \"smoke\",
      \"inputs\": [{\"path\": \"/data/in.pdb\", \"url\": \"$WORK/in.pdb\"}],
      \"outputs\": [{\"path\": \"/data/out.pdb\", \"url\": \"$WORK/out.pdb\"}],
      \"executors\": [{\"image\": \"$TES_IMAGE\", \"command\": [\"cp\", \"/data/in.pdb\", \"/data/out.pdb\"], \"workdir\": \"/data\"}]
    }" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
    STATE=QUEUED
    for _ in $(seq 1 300); do
        STATE=$(curl -sf "http://127.0.0.1:$PORT/v1/tasks/$TASK?view=MINIMAL" | python3 -c 'import sys,json; print(json.load(sys.stdin)["state"])')
        case "$STATE" in COMPLETE|EXECUTOR_ERROR|SYSTEM_ERROR|CANCELED) break;; esac
        sleep 1
    done
    check "TES task in $TES_IMAGE completed" test "$STATE" = COMPLETE
    check "bare-path output delivered" cmp "$PDB" "$WORK/out.pdb"
fi

echo "smoke: $pass checks passed"
