#!/usr/bin/env bash
# Full benchmark run: criterion (Rust) + Python baselines + rendered table.
set -euo pipefail
cd "$(dirname "$0")/.."
make fetch >/dev/null
cargo bench -p proteus-core --bench biophysics
validate/.venv/bin/python bench/collect_rust.py
validate/.venv/bin/python -c "import Bio" 2>/dev/null || uv pip install --python validate/.venv/bin/python biopython
validate/.venv/bin/python bench/python_baseline.py --runs "${RUNS:-5}"
validate/.venv/bin/python bench/render.py
