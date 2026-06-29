#!/usr/bin/env bash
# Simple launcher for T3/T4 real-vs-synthetic comparisons using the common T1 NO-VOID model.
# v15 uses 65_compare_gather_pairs_v16.py and shifts plot/output x coordinates back to local 0..71 m.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_JOBS="${MAX_JOBS:-4}" bash "$SCRIPT_DIR/67_run_T3_T4_real_vs_synthetic_novoid_v15.sh"
