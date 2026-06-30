#!/usr/bin/env bash
# Simple launcher for T3/T4 real-vs-synthetic comparisons using the common T1 NO-VOID model.
# v16 uses 65_compare_gather_pairs_v17.py and writes processed and f-k filtered SEG-Y gathers.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_JOBS="${MAX_JOBS:-4}" bash "$SCRIPT_DIR/67_run_T3_T4_real_vs_synthetic_novoid_v16.sh"
