#!/usr/bin/env bash
# Simple launcher for T3/T4 real-vs-synthetic NO-VOID comparisons.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_JOBS="${MAX_JOBS:-4}" bash "$SCRIPT_DIR/67_run_T3_T4_real_vs_synthetic_novoid_v11.sh"
