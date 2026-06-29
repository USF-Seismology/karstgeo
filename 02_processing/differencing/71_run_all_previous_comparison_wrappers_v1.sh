#!/usr/bin/env bash
# 71_run_all_previous_comparison_wrappers_v1.sh
#
# One-command bundle that runs the equivalent of the previous wrapper set using
# the consolidated 70_run_comparison_suite_v2.sh driver.
#
# Equivalent legacy coverage:
#   67_run_all_65_comparisons_and_movies_v17.sh
#       -> T1 synthetic cave/no-cave
#       -> T1 real vs synthetic no-void
#   67_wrapper_wrapper_T3_T4_real_vs_synthetic_novoid_v16.sh
#       -> T3/T4 real vs common T1 synthetic no-void
#   68_wrapper_rectangle_vs_polygon_single_shot_134p5m_v11.sh
#       -> Rectangle cave vs polygon cave single shot
#   69_wrapper_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v12.sh
#       -> Frequency suite synthetic NO_VOID vs POLYGON
#       -> Frequency suite real T1 single shot vs synthetic NO_VOID
#
# Examples:
#   bash 71_run_all_previous_comparison_wrappers_v1.sh
#   LIMIT=2 RUN_MOVIES=0 bash 71_run_all_previous_comparison_wrappers_v1.sh
#   STOP_ON_ERROR=0 bash 71_run_all_previous_comparison_wrappers_v1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${DRIVER:-$SCRIPT_DIR/70_run_comparison_suite_v2.sh}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

[[ -f "$DRIVER" ]] || { echo "ERROR: missing consolidated driver: $DRIVER" >&2; exit 1; }

run_step() {
  local label="$1"; shift
  echo
  echo "======================================================================"
  echo "Running: $label"
  echo "======================================================================"
  if [[ "$STOP_ON_ERROR" == "1" ]]; then
    bash "$DRIVER" "$@"
  else
    bash "$DRIVER" "$@" || echo "WARNING: step failed: $label" >&2
  fi
}

run_step "T1 synthetic cave/no-cave" \
  --mode synthetic_cave_nocave

run_step "T1 real vs synthetic no-void" \
  --mode real_vs_synthetic --line T1

run_step "T3/T4 real vs common T1 synthetic no-void" \
  --mode real_vs_synthetic --line T3,T4

run_step "Rectangle cave vs polygon cave" \
  --mode rectangle_vs_polygon

run_step "Frequency suite: synthetic NO_VOID vs POLYGON" \
  --mode polygon_vs_novoid --frequency-suite

run_step "Frequency suite: real T1 single shot vs synthetic NO_VOID" \
  --mode real_vs_novoid --frequency-suite

echo
echo "All requested comparison suites finished."
