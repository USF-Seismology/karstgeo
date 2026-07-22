#!/bin/bash
#
# run_audit_nodal_sds_availability.sh
#
# Example wrapper for auditing the Karst Geophysics nodal SDS archive.
#

#set -euo pipefail

SCRIPT="audit_nodal_sds_availability.py"

SDS_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_sds"
OUT_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_qc"

mkdir -p "${OUT_ROOT}"

# End date is exclusive. This includes May 16, 17, 18, 19, and 20.
START="2026-05-16"
END="2026-05-21"

# Vertical 500 Hz + 1000 Hz channels for T1, all nodal deployments.
python "${SCRIPT}" \
  --sds-root "${SDS_ROOT}" \
  --start "${START}" \
  --end "${END}" \
  --network T1 \
  --location 'N*' \
  --channel 'DPZ,GPZ' \
  --out-prefix "${OUT_ROOT}/T1_Z_availability" \
  --plot

# Uncomment for all three components:
# python "${SCRIPT}" \
#   --sds-root "${SDS_ROOT}" \
#   --start "${START}" \
#   --end "${END}" \
#   --network T1,T3 \
#   --location 'N*' \
#   --channel 'DP*,GP*' \
#   --out-prefix "${OUT_ROOT}/all_nodes_all_components_availability" \
#   --plot
