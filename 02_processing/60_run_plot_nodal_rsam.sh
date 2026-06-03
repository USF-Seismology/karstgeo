#!/bin/bash
#
# run_plot_nodal_rsam.sh
#
# Plot RSAM data already written by compute_nodal_rsam_from_sds_v2.py.
#

set -euo pipefail

SCRIPT="60_plot_nodal_rsam.py"

SAM_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_rsam"

NETWORK="T1"
CHANNEL="DPZ"
CHANNEL="*"

START="2026-05-16T00:00:00"
END="2026-05-20T00:00:00"

#SAM_DIR="${SAM_ROOT}"#/${NETWORK}_${LOCATION}_${CHANNEL}"
PLOT_DIR="${SAM_ROOT}/plots"

for LOCATION in "N1" "N2" "N3"; do
  for CHANNEL in "GPZ" "DPZ"; do
    #SAM_DIR="${SAM_ROOT}/${NETWORK}_${LOCATION}_${CHANNEL}"
    PLOT_DIR="${SAM_ROOT}/plots"

    echo
    echo "Plotting RSAM for ${NETWORK} ${LOCATION} ${CHANNEL}:"
    #echo "  SAM directory: ${SAM_DIR}"
    echo "  Plot directory: ${PLOT_DIR}"

    mkdir -p "${PLOT_DIR}"

    python "${SCRIPT}" \
      --sam-dir "${SAM_ROOT}" \
      --plot-dir "${PLOT_DIR}" \
      --network "${NETWORK}" \
      --location "${LOCATION}" \
      --channel "${CHANNEL}" \
      --start "${START}" \
      --end "${END}" \
      --sampling-interval 60 \
      --ext csv \
      --metrics mean LOW_5_20 MID_20_80 HIGH_80_240 \
      --kind line \
      --equal-scale \
      --outfile-prefix "${NETWORK}_${LOCATION}_${CHANNEL}_rsam"
  done
done
echo
echo "Finished plotting:"
echo "  ${PLOT_DIR}"
