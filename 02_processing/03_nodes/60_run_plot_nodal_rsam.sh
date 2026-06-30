#!/bin/bash
#
# run_plot_nodal_rsam.sh
#

set -euo pipefail

SCRIPT="60_plot_nodal_rsam.py"

SAM_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_rsam"

START="2026-05-16T00:00:00"
END="2026-05-20T00:00:00"

NETWORKS=("T1" "T3")
LOCATIONS=("N1" "N2" "N3")
CHANNELS=("DPE" "DPN" "DPZ" "GPZ")

PLOT_DIR="${SAM_ROOT}/plots"

mkdir -p "${PLOT_DIR}"

for NETWORK in "${NETWORKS[@]}"; do
  for LOCATION in "${LOCATIONS[@]}"; do
    for CHANNEL in "${CHANNELS[@]}"; do

      echo
      echo "============================================================"
      echo "Plotting RSAM:"
      echo "  Network : ${NETWORK}"
      echo "  Location: ${LOCATION}"
      echo "  Channel : ${CHANNEL}"
      echo "============================================================"

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
done

echo
echo "Finished plotting:"
echo "  ${PLOT_DIR}"
