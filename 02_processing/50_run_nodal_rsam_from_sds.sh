#!/bin/bash
#
# run_nodal_rsam.sh
#
# Compute RSAM for the karst nodal archive using FLOVOpy.
#
# Usage:
#   ./run_nodal_rsam.sh
#
# or edit the variables below.
#

set -euo pipefail

# ----------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------

SDS_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_sds"

OUT_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_rsam"

START="2026-05-16T00:00:00"
END="2026-05-20T00:00:00"

NETWORK="T1"
STATION="*"
#STATION="05726"
LOCATION="N*"
LOCATION="*"
CHANNEL="*Z"
#CHANNEL="*"

CHUNK_HOURS=1
SAMPLING_INTERVAL=60

# ----------------------------------------------------------------------
# Create output directories
# ----------------------------------------------------------------------

mkdir -p "${OUT_ROOT}"

#OUT_DIR="${OUT_ROOT}/${NETWORK}_Z"


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

python 50_compute_nodal_rsam_from_sds_v2.py \
    --sds-root "${SDS_ROOT}" \
    --out-dir "${OUT_ROOT}" \
    --network "${NETWORK}" \
    --station "${STATION}" \
    --location "${LOCATION}" \
    --channel "${CHANNEL}" \
    --start "${START}" \
    --end "${END}" \
    --chunk-hours "${CHUNK_HOURS}" \
    --sampling-interval "${SAMPLING_INTERVAL}" \
    --primary-filter 5 240 \
    --band LOW:5-20 \
    --band MID:20-80 \
    --band HIGH:80-240 \


echo
echo "Finished."
echo "Output directory:"
echo "  ${OUT_ROOT}"
