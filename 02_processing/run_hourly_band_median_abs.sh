#!/bin/bash
#
# run_hourly_band_median_abs.sh
#
# Example wrapper for simple hourly band-median amplitudes.
#

set -euo pipefail

SCRIPT="compute_hourly_band_median_abs.py"

SDS_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_sds"
OUT_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_qc"

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

mkdir -p "${OUT_ROOT}"

python "${SCRIPT}" \
  --sds-root "${SDS_ROOT}" \
  --network "${NETWORK}" \
  --station "${STATION}" \
  --location "${LOCATION}" \
  --channel "${CHANNEL}" \
  --start "${START}" \
  --end "${END}" \
  --chunk-hours "${CHUNK_HOURS}" \
  --read-buffer-seconds 30 \
  --band LOW:5-20 \
  --band MID:20-80 \
  --band HIGH:80-200 \
  --out "${OUT_ROOT}/T1_Z_hourly_band_median_abs.csv" \
  --overwrite
