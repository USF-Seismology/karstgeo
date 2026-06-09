#!/usr/bin/env bash
set -euo pipefail

SDS_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_sds"
OUT_ROOT="/Volumes/tachyon/LBSSP_DATA/nodal_rsam"

START="2026-05-16T00:00:00"
END="2026-05-20T00:00:00"

STATION="*"
CHANNELS=("DPZ" "DPN" "DPE")

CHUNK_HOURS=1
SAMPLING_INTERVAL=60

mkdir -p "${OUT_ROOT}"

for NETWORK in T1 T3; do

  if [[ "${NETWORK}" == "T1" ]]; then
    LOCATIONS=("N1" "N2" "N3")
  elif [[ "${NETWORK}" == "T3" ]]; then
    LOCATIONS=("N4")
  fi

  for LOCATION in "${LOCATIONS[@]}"; do
    for CHANNEL in "${CHANNELS[@]}"; do

      echo
      echo "============================================================"
      echo "Computing RSAM: network=${NETWORK}, location=${LOCATION}, channel=${CHANNEL}"
      echo "============================================================"

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
        --no-default-bands \
        --primary-filter 4 240 \
        --band B4_8:4-8 \
        --band B8_16:8-16 \
        --band B16_32:16-32 \
        --band B32_64:32-64 \
        --band B64_128:64-128 \
        --band B128_240:128-240 \
        --ext csv

    done
  done
done