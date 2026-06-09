#!/bin/bash
#
# run_plot_hourly_band_median_abs.sh
#

#set -euo pipefail

SCRIPT="plot_hourly_band_median_abs.py"

CSV="/Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_hourly_band_median_abs.csv"
OUT_DIR="/Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_hourly_band_plots"

python "${SCRIPT}" \
  --csv "${CSV}" \
  --out-dir "${OUT_DIR}" \
  --bands LOW_5_20_median_abs MID_20_80_median_abs HIGH_80_200_median_abs \
  --logy \
  --title-prefix "T1 vertical"

echo
echo "Finished:"
echo "  ${OUT_DIR}"
