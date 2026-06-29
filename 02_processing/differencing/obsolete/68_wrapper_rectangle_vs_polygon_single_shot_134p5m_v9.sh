#!/usr/bin/env bash
# 68_wrapper_rectangle_vs_polygon_single_shot_134p5m.sh
#
# Compare SPECFEM2D synthetic shot gathers:
#   RECTANGLE cave vs POLYGON cave
# for the single shot at x = 134.5 m.
#
# Requires same directory as:
#   65_compare_gather_pairs_v4.py
# Optional:
#   66_make_movies_from_shot_figures_v3.sh
#
# Run:
#   bash 68_wrapper_rectangle_vs_polygon_single_shot_134p5m.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v9.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v3.sh}"

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SOURCES_GROUNDED="${SOURCES_GROUNDED:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED}"

RECTANGLE_ROOT="$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_RECTANGLE_1m_50Hz_DX_DZ_0d5m_DT_1e-5s/single_shot_134d5m"
POLYGON_ROOT="$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_POLYGON_1m_50Hz_DX_DZ_0d5m_DT_1e-5s/single_shot_134d5m"

DATA_DIR="${DATA_DIR:-$POLYGON_ROOT/DATA}"
PAR_FILE="${PAR_FILE:-$POLYGON_ROOT/DATA/Par_file}"

OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/rectangle_vs_polygon_single_shot_134p5m}"
MOVIE_ROOT="${MOVIE_ROOT:-$BASE/02_Modelling/Seismic/differencing/movies_rectangle_vs_polygon_single_shot_134p5m}"

RUN_MOVIES="${RUN_MOVIES:-0}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"

mkdir -p "$OUT_ROOT/logs" "$MOVIE_ROOT"

[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE"; exit 1; }

if [[ ! -d "$DATA_DIR" ]]; then
  echo "WARNING: DATA_DIR not found: $DATA_DIR"
  echo "Trying rectangle DATA folder instead."
  DATA_DIR="$RECTANGLE_ROOT/DATA"
fi

if [[ ! -f "$PAR_FILE" ]]; then
  echo "WARNING: PAR_FILE not found: $PAR_FILE"
  echo "Trying rectangle Par_file instead."
  PAR_FILE="$RECTANGLE_ROOT/DATA/Par_file"
fi

COMMON_ARGS=(
  --mode synthetic_vs_synthetic
  --data-dir "$DATA_DIR"
  --par-file "$PAR_FILE"

  --reference-dir "$RECTANGLE_ROOT"
  --comparison-dir "$POLYGON_ROOT"
  --reference-label "Synthetic RECTANGLE cave"
  --comparison-label "Synthetic POLYGON cave"

  --scale-mode none
  --max-freq-hz "400"

  --write-diff-segy
  --write-combined-three-panel-products
  # Combined figures are preferred; standalone wiggle/overlay figures are off by default.
  --overlay-normalize pair
  --overlay-wiggle-scale "0.45"

  # Synthetic-vs-synthetic: preserve shared physical amplitudes/scales.
  # Do not independently normalize reference and comparison traces.
  --no-write-trace-normalized-figures
  --no-frequency-trace-normalization
  --no-combined-wiggle-trace-normalize

  --write-spectral-contours
  --spectral-contour-log10
  --spectral-contour-levels "24"

  --write-band-energy
  --band-energy-bands "10-30,30-80,80-150,150-400"
  --band-energy-window-s "0.05"
  --band-energy-step-s "0.01"
  --no-band-energy-normalize-per-trace

  --cave-extent-x-m "140.5,160.0"
  --limit "1"
)

if [[ "$WRITE_BANDPASS" == "1" ]]; then
  COMMON_ARGS+=(
    --write-diagnostic-bandpass
    --diagnostic-bandpass-fmin "25"
    --diagnostic-bandpass-fmax "400"
    --diagnostic-bandpass-corners "4"
    --diagnostic-bandpass-zerophase
  )
fi

if [[ "$WRITE_FK" == "1" ]]; then
  COMMON_ARGS+=(
    --write-fk-filtered
    --fk-min-velocity-mps "500"
    --fk-taper-width-mps "100"
    --fk-use-taper
  )
fi

COMPONENT_FILES=("Ux_file_single_v.su" "Uz_file_single_v.su")

component_name() {
  case "$1" in
    Ux*) echo "Ux" ;;
    Uz*) echo "Uz" ;;
    *) basename "$1" .su ;;
  esac
}

run_component() {
  local component_file="$1"
  local comp outdir log ref_file cmp_file
  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/$comp"
  log="$OUT_ROOT/logs/rectangle_vs_polygon_${comp}.log"
  mkdir -p "$outdir"

  ref_file="$RECTANGLE_ROOT/SURVEY_OUTPUT/shot_001_xs00134p5/$component_file"
  cmp_file="$POLYGON_ROOT/SURVEY_OUTPUT/shot_001_xs00134p5/$component_file"

  echo
  echo "================================================================================"
  echo "Running rectangle vs polygon comparison for $comp"
  echo "================================================================================"
  echo "Reference:  $ref_file"
  echo "Comparison: $cmp_file"
  echo "Output:     $outdir"
  echo "Log:        $log"

  if [[ ! -f "$ref_file" ]]; then
    echo "ERROR: missing reference component file for $comp: $ref_file" | tee "$log"
    return 10
  fi

  if [[ ! -f "$cmp_file" ]]; then
    echo "ERROR: missing comparison component file for $comp: $cmp_file" | tee "$log"
    return 11
  fi

  (
    echo "Command:"
    printf '%q ' "$PYTHON" "$ENGINE" \
      "${COMMON_ARGS[@]}" \
      --reference-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$component_file" \
      --comparison-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$component_file" \
      --output-dir "$outdir"
    echo
    echo

    "$PYTHON" "$ENGINE" \
      "${COMMON_ARGS[@]}" \
      --reference-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$component_file" \
      --comparison-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$component_file" \
      --output-dir "$outdir"
  ) >"$log" 2>&1

  local status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "Finished $comp"
  else
    echo "FAILED $comp with status $status; see $log"
  fi
  return "$status"
}

echo "Engine:         $ENGINE"
echo "Rectangle root: $RECTANGLE_ROOT"
echo "Polygon root:   $POLYGON_ROOT"
echo "DATA_DIR:       $DATA_DIR"
echo "PAR_FILE:       $PAR_FILE"
echo "OUT_ROOT:       $OUT_ROOT"
echo

echo "Checking paths:"
for p in \
  "$RECTANGLE_ROOT/DATA" \
  "$RECTANGLE_ROOT/SURVEY_OUTPUT/shot_001_xs00134p5" \
  "$POLYGON_ROOT/DATA" \
  "$POLYGON_ROOT/SURVEY_OUTPUT/shot_001_xs00134p5" \
  "$DATA_DIR/STATIONS" \
  "$DATA_DIR/SOURCES_LIST.txt" \
  "$PAR_FILE"; do
  [[ -e "$p" ]] && echo "  OK      $p" || echo "  MISSING $p"
done

STATUS=0
for component_file in "${COMPONENT_FILES[@]}"; do
  if ! run_component "$component_file"; then
    STATUS=1
  fi
done

if [[ "$STATUS" -ne 0 ]]; then
  echo
  echo "WARNING: one or more component runs failed."
  echo "Check logs in: $OUT_ROOT/logs"
fi

if [[ "$RUN_MOVIES" == "1" ]]; then
  if [[ -f "$MOVIE_SCRIPT" ]]; then
    echo
    echo "Making movies. Since this is a single shot, forcing MIN_FRAMES=1."
    MIN_FRAMES=1 FPS="${MOVIE_FPS:-1}" MAP_DEPTH=1 bash "$MOVIE_SCRIPT" "$OUT_ROOT" "$MOVIE_ROOT"
  else
    echo "WARNING: movie script not found: $MOVIE_SCRIPT"
  fi
fi

echo
echo "Done."
echo "Products: $OUT_ROOT"
echo "Logs:     $OUT_ROOT/logs"

exit "$STATUS"
