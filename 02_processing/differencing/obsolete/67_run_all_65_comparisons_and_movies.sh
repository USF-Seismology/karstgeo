#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v5.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v3.sh}"

MAX_JOBS="${MAX_JOBS:-4}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_REAL="${RUN_REAL:-1}"
RUN_MOVIES="${RUN_MOVIES:-1}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"
MOVIE_FPS="${MOVIE_FPS:-1}"

BASE="${BASE:-/Users/thompsong/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SOURCES_GROUNDED="$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED"
CAVE_MODEL="$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_VOID_150m_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"
NO_VOID_MODEL="$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s"
DATA_DIR="$NO_VOID_MODEL/DATA"
PAR_FILE="$CAVE_MODEL/DATA/Par_file"
REAL_DIR="$BASE/04_FieldData/051826/051826_Seismics_T1"

OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/comparison_products_v4}"
MOVIE_ROOT="${MOVIE_ROOT:-$BASE/02_Modelling/Seismic/differencing/movies_v4}"
mkdir -p "$OUT_ROOT/logs" "$MOVIE_ROOT"

[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE"; exit 1; }
if [[ "$RUN_MOVIES" == "1" ]]; then
  [[ -f "$MOVIE_SCRIPT" ]] || { echo "ERROR: missing movie script: $MOVIE_SCRIPT"; exit 1; }
fi

COMMON_ARGS=(
  --par-file "$PAR_FILE"
  --cave-extent-x-m "140.5,160.0"
  --max-freq-hz "400"

  --write-individual-wiggles
  --write-overlay-wiggles
  --overlay-wiggle-scale "0.45"

  --write-trace-normalized-figures
  --trace-normalize-method "maxabs"

  --write-spectral-contours
  --spectral-contour-log10
  --spectral-contour-levels "24"

  --write-band-energy
  --band-energy-bands "10-30,30-80,80-150,150-400"
  --band-energy-window-s "0.05"
  --band-energy-step-s "0.01"
  --band-energy-normalize-per-trace
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

if [[ -n "${LIMIT:-}" ]]; then
  COMMON_ARGS+=(--limit "$LIMIT")
fi

COMPONENT_FILES=("Ux_file_single_v.su" "Uz_file_single_v.su")

component_name() {
  case "$1" in
    Ux*) echo "Ux" ;;
    Uz*) echo "Uz" ;;
    *) basename "$1" .su ;;
  esac
}

wait_for_slot() {
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 2
  done
}

run_bg() {
  local log="$1"; shift
  wait_for_slot
  mkdir -p "$(dirname "$log")"
  echo
  echo "Starting job: $log"
  (
    echo "Command:"
    printf '%q ' "$@"
    echo
    echo
    "$@"
  ) >"$log" 2>&1 &
}

run_synthetic() {
  local component_file="$1"
  local comp outdir log
  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/synthetic_cave_vs_nocave/$comp"
  log="$OUT_ROOT/logs/synthetic_cave_vs_nocave_${comp}.log"
  mkdir -p "$outdir"
  run_bg "$log" "$PYTHON" "$ENGINE" \
    --mode synthetic_vs_synthetic \
    --data-dir "$DATA_DIR" \
    --reference-dir "$CAVE_MODEL" \
    --comparison-dir "$NO_VOID_MODEL" \
    --reference-pattern "SURVEY_OUTPUT/**/$component_file" \
    --comparison-pattern "SURVEY_OUTPUT/**/$component_file" \
    --reference-label "Synthetic WITH cave/void" \
    --comparison-label "Synthetic WITHOUT cave/void" \
    --output-dir "$outdir" \
    --scale-mode none \
    --overlay-normalize pair \
    --write-diff-segy \
    "${COMMON_ARGS[@]}"
}

run_real() {
  local component_file="$1"
  local comp outdir log
  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/real_vs_synthetic_novoid/$comp"
  log="$OUT_ROOT/logs/real_vs_synthetic_novoid_${comp}.log"
  mkdir -p "$outdir"
  run_bg "$log" "$PYTHON" "$ENGINE" \
    --mode real_vs_synthetic \
    --data-dir "$DATA_DIR" \
    --real-dir "$REAL_DIR" \
    --comparison-dir "$NO_VOID_MODEL" \
    --comparison-pattern "SURVEY_OUTPUT/**/$component_file" \
    --reference-label "Real Geode" \
    --comparison-label "Synthetic NO-VOID" \
    --output-dir "$outdir" \
    --real-first-file "3005" \
    --real-last-file "3046" \
    --real-shot-first-x-m "82.5" \
    --real-shot-dx-m "2" \
    --real-shot-duplicate-x-m "102.5" \
    --real-shot-duplicate-files "3015,3016" \
    --shot-match-tolerance-m "0.05" \
    --receiver-x-min "87" \
    --receiver-x-max "158" \
    --real-first-trace-x-m "87" \
    --real-dx-m "1" \
    --tmin "0.0" \
    --tmax "0.4" \
    --comparison-time-shift-ms "-31.6" \
    --scale-mode fixed \
    --fixed-scale-factor "2.96e7" \
    --scale-tmin "0.02" \
    --scale-tmax "0.12" \
    --demean \
    --detrend \
    --taper-fraction "0.05" \
    --highpass-hz "10" \
    --filter-corners "4" \
    --zerophase \
    --overlay-normalize trace \
    "${COMMON_ARGS[@]}"
}

make_movies() {
  local root="$1"
  local label="$2"
  local log="$OUT_ROOT/logs/movies_${label}.log"
  echo
  echo "Making movies for $label"
  FPS="$MOVIE_FPS" MAP_DEPTH=2 bash "$MOVIE_SCRIPT" "$root" "$MOVIE_ROOT" >"$log" 2>&1
  echo "Movie log: $log"
}

echo "Engine:       $ENGINE"
echo "Movie script: $MOVIE_SCRIPT"
echo "OUT_ROOT:     $OUT_ROOT"
echo "MOVIE_ROOT:   $MOVIE_ROOT"
echo "MAX_JOBS:     $MAX_JOBS"
echo "LIMIT:        ${LIMIT:-none}"
echo

echo "Checking key paths:"
for path in "$DATA_DIR/STATIONS" "$DATA_DIR/SOURCES_LIST.txt" "$PAR_FILE" "$CAVE_MODEL/SURVEY_OUTPUT" "$NO_VOID_MODEL/SURVEY_OUTPUT" "$REAL_DIR"; do
  [[ -e "$path" ]] && echo "  OK      $path" || echo "  MISSING $path"
done

if [[ "$RUN_SYNTHETIC" == "1" ]]; then
  for c in "${COMPONENT_FILES[@]}"; do run_synthetic "$c"; done
fi

if [[ "$RUN_REAL" == "1" ]]; then
  for c in "${COMPONENT_FILES[@]}"; do run_real "$c"; done
fi

echo
echo "Waiting for comparison jobs..."
wait
echo "All comparison jobs complete. Logs: $OUT_ROOT/logs"

if [[ "$RUN_MOVIES" == "1" ]]; then
  [[ "$RUN_SYNTHETIC" == "1" ]] && make_movies "$OUT_ROOT/synthetic_cave_vs_nocave" "synthetic_cave_vs_nocave"
  [[ "$RUN_REAL" == "1" ]] && make_movies "$OUT_ROOT/real_vs_synthetic_novoid" "real_vs_synthetic_novoid"
fi

echo
echo "Done."
echo "Products: $OUT_ROOT"
echo "Movies:   $MOVIE_ROOT"
