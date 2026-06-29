#!/usr/bin/env bash
# 70_run_comparison_suite_v2.sh
#
# Consolidated driver for karst seismic gather comparisons using
# 65_compare_gather_pairs_v17.py.
#
# Replaces/absorbs the common use cases from:
#   67_run_all_65_comparisons_and_movies_v17.sh
#   67_run_T3_T4_real_vs_synthetic_novoid_v16.sh
#   68_wrapper_rectangle_vs_polygon_single_shot_134p5m_v11.sh
#   69_wrapper_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v12.sh
#
# Usage examples:
#   bash 70_run_comparison_suite_v2.sh t1
#   bash 70_run_comparison_suite_v2.sh t3t4
#   bash 70_run_comparison_suite_v2.sh rectpoly
#   bash 70_run_comparison_suite_v2.sh freq
#   bash 70_run_comparison_suite_v2.sh all
#
# Common controls:
#   BASE=/path/to/2026KarstGeophysicsDEP
#   ENGINE=/path/to/65_compare_gather_pairs_v17.py
#   MOVIE_SCRIPT=/path/to/66_make_movies_from_shot_figures_v3.sh
#   MAX_JOBS=4
#   RUN_MOVIES=0|1
#   WRITE_FK=0|1
#   WRITE_BANDPASS=0|1
#   LIMIT=2
#   COMPONENTS="Ux_file_single_v.su Uz_file_single_v.su"
#
# Notes:
# - This script is intentionally bash-3.2 compatible for macOS /bin/bash.
# - It keeps experiment-specific output folders but centralizes the 65_* options.

set -euo pipefail

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
# Supports both legacy compact commands and a clearer option interface.
#
# Legacy examples:
#   bash 70_run_comparison_suite_v2.sh t1
#   bash 70_run_comparison_suite_v2.sh t3t4
#   bash 70_run_comparison_suite_v2.sh rectpoly
#   bash 70_run_comparison_suite_v2.sh freq
#
# Preferred examples:
#   bash 70_run_comparison_suite_v2.sh --mode real_vs_synthetic --line T1
#   bash 70_run_comparison_suite_v2.sh --mode real_vs_synthetic --line T3,T4
#   bash 70_run_comparison_suite_v2.sh --mode rectangle_vs_polygon
#   bash 70_run_comparison_suite_v2.sh --mode polygon_vs_novoid --frequency-suite
#   bash 70_run_comparison_suite_v2.sh --mode real_vs_novoid --frequency-suite

COMMAND=""
MODE=""
LINE=""
FREQUENCY_SUITE=0
RUN_BUNDLE=""

if [[ $# -eq 0 ]]; then
  COMMAND="help"
elif [[ "${1:-}" != --* ]]; then
  COMMAND="$1"
  shift || true
else
  COMMAND="options"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --line|--lines) LINE="${2:-}"; shift 2 ;;
    --frequency-suite|--freq-suite) FREQUENCY_SUITE=1; shift ;;
    --bundle) RUN_BUNDLE="${2:-}"; shift 2 ;;
    --help|-h) COMMAND="help"; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$COMMAND" == "options" ]]; then
  if [[ -n "$RUN_BUNDLE" ]]; then
    COMMAND="$RUN_BUNDLE"
  elif [[ "$MODE" == "real_vs_synthetic" ]]; then
    case "${LINE//,/ }" in
      T1|t1) COMMAND="real_t1" ;;
      T3|t3) COMMAND="real_t3" ;;
      T4|t4) COMMAND="real_t4" ;;
      *T3*T4*|*t3*t4*) COMMAND="real_t3t4" ;;
      *) echo "ERROR: --mode real_vs_synthetic requires --line T1, T3, T4, or T3,T4" >&2; exit 2 ;;
    esac
  elif [[ "$MODE" == "t1_synthetic_cave_nocave" || "$MODE" == "synthetic_cave_nocave" ]]; then
    COMMAND="t1_synthetic_cave_nocave"
  elif [[ "$MODE" == "rectangle_vs_polygon" ]]; then
    COMMAND="rectangle_vs_polygon"
  elif [[ "$MODE" == "polygon_vs_novoid" && "$FREQUENCY_SUITE" == "1" ]]; then
    COMMAND="polygon_novoid_frequency"
  elif [[ "$MODE" == "real_vs_novoid" && "$FREQUENCY_SUITE" == "1" ]]; then
    COMMAND="real_frequency"
  else
    echo "ERROR: unsupported option combination: --mode '$MODE' --line '$LINE' frequency_suite=$FREQUENCY_SUITE" >&2
    exit 2
  fi
fi

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v17.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v3.sh}"

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SPECFEM_ROOT="${SPECFEM_ROOT:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation}"
SOURCES_GROUNDED="${SOURCES_GROUNDED:-$SPECFEM_ROOT/SOURCES_GROUNDED}"
FREQ_SIM_ROOT="${FREQ_SIM_ROOT:-$SPECFEM_ROOT/FREQUENCY_TEST_9_LAYER_MODEL}"

MAX_JOBS="${MAX_JOBS:-4}"
RUN_MOVIES="${RUN_MOVIES:-0}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"
MOVIE_FPS="${MOVIE_FPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"

# Default outputs. Individual modes append a stable subfolder.
OUT_BASE="${OUT_BASE:-$BASE/02_Modelling/Seismic/differencing}"
MOVIE_BASE="${MOVIE_BASE:-$BASE/02_Modelling/Seismic/differencing}"

COMPONENTS_TEXT="${COMPONENTS:-Ux_file_single_v.su Uz_file_single_v.su}"
COMPONENT_FILES=()
for _c in $COMPONENTS_TEXT; do
  [[ -n "$_c" ]] && COMPONENT_FILES+=("$_c")
done
unset _c

# Frequency suite components: synthetic runs usually use both; real-vs-freq defaults to vertical.
SYN_FREQ_COMPONENTS_TEXT="${SYN_FREQ_COMPONENTS:-Ux_file_single_v.su Uz_file_single_v.su}"
SYN_FREQ_COMPONENT_FILES=()
for _c in $SYN_FREQ_COMPONENTS_TEXT; do
  [[ -n "$_c" ]] && SYN_FREQ_COMPONENT_FILES+=("$_c")
done
unset _c
REAL_FREQ_COMPONENTS_TEXT="${REAL_FREQ_COMPONENTS:-Uz_file_single_v.su}"
REAL_FREQ_COMPONENT_FILES=()
for _c in $REAL_FREQ_COMPONENTS_TEXT; do
  [[ -n "$_c" ]] && REAL_FREQ_COMPONENT_FILES+=("$_c")
done
unset _c

[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE" >&2; exit 1; }
if [[ "$RUN_MOVIES" == "1" && ! -f "$MOVIE_SCRIPT" ]]; then
  echo "ERROR: missing movie script: $MOVIE_SCRIPT" >&2
  exit 1
fi

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
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN=1; command not executed."
    else
      "$@"
    fi
  ) >"$log" 2>&1 &
}

make_movies_with_depth() {
  local root="$1" movie_root="$2" label="$3" depth="$4" log_root="$5"
  [[ "$RUN_MOVIES" == "1" ]] || return 0
  mkdir -p "$movie_root" "$log_root/logs"
  local log="$log_root/logs/movies_${label}.log"
  echo
  echo "Making movies for $label"
  FPS="$MOVIE_FPS" MAP_DEPTH="$depth" MIN_FRAMES="${MIN_MOVIE_FRAMES:-1}" bash "$MOVIE_SCRIPT" "$root" "$movie_root" >"$log" 2>&1 || {
    echo "WARNING: movie generation failed for $label; see $log" >&2
    return 1
  }
  echo "Movie log: $log"
}

# Shared options for products/diagnostics. This appends into COMMON_ARGS.
build_product_args() {
  COMMON_ARGS=(
    --max-freq-hz "${MAX_FREQ_HZ:-400}"
    --write-combined-three-panel-products
    --write-gather-segys
    --gather-segy-processed-dirname "${GATHER_SEGY_PROCESSED_DIRNAME:-processed}"
    --overlay-wiggle-scale "${OVERLAY_WIGGLE_SCALE:-0.45}"
    --write-spectral-contours
    --spectral-contour-log10
    --spectral-contour-levels "${SPECTRAL_CONTOUR_LEVELS:-24}"
    --write-band-energy
    --band-energy-bands "${BAND_ENERGY_BANDS:-10-30,30-80,80-150,150-400}"
    --band-energy-window-s "${BAND_ENERGY_WINDOW_S:-0.05}"
    --band-energy-step-s "${BAND_ENERGY_STEP_S:-0.01}"
  )
  if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
  fi
  if [[ "$WRITE_BANDPASS" == "1" ]]; then
    COMMON_ARGS+=(
      --write-diagnostic-bandpass
      --diagnostic-bandpass-fmin "${DIAGNOSTIC_BANDPASS_FMIN:-25}"
      --diagnostic-bandpass-fmax "${DIAGNOSTIC_BANDPASS_FMAX:-400}"
      --diagnostic-bandpass-corners "${DIAGNOSTIC_BANDPASS_CORNERS:-4}"
      --diagnostic-bandpass-zerophase
    )
  fi
  if [[ "$WRITE_FK" == "1" ]]; then
    COMMON_ARGS+=(
      --write-fk-filtered
      --write-fk-gather-segys
      --fk-min-velocity-mps "${FK_MIN_VELOCITY_MPS:-500}"
      --fk-taper-width-mps "${FK_TAPER_WIDTH_MPS:-100}"
      --fk-use-taper
      --fk-split-at-source
      --fk-spatial-taper-fraction "${FK_SPATIAL_TAPER_FRACTION:-0.05}"
      --fk-pad-factor "${FK_PAD_FACTOR:-2}"
    )
  fi
}

synthetic_norm_args() {
  NORM_ARGS=(
    --overlay-normalize pair
    --no-write-trace-normalized-figures
    --no-frequency-trace-normalization
    --no-band-energy-normalize-per-trace
    --no-combined-wiggle-trace-normalize
    --normalize-synthetic-source-factor
    --synthetic-source-target-factor "${SYNTHETIC_SOURCE_TARGET_FACTOR:-1e10}"
  )
}

real_norm_args() {
  NORM_ARGS=(
    --overlay-normalize trace
    --write-trace-normalized-figures
    --trace-normalize-method "${TRACE_NORMALIZE_METHOD:-maxabs}"
    --frequency-trace-normalization
    --band-energy-normalize-per-trace
    --combined-wiggle-trace-normalize
    --normalize-synthetic-source-factor
    --synthetic-source-target-factor "${SYNTHETIC_SOURCE_TARGET_FACTOR:-1e10}"
  )
}

real_processing_args() {
  PROCESSING_ARGS=(
    --tmin "${TMIN:-0.0}"
    --tmax "${TMAX:-0.4}"
    --comparison-time-shift-ms "${COMPARISON_TIME_SHIFT_MS:--31.6}"
    --scale-mode "${SCALE_MODE:-fixed}"
    --fixed-scale-factor "${FIXED_SCALE_FACTOR:-2.96e7}"
    --scale-tmin "${SCALE_TMIN:-0.02}"
    --scale-tmax "${SCALE_TMAX:-0.12}"
    --demean
    --detrend
    --taper-fraction "${TAPER_FRACTION:-0.05}"
    --highpass-hz "${HIGHPASS_HZ:-10}"
    --filter-corners "${FILTER_CORNERS:-4}"
    --zerophase
  )
}

# ----------------------------------------------------------------------------
# Mode: T1 standard products
# ----------------------------------------------------------------------------
run_t1() {
  local out_root="${OUT_ROOT:-$OUT_BASE/comparison_products_v18_consolidated}"
  local movie_root="${MOVIE_ROOT:-$MOVIE_BASE/movies_v18_consolidated}"
  local cave_model="${CAVE_MODEL:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_VOID_150m_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s}"
  local no_void_model="${NO_VOID_MODEL:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s}"
  local data_dir="${DATA_DIR:-$no_void_model/DATA}"
  local par_file="${PAR_FILE:-$cave_model/DATA/Par_file}"
  local real_dir="${REAL_DIR:-$BASE/04_FieldData/051826/051826_Seismics_T1}"
  local run_synthetic="${RUN_SYNTHETIC:-1}"
  local run_real="${RUN_REAL:-1}"

  mkdir -p "$out_root/logs" "$movie_root"
  echo "Mode:          t1"
  echo "Engine:        $ENGINE"
  echo "OUT_ROOT:      $out_root"
  echo "Cave model:    $cave_model"
  echo "No-void model: $no_void_model"
  echo "Real dir:      $real_dir"

  build_product_args

  local c comp outdir log
  if [[ "$run_synthetic" == "1" ]]; then
    synthetic_norm_args
    for c in "${COMPONENT_FILES[@]}"; do
      comp="$(component_name "$c")"
      outdir="$out_root/synthetic_cave_vs_nocave/$comp"
      log="$out_root/logs/synthetic_cave_vs_nocave_${comp}.log"
      mkdir -p "$outdir"
      run_bg "$log" "$PYTHON" "$ENGINE" \
        --mode synthetic_vs_synthetic \
        --data-dir "$data_dir" \
        --par-file "$par_file" \
        --reference-dir "$cave_model" \
        --comparison-dir "$no_void_model" \
        --reference-pattern "SURVEY_OUTPUT/**/$c" \
        --comparison-pattern "SURVEY_OUTPUT/**/$c" \
        --reference-label "Synthetic WITH cave/void" \
        --comparison-label "Synthetic WITHOUT cave/void" \
        --output-dir "$outdir" \
        --scale-mode none \
        --write-diff-segy \
        --cave-extent-x-m "${T1_CAVE_EXTENT_X_M:-140.5,160.0}" \
        "${COMMON_ARGS[@]}" \
        "${NORM_ARGS[@]}"
    done
  fi

  if [[ "$run_real" == "1" ]]; then
    real_norm_args
    real_processing_args
    for c in "${COMPONENT_FILES[@]}"; do
      comp="$(component_name "$c")"
      outdir="$out_root/real_vs_synthetic_novoid/$comp"
      log="$out_root/logs/real_vs_synthetic_novoid_${comp}.log"
      mkdir -p "$outdir"
      run_bg "$log" "$PYTHON" "$ENGINE" \
        --mode real_vs_synthetic \
        --data-dir "$data_dir" \
        --par-file "$par_file" \
        --real-dir "$real_dir" \
        --comparison-dir "$no_void_model" \
        --comparison-pattern "SURVEY_OUTPUT/**/$c" \
        --reference-label "Real Geode T1" \
        --comparison-label "Synthetic NO-VOID T1" \
        --output-dir "$outdir" \
        --real-first-file "${T1_REAL_FIRST_FILE:-3005}" \
        --real-last-file "${T1_REAL_LAST_FILE:-3046}" \
        --real-shot-first-x-m "${T1_REAL_SHOT_FIRST_X_M:-82.5}" \
        --real-shot-dx-m "${T1_REAL_SHOT_DX_M:-2}" \
        --real-shot-duplicate-x-m "${T1_REAL_DUPLICATE_X_M:-102.5}" \
        --real-shot-duplicate-files "${T1_REAL_DUPLICATE_FILES:-3015,3016}" \
        --shot-match-tolerance-m "${SHOT_MATCH_TOLERANCE_M:-0.05}" \
        --receiver-x-min "${T1_RECEIVER_X_MIN:-87}" \
        --receiver-x-max "${T1_RECEIVER_X_MAX:-158}" \
        --real-first-trace-x-m "${T1_REAL_FIRST_TRACE_X_M:-87}" \
        --real-dx-m "${REAL_DX_M:-1}" \
        "${PROCESSING_ARGS[@]}" \
        "${COMMON_ARGS[@]}" \
        "${NORM_ARGS[@]}"
    done
  fi

  wait
  [[ "$run_synthetic" == "1" ]] && make_movies_with_depth "$out_root/synthetic_cave_vs_nocave" "$movie_root" "synthetic_cave_vs_nocave" 2 "$out_root" || true
  [[ "$run_real" == "1" ]] && make_movies_with_depth "$out_root/real_vs_synthetic_novoid" "$movie_root" "real_vs_synthetic_novoid" 2 "$out_root" || true
  echo "Done: $out_root"
}

# ----------------------------------------------------------------------------
# Mode: T3/T4 real vs common T1 no-void model
# ----------------------------------------------------------------------------
line_config() {
  local line="$1"
  case "$line" in
    T3)
      REAL_DIR_LINE="$BASE/04_FieldData/051926/051926_Seismics_T3"
      REAL_FIRST_FILE_LINE="${T3_FIRST_FILE:-4001}"
      REAL_LAST_FILE_LINE="${T3_LAST_FILE:-4039}"
      LOCAL_SHOT0_LINE="${T3_SHOT_FIRST_X_M:--0.5}"
      SHOT_DX_LINE="${T3_SHOT_DX_M:-2}"
      MODEL_OFFSET_LINE="${T3_MODEL_X_OFFSET_M:-87}"
      LOCAL_RX_MIN_LINE="${T3_RECEIVER_X_MIN:-0}"
      LOCAL_RX_MAX_LINE="${T3_RECEIVER_X_MAX:-71}"
      LOCAL_FIRST_TRACE_LINE="${T3_FIRST_TRACE_X_M:-0}"
      DUP_X_LINE="${T3_DUP_X:-}"
      DUP_FILES_LINE="${T3_DUP_FILES:-}"
      ;;
    T4)
      REAL_DIR_LINE="$BASE/04_FieldData/051926/051926_Seismics_T4"
      REAL_FIRST_FILE_LINE="${T4_FIRST_FILE:-4040}"
      REAL_LAST_FILE_LINE="${T4_LAST_FILE:-4068}"
      LOCAL_SHOT0_LINE="${T4_SHOT_FIRST_X_M:--2.5}"
      SHOT_DX_LINE="${T4_SHOT_DX_M:-2}"
      MODEL_OFFSET_LINE="${T4_MODEL_X_OFFSET_M:-87}"
      LOCAL_RX_MIN_LINE="${T4_RECEIVER_X_MIN:-0}"
      LOCAL_RX_MAX_LINE="${T4_RECEIVER_X_MAX:-71}"
      LOCAL_FIRST_TRACE_LINE="${T4_FIRST_TRACE_X_M:-0}"
      DUP_X_LINE="${T4_DUP_X:-}"
      DUP_FILES_LINE="${T4_DUP_FILES:-}"
      ;;
    *) echo "ERROR: unsupported line '$line'" >&2; return 1 ;;
  esac
  REAL_SHOT0_MODEL_LINE=$(awk -v a="$LOCAL_SHOT0_LINE" -v b="$MODEL_OFFSET_LINE" 'BEGIN{printf "%.10g", a+b}')
  RX_MIN_MODEL_LINE=$(awk -v a="$LOCAL_RX_MIN_LINE" -v b="$MODEL_OFFSET_LINE" 'BEGIN{printf "%.10g", a+b}')
  RX_MAX_MODEL_LINE=$(awk -v a="$LOCAL_RX_MAX_LINE" -v b="$MODEL_OFFSET_LINE" 'BEGIN{printf "%.10g", a+b}')
  FIRST_TRACE_MODEL_LINE=$(awk -v a="$LOCAL_FIRST_TRACE_LINE" -v b="$MODEL_OFFSET_LINE" 'BEGIN{printf "%.10g", a+b}')
  DISPLAY_SHIFT_LINE=$(awk -v b="$MODEL_OFFSET_LINE" 'BEGIN{printf "%.10g", -b}')
}

run_t3t4() {
  local out_root="${OUT_ROOT:-$OUT_BASE/real_vs_synthetic_novoid_T3_T4_v18_consolidated}"
  local movie_root="${MOVIE_ROOT:-$MOVIE_BASE/movies_real_vs_synthetic_novoid_T3_T4_v18_consolidated}"
  local lines="${LINES:-T3 T4}"
  local no_void_model="${NO_VOID_MODEL_COMMON:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s}"
  local data_dir="$no_void_model/DATA"
  local par_file="$no_void_model/DATA/Par_file"
  mkdir -p "$out_root/logs" "$movie_root"

  echo "Mode:                 t3t4"
  echo "OUT_ROOT:             $out_root"
  echo "Common T1 no-void:    $no_void_model"
  echo "Lines:                $lines"

  build_product_args
  real_norm_args
  real_processing_args

  local line c comp outdir log extra_real_args
  for line in $lines; do
    line_config "$line"
    for c in "${COMPONENT_FILES[@]}"; do
      comp="$(component_name "$c")"
      outdir="$out_root/$line/real_vs_synthetic_novoid/$comp"
      log="$out_root/logs/${line}_real_vs_synthetic_novoid_${comp}.log"
      mkdir -p "$outdir"
      extra_real_args=()
      if [[ -n "$DUP_X_LINE" && -n "$DUP_FILES_LINE" ]]; then
        extra_real_args+=(--real-shot-duplicate-x-m "$DUP_X_LINE" --real-shot-duplicate-files "$DUP_FILES_LINE")
      else
        extra_real_args+=(--real-shot-duplicate-x-m "-999999" --real-shot-duplicate-files "")
      fi
      run_bg "$log" "$PYTHON" "$ENGINE" \
        --mode real_vs_synthetic \
        --data-dir "$data_dir" \
        --par-file "$par_file" \
        --real-dir "$REAL_DIR_LINE" \
        --comparison-dir "$no_void_model" \
        --comparison-pattern "SURVEY_OUTPUT/**/$c" \
        --reference-label "Real Geode $line" \
        --comparison-label "Synthetic NO-VOID T1 model" \
        --output-dir "$outdir" \
        --real-first-file "$REAL_FIRST_FILE_LINE" \
        --real-last-file "$REAL_LAST_FILE_LINE" \
        --real-shot-first-x-m "$REAL_SHOT0_MODEL_LINE" \
        --real-shot-dx-m "$SHOT_DX_LINE" \
        "${extra_real_args[@]}" \
        --shot-match-tolerance-m "${SHOT_MATCH_TOLERANCE_M:-0.05}" \
        --receiver-x-min "$RX_MIN_MODEL_LINE" \
        --receiver-x-max "$RX_MAX_MODEL_LINE" \
        --real-first-trace-x-m "$FIRST_TRACE_MODEL_LINE" \
        --real-dx-m "${REAL_DX_M:-1}" \
        --display-x-shift-m "$DISPLAY_SHIFT_LINE" \
        "${PROCESSING_ARGS[@]}" \
        "${COMMON_ARGS[@]}" \
        "${NORM_ARGS[@]}"
    done
  done

  wait
  for line in $lines; do
    make_movies_with_depth "$out_root/$line/real_vs_synthetic_novoid" "$movie_root" "${line}_real_vs_synthetic_novoid" 3 "$out_root" || true
  done
  echo "Done: $out_root"
}

# ----------------------------------------------------------------------------
# Mode: rectangle vs polygon, single shot x=134.5 m
# ----------------------------------------------------------------------------
run_rectpoly() {
  local out_root="${OUT_ROOT:-$OUT_BASE/rectangle_vs_polygon_single_shot_134p5m_v18_consolidated}"
  local movie_root="${MOVIE_ROOT:-$MOVIE_BASE/movies_rectangle_vs_polygon_single_shot_134p5m_v18_consolidated}"
  local rect_root="${RECTANGLE_ROOT:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_RECTANGLE_1m_50Hz_DX_DZ_0d5m_DT_1e-5s/single_shot_134d5m}"
  local poly_root="${POLYGON_ROOT:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_POLYGON_1m_50Hz_DX_DZ_0d5m_DT_1e-5s/single_shot_134d5m}"
  local data_dir="${DATA_DIR:-$poly_root/DATA}"
  local par_file="${PAR_FILE:-$poly_root/DATA/Par_file}"
  mkdir -p "$out_root/logs" "$movie_root"

  [[ -d "$data_dir" ]] || data_dir="$rect_root/DATA"
  [[ -f "$par_file" ]] || par_file="$rect_root/DATA/Par_file"

  echo "Mode:          rectpoly"
  echo "OUT_ROOT:      $out_root"
  echo "Rectangle:     $rect_root"
  echo "Polygon:       $poly_root"

  build_product_args
  synthetic_norm_args

  local c comp outdir log
  for c in "${COMPONENT_FILES[@]}"; do
    comp="$(component_name "$c")"
    outdir="$out_root/$comp"
    log="$out_root/logs/rectangle_vs_polygon_${comp}.log"
    mkdir -p "$outdir"
    run_bg "$log" "$PYTHON" "$ENGINE" \
      --mode synthetic_vs_synthetic \
      --data-dir "$data_dir" \
      --par-file "$par_file" \
      --reference-dir "$rect_root" \
      --comparison-dir "$poly_root" \
      --reference-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$c" \
      --comparison-pattern "SURVEY_OUTPUT/shot_001_xs00134p5/$c" \
      --reference-label "Rectangle cave" \
      --comparison-label "Polygon cave" \
      --output-dir "$outdir" \
      --scale-mode none \
      --write-diff-segy \
      --cave-extent-x-m "${RECTPOLY_CAVE_EXTENT_X_M:-140.5,160.0}" \
      --limit "1" \
      "${COMMON_ARGS[@]}" \
      "${NORM_ARGS[@]}"
  done
  wait
  make_movies_with_depth "$out_root" "$movie_root" "rectangle_vs_polygon" 1 "$out_root" || true
  echo "Done: $out_root"
}

# ----------------------------------------------------------------------------
# Mode: frequency suite polygon/no-void and optional real single shot
# ----------------------------------------------------------------------------
model_label() {
  case "$1" in
    NO_VOID) echo "Synthetic NO_VOID" ;;
    POLYGON) echo "Synthetic POLYGON cave" ;;
    RECTANGLE) echo "Synthetic RECTANGLE cave" ;;
    *) echo "Synthetic $1" ;;
  esac
}

is_void_model() {
  case "$1" in POLYGON|RECTANGLE|VOID|WITH_VOID|CAVE) return 0 ;; *) return 1 ;; esac
}

synthetic_cave_shade_panels() {
  local ref_model="$1" cmp_model="$2" panels=()
  if is_void_model "$ref_model"; then panels+=(reference); fi
  if is_void_model "$cmp_model"; then panels+=(comparison); fi
  panels+=(difference)
  local IFS=,
  echo "${panels[*]}"
}

resolve_freq_relpath() {
  local root="$1" component_file="$2" shot_dir="${SHOT_DIR:-auto}" rel=""
  if [[ "$shot_dir" == "auto" ]]; then
    for candidate in "OUTPUT_FILES/$component_file" "${SURVEY_SHOT_DIR:-SURVEY_OUTPUT/shot_001_xs00134p5}/$component_file" "$component_file"; do
      if [[ -f "$root/$candidate" ]]; then
        rel="$candidate"
        break
      fi
    done
  else
    rel="$shot_dir/$component_file"
  fi
  [[ -n "$rel" ]] || rel="OUTPUT_FILES/$component_file"
  echo "$rel"
}

run_freq() {
  local out_root="${OUT_ROOT:-$OUT_BASE/frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v18_consolidated}"
  local movie_root="${MOVIE_ROOT:-$MOVIE_BASE/movies_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v18_consolidated}"
  local ref_model="${REFERENCE_MODEL:-NO_VOID}"
  local cmp_model="${COMPARISON_MODEL:-POLYGON}"
  local freqs_text="${FREQS:-30 40 50 60 70 80 100}"
  freqs_text="${freqs_text//,/ }"
  local run_synth="${RUN_SYNTHETIC:-1}"
  local run_real="${RUN_REAL_NOVOID:-1}"
  local real_dir="${REAL_DIR:-$BASE/04_FieldData/051826/051826_Seismics_T1}"
  mkdir -p "$out_root/logs" "$movie_root"

  echo "Mode:          freq"
  echo "SIM_ROOT:      $FREQ_SIM_ROOT"
  echo "OUT_ROOT:      $out_root"
  echo "Models:        $ref_model vs $cmp_model"
  echo "Frequencies:   $freqs_text"

  build_product_args

  local freq c comp ref_root cmp_root ref_rel cmp_rel data_dir par_file outdir log shade args novoid_root syn_rel

  if [[ "$run_synth" == "1" ]]; then
    synthetic_norm_args
    for freq in $freqs_text; do
      for c in "${SYN_FREQ_COMPONENT_FILES[@]}"; do
        comp="$(component_name "$c")"
        ref_root="$FREQ_SIM_ROOT/$ref_model/${freq}Hz"
        cmp_root="$FREQ_SIM_ROOT/$cmp_model/${freq}Hz"
        ref_rel="$(resolve_freq_relpath "$ref_root" "$c")"
        cmp_rel="$(resolve_freq_relpath "$cmp_root" "$c")"
        data_dir="$ref_root/DATA"; [[ -d "$data_dir" ]] || data_dir="$cmp_root/DATA"
        par_file="$ref_root/DATA/Par_file"; [[ -f "$par_file" ]] || par_file="$cmp_root/DATA/Par_file"
        outdir="$out_root/synthetic_${ref_model}_vs_${cmp_model}/${freq}Hz/$comp"
        log="$out_root/logs/synthetic_${freq}Hz_${ref_model}_vs_${cmp_model}_${comp}.log"
        shade="$(synthetic_cave_shade_panels "$ref_model" "$cmp_model")"
        mkdir -p "$outdir"
        run_bg "$log" "$PYTHON" "$ENGINE" \
          --mode synthetic_vs_synthetic \
          --data-dir "$data_dir" \
          --par-file "$par_file" \
          --reference-dir "$ref_root" \
          --comparison-dir "$cmp_root" \
          --reference-pattern "$ref_rel" \
          --comparison-pattern "$cmp_rel" \
          --reference-label "$(model_label "$ref_model") ${freq}Hz" \
          --comparison-label "$(model_label "$cmp_model") ${freq}Hz" \
          --output-dir "$outdir" \
          --scale-mode none \
          --write-diff-segy \
          --cave-extent-x-m "${FREQ_CAVE_EXTENT_X_M:-115,125}" \
          --cave-shade-panels "$shade" \
          "${COMMON_ARGS[@]}" \
          "${NORM_ARGS[@]}"
      done
    done
  fi

  if [[ "$run_real" == "1" ]]; then
    real_norm_args
    real_processing_args
    for freq in $freqs_text; do
      for c in "${REAL_FREQ_COMPONENT_FILES[@]}"; do
        comp="$(component_name "$c")"
        novoid_root="$FREQ_SIM_ROOT/NO_VOID/${freq}Hz"
        syn_rel="${REAL_SYN_PATTERN_TEMPLATE:-SURVEY_OUTPUT/shot_001_xs00134p5/{component}}"
        syn_rel="${syn_rel//\{component\}/$c}"
        data_dir="$novoid_root/DATA"
        par_file="$novoid_root/DATA/Par_file"
        outdir="$out_root/real_T1_1m_vs_synthetic_NO_VOID/${freq}Hz/$comp"
        log="$out_root/logs/real_T1_1m_vs_NO_VOID_${freq}Hz_${comp}.log"
        mkdir -p "$outdir"
        run_bg "$log" "$PYTHON" "$ENGINE" \
          --mode real_vs_synthetic \
          --data-dir "$data_dir" \
          --par-file "$par_file" \
          --real-dir "$real_dir" \
          --real-first-file "${REAL_FIRST_FILE:-3032}" \
          --real-last-file "${REAL_LAST_FILE:-3032}" \
          --real-shot-first-x-m "${REAL_SHOT_FIRST_X_M:-134.5}" \
          --real-shot-dx-m "${REAL_SHOT_DX_M:-2.0}" \
          --real-shot-duplicate-x-m "${REAL_SHOT_DUPLICATE_X_M:-102.5}" \
          --real-shot-duplicate-files "${REAL_SHOT_DUPLICATE_FILES:-3016}" \
          --real-first-trace-x-m "${REAL_FIRST_TRACE_X_M:-87.0}" \
          --real-dx-m "${REAL_DX_M:-1.0}" \
          --shot-match-tolerance-m "${SHOT_MATCH_TOLERANCE_M:-0.25}" \
          --comparison-dir "$novoid_root" \
          --comparison-pattern "$syn_rel" \
          --reference-label "Real T1 1-m refraction" \
          --comparison-label "Synthetic NO_VOID ${freq}Hz" \
          --output-dir "$outdir" \
          "${PROCESSING_ARGS[@]}" \
          "${COMMON_ARGS[@]}" \
          "${NORM_ARGS[@]}"
      done
    done
  fi

  wait
  # The standard movie script still works for many products, though the dedicated
  # old 69-wrapper frequency movie maker had more careful frequency ordering.
  # Set RUN_MOVIES=1 to attempt movies; otherwise use the existing PNG folders.
  [[ "$run_synth" == "1" ]] && make_movies_with_depth "$out_root/synthetic_${ref_model}_vs_${cmp_model}" "$movie_root" "frequency_synthetic_${ref_model}_vs_${cmp_model}" 3 "$out_root" || true
  [[ "$run_real" == "1" ]] && make_movies_with_depth "$out_root/real_T1_1m_vs_synthetic_NO_VOID" "$movie_root" "frequency_real_vs_NO_VOID" 3 "$out_root" || true
  echo "Done: $out_root"
}


# Convenience modes that run only one half of an existing grouped function.
run_t1_synthetic_only() {
  RUN_SYNTHETIC=1 RUN_REAL=0 run_t1
}

run_real_t1_only() {
  RUN_SYNTHETIC=0 RUN_REAL=1 run_t1
}

run_real_t3_only() {
  LINES="T3" run_t3t4
}

run_real_t4_only() {
  LINES="T4" run_t3t4
}

run_polygon_novoid_frequency_only() {
  RUN_SYNTHETIC=1 RUN_REAL_NOVOID=0 REFERENCE_MODEL="${REFERENCE_MODEL:-NO_VOID}" COMPARISON_MODEL="${COMPARISON_MODEL:-POLYGON}" run_freq
}

run_real_frequency_only() {
  RUN_SYNTHETIC=0 RUN_REAL_NOVOID=1 run_freq
}

run_all_previous_wrappers_equivalent() {
  echo "Running full bundle equivalent to previous wrappers:"
  echo "  1. T1 synthetic cave/no-cave"
  echo "  2. T1 real vs synthetic no-void"
  echo "  3. T3/T4 real vs common T1 synthetic no-void"
  echo "  4. Rectangle cave vs polygon cave"
  echo "  5. Frequency suite: NO_VOID vs POLYGON"
  echo "  6. Frequency suite: real T1 single shot vs NO_VOID"
  run_t1
  run_t3t4
  run_rectpoly
  run_freq
}

usage() {
  cat <<USAGE
Usage:
  bash $(basename "$0") <legacy-mode>
  bash $(basename "$0") --mode <mode> [--line LINE] [--frequency-suite]

Preferred modes:
  --mode synthetic_cave_nocave
      T1 synthetic cave model vs T1 synthetic no-void model.

  --mode real_vs_synthetic --line T1
      T1 real data vs T1 synthetic no-void model.

  --mode real_vs_synthetic --line T3
  --mode real_vs_synthetic --line T4
  --mode real_vs_synthetic --line T3,T4
      T3/T4 real data vs common T1 synthetic no-void model, with plotted
      coordinates shifted back to local 0..71 m.

  --mode rectangle_vs_polygon
      Single-shot synthetic rectangle-cave vs polygon-cave comparison.

  --mode polygon_vs_novoid --frequency-suite
      Frequency suite synthetic NO_VOID vs POLYGON, 30..100 Hz by default.

  --mode real_vs_novoid --frequency-suite
      Frequency suite real T1 single shot vs synthetic NO_VOID, 30..100 Hz.

Legacy aliases:
  t1                         T1 synthetic cave/no-cave AND T1 real-vs-no-void.
  t3t4                       T3/T4 real-vs-no-void.
  rectpoly                   Rectangle-vs-polygon.
  freq                       Both frequency-suite products.
  all                        Equivalent of all previous wrappers.

Common environment controls:
  BASE=/path/to/2026KarstGeophysicsDEP
  ENGINE=/path/to/65_compare_gather_pairs_v17.py
  MOVIE_SCRIPT=/path/to/66_make_movies_from_shot_figures_v3.sh
  MAX_JOBS=4
  LIMIT=2
  RUN_MOVIES=0|1
  WRITE_FK=0|1
  WRITE_BANDPASS=0|1
  COMPONENTS="Ux_file_single_v.su Uz_file_single_v.su"
  FREQS="30 40 50 60 70 80 100"

Examples:
  LIMIT=2 RUN_MOVIES=0 bash $(basename "$0") --mode real_vs_synthetic --line T3,T4
  RUN_MOVIES=1 bash $(basename "$0") --mode polygon_vs_novoid --frequency-suite
  COMPONENTS="Uz_file_single_v.su" bash $(basename "$0") --mode real_vs_synthetic --line T1
  WRITE_FK=0 bash $(basename "$0") --mode rectangle_vs_polygon
  bash $(basename "$0") all

USAGE
}

case "$COMMAND" in
  # Preferred explicit commands generated by option parser.
  t1_synthetic_cave_nocave|synthetic_cave_nocave) run_t1_synthetic_only ;;
  real_t1) run_real_t1_only ;;
  real_t3) run_real_t3_only ;;
  real_t4) run_real_t4_only ;;
  real_t3t4) run_t3t4 ;;
  rectangle_vs_polygon) run_rectpoly ;;
  polygon_novoid_frequency) run_polygon_novoid_frequency_only ;;
  real_frequency|real_novoid_frequency) run_real_frequency_only ;;

  # Legacy aliases.
  t1) run_t1 ;;
  t3t4) run_t3t4 ;;
  rectpoly) run_rectpoly ;;
  freq|frequency) run_freq ;;
  all|all_previous|previous_wrappers) run_all_previous_wrappers_equivalent ;;

  help|--help|-h) usage ;;
  *) echo "ERROR: unknown mode: $COMMAND" >&2; usage; exit 2 ;;
esac
