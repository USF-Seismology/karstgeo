#!/usr/bin/env bash
set -euo pipefail

# Run real-vs-synthetic NO-VOID comparisons for T3 and/or T4.
# Based on 67_run_all_65_comparisons_and_movies_v10.sh, but line-aware and using
# 65_compare_gather_pairs_v16.py.
#
# Default run:
#   bash 67_run_T3_T4_real_vs_synthetic_novoid_v15.sh
#
# Common controls:
#   LINES="T3 T4"              # which lines to process
#   MAX_JOBS=4                 # parallel component/line jobs
#   RUN_MOVIES=1               # make movies after comparisons
#   LIMIT=3                    # optional quick test, limits matched shots
#   ENGINE=/path/to/65_compare_gather_pairs_v16.py
#   MOVIE_SCRIPT=/path/to/66_make_movies_from_shot_figures_v3.sh
#   BASE=/path/to/2026KarstGeophysicsDEP
#
# Synthetic model root:
#   Felix has only run the T1 no-void synthetic model, so by default T3/T4
#   real data are compared against that common T1 no-void model.
#
#   NO_VOID_MODEL_COMMON=/path/to/T1/no_void/model
#
# Geometry remapping:
#   T3/T4 field coordinates are local 0..71 m receiver coordinates, whereas the
#   available T1 no-void synthetic model is in the T1 model coordinate frame
#   where the equivalent 72-receiver spread is normally 87..158 m.
#   Therefore this script defaults to adding +87 m to real shot and receiver
#   coordinates before matching synthetic source/receiver positions, then subtracting the same offset for display so plots can show local 0..71 m coordinates. Override with:
#
#   T3_MODEL_X_OFFSET_M=87
#   T4_MODEL_X_OFFSET_M=87
#
# If you later get true T3/T4 no-void synthetic models in local coordinates, set:
#   USE_COMMON_T1_NO_VOID=0
#   NO_VOID_MODEL_T3=/path/to/T3/no_void/model
#   NO_VOID_MODEL_T4=/path/to/T4/no_void/model
#   T3_MODEL_X_OFFSET_M=0
#   T4_MODEL_X_OFFSET_M=0
#
# T4 note:
#   Jochen metadata described T4 as files 4040-4070, shots every 2 m from
#   -2.5 to 53.5, then final shot at 71.5. That is not representable by the
#   current v11 linear real-shot mapping. By default this script uses files
#   4040-4068 for the regular -2.5..53.5 sequence and skips the final storm-affected
#   shot. Set T4_LAST_FILE=4070 only if your metadata/engine mapping has been patched.

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v16.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v3.sh}"

MAX_JOBS="${MAX_JOBS:-4}"
RUN_REAL="${RUN_REAL:-1}"
RUN_MOVIES="${RUN_MOVIES:-1}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"
MOVIE_FPS="${MOVIE_FPS:-1}"
LINES="${LINES:-T3 T4}"

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SOURCES_GROUNDED="${SOURCES_GROUNDED:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED}"
USE_COMMON_T1_NO_VOID="${USE_COMMON_T1_NO_VOID:-1}"
NO_VOID_MODEL_COMMON="${NO_VOID_MODEL_COMMON:-$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NO_VOID_T1_1m_50Hz_DX_DZ_0d5m_DT_1e-5s}"

OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/real_vs_synthetic_novoid_T3_T4_v15}"
MOVIE_ROOT="${MOVIE_ROOT:-$BASE/02_Modelling/Seismic/differencing/movies_real_vs_synthetic_novoid_T3_T4_v15}"
mkdir -p "$OUT_ROOT/logs" "$MOVIE_ROOT"

[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE"; exit 1; }
if [[ "$RUN_MOVIES" == "1" ]]; then
  [[ -f "$MOVIE_SCRIPT" ]] || { echo "ERROR: missing movie script: $MOVIE_SCRIPT"; exit 1; }
fi

BASE_COMMON_ARGS=(
  --max-freq-hz "400"

  --write-combined-three-panel-products
  --overlay-wiggle-scale "0.45"

  --write-spectral-contours
  --spectral-contour-log10
  --spectral-contour-levels "24"

  --write-band-energy
  --band-energy-bands "10-30,30-80,80-150,150-400"
  --band-energy-window-s "0.05"
  --band-energy-step-s "0.01"
)

DIAGNOSTIC_ARGS=()
if [[ "$WRITE_BANDPASS" == "1" ]]; then
  DIAGNOSTIC_ARGS+=(
    --write-diagnostic-bandpass
    --diagnostic-bandpass-fmin "25"
    --diagnostic-bandpass-fmax "400"
    --diagnostic-bandpass-corners "4"
    --diagnostic-bandpass-zerophase
  )
fi
if [[ "$WRITE_FK" == "1" ]]; then
  DIAGNOSTIC_ARGS+=(
    --write-fk-filtered
    --fk-min-velocity-mps "500"
    --fk-taper-width-mps "100"
    --fk-use-taper
  )
fi
if [[ -n "${LIMIT:-}" ]]; then
  BASE_COMMON_ARGS+=(--limit "$LIMIT")
fi

COMPONENT_FILES=("Ux_file_single_v.su" "Uz_file_single_v.su")

REAL_NORMALIZATION_ARGS=(
  --overlay-normalize trace
  --write-trace-normalized-figures
  --trace-normalize-method "maxabs"
  --frequency-trace-normalization
  --band-energy-normalize-per-trace
  --combined-wiggle-trace-normalize
)

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

find_novoid_model() {
  local line="$1"

  # Default for this analysis: use the existing T1 no-void synthetic model
  # for T3/T4 real data, because no true T3/T4 no-void synthetics exist yet.
  if [[ "$USE_COMMON_T1_NO_VOID" == "1" ]]; then
    echo "$NO_VOID_MODEL_COMMON"
    return 0
  fi

  local env_name="NO_VOID_MODEL_${line}"
  local explicit="${!env_name:-}"
  if [[ -n "$explicit" ]]; then
    echo "$explicit"
    return 0
  fi

  # macOS ships bash 3.2, which does not have mapfile/readarray.
  # Use a portable read loop instead.
  local matches=()
  local m
  while IFS= read -r m; do
    [[ -n "$m" ]] && matches+=("$m")
  done < <(
    find "$SOURCES_GROUNDED" -maxdepth 1 -type d \
      \( -iname "*NO*VOID*${line}*50Hz*" -o -iname "*NO_VOID*${line}*50Hz*" \) \
      | sort
  )
  if (( ${#matches[@]} == 1 )); then
    echo "${matches[0]}"
    return 0
  fi

  echo "ERROR: could not uniquely identify NO_VOID synthetic model for $line under:" >&2
  echo "  $SOURCES_GROUNDED" >&2
  echo "Set NO_VOID_MODEL_${line}=/path/to/model and rerun, or USE_COMMON_T1_NO_VOID=1." >&2
  if (( ${#matches[@]} > 1 )); then
    echo "Matches found:" >&2
    printf '  %s\n' "${matches[@]}" >&2
  else
    echo "Nearby candidates:" >&2
    find "$SOURCES_GROUNDED" -maxdepth 1 -type d -iname "*${line}*" | sort | sed 's/^/  /' >&2 || true
  fi
  return 1
}

line_config() {
  local line="$1"
  case "$line" in
    T3)
      REAL_DIR="$BASE/04_FieldData/051926/051926_Seismics_T3"
      REAL_FIRST_FILE="${T3_FIRST_FILE:-4001}"
      REAL_LAST_FILE="${T3_LAST_FILE:-4039}"
      REAL_LOCAL_SHOT_FIRST_X_M="${T3_SHOT_FIRST_X_M:--0.5}"
      MODEL_X_OFFSET_M="${T3_MODEL_X_OFFSET_M:-87}"
      REAL_SHOT_FIRST_X_M=$(awk -v a="$REAL_LOCAL_SHOT_FIRST_X_M" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      REAL_SHOT_DX_M="${T3_SHOT_DX_M:-2}"
      REAL_DUP_X="${T3_DUP_X:-}"       # none by default
      REAL_DUP_FILES="${T3_DUP_FILES:-}"
      RECEIVER_LOCAL_X_MIN="${T3_RECEIVER_X_MIN:-0}"
      RECEIVER_LOCAL_X_MAX="${T3_RECEIVER_X_MAX:-71}"
      REAL_LOCAL_FIRST_TRACE_X_M="${T3_FIRST_TRACE_X_M:-0}"
      RECEIVER_X_MIN=$(awk -v a="$RECEIVER_LOCAL_X_MIN" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      RECEIVER_X_MAX=$(awk -v a="$RECEIVER_LOCAL_X_MAX" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      REAL_FIRST_TRACE_X_M=$(awk -v a="$REAL_LOCAL_FIRST_TRACE_X_M" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      ;;
    T4)
      REAL_DIR="$BASE/04_FieldData/051926/051926_Seismics_T4"
      REAL_FIRST_FILE="${T4_FIRST_FILE:-4040}"
      # Default stops at 4068, giving -2.5..53.5 every 2 m.
      # The final 71.5 m thunderstorm shot needs a non-linear shot map patch in v11.
      REAL_LAST_FILE="${T4_LAST_FILE:-4068}"
      REAL_LOCAL_SHOT_FIRST_X_M="${T4_SHOT_FIRST_X_M:--2.5}"
      MODEL_X_OFFSET_M="${T4_MODEL_X_OFFSET_M:-87}"
      REAL_SHOT_FIRST_X_M=$(awk -v a="$REAL_LOCAL_SHOT_FIRST_X_M" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      REAL_SHOT_DX_M="${T4_SHOT_DX_M:-2}"
      REAL_DUP_X="${T4_DUP_X:-}"
      REAL_DUP_FILES="${T4_DUP_FILES:-}"
      RECEIVER_LOCAL_X_MIN="${T4_RECEIVER_X_MIN:-0}"
      RECEIVER_LOCAL_X_MAX="${T4_RECEIVER_X_MAX:-71}"
      REAL_LOCAL_FIRST_TRACE_X_M="${T4_FIRST_TRACE_X_M:-0}"
      RECEIVER_X_MIN=$(awk -v a="$RECEIVER_LOCAL_X_MIN" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      RECEIVER_X_MAX=$(awk -v a="$RECEIVER_LOCAL_X_MAX" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      REAL_FIRST_TRACE_X_M=$(awk -v a="$REAL_LOCAL_FIRST_TRACE_X_M" -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", a+b}')
      ;;
    *)
      echo "ERROR: unsupported line: $line" >&2
      return 1
      ;;
  esac
}

run_real_line_component() {
  local line="$1"
  local component_file="$2"
  local comp no_void_model data_dir par_file outdir log

  line_config "$line"
  no_void_model="$(find_novoid_model "$line")"
  data_dir="$no_void_model/DATA"
  par_file="$no_void_model/DATA/Par_file"

  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/${line}/real_vs_synthetic_novoid/$comp"
  log="$OUT_ROOT/logs/${line}_real_vs_synthetic_novoid_${comp}.log"
  mkdir -p "$outdir"

  extra_real_args=()
  if [[ -n "$REAL_DUP_X" && -n "$REAL_DUP_FILES" ]]; then
    extra_real_args+=(--real-shot-duplicate-x-m "$REAL_DUP_X" --real-shot-duplicate-files "$REAL_DUP_FILES")
  else
    # Use a harmless impossible duplicate so defaults from the engine do not affect T3/T4.
    extra_real_args+=(--real-shot-duplicate-x-m "-999999" --real-shot-duplicate-files "")
  fi

  run_bg "$log" "$PYTHON" "$ENGINE" \
    --mode real_vs_synthetic \
    --data-dir "$data_dir" \
    --par-file "$par_file" \
    --real-dir "$REAL_DIR" \
    --comparison-dir "$no_void_model" \
    --comparison-pattern "SURVEY_OUTPUT/**/$component_file" \
    --reference-label "Real Geode $line" \
    --comparison-label "Synthetic NO-VOID T1 model" \
    --output-dir "$outdir" \
    --real-first-file "$REAL_FIRST_FILE" \
    --real-last-file "$REAL_LAST_FILE" \
    --real-shot-first-x-m "$REAL_SHOT_FIRST_X_M" \
    --real-shot-dx-m "$REAL_SHOT_DX_M" \
    "${extra_real_args[@]}" \
    --shot-match-tolerance-m "${SHOT_MATCH_TOLERANCE_M:-0.05}" \
    --receiver-x-min "$RECEIVER_X_MIN" \
    --receiver-x-max "$RECEIVER_X_MAX" \
    --real-first-trace-x-m "$REAL_FIRST_TRACE_X_M" \
    --real-dx-m "${REAL_DX_M:-1}" \
    --display-x-shift-m "${DISPLAY_X_SHIFT_M:-$(awk -v b="$MODEL_X_OFFSET_M" 'BEGIN{printf "%.10g", -b}')}" \
    --tmin "${TMIN:-0.0}" \
    --tmax "${TMAX:-0.4}" \
    --comparison-time-shift-ms "${COMPARISON_TIME_SHIFT_MS:--31.6}" \
    --scale-mode "${SCALE_MODE:-fixed}" \
    --fixed-scale-factor "${FIXED_SCALE_FACTOR:-2.96e7}" \
    --scale-tmin "${SCALE_TMIN:-0.02}" \
    --scale-tmax "${SCALE_TMAX:-0.12}" \
    --demean \
    --detrend \
    --taper-fraction "${TAPER_FRACTION:-0.05}" \
    --highpass-hz "${HIGHPASS_HZ:-10}" \
    --filter-corners "${FILTER_CORNERS:-4}" \
    --zerophase \
    --normalize-synthetic-source-factor \
    --synthetic-source-target-factor "${SYNTHETIC_SOURCE_TARGET_FACTOR:-1e10}" \
    "${BASE_COMMON_ARGS[@]}" \
    "${REAL_NORMALIZATION_ARGS[@]}" \
    "${DIAGNOSTIC_ARGS[@]}"
}

make_movies() {
  local root="$1"
  local label="$2"
  local log="$OUT_ROOT/logs/movies_${label}.log"
  echo
  echo "Making movies for $label"
  FPS="$MOVIE_FPS" MAP_DEPTH=3 bash "$MOVIE_SCRIPT" "$root" "$MOVIE_ROOT" >"$log" 2>&1
  echo "Movie log: $log"
}

echo "Engine:          $ENGINE"
echo "Movie script:    $MOVIE_SCRIPT"
echo "SOURCES_GROUNDED:$SOURCES_GROUNDED"
echo "USE_COMMON_T1_NO_VOID: $USE_COMMON_T1_NO_VOID"
echo "NO_VOID_MODEL_COMMON:  $NO_VOID_MODEL_COMMON"
echo "OUT_ROOT:        $OUT_ROOT"
echo "MOVIE_ROOT:      $MOVIE_ROOT"
echo "LINES:           $LINES"
echo "MAX_JOBS:        $MAX_JOBS"
echo "LIMIT:           ${LIMIT:-none}"
echo

echo "Checking core paths:"
[[ -d "$SOURCES_GROUNDED" ]] && echo "  OK      $SOURCES_GROUNDED" || echo "  MISSING $SOURCES_GROUNDED"
for line in $LINES; do
  line_config "$line"
  no_void_model="$(find_novoid_model "$line" || true)"
  echo "  $line real dir:      $([[ -d "$REAL_DIR" ]] && echo OK || echo MISSING) $REAL_DIR"
  echo "  $line no-void model: ${no_void_model:-NOT FOUND}"
  if [[ -n "$no_void_model" ]]; then
    for path in "$no_void_model/DATA/STATIONS" "$no_void_model/DATA/SOURCES_LIST.txt" "$no_void_model/DATA/Par_file" "$no_void_model/SURVEY_OUTPUT"; do
      [[ -e "$path" ]] && echo "    OK      $path" || echo "    MISSING $path"
    done
  fi
  echo "  $line real files:    $REAL_FIRST_FILE to $REAL_LAST_FILE"
  echo "  $line geometry:      local shot0=${REAL_LOCAL_SHOT_FIRST_X_M:-?} dx=$REAL_SHOT_DX_M, local receivers=${RECEIVER_LOCAL_X_MIN:-?}..${RECEIVER_LOCAL_X_MAX:-?}"
  echo "  $line model frame:   offset=${MODEL_X_OFFSET_M:-?}; shot0=$REAL_SHOT_FIRST_X_M; receivers=$RECEIVER_X_MIN..$RECEIVER_X_MAX"
  echo "  $line display frame: shift=${DISPLAY_X_SHIFT_M:--$MODEL_X_OFFSET_M}; expected receivers=${RECEIVER_LOCAL_X_MIN:-?}..${RECEIVER_LOCAL_X_MAX:-?}"
  echo
done

if [[ "$RUN_REAL" == "1" ]]; then
  for line in $LINES; do
    for c in "${COMPONENT_FILES[@]}"; do
      run_real_line_component "$line" "$c"
    done
  done
fi

echo
echo "Waiting for comparison jobs..."
wait
echo "All comparison jobs complete. Logs: $OUT_ROOT/logs"

if [[ "$RUN_MOVIES" == "1" ]]; then
  for line in $LINES; do
    make_movies "$OUT_ROOT/${line}/real_vs_synthetic_novoid" "${line}_real_vs_synthetic_novoid"
  done
fi

echo
echo "Done."
echo "Products: $OUT_ROOT"
echo "Movies:   $MOVIE_ROOT"
