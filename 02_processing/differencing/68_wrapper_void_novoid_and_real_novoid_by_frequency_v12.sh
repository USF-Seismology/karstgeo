#!/usr/bin/env bash
# 68_wrapper_void_novoid_and_real_novoid_by_frequency_v12.sh
#
# Runs two comparison families, by Ricker source frequency:
#   1) synthetic WITH-void vs synthetic NO-void
#   2) real T1 1-m refraction vs synthetic NO-void
#
# Cave shading policy:
#   - WITH-void vs NO-void: shade only the synthetic WITH-void panel and the difference panel.
#   - Real vs NO-void: no cave shading at all.
#   - No-void-only panels are never shaded.
#
# f-k products:
#   - Enables --write-fk-filtered with vmin=500 m/s.
#   - Each shot directory should contain a subfolder like fk_vmin_500mps/.
#
# Edit/override these environment variables as needed before running:
#   FREQS="25 50 100 150"
#   REAL_DIR=/path/to/T1_1m_SEG2
#   VOID_ROOT_50=/path/to/with_void_50Hz_run
#   NOVOID_ROOT_50=/path/to/no_void_50Hz_run
#
# Example:
#   ENGINE=/path/to/65_compare_gather_pairs_v12.py \
#   REAL_DIR=/path/to/SEG2 \
#   bash 68_wrapper_void_novoid_and_real_novoid_by_frequency_v12.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v12.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v3.sh}"

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SOURCES_GROUNDED="${SOURCES_GROUNDED:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/SOURCES_GROUNDED}"
OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/frequency_sweep_void_novoid_and_real_novoid_v12}"
REAL_DIR="${REAL_DIR:-}"

# Frequencies to run. Override, e.g. FREQS="25 50 100 150".
read -r -a FREQ_ARRAY <<< "${FREQS:-25 50 100 150}"

# Components for synthetic-vs-synthetic. Real-vs-synthetic usually uses vertical only.
SYN_COMPONENT_FILES=("${SYN_COMPONENT_FILES[@]:-Ux_file_single_v.su Uz_file_single_v.su}")
REAL_COMPONENT_FILES=("${REAL_COMPONENT_FILES[@]:-Uz_file_single_v.su}")

RUN_SYN_VOID_NOVOID="${RUN_SYN_VOID_NOVOID:-1}"
RUN_REAL_NOVOID="${RUN_REAL_NOVOID:-1}"
RUN_MOVIES="${RUN_MOVIES:-0}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"

# Real T1 1-m refraction defaults; override if using a different line/file range.
REAL_FIRST_FILE="${REAL_FIRST_FILE:-3005}"
REAL_LAST_FILE="${REAL_LAST_FILE:-3046}"
REAL_SHOT_FIRST_X_M="${REAL_SHOT_FIRST_X_M:-82.5}"
REAL_SHOT_DX_M="${REAL_SHOT_DX_M:-2.0}"
REAL_SHOT_DUPLICATE_X_M="${REAL_SHOT_DUPLICATE_X_M:-102.5}"
REAL_SHOT_DUPLICATE_FILES="${REAL_SHOT_DUPLICATE_FILES:-3015,3016}"
REAL_FIRST_TRACE_X_M="${REAL_FIRST_TRACE_X_M:-87.0}"
REAL_DX_M="${REAL_DX_M:-1.0}"
SHOT_MATCH_TOLERANCE_M="${SHOT_MATCH_TOLERANCE_M:-0.25}"

CAVE_EXTENT_X_M="${CAVE_EXTENT_X_M:-140.5,160.0}"
MAX_FREQ_HZ="${MAX_FREQ_HZ:-400}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUT_ROOT/logs"
[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE"; exit 1; }

component_name() {
  case "$1" in
    Ux*) echo "Ux" ;;
    Uz*) echo "Uz" ;;
    *) basename "$1" .su ;;
  esac
}

# Candidate run-root discovery. Explicit env vars win:
#   VOID_ROOT_50=/... and NOVOID_ROOT_50=/...
# Otherwise we try common names used in this project.
root_for_freq() {
  local kind="$1" freq="$2" var val
  if [[ "$kind" == "void" ]]; then
    var="VOID_ROOT_${freq}"
  else
    var="NOVOID_ROOT_${freq}"
  fi
  val="${!var:-}"
  if [[ -n "$val" ]]; then
    echo "$val"
    return 0
  fi

  local candidates=()
  if [[ "$kind" == "void" ]]; then
    candidates=(
      "$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_RECTANGLE_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_RECTANGLE_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/9_LAYER_MODEL_RECTANGLE_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
    )
  else
    candidates=(
      "$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NOVOID_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/9_LAYER_MODEL_TOPO_NO_VOID_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/9_LAYER_MODEL_NOVOID_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/9_LAYER_MODEL_NO_VOID_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
      "$SOURCES_GROUNDED/T1_9_LAYER_MODEL_TOPO_NOVOID_1m_${freq}Hz_DX_DZ_0d5m_DT_1e-5s"
    )
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -d "$c" ]]; then
      echo "$c"
      return 0
    fi
  done

  # Return first candidate to make the missing path visible in logs.
  echo "${candidates[0]}"
}

common_args() {
  local data_dir="$1" par_file="$2"
  local args=(
    --data-dir "$data_dir"
    --par-file "$par_file"
    --scale-mode none
    --max-freq-hz "$MAX_FREQ_HZ"
    --write-diff-segy
    --write-combined-three-panel-products
    --overlay-normalize pair
    --overlay-wiggle-scale "0.45"
    --write-spectral-contours
    --spectral-contour-log10
    --spectral-contour-levels "24"
    --write-band-energy
    --band-energy-bands "10-30,30-80,80-150,150-400"
    --band-energy-window-s "0.05"
    --band-energy-step-s "0.01"
    --no-band-energy-normalize-per-trace
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "$WRITE_BANDPASS" == "1" ]]; then
    args+=(
      --write-diagnostic-bandpass
      --diagnostic-bandpass-fmin "25"
      --diagnostic-bandpass-fmax "400"
      --diagnostic-bandpass-corners "4"
      --diagnostic-bandpass-zerophase
    )
  fi
  if [[ "$WRITE_FK" == "1" ]]; then
    args+=(
      --write-fk-filtered
      --fk-min-velocity-mps "500"
      --fk-taper-width-mps "100"
      --fk-use-taper
      --fk-split-at-source
    )
  fi
  printf '%s\0' "${args[@]}"
}

run_cmd_logged() {
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  {
    echo "Command:"
    printf '%q ' "$@"
    echo
    echo
    "$@"
  } > "$log" 2>&1
}

run_syn_void_novoid() {
  local freq="$1" component_file="$2" void_root="$3" novoid_root="$4"
  local comp outdir log data_dir par_file
  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/synthetic_void_vs_novoid/${freq}Hz/$comp"
  log="$OUT_ROOT/logs/syn_void_vs_novoid_${freq}Hz_${comp}.log"
  data_dir="$void_root/DATA"
  par_file="$void_root/DATA/Par_file"

  echo "Synthetic void vs no-void: ${freq} Hz, ${comp}"
  if [[ ! -d "$void_root" || ! -d "$novoid_root" ]]; then
    echo "  SKIP: missing root(s):" | tee "$log"
    echo "    void:   $void_root" | tee -a "$log"
    echo "    novoid: $novoid_root" | tee -a "$log"
    return 0
  fi

  mapfile -d '' args < <(common_args "$data_dir" "$par_file")
  run_cmd_logged "$log" "$PYTHON" "$ENGINE" \
    --mode synthetic_vs_synthetic \
    "${args[@]}" \
    --reference-dir "$void_root" \
    --comparison-dir "$novoid_root" \
    --reference-pattern "SURVEY_OUTPUT/**/$component_file" \
    --comparison-pattern "SURVEY_OUTPUT/**/$component_file" \
    --reference-label "Synthetic WITH void ${freq} Hz" \
    --comparison-label "Synthetic NO void ${freq} Hz" \
    --cave-extent-x-m "$CAVE_EXTENT_X_M" \
    --cave-shade-panels "reference,difference" \
    --output-dir "$outdir"

  if [[ "$WRITE_FK" == "1" ]]; then
    if find "$outdir" -type d -name 'fk_vmin_*mps' | grep -q .; then
      echo "  f-k folders found under $outdir"
    else
      echo "  WARNING: no f-k folder found. Check $log"
    fi
  fi
}

run_real_novoid() {
  local freq="$1" component_file="$2" novoid_root="$3"
  local comp outdir log data_dir par_file
  comp="$(component_name "$component_file")"
  outdir="$OUT_ROOT/real_T1_1m_vs_synthetic_novoid/${freq}Hz/$comp"
  log="$OUT_ROOT/logs/real_vs_novoid_${freq}Hz_${comp}.log"
  data_dir="$novoid_root/DATA"
  par_file="$novoid_root/DATA/Par_file"

  echo "Real T1 1-m vs synthetic no-void: ${freq} Hz, ${comp}"
  if [[ -z "$REAL_DIR" || ! -d "$REAL_DIR" ]]; then
    echo "  SKIP: REAL_DIR is unset or missing: ${REAL_DIR:-<unset>}" | tee "$log"
    return 0
  fi
  if [[ ! -d "$novoid_root" ]]; then
    echo "  SKIP: missing no-void root: $novoid_root" | tee "$log"
    return 0
  fi

  mapfile -d '' args < <(common_args "$data_dir" "$par_file")
  run_cmd_logged "$log" "$PYTHON" "$ENGINE" \
    --mode real_vs_synthetic \
    "${args[@]}" \
    --real-dir "$REAL_DIR" \
    --real-first-file "$REAL_FIRST_FILE" \
    --real-last-file "$REAL_LAST_FILE" \
    --real-shot-first-x-m "$REAL_SHOT_FIRST_X_M" \
    --real-shot-dx-m "$REAL_SHOT_DX_M" \
    --real-shot-duplicate-x-m "$REAL_SHOT_DUPLICATE_X_M" \
    --real-shot-duplicate-files "$REAL_SHOT_DUPLICATE_FILES" \
    --real-first-trace-x-m "$REAL_FIRST_TRACE_X_M" \
    --real-dx-m "$REAL_DX_M" \
    --shot-match-tolerance-m "$SHOT_MATCH_TOLERANCE_M" \
    --comparison-dir "$novoid_root" \
    --comparison-pattern "SURVEY_OUTPUT/**/$component_file" \
    --reference-label "Real T1 1-m" \
    --comparison-label "Synthetic NO void ${freq} Hz" \
    --output-dir "$outdir"

  if [[ "$WRITE_FK" == "1" ]]; then
    if find "$outdir" -type d -name 'fk_vmin_*mps' | grep -q .; then
      echo "  f-k folders found under $outdir"
    else
      echo "  WARNING: no f-k folder found. Check $log"
    fi
  fi
}

echo "Engine:          $ENGINE"
echo "OUT_ROOT:        $OUT_ROOT"
echo "SOURCES_GROUNDED:$SOURCES_GROUNDED"
echo "FREQS:           ${FREQ_ARRAY[*]}"
echo "REAL_DIR:        ${REAL_DIR:-<unset>}"
echo "CAVE_EXTENT_X_M: $CAVE_EXTENT_X_M"
echo

STATUS=0
for freq in "${FREQ_ARRAY[@]}"; do
  void_root="$(root_for_freq void "$freq")"
  novoid_root="$(root_for_freq novoid "$freq")"
  echo "================================================================================"
  echo "Frequency ${freq} Hz"
  echo "  void root:   $void_root"
  echo "  no-void root:$novoid_root"
  echo "================================================================================"

  if [[ "$RUN_SYN_VOID_NOVOID" == "1" ]]; then
    for component_file in "${SYN_COMPONENT_FILES[@]}"; do
      run_syn_void_novoid "$freq" "$component_file" "$void_root" "$novoid_root" || STATUS=1
    done
  fi

  if [[ "$RUN_REAL_NOVOID" == "1" ]]; then
    for component_file in "${REAL_COMPONENT_FILES[@]}"; do
      run_real_novoid "$freq" "$component_file" "$novoid_root" || STATUS=1
    done
  fi
  echo
done

if [[ "$RUN_MOVIES" == "1" ]]; then
  if [[ -f "$MOVIE_SCRIPT" ]]; then
    MOVIE_ROOT="${MOVIE_ROOT:-$OUT_ROOT/movies}"
    FPS="${MOVIE_FPS:-4}" MAP_DEPTH=1 bash "$MOVIE_SCRIPT" "$OUT_ROOT" "$MOVIE_ROOT"
  else
    echo "WARNING: movie script not found: $MOVIE_SCRIPT"
  fi
fi

echo "Done. Products: $OUT_ROOT"
echo "Logs: $OUT_ROOT/logs"
exit "$STATUS"
