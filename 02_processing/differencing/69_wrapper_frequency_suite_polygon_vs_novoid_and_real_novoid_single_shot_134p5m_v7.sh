#!/usr/bin/env bash
# 69_wrapper_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v7.sh
#
# Rebased on 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v2.sh.
# Preserves Felix's frequency array: 30 40 50 60 70 80 100 Hz.
#
# Runs:
#   1) synthetic NO_VOID vs synthetic POLYGON, by frequency and component
#   2) real T1 1-m refraction vs synthetic NO_VOID, by frequency
#
# f-k products are enabled by default at vmin=500 m/s.
# Cave shading is applied only to the synthetic cave panel and difference panel.
# Real-vs-NO_VOID plots have no cave shading.

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v13.py}"
if [[ ! -f "$ENGINE" && -f "$SCRIPT_DIR/65_compare_gather_pairs_v11.py" ]]; then
  echo "WARNING: v12 engine not found; falling back to v11. Cave panel-specific shading may not work." >&2
  ENGINE="$SCRIPT_DIR/65_compare_gather_pairs_v11.py"
fi

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SIM_ROOT="${SIM_ROOT:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/FREQUENCY_TEST_9_LAYER_MODEL}"
OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v7}"
MOVIE_ROOT="${MOVIE_ROOT:-$BASE/02_Modelling/Seismic/differencing/movies_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v7}"

REFERENCE_MODEL="${REFERENCE_MODEL:-NO_VOID}"
COMPARISON_MODEL="${COMPARISON_MODEL:-POLYGON}"

# Correct frequency suite from the uploaded 69-wrapper.
FREQS=(${FREQS:-30 40 50 60 70 80 100})
SYN_COMPONENT_FILES=(${SYN_COMPONENT_FILES:-Ux_file_single_v.su Uz_file_single_v.su})
REAL_COMPONENT_FILES=(${REAL_COMPONENT_FILES:-Uz_file_single_v.su})

SHOT_DIR="${SHOT_DIR:-auto}"
SURVEY_SHOT_DIR="${SURVEY_SHOT_DIR:-SURVEY_OUTPUT/shot_001_xs00134p5}"
CAVE_EXTENT_X_M="${CAVE_EXTENT_X_M:-115,125}"
MAX_FREQ_HZ="${MAX_FREQ_HZ:-400}"

RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_REAL_NOVOID="${RUN_REAL_NOVOID:-1}"
RUN_MOVIES="${RUN_MOVIES:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
MOVIE_FPS="${MOVIE_FPS:-1}"
MIN_MOVIE_FRAMES="${MIN_MOVIE_FRAMES:-2}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"

# Real T1 1-m refraction defaults.
#
# REAL_DIR can be supplied explicitly. Default is the T1 field-data folder.
# Set RUN_REAL_NOVOID=0 if you want to skip real comparisons.
DEFAULT_REAL_DIR="$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/04_FieldData/051826/051826_Seismics_T1"
REAL_DIR="${REAL_DIR:-$DEFAULT_REAL_DIR}"
AUTO_FIND_REAL_DIR="${AUTO_FIND_REAL_DIR:-1}"
REAL_SEARCH_ROOTS=(
  "${REAL_SEARCH_ROOTS:-}"
  "$BASE"
  "$HOME/Library/CloudStorage/Box-Box"
  "/Volumes/tachyon/LBSSP_DATA"
  "/Volumes/Tachyon/LBSSP_DATA"
  "/Volumes/tachyon"
  "/Volumes/Tachyon"
)
# Pattern used for the synthetic side of real-vs-synthetic. The frequency-test
# suite only exists for the single 134.5 m shot, so keep this fixed to that shot.
# Use {component} as a placeholder for Ux_file_single_v.su or Uz_file_single_v.su.
REAL_SYN_PATTERN_TEMPLATE="${REAL_SYN_PATTERN_TEMPLATE:-SURVEY_OUTPUT/shot_001_xs00134p5/{component}}"
REAL_FIRST_FILE="${REAL_FIRST_FILE:-3005}"
REAL_LAST_FILE="${REAL_LAST_FILE:-3046}"
REAL_SHOT_FIRST_X_M="${REAL_SHOT_FIRST_X_M:-82.5}"
REAL_SHOT_DX_M="${REAL_SHOT_DX_M:-2.0}"
REAL_SHOT_DUPLICATE_X_M="${REAL_SHOT_DUPLICATE_X_M:-102.5}"
REAL_SHOT_DUPLICATE_FILES="${REAL_SHOT_DUPLICATE_FILES:-3015,3016}"
REAL_FIRST_TRACE_X_M="${REAL_FIRST_TRACE_X_M:-87.0}"
REAL_DX_M="${REAL_DX_M:-1.0}"
SHOT_MATCH_TOLERANCE_M="${SHOT_MATCH_TOLERANCE_M:-0.25}"

mkdir -p "$OUT_ROOT/logs" "$MOVIE_ROOT"
[[ -f "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE"; exit 1; }

component_name() {
  case "$1" in
    Ux*) echo "Ux" ;;
    Uz*) echo "Uz" ;;
    *) basename "$1" .su ;;
  esac
}

model_label() {
  case "$1" in
    NO_VOID) echo "Synthetic NO_VOID" ;;
    POLYGON) echo "Synthetic POLYGON cave" ;;
    RECTANGLE) echo "Synthetic RECTANGLE cave" ;;
    *) echo "Synthetic $1" ;;
  esac
}

is_void_model() {
  case "$1" in
    POLYGON|RECTANGLE|VOID|WITH_VOID|CAVE) return 0 ;;
    *) return 1 ;;
  esac
}

synthetic_cave_shade_panels() {
  local panels=()
  if is_void_model "$REFERENCE_MODEL"; then panels+=(reference); fi
  if is_void_model "$COMPARISON_MODEL"; then panels+=(comparison); fi
  panels+=(difference)
  local IFS=,
  echo "${panels[*]}"
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

build_common_args() {
  # Avoid `local -n` namerefs so this wrapper works with macOS /bin/bash 3.2.
  COMMON_ARGS=(
    --scale-mode none
    --max-freq-hz "$MAX_FREQ_HZ"
    --write-diff-segy
    --write-combined-three-panel-products
    --overlay-normalize pair
    --overlay-wiggle-scale "0.45"
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
  )
  if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
  fi
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
      --fk-split-at-source
    )
  fi
}

resolve_component_relpath() {
  local root="$1" component_file="$2" rel=""
  if [[ "$SHOT_DIR" == "auto" ]]; then
    for candidate in "OUTPUT_FILES/$component_file" "$SURVEY_SHOT_DIR/$component_file" "$component_file"; do
      if [[ -f "$root/$candidate" ]]; then
        rel="$candidate"
        break
      fi
    done
  else
    rel="$SHOT_DIR/$component_file"
  fi
  [[ -n "$rel" ]] || rel="OUTPUT_FILES/$component_file"
  echo "$rel"
}


find_real_dir_auto() {
  # Return first directory that contains both first and last expected SEG-2 files.
  local root hit dir first_name last_name
  first_name="${REAL_FIRST_FILE}.dat"
  last_name="${REAL_LAST_FILE}.dat"

  # Prefer likely directory names before a broader find.
  local likely=(
    "$BASE/01_Field_Data"
    "$BASE/00_Field_Data"
    "$BASE/02_RealData"
    "$BASE/02_Modelling/Seismic/real"
    "/Volumes/tachyon/LBSSP_DATA"
    "/Volumes/Tachyon/LBSSP_DATA"
  )
  for root in "${likely[@]}"; do
    [[ -n "$root" && -d "$root" ]] || continue
    while IFS= read -r hit; do
      dir="$(dirname "$hit")"
      if [[ -f "$dir/$last_name" ]]; then
        echo "$dir"
        return 0
      fi
    done < <(find "$root" -maxdepth 8 -type f -name "$first_name" 2>/dev/null | head -n 20)
  done

  for root in "${REAL_SEARCH_ROOTS[@]}"; do
    [[ -n "$root" && -d "$root" ]] || continue
    while IFS= read -r hit; do
      dir="$(dirname "$hit")"
      if [[ -f "$dir/$last_name" ]]; then
        echo "$dir"
        return 0
      fi
    done < <(find "$root" -maxdepth 8 -type f -name "$first_name" 2>/dev/null | head -n 20)
  done
  return 1
}

resolve_real_dir_or_fail() {
  if [[ -n "$REAL_DIR" && -d "$REAL_DIR" ]]; then
    return 0
  fi
  if [[ "$AUTO_FIND_REAL_DIR" == "1" ]]; then
    local found
    found="$(find_real_dir_auto || true)"
    if [[ -n "$found" && -d "$found" ]]; then
      REAL_DIR="$found"
      echo "Auto-detected REAL_DIR: $REAL_DIR"
      return 0
    fi
  fi
  echo "ERROR: RUN_REAL_NOVOID=1 but REAL_DIR is unset or missing." >&2
  echo "Set REAL_DIR to the folder containing ${REAL_FIRST_FILE}.dat ... ${REAL_LAST_FILE}.dat, e.g.:" >&2
  echo "  REAL_DIR=/path/to/T1_1m_SEG2 bash $(basename "$0")" >&2
  echo "or skip with:" >&2
  echo "  RUN_REAL_NOVOID=0 bash $(basename "$0")" >&2
  return 1
}

real_synthetic_pattern_for_component() {
  local component_file="$1"
  local pattern="$REAL_SYN_PATTERN_TEMPLATE"
  pattern="${pattern//\{component\}/$component_file}"
  echo "$pattern"
}

choose_data_and_par() {
  local preferred_root="$1" fallback_root="$2"
  local data_dir="$preferred_root/DATA"
  local par_file="$preferred_root/DATA/Par_file"
  [[ -d "$data_dir" ]] || data_dir="$fallback_root/DATA"
  [[ -f "$par_file" ]] || par_file="$fallback_root/DATA/Par_file"
  echo "$data_dir|$par_file"
}

run_synthetic_frequency_component() {
  local freq="$1" component_file="$2"
  local comp ref_root cmp_root ref_rel cmp_rel ref_file cmp_file data_par data_dir par_file outdir log shade_panels

  comp="$(component_name "$component_file")"
  ref_root="$SIM_ROOT/$REFERENCE_MODEL/${freq}Hz"
  cmp_root="$SIM_ROOT/$COMPARISON_MODEL/${freq}Hz"

  ref_rel="$(resolve_component_relpath "$ref_root" "$component_file")"
  cmp_rel="$(resolve_component_relpath "$cmp_root" "$component_file")"
  ref_file="$ref_root/$ref_rel"
  cmp_file="$cmp_root/$cmp_rel"

  data_par="$(choose_data_and_par "$ref_root" "$cmp_root")"
  data_dir="${data_par%%|*}"
  par_file="${data_par#*|}"

  outdir="$OUT_ROOT/synthetic_${REFERENCE_MODEL}_vs_${COMPARISON_MODEL}/${freq}Hz/$comp"
  log="$OUT_ROOT/logs/synthetic_${freq}Hz_${REFERENCE_MODEL}_vs_${COMPARISON_MODEL}_${comp}.log"
  shade_panels="$(synthetic_cave_shade_panels)"

  echo
  echo "================================================================================"
  echo "Synthetic frequency ${freq} Hz, component $comp"
  echo "Reference:  $ref_file"
  echo "Comparison: $cmp_file"
  echo "Output:     $outdir"
  echo "Cave shade panels: $shade_panels"
  echo "================================================================================"

  if [[ ! -f "$ref_file" ]]; then echo "ERROR: missing reference file: $ref_file" | tee "$log"; return 10; fi
  if [[ ! -f "$cmp_file" ]]; then echo "ERROR: missing comparison file: $cmp_file" | tee "$log"; return 11; fi
  if [[ ! -d "$data_dir" ]]; then echo "ERROR: missing DATA directory: $data_dir" | tee "$log"; return 12; fi
  if [[ ! -f "$par_file" ]]; then echo "ERROR: missing Par_file: $par_file" | tee "$log"; return 13; fi

  local args=(
    "$PYTHON" "$ENGINE"
    --mode synthetic_vs_synthetic
    --data-dir "$data_dir"
    --par-file "$par_file"
    --reference-dir "$ref_root"
    --comparison-dir "$cmp_root"
    --reference-pattern "$ref_rel"
    --comparison-pattern "$cmp_rel"
    --reference-label "$(model_label "$REFERENCE_MODEL") ${freq}Hz"
    --comparison-label "$(model_label "$COMPARISON_MODEL") ${freq}Hz"
    --output-dir "$outdir"
    --cave-extent-x-m "$CAVE_EXTENT_X_M"
    --cave-shade-panels "$shade_panels"
  )
  build_common_args
  args+=("${COMMON_ARGS[@]}")
  run_bg "$log" "${args[@]}"
}

run_real_novoid_frequency_component() {
  local freq="$1" component_file="$2"
  local comp novoid_root syn_rel syn_file data_dir par_file outdir log

  comp="$(component_name "$component_file")"
  novoid_root="$SIM_ROOT/NO_VOID/${freq}Hz"
  syn_rel="$(real_synthetic_pattern_for_component "$component_file")"
  # Single 134.5 m synthetic no-void frequency-test shot.
  syn_file="$novoid_root/$syn_rel"
  data_dir="$novoid_root/DATA"
  par_file="$novoid_root/DATA/Par_file"

  outdir="$OUT_ROOT/real_T1_1m_vs_synthetic_NO_VOID/${freq}Hz/$comp"
  log="$OUT_ROOT/logs/real_T1_1m_vs_NO_VOID_${freq}Hz_${comp}.log"

  echo
  echo "================================================================================"
  echo "Real T1 1-m vs synthetic NO_VOID, frequency ${freq} Hz, component $comp"
  echo "Real dir:   ${REAL_DIR:-<unset>}"
  echo "Synthetic:  $syn_file"
  echo "Output:     $outdir"
  echo "Cave shading: disabled"
  echo "================================================================================"

  if [[ -z "$REAL_DIR" || ! -d "$REAL_DIR" ]]; then
    echo "ERROR: REAL_DIR is unset or missing: ${REAL_DIR:-<unset>}" | tee "$log"
    return 20
  fi
  if [[ ! -f "$syn_file" ]]; then echo "ERROR: missing synthetic no-void file: $syn_file" | tee "$log"; return 21; fi
  if [[ ! -d "$data_dir" ]]; then echo "ERROR: missing DATA directory: $data_dir" | tee "$log"; return 22; fi
  if [[ ! -f "$par_file" ]]; then echo "ERROR: missing Par_file: $par_file" | tee "$log"; return 23; fi

  local args=(
    "$PYTHON" "$ENGINE"
    --mode real_vs_synthetic
    --data-dir "$data_dir"
    --par-file "$par_file"
    --real-dir "$REAL_DIR"
    --real-first-file "$REAL_FIRST_FILE"
    --real-last-file "$REAL_LAST_FILE"
    --real-shot-first-x-m "$REAL_SHOT_FIRST_X_M"
    --real-shot-dx-m "$REAL_SHOT_DX_M"
    --real-shot-duplicate-x-m "$REAL_SHOT_DUPLICATE_X_M"
    --real-shot-duplicate-files "$REAL_SHOT_DUPLICATE_FILES"
    --real-first-trace-x-m "$REAL_FIRST_TRACE_X_M"
    --real-dx-m "$REAL_DX_M"
    --shot-match-tolerance-m "$SHOT_MATCH_TOLERANCE_M"
    --comparison-dir "$novoid_root"
    --comparison-pattern "$syn_rel"
    --reference-label "Real T1 1-m refraction"
    --comparison-label "Synthetic NO_VOID ${freq}Hz"
    --output-dir "$outdir"
  )
  build_common_args
  args+=("${COMMON_ARGS[@]}")
  run_bg "$log" "${args[@]}"
}

make_frequency_movies() {
  command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found; cannot make movies"; return 1; }
  command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found; cannot make movies"; return 1; }

  ROOT="$OUT_ROOT" OUTDIR="$MOVIE_ROOT" FPS="$MOVIE_FPS" MIN_FRAMES="$MIN_MOVIE_FRAMES" EXT="png" FREQS_CSV="$(IFS=,; echo "${FREQS[*]}")" DRY_RUN="$DRY_RUN" python3 <<'PY'
from __future__ import annotations
import csv, json, os, re, shutil, subprocess
from collections import defaultdict
from pathlib import Path
root = Path(os.environ["ROOT"]).expanduser().resolve()
outdir = Path(os.environ["OUTDIR"]).expanduser().resolve()
fps = float(os.environ.get("FPS", "1"))
min_frames = int(os.environ.get("MIN_FRAMES", "2"))
ext = os.environ.get("EXT", "png").lstrip(".")
freq_order = [int(x) for x in os.environ.get("FREQS_CSV", "").split(",") if x.strip()]
freq_rank = {f: i for i, f in enumerate(freq_order)}
dry_run = os.environ.get("DRY_RUN", "0") == "1"
outdir.mkdir(parents=True, exist_ok=True)
workdir = outdir / "_frames_tmp"
if workdir.exists(): shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)
plan_json = outdir / "frequency_movie_plan.json"
manifest_csv = outdir / "frequency_movie_manifest.csv"
freq_re = re.compile(r"^(\d+(?:\.\d+)?)Hz$")
def safe_name(text): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "movie"
def freq_sort_key(freq_label):
    m = freq_re.match(freq_label)
    if not m: return (999999, freq_label)
    f = int(float(m.group(1)))
    return (freq_rank.get(f, 100000 + f), f)
def classify_png(path):
    rel = path.relative_to(root); parts = rel.parts
    freq_label = next((p for p in parts if freq_re.match(p)), None)
    if freq_label is None: return None
    comp = next((p for p in parts if p in ("Ux", "Uz")), None)
    if comp is None: return None
    comp_idx = parts.index(comp); after_comp = parts[comp_idx + 1:]
    if len(after_comp) < 2: return None
    shot_folder = after_comp[0]
    product = "baseline" if len(after_comp) == 2 else "/".join(after_comp[1:-1])
    figure_name = after_comp[-1]
    family = "/".join(parts[:max(0, comp_idx-1)])
    return family, comp, shot_folder, product, figure_name, freq_label
groups = defaultdict(list)
for p in root.rglob(f"*.{ext}"):
    if any(part in {"_movies", "_frames_tmp", "movies"} for part in p.parts): continue
    cls = classify_png(p)
    if cls is None: continue
    family, comp, shot_folder, product, figure_name, freq_label = cls
    groups[(family, comp, shot_folder, product, figure_name)].append((freq_label, p))
movies, rows = [], []
for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
    family, comp, shot_folder, product, figure_name = key
    items = sorted(items, key=lambda item: freq_sort_key(item[0]))
    if len(items) < min_frames: continue
    movie = outdir / comp / (safe_name(f"{family}_{comp}_{shot_folder}_{product}_{Path(figure_name).stem}_frequency_sweep") + ".mp4")
    frames = [p for _, p in items]
    movies.append({"family":family,"component":comp,"shot_folder":shot_folder,"product":product,"figure_name":figure_name,"movie":str(movie),"frames":[str(p) for p in frames],"frequencies":[f for f,_ in items]})
    for i, (freq_label, p) in enumerate(items, 1): rows.append({"movie":str(movie),"family":family,"component":comp,"shot_folder":shot_folder,"product":product,"figure_name":figure_name,"frame_index":i,"frequency":freq_label,"frame_path":str(p)})
plan_json.write_text(json.dumps(movies, indent=2), encoding="utf-8")
with manifest_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["movie","family","component","shot_folder","product","figure_name","frame_index","frequency","frame_path"]); w.writeheader(); w.writerows(rows)
print(f"Detected {len(movies)} frequency movies")
for mi, m in enumerate(movies, 1):
    movie = Path(m["movie"]); movie.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = workdir / f"{mi:03d}_{movie.stem}"; frame_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(m["frames"][0]).suffix.lower()
    for i, src in enumerate(m["frames"], 1): shutil.copy2(Path(src), frame_dir / f"frame_{i:05d}{suffix}")
    pattern = frame_dir / f"frame_%05d{suffix}"
    print(f"Making movie: {movie}")
    if dry_run: continue
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","warning","-y","-framerate",str(fps),"-i",str(pattern),"-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-preset",os.environ.get("PRESET","slow"),"-crf",os.environ.get("CRF","18"),"-pix_fmt","yuv420p","-movflags","+faststart",str(movie)], check=True)
shutil.rmtree(workdir, ignore_errors=True)
print(f"Movies written to: {outdir}")
print(f"Manifest written to: {manifest_csv}")
print(f"Plan written to: {plan_json}")
PY
}

STATUS=0

if [[ "$RUN_REAL_NOVOID" == "1" ]]; then
  if ! resolve_real_dir_or_fail; then
    exit 2
  fi
fi

echo "Engine:           $ENGINE"
echo "SIM_ROOT:         $SIM_ROOT"
echo "OUT_ROOT:         $OUT_ROOT"
echo "MOVIE_ROOT:       $MOVIE_ROOT"
echo "Synthetic ref:    $REFERENCE_MODEL  (left panel)"
echo "Synthetic comp:   $COMPARISON_MODEL (middle panel)"
echo "Synthetic diff:   $REFERENCE_MODEL - $COMPARISON_MODEL"
echo "Frequencies:      ${FREQS[*]} Hz"
echo "Shot dir:         $SHOT_DIR"
echo "Survey shot dir:  $SURVEY_SHOT_DIR"
echo "Cave extent:      $CAVE_EXTENT_X_M m"
echo "Synthetic cave shade panels: $(synthetic_cave_shade_panels)"
echo "WRITE_FK:         $WRITE_FK  (vmin=500 m/s)"
echo "REAL_DIR:         ${REAL_DIR:-<unset>}"
echo "REAL_SYN_PATTERN: ${REAL_SYN_PATTERN_TEMPLATE}"
echo "MAX_JOBS:         $MAX_JOBS"
echo "RUN_SYNTHETIC:    $RUN_SYNTHETIC"
echo "RUN_REAL_NOVOID:  $RUN_REAL_NOVOID"
echo "RUN_MOVIES:       $RUN_MOVIES"
echo

echo "Checking key synthetic paths:"
for freq in "${FREQS[@]}"; do
  for model in "$REFERENCE_MODEL" "$COMPARISON_MODEL" "NO_VOID"; do
    root="$SIM_ROOT/$model/${freq}Hz"
    [[ -d "$root" ]] && echo "  OK      $root" || echo "  MISSING $root"
    [[ -d "$root/DATA" ]] && echo "  OK      $root/DATA" || echo "  MISSING $root/DATA"
    if [[ "$SHOT_DIR" == "auto" ]]; then
      [[ -d "$root/OUTPUT_FILES" ]] && echo "  OK      $root/OUTPUT_FILES" || echo "  MISSING $root/OUTPUT_FILES"
      [[ -d "$root/$SURVEY_SHOT_DIR" ]] && echo "  OK      $root/$SURVEY_SHOT_DIR" || true
    else
      [[ -d "$root/$SHOT_DIR" ]] && echo "  OK      $root/$SHOT_DIR" || echo "  MISSING $root/$SHOT_DIR"
    fi
  done
done

if [[ "$RUN_SYNTHETIC" == "1" ]]; then
  for freq in "${FREQS[@]}"; do
    for component_file in "${SYN_COMPONENT_FILES[@]}"; do
      if ! run_synthetic_frequency_component "$freq" "$component_file"; then STATUS=1; fi
    done
  done
fi

if [[ "$RUN_REAL_NOVOID" == "1" ]]; then
  for freq in "${FREQS[@]}"; do
    for component_file in "${REAL_COMPONENT_FILES[@]}"; do
      if ! run_real_novoid_frequency_component "$freq" "$component_file"; then STATUS=1; fi
    done
  done
fi

if [[ "$RUN_SYNTHETIC" == "1" || "$RUN_REAL_NOVOID" == "1" ]]; then
  echo
  echo "Waiting for comparison jobs..."
  wait || STATUS=1
  echo "Comparison jobs complete. Logs: $OUT_ROOT/logs"
fi

if [[ "$RUN_MOVIES" == "1" ]]; then
  if ! make_frequency_movies; then STATUS=1; fi
fi

echo
echo "Done."
echo "Products: $OUT_ROOT"
echo "Logs:     $OUT_ROOT/logs"
echo "Movies:   $MOVIE_ROOT"
exit "$STATUS"
