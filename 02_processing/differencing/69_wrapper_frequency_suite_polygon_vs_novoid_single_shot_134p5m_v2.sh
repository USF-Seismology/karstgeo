#!/usr/bin/env bash
# 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v1.sh
#
# Compare SPECFEM2D synthetic shot gathers for a single shot at x = 134.5 m
# across Felix's Ricker source-frequency suite:
#
#   FREQUENCY_TEST_9_LAYER_MODEL/NO_VOID/{30,40,50,60,70,80,100}Hz
#   FREQUENCY_TEST_9_LAYER_MODEL/POLYGON/{30,40,50,60,70,80,100}Hz
#
# It auto-detects either:
#   OUTPUT_FILES/Ux_file_single_v.su
# or older survey-style:
#   SURVEY_OUTPUT/shot_001_xs00134p5/Ux_file_single_v.su
#
# For each frequency and component, this calls 65_compare_gather_pairs_v10.py
# and writes the usual three-panel products:
#
#   left   = reference model
#   middle = comparison model
#   right  = reference - comparison
#
# Default panel order here is:
#   reference  = NO_VOID
#   comparison = POLYGON
# so the right panel is NO_VOID - POLYGON.
#
# To reverse sign and panel order, run with:
#   REFERENCE_MODEL=POLYGON COMPARISON_MODEL=NO_VOID bash 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v1.sh
#
# Run examples:
#   bash 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v1.sh
#   RUN_MOVIES=1 bash 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v1.sh
#   MAX_JOBS=4 RUN_MOVIES=1 bash 69_wrapper_frequency_suite_polygon_vs_novoid_single_shot_134p5m_v1.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$SCRIPT_DIR/65_compare_gather_pairs_v11.py}"

# Convenience fallback for ChatGPT-downloaded files that sometimes get "(1)" suffixes.
if [[ ! -f "$ENGINE" && -f "$SCRIPT_DIR/65_compare_gather_pairs_v10(1).py" ]]; then
  ENGINE="$SCRIPT_DIR/65_compare_gather_pairs_v10(1).py"
fi

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
SIM_ROOT="${SIM_ROOT:-$BASE/02_Modelling/Seismic/specfem2d/felix/T1_9_LayerModel_Simulation/FREQUENCY_TEST_9_LAYER_MODEL}"

OUT_ROOT="${OUT_ROOT:-$BASE/02_Modelling/Seismic/differencing/frequency_suite_polygon_vs_novoid_single_shot_134p5m}"
MOVIE_ROOT="${MOVIE_ROOT:-$BASE/02_Modelling/Seismic/differencing/movies_frequency_suite_polygon_vs_novoid_single_shot_134p5m}"

# Model names must match subfolders under $SIM_ROOT.
REFERENCE_MODEL="${REFERENCE_MODEL:-NO_VOID}"
COMPARISON_MODEL="${COMPARISON_MODEL:-POLYGON}"

# Frequencies are subfolder stems, e.g. 30Hz, 40Hz, ...
FREQS=(${FREQS:-30 40 50 60 70 80 100})
COMPONENT_FILES=(${COMPONENT_FILES:-Ux_file_single_v.su Uz_file_single_v.su})

# For these frequency-test runs Felix usually has single-shot output directly in OUTPUT_FILES.
# Some older survey-style runs use SURVEY_OUTPUT/shot_001_xs00134p5.
# Leave SHOT_DIR=auto unless you know which layout you want.
SHOT_DIR="${SHOT_DIR:-auto}"
SURVEY_SHOT_DIR="${SURVEY_SHOT_DIR:-SURVEY_OUTPUT/shot_001_xs00134p5}"
CAVE_EXTENT_X_M="${CAVE_EXTENT_X_M:-115,125}"
MAX_FREQ_HZ="${MAX_FREQ_HZ:-400}"

RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
RUN_MOVIES="${RUN_MOVIES:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
MOVIE_FPS="${MOVIE_FPS:-1}"
MIN_MOVIE_FRAMES="${MIN_MOVIE_FRAMES:-2}"
WRITE_BANDPASS="${WRITE_BANDPASS:-1}"
WRITE_FK="${WRITE_FK:-1}"
DRY_RUN="${DRY_RUN:-0}"

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
    *) echo "Synthetic $1" ;;
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

COMMON_ARGS=(
  --mode synthetic_vs_synthetic
  --scale-mode none
  --max-freq-hz "$MAX_FREQ_HZ"
  --write-diff-segy
  --write-combined-three-panel-products

  # Synthetic-vs-synthetic: preserve shared physical amplitudes/scales.
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

  --cave-extent-x-m "$CAVE_EXTENT_X_M"
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

run_one_frequency_component() {
  local freq="$1"
  local component_file="$2"
  local comp ref_root cmp_root data_dir par_file outdir log ref_file cmp_file

  comp="$(component_name "$component_file")"
  ref_root="$SIM_ROOT/$REFERENCE_MODEL/${freq}Hz"
  cmp_root="$SIM_ROOT/$COMPARISON_MODEL/${freq}Hz"

  # Use the reference model's DATA/Par_file by default. If absent, fall back to comparison.
  data_dir="$ref_root/DATA"
  par_file="$ref_root/DATA/Par_file"
  [[ -d "$data_dir" ]] || data_dir="$cmp_root/DATA"
  [[ -f "$par_file" ]] || par_file="$cmp_root/DATA/Par_file"

  outdir="$OUT_ROOT/${freq}Hz/$comp"
  log="$OUT_ROOT/logs/${freq}Hz_${REFERENCE_MODEL}_vs_${COMPARISON_MODEL}_${comp}.log"
  mkdir -p "$outdir"

  # Resolve the relative file paths separately for reference and comparison.
  # This avoids silently using stale/empty SURVEY_OUTPUT products when the actual
  # frequency-test output is in OUTPUT_FILES.
  if [[ "$SHOT_DIR" == "auto" ]]; then
    ref_rel=""
    cmp_rel=""
    for candidate in "OUTPUT_FILES/$component_file" "$SURVEY_SHOT_DIR/$component_file" "$component_file"; do
      if [[ -z "$ref_rel" && -f "$ref_root/$candidate" ]]; then
        ref_rel="$candidate"
      fi
      if [[ -z "$cmp_rel" && -f "$cmp_root/$candidate" ]]; then
        cmp_rel="$candidate"
      fi
    done
  else
    ref_rel="$SHOT_DIR/$component_file"
    cmp_rel="$SHOT_DIR/$component_file"
  fi

  if [[ -z "${ref_rel:-}" ]]; then
    ref_rel="OUTPUT_FILES/$component_file"
  fi
  if [[ -z "${cmp_rel:-}" ]]; then
    cmp_rel="OUTPUT_FILES/$component_file"
  fi

  ref_file="$ref_root/$ref_rel"
  cmp_file="$cmp_root/$cmp_rel"

  echo
  echo "================================================================================"
  echo "Frequency ${freq} Hz, component $comp"
  echo "Reference:  $ref_file"
  echo "Comparison: $cmp_file"
  echo "Output:     $outdir"
  echo "Log:        $log"
  echo "================================================================================"

  if [[ ! -f "$ref_file" ]]; then
    echo "ERROR: missing reference file: $ref_file" | tee "$log"
    return 10
  fi
  if [[ ! -f "$cmp_file" ]]; then
    echo "ERROR: missing comparison file: $cmp_file" | tee "$log"
    return 11
  fi
  if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: missing DATA directory: $data_dir" | tee "$log"
    return 12
  fi
  if [[ ! -f "$par_file" ]]; then
    echo "ERROR: missing Par_file: $par_file" | tee "$log"
    return 13
  fi

  run_bg "$log" "$PYTHON" "$ENGINE" \
    --data-dir "$data_dir" \
    --par-file "$par_file" \
    --reference-dir "$ref_root" \
    --comparison-dir "$cmp_root" \
    --reference-pattern "$ref_rel" \
    --comparison-pattern "$cmp_rel" \
    --reference-label "$(model_label "$REFERENCE_MODEL") ${freq}Hz" \
    --comparison-label "$(model_label "$COMPARISON_MODEL") ${freq}Hz" \
    --output-dir "$outdir" \
    "${COMMON_ARGS[@]}"
}

make_frequency_movies() {
  command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found; cannot make movies"; return 1; }
  command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found; cannot make movies"; return 1; }

  echo
  echo "Making frequency movies from: $OUT_ROOT"
  echo "Movie output: $MOVIE_ROOT"

  ROOT="$OUT_ROOT" \
  OUTDIR="$MOVIE_ROOT" \
  FPS="$MOVIE_FPS" \
  MIN_FRAMES="$MIN_MOVIE_FRAMES" \
  EXT="png" \
  FREQS_CSV="$(IFS=,; echo "${FREQS[*]}")" \
  python3 <<'PY'
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

crf = os.environ.get("CRF", "18")
preset = os.environ.get("PRESET", "slow")
dry_run = os.environ.get("DRY_RUN", "0") == "1"
keep_frames = os.environ.get("KEEP_FRAMES", "0") == "1"

outdir.mkdir(parents=True, exist_ok=True)
workdir = outdir / "_frames_tmp"
if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)

plan_json = outdir / "frequency_movie_plan.json"
manifest_csv = outdir / "frequency_movie_manifest.csv"

freq_re = re.compile(r"^(\d+(?:\.\d+)?)Hz$")
shot_re = re.compile(r"^\d{3}[_-]|^\d{3,4}")

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "movie"

def freq_sort_key(freq_label: str):
    m = freq_re.match(freq_label)
    if not m:
        return (999999, freq_label)
    f = int(float(m.group(1)))
    return (freq_rank.get(f, 100000 + f), f)

def classify_png(path: Path):
    rel = path.relative_to(root)
    parts = rel.parts

    freq_label = None
    freq_idx = None
    for i, part in enumerate(parts):
        if freq_re.match(part):
            freq_label = part
            freq_idx = i
            break
    if freq_label is None:
        return None

    comp = "all"
    comp_idx = None
    for i, part in enumerate(parts):
        if part in ("Ux", "Uz"):
            comp = part
            comp_idx = i
            break
    if comp_idx is None:
        return None

    after_comp = parts[comp_idx + 1:]
    if len(after_comp) < 2:
        return None

    # 65_compare writes: component/shot_folder/product_or_png
    # Examples:
    #   30Hz/Ux/001_xs00134p5/combined_image_....png
    #   30Hz/Ux/001_xs00134p5/diagnostic_bandpass_25_400Hz/combined_image_....png
    shot_folder = after_comp[0]
    if len(after_comp) == 2:
        product = "baseline"
        figure_name = after_comp[-1]
    else:
        product = "/".join(after_comp[1:-1])
        figure_name = after_comp[-1]

    return comp, shot_folder, product, figure_name, freq_label

groups = defaultdict(list)
for p in root.rglob(f"*.{ext}"):
    if any(part in {"_movies", "_frames_tmp", "movies"} for part in p.parts):
        continue
    cls = classify_png(p)
    if cls is None:
        continue
    comp, shot_folder, product, figure_name, freq_label = cls
    groups[(comp, shot_folder, product, figure_name)].append((freq_label, p))

movies = []
rows = []
for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
    comp, shot_folder, product, figure_name = key
    items = sorted(items, key=lambda item: freq_sort_key(item[0]))
    if len(items) < min_frames:
        continue

    movie_name = safe_name(f"{comp}_{shot_folder}_{product}_{Path(figure_name).stem}_frequency_sweep") + ".mp4"
    movie = outdir / comp / movie_name
    frames = [p for _, p in items]
    movies.append({
        "component": comp,
        "shot_folder": shot_folder,
        "product": product,
        "figure_name": figure_name,
        "movie": str(movie),
        "frames": [str(p) for p in frames],
        "frequencies": [f for f, _ in items],
    })
    for i, (freq_label, p) in enumerate(items, 1):
        rows.append({
            "movie": str(movie),
            "component": comp,
            "shot_folder": shot_folder,
            "product": product,
            "figure_name": figure_name,
            "frame_index": i,
            "frequency": freq_label,
            "frame_path": str(p),
        })

plan_json.write_text(json.dumps(movies, indent=2), encoding="utf-8")
with manifest_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["movie", "component", "shot_folder", "product", "figure_name", "frame_index", "frequency", "frame_path"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Detected {len(movies)} frequency movies")
for m in movies:
    print(f"  {Path(m['movie']).name}: {len(m['frames'])} frames [{', '.join(m['frequencies'])}]")

if not movies:
    print(f"No movies created. Manifest: {manifest_csv}")
    raise SystemExit(0)

for mi, m in enumerate(movies, 1):
    movie = Path(m["movie"])
    movie.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = workdir / f"{mi:03d}_{movie.stem}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(m["frames"][0]).suffix.lower()
    for i, src in enumerate(m["frames"], 1):
        shutil.copy2(Path(src), frame_dir / f"frame_{i:05d}{suffix}")
    pattern = frame_dir / f"frame_%05d{suffix}"
    print(f"\nMaking movie: {movie}")
    print(f"  component={m['component']} product={m['product']} figure={m['figure_name']}")
    print(f"  frames={len(m['frames'])}: {', '.join(m['frequencies'])}")
    if dry_run:
        print("  DRY_RUN=1; skipping ffmpeg")
        continue
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", str(fps), "-i", str(pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie),
    ]
    subprocess.run(cmd, check=True)

if keep_frames:
    print(f"Temporary frames kept at: {workdir}")
else:
    shutil.rmtree(workdir, ignore_errors=True)
print(f"\nMovies written to: {outdir}")
print(f"Manifest written to: {manifest_csv}")
print(f"Plan written to:     {plan_json}")
PY
}

echo "Engine:           $ENGINE"
echo "SIM_ROOT:         $SIM_ROOT"
echo "OUT_ROOT:         $OUT_ROOT"
echo "MOVIE_ROOT:       $MOVIE_ROOT"
echo "Reference model:  $REFERENCE_MODEL  (left panel)"
echo "Comparison model: $COMPARISON_MODEL (middle panel)"
echo "Difference panel: $REFERENCE_MODEL - $COMPARISON_MODEL"
echo "Frequencies:      ${FREQS[*]} Hz"
echo "Shot dir:         $SHOT_DIR"
echo "Survey shot dir:  $SURVEY_SHOT_DIR"
echo "Cave extent:      $CAVE_EXTENT_X_M m"
echo "MAX_JOBS:         $MAX_JOBS"
echo "RUN_COMPARISONS:  $RUN_COMPARISONS"
echo "RUN_MOVIES:       $RUN_MOVIES"
echo

echo "Checking key paths:"
for freq in "${FREQS[@]}"; do
  for model in "$REFERENCE_MODEL" "$COMPARISON_MODEL"; do
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

STATUS=0
if [[ "$RUN_COMPARISONS" == "1" ]]; then
  for freq in "${FREQS[@]}"; do
    for component_file in "${COMPONENT_FILES[@]}"; do
      if ! run_one_frequency_component "$freq" "$component_file"; then
        STATUS=1
      fi
    done
  done

  echo
  echo "Waiting for comparison jobs..."
  wait || STATUS=1
  echo "Comparison jobs complete. Logs: $OUT_ROOT/logs"
fi

if [[ "$RUN_MOVIES" == "1" ]]; then
  if ! make_frequency_movies; then
    STATUS=1
  fi
fi

echo
echo "Done."
echo "Products: $OUT_ROOT"
echo "Logs:     $OUT_ROOT/logs"
echo "Movies:   $MOVIE_ROOT"
exit "$STATUS"
