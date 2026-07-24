#!/usr/bin/env bash
# 102_make_virtual_T1_animation.sh
#
# Build PNG products from notebook-100 per-shot SEG-Y gathers and encode MP4
# animations in ascending source-position order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/tachyon/LBSSP_DATA}"
SEGY_DIR="${SEGY_DIR:-$PROJECT_ROOT/100_virtual_T1_SEGY/per_shot_segy}"
FRAME_ROOT="${FRAME_ROOT:-$PROJECT_ROOT/101_virtual_T1_animation_frames}"
MOVIE_ROOT="${MOVIE_ROOT:-$PROJECT_ROOT/102_virtual_T1_movies}"

PLOTTER="${PLOTTER:-$SCRIPT_DIR/101_plot_virtual_T1_shot_gathers.py}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/../02_differencing/66_make_movies_from_shot_figures_v4.sh}"

PYTHON="${PYTHON:-python}"
FPS="${FPS:-4}"
TMIN_S="${TMIN_S:-0.0}"
TMAX_S="${TMAX_S:-1.25}"
CLIP_PERCENTILE="${CLIP_PERCENTILE:-99.0}"
WIGGLE_SCALE="${WIGGLE_SCALE:-0.45}"
DPI="${DPI:-180}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-}"

[[ -d "$SEGY_DIR" ]] || {
  echo "ERROR: missing per-shot SEG-Y directory: $SEGY_DIR" >&2
  exit 1
}
[[ -f "$PLOTTER" ]] || {
  echo "ERROR: missing plotter: $PLOTTER" >&2
  exit 1
}
[[ -f "$MOVIE_SCRIPT" ]] || {
  echo "ERROR: missing movie script: $MOVIE_SCRIPT" >&2
  exit 1
}

mkdir -p "$FRAME_ROOT" "$MOVIE_ROOT"

PLOT_ARGS=(
  --input-dir "$SEGY_DIR"
  --output-dir "$FRAME_ROOT"
  --component Z
  --tmin-s "$TMIN_S"
  --tmax-s "$TMAX_S"
  --clip-percentile "$CLIP_PERCENTILE"
  --wiggle-scale "$WIGGLE_SCALE"
  --dpi "$DPI"
)

if [[ "$OVERWRITE" == "1" ]]; then
  PLOT_ARGS+=(--overwrite)
fi

if [[ -n "$LIMIT" ]]; then
  PLOT_ARGS+=(--limit "$LIMIT")
fi

echo "Generating animation frames..."
"$PYTHON" "$PLOTTER" "${PLOT_ARGS[@]}"

echo
echo "Encoding movies in ascending source-position order..."
FPS="$FPS" \
MAP_DEPTH=1 \
MIN_FRAMES=2 \
SORT_MODE=source_x \
REQUIRE_SOURCE_X=1 \
NO_MAP_OUTPUT=1 \
  bash "$MOVIE_SCRIPT" "$FRAME_ROOT" "$MOVIE_ROOT"

echo
echo "Virtual T1 animation products:"
echo "  Frames: $FRAME_ROOT"
echo "  Movies: $MOVIE_ROOT"
echo "  Order manifest: $MOVIE_ROOT/movie_manifest.csv"
