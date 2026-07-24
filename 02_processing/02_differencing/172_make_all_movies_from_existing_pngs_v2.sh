#!/usr/bin/env bash
# 172_make_all_movies_from_existing_pngs_v2.sh
#
# Rebuild movies from existing comparison PNGs without rerunning the comparison
# engine. Frames are ordered by source coordinate parsed from folder names such
# as x0082p500m, not by folder creation number or modification time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${BASE:-$HOME/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP}"
DIFF_ROOT="${DIFF_ROOT:-$BASE/02_Modelling/Seismic/differencing_final}"
MOVIE_SCRIPT="${MOVIE_SCRIPT:-$SCRIPT_DIR/66_make_movies_from_shot_figures_v4.sh}"

MOVIE_FPS="${MOVIE_FPS:-1}"
MIN_MOVIE_FRAMES="${MIN_MOVIE_FRAMES:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
ONLY="${ONLY:-t1_synth t1_real t3t4 rectpoly freq_synth freq_real}"

# New in v2: source-coordinate ordering is explicit.
SORT_MODE="${SORT_MODE:-source_x}"
REQUIRE_SOURCE_X="${REQUIRE_SOURCE_X:-1}"

[[ -f "$MOVIE_SCRIPT" ]] || {
  echo "ERROR: missing movie script: $MOVIE_SCRIPT" >&2
  exit 1
}

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: ffmpeg not found. Install ffmpeg or add it to PATH." >&2
  exit 1
}

run_movies() {
  local key="$1"
  local root="$2"
  local movie_root="$3"
  local label="$4"
  local depth="$5"

  case " $ONLY " in
    *" $key "*) ;;
    *) return 0 ;;
  esac

  echo
  echo "======================================================================"
  echo "Movies: $label"
  echo "Root:   $root"
  echo "Out:    $movie_root"
  echo "Depth:  $depth"
  echo "Sort:   $SORT_MODE"
  echo "======================================================================"

  if [[ ! -d "$root" ]]; then
    echo "WARNING: missing PNG root, skipping: $root" >&2
    return 0
  fi

  mkdir -p "$movie_root"
  local log="$movie_root/movies_${key}.log"

  if [[ "$STOP_ON_ERROR" == "1" ]]; then
    FPS="$MOVIE_FPS" \
    MAP_DEPTH="$depth" \
    MIN_FRAMES="$MIN_MOVIE_FRAMES" \
    SORT_MODE="$SORT_MODE" \
    REQUIRE_SOURCE_X="$REQUIRE_SOURCE_X" \
      bash "$MOVIE_SCRIPT" "$root" "$movie_root" 2>&1 | tee "$log"
  else
    FPS="$MOVIE_FPS" \
    MAP_DEPTH="$depth" \
    MIN_FRAMES="$MIN_MOVIE_FRAMES" \
    SORT_MODE="$SORT_MODE" \
    REQUIRE_SOURCE_X="$REQUIRE_SOURCE_X" \
      bash "$MOVIE_SCRIPT" "$root" "$movie_root" 2>&1 | tee "$log" || \
      echo "WARNING: movie generation failed for $label; see $log" >&2
  fi
}

T1_ROOT="$DIFF_ROOT/comparison_products_final"
T1_MOVIES="$DIFF_ROOT/movies_final"

T3T4_ROOT="$DIFF_ROOT/real_vs_synthetic_novoid_T3_T4_v18_consolidated"
T3T4_MOVIES="$DIFF_ROOT/movies_real_vs_synthetic_novoid_T3_T4_v18_consolidated"

RECTPOLY_ROOT="$DIFF_ROOT/rectangle_vs_polygon_single_shot_134p5m_v18_consolidated"
RECTPOLY_MOVIES="$DIFF_ROOT/movies_rectangle_vs_polygon_single_shot_134p5m_v18_consolidated"

FREQ_ROOT="$DIFF_ROOT/frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v18_consolidated"
FREQ_MOVIES="$DIFF_ROOT/movies_frequency_suite_polygon_vs_novoid_and_real_novoid_single_shot_134p5m_v18_consolidated"

run_movies \
  "t1_synth" \
  "$T1_ROOT/synthetic_cave_vs_nocave" \
  "$T1_MOVIES" \
  "T1 synthetic cave/no-cave" \
  2

run_movies \
  "t1_real" \
  "$T1_ROOT/real_vs_synthetic_novoid" \
  "$T1_MOVIES" \
  "T1 real vs synthetic no-void" \
  2

run_movies \
  "t3t4" \
  "$T3T4_ROOT/T3/real_vs_synthetic_novoid" \
  "$T3T4_MOVIES" \
  "T3 real vs synthetic no-void" \
  3

run_movies \
  "t3t4" \
  "$T3T4_ROOT/T4/real_vs_synthetic_novoid" \
  "$T3T4_MOVIES" \
  "T4 real vs synthetic no-void" \
  3

run_movies \
  "rectpoly" \
  "$RECTPOLY_ROOT" \
  "$RECTPOLY_MOVIES" \
  "Rectangle cave vs polygon cave" \
  1

run_movies \
  "freq_synth" \
  "$FREQ_ROOT/synthetic_NO_VOID_vs_POLYGON" \
  "$FREQ_MOVIES" \
  "Frequency suite synthetic NO_VOID vs POLYGON" \
  3

run_movies \
  "freq_real" \
  "$FREQ_ROOT/real_T1_1m_vs_synthetic_NO_VOID" \
  "$FREQ_MOVIES" \
  "Frequency suite real T1 vs NO_VOID" \
  3

echo
echo "Movie-only pass finished."
