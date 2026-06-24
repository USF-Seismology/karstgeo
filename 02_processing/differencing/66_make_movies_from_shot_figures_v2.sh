#!/usr/bin/env bash
# 66_make_movies_from_shot_figures_v2.sh
#
# Make one MP4 movie per repeated PNG figure type in a shot-gather output tree.
#
# Usage:
#   bash 66_make_movies_from_shot_figures_v2.sh ROOT_DIR [MOVIE_ROOT]
#
# If MOVIE_ROOT is omitted:
#   movies go to ROOT_DIR/_movies
#
# If MOVIE_ROOT is supplied:
#   movies go to MOVIE_ROOT/<last MAP_DEPTH parts of ROOT_DIR>
#
# This lets you map several result folders into a common movie tree, e.g.:
#
#   MAP_DEPTH=2 bash 66_make_movies_from_shot_figures_v2.sh \
#     /.../synthetic_cave_vs_nocave_comparison_v2/Uz \
#     /.../differencing/movies
#
# writes to:
#
#   /.../differencing/movies/synthetic_cave_vs_nocave_comparison_v2/Uz/
#
# Defaults:
#   FPS=1
#   EXT=png
#   MIN_FRAMES=2
#   MAP_DEPTH=2
#
# Optional environment variables:
#   FPS=1
#   EXT=png
#   MIN_FRAMES=2
#   MAP_DEPTH=2
#   INCLUDE_REGEX='.*'
#   EXCLUDE_REGEX='(^$)'
#   CRF=18
#   PRESET=slow
#   DRY_RUN=1
#   KEEP_FRAMES=1
#   NO_MAP_OUTPUT=1     # use MOVIE_ROOT exactly, without appending path tail

set -euo pipefail

ROOT="${1:-}"
MOVIE_ROOT="${2:-}"

if [[ -z "$ROOT" ]]; then
  echo "Usage: bash $0 ROOT_DIR [MOVIE_ROOT]"
  exit 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: ROOT_DIR does not exist: $ROOT"
  exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"

FPS="${FPS:-1}"
EXT="${EXT:-png}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAP_DEPTH="${MAP_DEPTH:-2}"
INCLUDE_REGEX="${INCLUDE_REGEX:-.*}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-(^$)}"
CRF="${CRF:-18}"
PRESET="${PRESET:-slow}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"
NO_MAP_OUTPUT="${NO_MAP_OUTPUT:-0}"

if [[ -z "$MOVIE_ROOT" ]]; then
  OUTDIR="$ROOT/_movies"
else
  mkdir -p "$MOVIE_ROOT"
  MOVIE_ROOT="$(cd "$MOVIE_ROOT" && pwd)"
  if [[ "$NO_MAP_OUTPUT" == "1" ]]; then
    OUTDIR="$MOVIE_ROOT"
  else
    # Append the last MAP_DEPTH parts of ROOT to MOVIE_ROOT.
    TAIL="$(python3 - "$ROOT" "$MAP_DEPTH" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
n = int(sys.argv[2])
parts = p.parts[-n:] if n > 0 else ()
print(str(Path(*parts)) if parts else "")
PY
)"
    OUTDIR="$MOVIE_ROOT/$TAIL"
  fi
fi

mkdir -p "$OUTDIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with:"
  echo "  conda install -c conda-forge ffmpeg"
  echo "or:"
  echo "  brew install ffmpeg"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  exit 1
fi

echo "Root:       $ROOT"
echo "Outdir:     $OUTDIR"
echo "FPS:        $FPS"
echo "Extension:  $EXT"
echo "Min frames: $MIN_FRAMES"
echo "Map depth:  $MAP_DEPTH"
echo

export ROOT OUTDIR FPS EXT MIN_FRAMES INCLUDE_REGEX EXCLUDE_REGEX CRF PRESET DRY_RUN KEEP_FRAMES

python3 <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["ROOT"])
outdir = Path(os.environ["OUTDIR"])
fps = float(os.environ["FPS"])
ext = os.environ["EXT"].lstrip(".")
min_frames = int(os.environ["MIN_FRAMES"])
include_re = re.compile(os.environ["INCLUDE_REGEX"])
exclude_re = re.compile(os.environ["EXCLUDE_REGEX"])
crf = os.environ["CRF"]
preset = os.environ["PRESET"]
dry_run = os.environ["DRY_RUN"] == "1"
keep_frames = os.environ["KEEP_FRAMES"] == "1"

outdir.mkdir(parents=True, exist_ok=True)
workdir = outdir / "_frames_tmp"
if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)

plan_json = outdir / "movie_plan.json"
manifest_csv = outdir / "movie_manifest.csv"

def natural_key(text: str):
    parts = re.split(r"(\d+(?:\.\d+)?)", text)
    key = []
    for part in parts:
        if not part:
            continue
        try:
            key.append(float(part))
        except ValueError:
            key.append(part.lower())
    return key

def safe_name(text: str) -> str:
    stem = Path(text).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "figure"

# Group by exact PNG filename.
groups: dict[str, list[Path]] = defaultdict(list)
for p in root.rglob(f"*.{ext}"):
    if "_movies" in p.parts or "_frames_tmp" in p.parts or "movies" in p.parts:
        continue
    if not include_re.search(p.name):
        continue
    if exclude_re.search(p.name):
        continue
    groups[p.name].append(p)

movies = []
manifest_rows = []

for fname, files in sorted(groups.items(), key=lambda kv: natural_key(kv[0])):
    # If root contains Ux and Uz, split movies by component.
    component_groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        rel = p.relative_to(root)
        comp = "all"
        for part in rel.parts:
            if part in ("Ux", "Uz"):
                comp = part
                break
        component_groups[comp].append(p)

    for comp, cfiles in sorted(component_groups.items()):
        cfiles = sorted(cfiles, key=lambda p: natural_key(p.parent.name) + natural_key(str(p)))
        if len(cfiles) < min_frames:
            continue

        suffix = "" if comp == "all" else f"_{comp}"
        movie_path = outdir / f"{safe_name(fname)}{suffix}.mp4"
        movie = {
            "figure_name": fname,
            "component": comp,
            "movie": str(movie_path),
            "frames": [str(p) for p in cfiles],
        }
        movies.append(movie)

        for i, p in enumerate(cfiles, start=1):
            manifest_rows.append({
                "movie": str(movie_path),
                "figure_name": fname,
                "component": comp,
                "frame_index": i,
                "shot_folder": p.parent.name,
                "frame_path": str(p),
            })

plan_json.write_text(json.dumps(movies, indent=2), encoding="utf-8")

with manifest_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["movie", "figure_name", "component", "frame_index", "shot_folder", "frame_path"],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Detected {len(movies)} movies")
for m in movies:
    print(f"  {Path(m['movie']).name}: {len(m['frames'])} frames")

if not movies:
    print("No movies to create. Try lowering MIN_FRAMES or changing INCLUDE_REGEX.")
    print(f"Manifest: {manifest_csv}")
    raise SystemExit(0)

for mi, movie in enumerate(movies, start=1):
    movie_path = Path(movie["movie"])
    movie_path.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = workdir / f"{mi:03d}_{movie_path.stem}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(movie["frames"][0]).suffix.lower()
    for i, src_str in enumerate(movie["frames"], start=1):
        src = Path(src_str)
        dst = frame_dir / f"frame_{i:05d}{suffix}"
        shutil.copy2(src, dst)

    pattern = frame_dir / f"frame_%05d{suffix}"

    print()
    print("Making movie:")
    print(f"  figure: {movie['figure_name']}")
    print(f"  frames: {len(movie['frames'])}")
    print(f"  output: {movie_path}")

    if dry_run:
        print("  DRY_RUN=1; skipping ffmpeg")
        continue

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-framerate", str(fps),
        "-i", str(pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(movie_path),
    ]
    subprocess.run(cmd, check=True)

if not keep_frames:
    shutil.rmtree(workdir, ignore_errors=True)
else:
    print(f"Temporary frames kept at: {workdir}")

print()
print(f"Movies written to: {outdir}")
print(f"Manifest written to: {manifest_csv}")
print(f"Plan written to:     {plan_json}")
PY
