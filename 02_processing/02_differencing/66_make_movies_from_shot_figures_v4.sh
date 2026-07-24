#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"
MOVIE_ROOT="${2:-}"

[[ -n "$ROOT" ]] || {
  echo "Usage: bash $0 ROOT_DIR [MOVIE_ROOT]"
  exit 2
}
[[ -d "$ROOT" ]] || {
  echo "ERROR: ROOT_DIR does not exist: $ROOT" >&2
  exit 1
}

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

# New in v4:
#   SORT_MODE=source_x  -> sort by source coordinate parsed from folder/path
#   SORT_MODE=natural   -> legacy natural-name ordering
SORT_MODE="${SORT_MODE:-source_x}"
REQUIRE_SOURCE_X="${REQUIRE_SOURCE_X:-0}"

if [[ -z "$MOVIE_ROOT" ]]; then
  OUTDIR="$ROOT/_movies"
else
  mkdir -p "$MOVIE_ROOT"
  MOVIE_ROOT="$(cd "$MOVIE_ROOT" && pwd)"
  if [[ "$NO_MAP_OUTPUT" == "1" ]]; then
    OUTDIR="$MOVIE_ROOT"
  else
    TAIL="$(python3 - "$ROOT" "$MAP_DEPTH" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
n = int(sys.argv[2])
print(str(Path(*p.parts[-n:])) if n > 0 else "")
PY
)"
    OUTDIR="$MOVIE_ROOT/$TAIL"
  fi
fi

mkdir -p "$OUTDIR"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: ffmpeg not found" >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 not found" >&2
  exit 1
}

echo "Root:             $ROOT"
echo "Outdir:           $OUTDIR"
echo "FPS:              $FPS"
echo "Extension:        $EXT"
echo "Min frames:       $MIN_FRAMES"
echo "Map depth:        $MAP_DEPTH"
echo "Sort mode:        $SORT_MODE"
echo "Require source x: $REQUIRE_SOURCE_X"
echo

export ROOT OUTDIR FPS EXT MIN_FRAMES INCLUDE_REGEX EXCLUDE_REGEX
export CRF PRESET DRY_RUN KEEP_FRAMES SORT_MODE REQUIRE_SOURCE_X

python3 <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
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
sort_mode = os.environ["SORT_MODE"].strip().lower()
require_source_x = os.environ["REQUIRE_SOURCE_X"] == "1"

outdir.mkdir(parents=True, exist_ok=True)
workdir = outdir / "_frames_tmp"
if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)

plan_json = outdir / "movie_plan.json"
manifest_csv = outdir / "movie_manifest.csv"


def natural_key(text):
    out = []
    for part in re.split(r"(\d+(?:\.\d+)?)", str(text)):
        if not part:
            continue
        try:
            out.append((0, float(part)))
        except ValueError:
            out.append((1, part.lower()))
    return out


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "figure"


def is_shot(name):
    return (
        bool(re.match(r"^\d{3}[_-]", name))
        or bool(re.match(r"^\d{3,4}", name))
        or "_Source_" in name
    )


def parse_source_x(text):
    """
    Parse source coordinates such as:
        x0082p500m -> 82.5
        x0010p000m -> 10.0
        x00010.0m  -> 10.0
        x00124.5m  -> 124.5
    The last matching source token is used.
    """
    patterns = [
        re.compile(r"(?:^|[_-])x(?P<int>\d+)p(?P<frac>\d+)m(?:$|[_-])", re.I),
        re.compile(r"(?:^|[_-])x(?P<value>\d+(?:\.\d+)?)m(?:$|[_-])", re.I),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(str(text)))

    if not matches:
        return None

    match = sorted(matches, key=lambda m: m.start())[-1]

    if "value" in match.groupdict() and match.group("value") is not None:
        return float(match.group("value"))

    integer = int(match.group("int"))
    fraction_text = match.group("frac")
    fraction = int(fraction_text) / (10 ** len(fraction_text))
    return float(integer + fraction)


def source_x_from_path(path, shot_folder):
    candidates = [shot_folder, path.name, *reversed(path.parts)]
    for candidate in candidates:
        value = parse_source_x(candidate)
        if value is not None:
            return value
    return None


def classify(path):
    rel = path.relative_to(root)
    component = "all"

    for part in rel.parts:
        if part in ("Ux", "Uz", "Z", "N", "E"):
            component = part
            break

    parent = path.parent.name
    grand = path.parent.parent.name if path.parent.parent != path.parent else ""

    if is_shot(parent):
        product = "baseline"
        shot = parent
    elif is_shot(grand):
        product = parent
        shot = grand
    else:
        product = "other"
        shot = parent

    return component, product, shot, path.name


groups = defaultdict(list)

for path in root.rglob(f"*.{ext}"):
    if any(part in ("_movies", "_frames_tmp", "movies") for part in path.parts):
        continue
    if not include_re.search(path.name):
        continue
    if exclude_re.search(path.name):
        continue

    component, product, shot, figure = classify(path)
    source_x_m = source_x_from_path(path, shot)

    if require_source_x and source_x_m is None:
        print(f"WARNING: no source coordinate parsed; skipping: {path}")
        continue

    groups[(component, product, figure)].append(
        {
            "shot": shot,
            "path": path,
            "source_x_m": source_x_m,
        }
    )


def item_sort_key(item):
    if sort_mode == "source_x":
        source_x = item["source_x_m"]
        missing_flag = 1 if source_x is None else 0
        source_value = float("inf") if source_x is None else float(source_x)
        return (
            missing_flag,
            source_value,
            natural_key(item["shot"]),
            natural_key(str(item["path"])),
        )

    return (
        natural_key(item["shot"]),
        natural_key(str(item["path"])),
    )


movies = []
rows = []

for (component, product, figure), items in sorted(
    groups.items(),
    key=lambda kv: natural_key("_".join(kv[0])),
):
    items = sorted(items, key=item_sort_key)

    if len(items) < min_frames:
        continue

    movie = outdir / (
        safe_name(f"{product}_{Path(figure).stem}_{component}") + ".mp4"
    )

    frames = [item["path"] for item in items]

    movies.append(
        {
            "component": component,
            "product": product,
            "figure_name": figure,
            "movie": str(movie),
            "sort_mode": sort_mode,
            "frames": [
                {
                    "path": str(item["path"]),
                    "shot_folder": item["shot"],
                    "source_x_m": item["source_x_m"],
                }
                for item in items
            ],
        }
    )

    for frame_index, item in enumerate(items, start=1):
        rows.append(
            {
                "movie": str(movie),
                "component": component,
                "product": product,
                "figure_name": figure,
                "frame_index": frame_index,
                "shot_folder": item["shot"],
                "source_x_m": item["source_x_m"],
                "frame_path": str(item["path"]),
            }
        )

plan_json.write_text(json.dumps(movies, indent=2), encoding="utf-8")

with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "movie",
        "component",
        "product",
        "figure_name",
        "frame_index",
        "shot_folder",
        "source_x_m",
        "frame_path",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Detected {len(movies)} movies")

for movie in movies:
    frame_records = movie["frames"]
    first_x = frame_records[0]["source_x_m"]
    last_x = frame_records[-1]["source_x_m"]
    print(
        f"  {Path(movie['movie']).name}: {len(frame_records)} frames"
        f"  source_x={first_x} to {last_x}"
    )

if not movies:
    print(f"No movies to create. Manifest: {manifest_csv}")
    raise SystemExit(0)

for movie_index, movie_record in enumerate(movies, start=1):
    movie = Path(movie_record["movie"])
    movie.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = workdir / f"{movie_index:03d}_{movie.stem}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_records = movie_record["frames"]
    suffix = Path(frame_records[0]["path"]).suffix.lower()

    for frame_index, record in enumerate(frame_records, start=1):
        source = Path(record["path"])
        shutil.copy2(
            source,
            frame_dir / f"frame_{frame_index:05d}{suffix}",
        )

    pattern = frame_dir / f"frame_%05d{suffix}"

    print(
        f"\nMaking movie: {movie}\n"
        f"  product={movie_record['product']}\n"
        f"  figure={movie_record['figure_name']}\n"
        f"  frames={len(frame_records)}\n"
        f"  sort_mode={sort_mode}"
    )

    if dry_run:
        print("  DRY_RUN=1; skipping ffmpeg")
        continue

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]

    subprocess.run(command, check=True)

if not keep_frames:
    shutil.rmtree(workdir, ignore_errors=True)
else:
    print(f"Temporary frames kept at: {workdir}")

print(f"\nMovies written to: {outdir}")
print(f"Manifest written to: {manifest_csv}")
print(f"Plan written to:     {plan_json}")
PY
