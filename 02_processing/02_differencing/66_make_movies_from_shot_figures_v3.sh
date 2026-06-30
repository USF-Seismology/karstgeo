#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-}"
MOVIE_ROOT="${2:-}"
[[ -n "$ROOT" ]] || { echo "Usage: bash $0 ROOT_DIR [MOVIE_ROOT]"; exit 2; }
[[ -d "$ROOT" ]] || { echo "ERROR: ROOT_DIR does not exist: $ROOT"; exit 1; }
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
    TAIL="$(python3 - "$ROOT" "$MAP_DEPTH" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); n=int(sys.argv[2])
print(str(Path(*p.parts[-n:])) if n>0 else "")
PY
)"
    OUTDIR="$MOVIE_ROOT/$TAIL"
  fi
fi
mkdir -p "$OUTDIR"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

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
import csv, json, os, re, shutil, subprocess
from collections import defaultdict
from pathlib import Path

root=Path(os.environ["ROOT"])
outdir=Path(os.environ["OUTDIR"])
fps=float(os.environ["FPS"])
ext=os.environ["EXT"].lstrip(".")
min_frames=int(os.environ["MIN_FRAMES"])
include_re=re.compile(os.environ["INCLUDE_REGEX"])
exclude_re=re.compile(os.environ["EXCLUDE_REGEX"])
crf=os.environ["CRF"]
preset=os.environ["PRESET"]
dry_run=os.environ["DRY_RUN"]=="1"
keep_frames=os.environ["KEEP_FRAMES"]=="1"

outdir.mkdir(parents=True, exist_ok=True)
workdir=outdir/"_frames_tmp"
if workdir.exists(): shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)
plan_json=outdir/"movie_plan.json"
manifest_csv=outdir/"movie_manifest.csv"

def natural_key(text):
    out=[]
    for p in re.split(r"(\d+(?:\.\d+)?)", str(text)):
        if not p: continue
        try: out.append(float(p))
        except ValueError: out.append(p.lower())
    return out

def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+","_",str(text)).strip("_") or "figure"

def is_shot(name):
    return bool(re.match(r"^\d{3}[_-]", name)) or bool(re.match(r"^\d{3,4}", name))

def classify(p):
    rel=p.relative_to(root)
    comp="all"
    for part in rel.parts:
        if part in ("Ux","Uz"):
            comp=part
            break
    parent=p.parent.name
    grand=p.parent.parent.name if p.parent.parent != p.parent else ""
    if is_shot(parent):
        product="baseline"; shot=parent
    elif is_shot(grand):
        product=parent; shot=grand
    else:
        product="other"; shot=parent
    return comp, product, shot, p.name

groups=defaultdict(list)
for p in root.rglob(f"*.{ext}"):
    if any(x in p.parts for x in ("_movies","_frames_tmp","movies")):
        continue
    if not include_re.search(p.name): continue
    if exclude_re.search(p.name): continue
    comp, product, shot, fig=classify(p)
    groups[(comp,product,fig)].append((shot,p))

movies=[]; rows=[]
for (comp,product,fig), items in sorted(groups.items(), key=lambda kv:natural_key("_".join(kv[0]))):
    items=sorted(items, key=lambda item:natural_key(item[0])+natural_key(str(item[1])))
    if len(items) < min_frames: continue
    movie=outdir/(safe_name(f"{product}_{Path(fig).stem}_{comp}")+".mp4")
    frames=[p for _,p in items]
    movies.append({"component":comp,"product":product,"figure_name":fig,"movie":str(movie),"frames":[str(p) for p in frames]})
    for i,(shot,p) in enumerate(items,1):
        rows.append({"movie":str(movie),"component":comp,"product":product,"figure_name":fig,
                     "frame_index":i,"shot_folder":shot,"frame_path":str(p)})

plan_json.write_text(json.dumps(movies,indent=2), encoding="utf-8")
with manifest_csv.open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["movie","component","product","figure_name","frame_index","shot_folder","frame_path"])
    w.writeheader(); w.writerows(rows)

print(f"Detected {len(movies)} movies")
for m in movies:
    print(f"  {Path(m['movie']).name}: {len(m['frames'])} frames")

if not movies:
    print(f"No movies to create. Manifest: {manifest_csv}")
    raise SystemExit(0)

for mi,m in enumerate(movies,1):
    movie=Path(m["movie"]); movie.parent.mkdir(parents=True, exist_ok=True)
    frame_dir=workdir/f"{mi:03d}_{movie.stem}"; frame_dir.mkdir(parents=True, exist_ok=True)
    suffix=Path(m["frames"][0]).suffix.lower()
    for i,src in enumerate(m["frames"],1):
        shutil.copy2(Path(src), frame_dir/f"frame_{i:05d}{suffix}")
    pattern=frame_dir/f"frame_%05d{suffix}"
    print(f"\nMaking movie: {movie}\n  product={m['product']} figure={m['figure_name']} frames={len(m['frames'])}")
    if dry_run:
        print("  DRY_RUN=1; skipping ffmpeg")
        continue
    cmd=["ffmpeg","-hide_banner","-loglevel","warning","-y","-framerate",str(fps),"-i",str(pattern),
         "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-preset",preset,
         "-crf",str(crf),"-pix_fmt","yuv420p","-movflags","+faststart",str(movie)]
    subprocess.run(cmd, check=True)

if not keep_frames:
    shutil.rmtree(workdir, ignore_errors=True)
else:
    print(f"Temporary frames kept at: {workdir}")
print(f"\nMovies written to: {outdir}")
print(f"Manifest written to: {manifest_csv}")
print(f"Plan written to:     {plan_json}")
PY
