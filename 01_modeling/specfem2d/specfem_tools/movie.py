from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def jpg_to_movie(input_dir="OUTPUT_FILES", pattern="forward_image*.jpg", output="specfem_movie.mp4", fps=10, overwrite=True, keep_listfile=True, verbose=True):
    input_dir = Path(input_dir).expanduser()
    output = Path(output).expanduser()
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")
    listfile = input_dir / "movie_frames.txt"
    with listfile.open("w") as f:
        for file in files:
            f.write(f"file '{file.resolve()}'\n")
    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-r", str(fps), "-f", "concat", "-safe", "0", "-i", str(listfile), "-pix_fmt", "yuv420p", str(output)]
    if verbose:
        print(f"Found {len(files)} frames"); print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    if not keep_listfile:
        listfile.unlink(missing_ok=True)
    return output
