#!/usr/bin/env python3
from __future__ import annotations
import argparse
from specfem_tools.movie import jpg_to_movie


def main():
    p = argparse.ArgumentParser(description="Make movie from SPECFEM2D forward_image*.jpg files.")
    p.add_argument("--input-dir", default="OUTPUT_FILES")
    p.add_argument("--pattern", default="forward_image*.jpg")
    p.add_argument("--output", default="specfem_movie.mp4")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    print(jpg_to_movie(args.input_dir, args.pattern, args.output, args.fps))
if __name__ == "__main__": main()
