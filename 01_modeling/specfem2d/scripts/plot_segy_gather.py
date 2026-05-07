#!/usr/bin/env python3
from __future__ import annotations
import argparse
from segy_tools.gather import plot_wiggle_gather_from_segy, plot_image_gather_from_segy


def main():
    p = argparse.ArgumentParser(description="Plot wiggle/image gather from SEG-Y.")
    p.add_argument("segy")
    p.add_argument("--kind", choices=["wiggle", "image"], default="wiggle")
    p.add_argument("--outfile")
    p.add_argument("--tmin", type=float); p.add_argument("--tmax", type=float)
    p.add_argument("--omin", type=float); p.add_argument("--omax", type=float)
    p.add_argument("--scale", type=float, default=0.8)
    p.add_argument("--receiver-spacing", type=float)
    p.add_argument("--first-receiver-x", type=float, default=0.0)
    p.add_argument("--source-x", type=float)
    p.add_argument("--cave-x-min", type=float); p.add_argument("--cave-x-max", type=float)
    args = p.parse_args()
    cave = {"x_min_m": args.cave_x_min, "x_max_m": args.cave_x_max} if args.cave_x_min is not None and args.cave_x_max is not None else None
    common = dict(outfile=args.outfile, tmin=args.tmin, tmax=args.tmax, omin=args.omin, omax=args.omax, fallback_receiver_spacing_m=args.receiver_spacing, fallback_first_receiver_x_m=args.first_receiver_x, fallback_source_x_m=args.source_x, cave=cave)
    if args.kind == "wiggle":
        plot_wiggle_gather_from_segy(args.segy, scale=args.scale, **common)
    else:
        plot_image_gather_from_segy(args.segy, **common)
    if args.outfile: print(f"wrote {args.outfile}")

if __name__ == "__main__":
    main()
