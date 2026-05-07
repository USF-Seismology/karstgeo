#!/usr/bin/env python3
from __future__ import annotations
import argparse
from specfem_tools.converters import Geometry, Timing, convert_sem_output_to_segy


def main():
    p = argparse.ArgumentParser(description="Convert one SPECFEM OUTPUT_FILES SEM gather to SEG-Y.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--component", choices=["X", "Z", "BXX", "BXZ"], default="Z")
    p.add_argument("--extension", default="semv")
    p.add_argument("--shot-number", type=int, default=1)
    p.add_argument("--source-x", type=float, default=None)
    p.add_argument("--first-receiver-x", type=float, default=0.0)
    p.add_argument("--receiver-spacing", type=float, default=1.0)
    p.add_argument("--receiver-z", type=float, default=0.0)
    p.add_argument("--first-shot-x", type=float, default=0.0)
    p.add_argument("--shot-spacing", type=float, default=1.0)
    p.add_argument("--source-z", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--t0", type=float, default=None)
    args = p.parse_args()
    geom = Geometry(args.first_receiver_x, args.receiver_spacing, args.receiver_z, args.first_shot_x, args.shot_spacing, args.source_z)
    timing = Timing(dt_s=args.dt, t0_s=args.t0)
    convert_sem_output_to_segy(args.input, args.output, component=args.component, extension=args.extension, shot_number=args.shot_number, source_x_m=args.source_x, geom=geom, timing=timing)

if __name__ == "__main__":
    main()
