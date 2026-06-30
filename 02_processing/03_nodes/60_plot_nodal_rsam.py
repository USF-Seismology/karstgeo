#!/usr/bin/env python
"""
plot_nodal_rsam.py

Read RSAM files written by FLOVOpy's RSAM.write() and generate plots using
RSAM.plot().

This does NOT read waveform/SDS data. It reads the RSAM products already written
to disk, e.g.

    SAM_DIR/
      RSAM/
        T1/
          RSAM_05726_2026_60s.csv
          ...

Example:

    python 60_plot_nodal_rsam.py \
      --sam-dir /Volumes/tachyon/LBSSP_DATA/nodal_rsam \
      --plot-dir /Volumes/tachyon/LBSSP_DATA/nodal_rsam/plots \
      --network T1 \
      --start 2026-05-17T08:00:00 \
      --end 2026-05-17T09:00:00 \
      --sampling-interval 60 \
      --ext csv \
      --metrics mean median rms LOW_5_20 MID_20_80 HIGH_80_240 \
      --kind stream

Notes:
  - `--sam-dir` should be the same directory passed as `--out-dir` when computing RSAM.
  - The plotting itself is done by RSAM.plot().
  - For `kind=stream`, RSAM.plot() writes one PNG per metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from obspy import UTCDateTime

from flovopy.processing.sam import RSAM


def utc(text: str) -> UTCDateTime:
    """Parse UTCDateTime from command-line text."""
    return UTCDateTime(text)


def summarize_rsam(rsam: RSAM, max_ids: int = 10):
    """Print a useful summary of the loaded RSAM object without relying on __str__."""
    dataframes = getattr(rsam, "dataframes", {})
    print("=" * 80)
    print("Loaded RSAM object")
    print(f"Number of trace IDs: {len(dataframes)}")

    if not dataframes:
        print("No dataframes loaded.")
        print("=" * 80)
        return

    print("Trace IDs:")
    for i, (seed_id, df) in enumerate(dataframes.items()):
        if i >= max_ids:
            print(f"  ... {len(dataframes) - max_ids} more")
            break
        print(f"  {seed_id}: {len(df)} rows")

    # Show available columns from first non-empty dataframe.
    for seed_id, df in dataframes.items():
        if df is not None and len(df):
            print(f"Columns in {seed_id}:")
            print("  " + ", ".join(str(c) for c in df.columns))
            break

    print("=" * 80)


def filter_trace_ids(
    rsam: RSAM,
    station: Optional[str] = None,
    location: Optional[str] = None,
    channel: Optional[str] = None,
) -> RSAM:
    """
    Return a selected RSAM object by matching SEED id parts.

    Matching is simple:
      NET.STA.LOC.CHA
    """
    if not any([station, location, channel]):
        return rsam

    selected = []

    for seed_id in rsam.dataframes.keys():
        parts = seed_id.split(".")
        if len(parts) != 4:
            continue

        _net, sta, loc, cha = parts

        if station and station != "*" and sta != station:
            continue
        if location and location != "*" and loc != location:
            continue
        if channel and channel != "*" and cha != channel:
            continue

        selected.append(seed_id)

    print(f"Selected {len(selected)} trace IDs after station/location/channel filtering.")
    return rsam.select_ids(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Read FLOVOpy RSAM files and plot them using RSAM.plot()."
    )

    parser.add_argument("--sam-dir", required=True, type=Path)
    parser.add_argument("--plot-dir", required=True, type=Path)

    parser.add_argument("--start", required=True, type=utc)
    parser.add_argument("--end", required=True, type=utc)

    parser.add_argument("--network", default="*")
    parser.add_argument("--station", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--channel", default=None)

    parser.add_argument("--sampling-interval", type=int, default=60)
    parser.add_argument("--ext", choices=["csv", "pickle"], default="csv")

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean"],
        help=(
            "Metrics/columns to plot. Examples: mean median rms max "
            "LOW_5_20 MID_20_80 HIGH_80_200. Use 'bands' to let RSAM.plot() "
            "auto-detect classic band triads, if present."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["stream", "line", "scatter"],
        default="stream",
    )
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--equal-scale", action="store_true")
    parser.add_argument(
        "--ylims",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument("--outfile-prefix", default="rsam")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading RSAM from: {args.sam_dir}")
    print(f"Time range: {args.start} to {args.end}")

    rsam = RSAM.read(
        args.start,
        args.end,
        SAM_DIR=str(args.sam_dir),
        network=args.network,
        sampling_interval=args.sampling_interval,
        ext=args.ext,
        verbose=args.verbose,
    )

    print(rsam)
    #summarize_rsam(rsam)

    rsam = filter_trace_ids(
        rsam,
        station=args.station,
        location=args.location,
        channel=args.channel,
    )

    summarize_rsam(rsam)

    if not getattr(rsam, "dataframes", None):
        raise RuntimeError("No RSAM dataframes available to plot.")

    outfile = args.plot_dir / f"{args.outfile_prefix}.png"

    print(f"Plotting metrics: {args.metrics}")
    print(f"Plot kind: {args.kind}")
    print(f"Output base: {outfile}")

    # Let the RSAM class handle plotting.
    rsam.plot(
        metrics=args.metrics,
        kind=args.kind,
        logy=args.logy,
        equal_scale=args.equal_scale,
        outfile=str(outfile),
        ylims=args.ylims,
        trim_to_data=True,
    )

    print(f"Done. Plots written in: {args.plot_dir}")


if __name__ == "__main__":
    main()
