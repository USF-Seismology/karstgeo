#!/usr/bin/env python
"""
audit_nodal_sds_availability.py

Audit an SDS archive using FLOVOpy's EnhancedSDSClient and report
availability per SEED id per UTC day.

Outputs:
  1. Wide CSV: one row per day, one column per SEED id, values 0..1
  2. Long CSV: one row per day/SEED id, with percent availability
  3. Summary CSV: one row per SEED id, with mean/min/max availability
  4. Optional PNG heatmap using EnhancedSDSClient.plot_availability()

Example:

python audit_nodal_sds_availability.py \
  --sds-root /Volumes/tachyon/LBSSP_DATA/nodal_sds \
  --start 2026-05-16 \
  --end 2026-05-20 \
  --network T1 \
  --location 'N*' \
  --channel 'DPZ,GPZ' \
  --out-prefix /Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_availability \
  --plot

Notes:
  - --end is exclusive in the day loop. Use 2026-05-21 to include May 20.
  - Selectors accept comma-separated lists and shell-style wildcards.
  - If --trace-ids is supplied, discovery is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from fnmatch import fnmatch
from typing import Iterable, Sequence

import pandas as pd
from obspy import UTCDateTime

from flovopy.enhanced.sdsclient import EnhancedSDSClient


def parse_list(text: str | None) -> list[str]:
    """Parse comma-separated selector values."""
    if text is None:
        return ["*"]
    vals = [v.strip() for v in str(text).split(",") if v.strip()]
    return vals if vals else ["*"]


def seed_id_matches(seed_id: str, networks, stations, locations, channels) -> bool:
    """Return True if NET.STA.LOC.CHA matches selector lists."""
    parts = seed_id.split(".")
    if len(parts) != 4:
        return False
    net, sta, loc, cha = parts

    def any_match(value, patterns):
        return any(fnmatch(value, p) for p in patterns)

    return (
        any_match(net, networks)
        and any_match(sta, stations)
        and any_match(loc, locations)
        and any_match(cha, channels)
    )


def filter_trace_ids(trace_ids: Sequence[str], networks, stations, locations, channels) -> list[str]:
    """Filter trace IDs with shell-style wildcard selectors."""
    return sorted(
        tid for tid in trace_ids
        if seed_id_matches(tid, networks, stations, locations, channels)
    )


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide availability table to long format."""
    if wide.empty:
        return pd.DataFrame(columns=["date", "seed_id", "availability", "availability_percent"])

    long = wide.melt(
        id_vars=["date"],
        var_name="seed_id",
        value_name="availability",
    )
    long["availability_percent"] = 100.0 * long["availability"].astype(float)

    parts = long["seed_id"].str.split(".", expand=True)
    if parts.shape[1] == 4:
        long.insert(1, "network", parts[0])
        long.insert(2, "station", parts[1])
        long.insert(3, "location", parts[2])
        long.insert(4, "channel", parts[3])

    return long


def summarize_long(long: pd.DataFrame) -> pd.DataFrame:
    """Summarize availability by SEED id."""
    if long.empty:
        return pd.DataFrame()

    grp = long.groupby("seed_id", as_index=False)
    summary = grp.agg(
        network=("network", "first") if "network" in long else ("seed_id", "first"),
        station=("station", "first") if "station" in long else ("seed_id", "first"),
        location=("location", "first") if "location" in long else ("seed_id", "first"),
        channel=("channel", "first") if "channel" in long else ("seed_id", "first"),
        n_days=("availability", "size"),
        n_days_with_data=("availability", lambda x: int((x > 0).sum())),
        mean_availability=("availability", "mean"),
        min_availability=("availability", "min"),
        max_availability=("availability", "max"),
    )
    summary["mean_availability_percent"] = 100.0 * summary["mean_availability"]
    summary["min_availability_percent"] = 100.0 * summary["min_availability"]
    summary["max_availability_percent"] = 100.0 * summary["max_availability"]

    return summary.sort_values(
        ["network", "location", "station", "channel"],
        kind="stable",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audit SDS archive availability per SEED id per UTC day."
    )
    parser.add_argument("--sds-root", required=True, type=Path)
    parser.add_argument("--start", required=True, help="UTC start day, inclusive, e.g. 2026-05-16")
    parser.add_argument("--end", required=True, help="UTC end day, exclusive, e.g. 2026-05-21")
    parser.add_argument("--network", default="*", help="Comma-separated networks, wildcards allowed")
    parser.add_argument("--station", default="*", help="Comma-separated stations, wildcards allowed")
    parser.add_argument("--location", default="*", help="Comma-separated locations, wildcards allowed")
    parser.add_argument("--channel", default="*", help="Comma-separated channels, wildcards allowed")
    parser.add_argument(
        "--trace-ids",
        default=None,
        help="Optional comma-separated explicit NET.STA.LOC.CHA list. Skips discovery.",
    )
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--skip-low-rate-channels", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = UTCDateTime(args.start)
    end = UTCDateTime(args.end)

    client = EnhancedSDSClient(args.sds_root)

    networks = parse_list(args.network)
    stations = parse_list(args.station)
    locations = parse_list(args.location)
    channels = parse_list(args.channel)

    if args.trace_ids:
        trace_ids = [t.strip() for t in args.trace_ids.split(",") if t.strip()]
    else:
        print("Discovering trace IDs...")
        trace_ids = client.iter_trace_ids(
            start,
            end,
            skip_low_rate=args.skip_low_rate_channels,
        )
        trace_ids = filter_trace_ids(trace_ids, networks, stations, locations, channels)

    print(f"Found {len(trace_ids)} trace IDs matching selectors.")
    for tid in trace_ids[:20]:
        print(" ", tid)
    if len(trace_ids) > 20:
        print(f"  ... {len(trace_ids) - 20} more")

    print("Computing daily availability...")
    wide, trace_ids = client.get_availability(
        startday=start,
        endday=end,
        trace_ids=trace_ids,
        skip_low_rate_channels=args.skip_low_rate_channels,
        progress=not args.no_progress,
        verbose=args.verbose,
    )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    wide_csv = out_prefix.with_name(out_prefix.name + "_wide.csv")
    long_csv = out_prefix.with_name(out_prefix.name + "_long.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")

    wide.to_csv(wide_csv, index=False)

    long = wide_to_long(wide)
    long.to_csv(long_csv, index=False)

    summary = summarize_long(long)
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote wide table:    {wide_csv}")
    print(f"Wrote long table:    {long_csv}")
    print(f"Wrote summary table: {summary_csv}")

    if not summary.empty:
        print("\nLowest mean availability:")
        print(
            summary[
                ["seed_id", "n_days_with_data", "mean_availability_percent", "min_availability_percent", "max_availability_percent"]
            ].head(20).to_string(index=False)
        )

    if args.plot:
        png = out_prefix.with_name(out_prefix.name + "_heatmap.png")
        print(f"Writing heatmap: {png}")
        client.plot_availability(
            availability_df=wide,
            outfile=str(png),
            progress=False,
            verbose=args.verbose,
        )

    print("Done.")


if __name__ == "__main__":
    main()
