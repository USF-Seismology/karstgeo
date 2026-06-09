#!/usr/bin/env python
"""
compute_hourly_band_median_abs.py

Read nodal data from an SDS archive with FLOVOpy's EnhancedSDSClient, then for
each hourly chunk and each trace:

  1. remove masked/null/non-finite samples
  2. detrend/demean/taper
  3. filter into user-defined passbands
  4. compute the median absolute amplitude in each band

This intentionally bypasses the FLOVOpy RSAM/SAM classes. The output is a
simple long-format CSV with one row per hour per SEED id.

Example:

python compute_hourly_band_median_abs.py \
  --sds-root /Volumes/tachyon/LBSSP_DATA/nodal_sds \
  --network T1 \
  --station '*' \
  --location 'N*' \
  --channel '*Z' \
  --start 2026-05-16T00:00:00 \
  --end 2026-05-20T00:00:00 \
  --band LOW:5-20 \
  --band MID:20-80 \
  --band HIGH:80-200 \
  --out /Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_hourly_band_median_abs.csv

For 500 Hz data, keep fmax comfortably below Nyquist; 200 Hz is safer than 240 Hz.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

from flovopy.enhanced.sdsclient import EnhancedSDSClient


DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "LOW_5_20": (5.0, 20.0),
    "MID_20_80": (20.0, 80.0),
    "HIGH_80_200": (80.0, 200.0),
}


def utc(text: str) -> UTCDateTime:
    """Parse a UTCDateTime from command-line text."""
    return UTCDateTime(text)


def parse_band_arg(text: str) -> Tuple[str, Tuple[float, float]]:
    """
    Parse one band argument of form NAME:FMIN-FMAX.

    Example:
        LOW:5-20
    """
    if ":" not in text or "-" not in text:
        raise argparse.ArgumentTypeError(
            f"Band must be NAME:FMIN-FMAX, got {text!r}"
        )
    name, limits = text.split(":", 1)
    fmin, fmax = limits.split("-", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Band name cannot be empty")
    return name, (float(fmin), float(fmax))


def iter_chunks(
    start: UTCDateTime,
    end: UTCDateTime,
    chunk_seconds: float,
) -> Iterable[Tuple[UTCDateTime, UTCDateTime]]:
    """Yield non-overlapping chunk windows."""
    t0 = UTCDateTime(start)
    while t0 < end:
        t1 = min(t0 + chunk_seconds, end)
        yield t0, t1
        t0 = t1


def safe_read_waveforms(
    client: EnhancedSDSClient,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
) -> Stream:
    """Read waveforms, returning an empty Stream on failure."""
    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
    except Exception as exc:
        print(f"  read failed for {starttime} to {endtime}: {exc}")
        return Stream()

    if len(st) == 0:
        return Stream()

    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception as exc:
        print(f"  merge warning: {exc}")

    return st


def clean_trace_for_filtering(
    tr: Trace,
    target_start: UTCDateTime,
    target_end: UTCDateTime,
    taper_percentage: float = 0.01,
) -> Trace | None:
    """
    Return a cleaned copy of a trace suitable for filtering.

    This removes null/non-finite samples by interpolation, trims exactly to the
    target window, detrends, demeans, and tapers.
    """
    tr = tr.copy()

    # Trim exactly to the requested chunk.
    try:
        tr.trim(target_start, target_end, pad=False)
    except Exception as exc:
        print(f"  trim failed for {tr.id}: {exc}")
        return None

    if tr.stats.npts < 10:
        return None

    data = tr.data

    # Convert masked arrays to NaNs first.
    if np.ma.isMaskedArray(data):
        data = data.astype(float).filled(np.nan)
    else:
        data = np.asarray(data, dtype=float)

    if data.size < 10:
        return None

    finite = np.isfinite(data)
    n_finite = int(finite.sum())

    if n_finite < 10:
        print(f"  skipping {tr.id}: too few finite samples ({n_finite})")
        return None

    # Interpolate over NaNs/Infs.
    if n_finite < data.size:
        idx = np.arange(data.size)
        data[~finite] = np.interp(idx[~finite], idx[finite], data[finite])

    tr.data = data

    try:
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(max_percentage=taper_percentage, type="hann")
    except Exception as exc:
        print(f"  preprocessing failed for {tr.id}: {exc}")
        return None

    return tr


def median_abs_for_band(
    tr: Trace,
    fmin: float,
    fmax: float,
    corners: int = 4,
    zerophase: bool = True,
) -> float:
    """Filter a trace and return median(abs(data))."""
    fs = float(tr.stats.sampling_rate)
    nyq = 0.5 * fs

    if fmin <= 0 or fmax <= fmin:
        return np.nan

    if fmax >= 0.95 * nyq:
        # Avoid unstable filters too close to Nyquist.
        return np.nan

    tb = tr.copy()

    try:
        tb.filter(
            "bandpass",
            freqmin=float(fmin),
            freqmax=float(fmax),
            corners=int(corners),
            zerophase=bool(zerophase),
        )
    except Exception as exc:
        print(f"  filter failed for {tr.id} {fmin}-{fmax} Hz: {exc}")
        return np.nan

    data = np.asarray(tb.data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return np.nan

    return float(np.nanmedian(np.abs(data)))


def compute_hourly_metrics_for_stream(
    st: Stream,
    target_start: UTCDateTime,
    target_end: UTCDateTime,
    bands: Dict[str, Tuple[float, float]],
    corners: int,
    zerophase: bool,
    taper_percentage: float,
) -> list[dict]:
    """Compute one row per trace for one time chunk."""
    rows = []

    for tr in st:
        clean = clean_trace_for_filtering(
            tr,
            target_start=target_start,
            target_end=target_end,
            taper_percentage=taper_percentage,
        )

        if clean is None:
            continue

        parts = clean.id.split(".")
        if len(parts) == 4:
            net, sta, loc, cha = parts
        else:
            net, sta, loc, cha = "", "", "", ""

        row = {
            "time": target_start.datetime.isoformat(),
            "starttime": str(target_start),
            "endtime": str(target_end),
            "seed_id": clean.id,
            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,
            "sampling_rate": float(clean.stats.sampling_rate),
            "npts": int(clean.stats.npts),
            "duration_s": float(clean.stats.endtime - clean.stats.starttime),
        }

        # Also include unfiltered median absolute amplitude after detrend/taper.
        base_data = np.asarray(clean.data, dtype=float)
        base_data = base_data[np.isfinite(base_data)]
        row["RAW_median_abs"] = (
            float(np.nanmedian(np.abs(base_data))) if base_data.size else np.nan
        )

        for band_name, (fmin, fmax) in bands.items():
            row[f"{band_name}_median_abs"] = median_abs_for_band(
                clean,
                fmin=fmin,
                fmax=fmax,
                corners=corners,
                zerophase=zerophase,
            )

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute hourly median absolute amplitude in several passbands "
            "from an SDS archive using EnhancedSDSClient."
        )
    )

    parser.add_argument("--sds-root", required=True, type=Path)
    parser.add_argument("--network", default="*")
    parser.add_argument("--station", default="*")
    parser.add_argument("--location", default="*")
    parser.add_argument("--channel", default="*Z")

    parser.add_argument("--start", required=True, type=utc)
    parser.add_argument("--end", required=True, type=utc)

    parser.add_argument(
        "--chunk-hours",
        type=float,
        default=1.0,
        help="Chunk length in hours. Default 1 hour.",
    )
    parser.add_argument(
        "--read-buffer-seconds",
        type=float,
        default=30.0,
        help=(
            "Read a small buffer around each chunk before trimming/filtering. "
            "Default 30 s. Increase if using lower filter frequencies."
        ),
    )

    parser.add_argument(
        "--band",
        action="append",
        type=parse_band_arg,
        default=[],
        help=(
            "Band as NAME:FMIN-FMAX. Repeatable. "
            "Defaults to LOW_5_20, MID_20_80, HIGH_80_200."
        ),
    )
    parser.add_argument(
        "--no-default-bands",
        action="store_true",
        help="Do not include default bands.",
    )

    parser.add_argument("--corners", type=int, default=4)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal filtering instead of zerophase filtering.",
    )
    parser.add_argument("--taper-percentage", type=float, default=0.01)

    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )

    args = parser.parse_args()

    bands: Dict[str, Tuple[float, float]] = {}
    if not args.no_default_bands:
        bands.update(DEFAULT_BANDS)
    for name, limits in args.band:
        bands[name] = limits

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.exists() and args.overwrite:
        args.out.unlink()
    elif args.out.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.out} already exists. Use --overwrite or choose another file."
        )

    client = EnhancedSDSClient(args.sds_root)
    chunk_seconds = float(args.chunk_hours) * 3600.0

    wrote_header = False
    total_rows = 0

    print("=" * 80)
    print("Hourly band median absolute amplitude")
    print(f"SDS root: {args.sds_root}")
    print(f"Selectors: {args.network}.{args.station}.{args.location}.{args.channel}")
    print(f"Time range: {args.start} to {args.end}")
    print(f"Chunk length: {chunk_seconds} s")
    print(f"Read buffer: {args.read_buffer_seconds} s")
    print(f"Bands: {bands}")
    print(f"Output: {args.out}")
    print("=" * 80)

    for target_start, target_end in iter_chunks(args.start, args.end, chunk_seconds):
        read_start = target_start - args.read_buffer_seconds
        read_end = target_end + args.read_buffer_seconds

        print("\n" + "-" * 80)
        print(f"Target window: {target_start} to {target_end}")
        print(f"Read window:   {read_start} to {read_end}")

        st = safe_read_waveforms(
            client,
            network=args.network,
            station=args.station,
            location=args.location,
            channel=args.channel,
            starttime=read_start,
            endtime=read_end,
        )

        if len(st) == 0:
            print("  no data")
            continue

        print(f"  read {len(st)} trace(s)")

        rows = compute_hourly_metrics_for_stream(
            st,
            target_start=target_start,
            target_end=target_end,
            bands=bands,
            corners=args.corners,
            zerophase=not args.causal,
            taper_percentage=args.taper_percentage,
        )

        if not rows:
            print("  no metric rows")
            continue

        df = pd.DataFrame(rows)
        df.sort_values(["time", "network", "location", "station", "channel"], inplace=True)

        df.to_csv(
            args.out,
            mode="a",
            header=not wrote_header,
            index=False,
        )

        wrote_header = True
        total_rows += len(df)

        print(f"  wrote {len(df)} row(s); cumulative {total_rows}")

    print("\nDone.")
    print(f"Wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
