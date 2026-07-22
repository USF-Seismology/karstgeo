#!/usr/bin/env python
"""
compute_nodal_rsam_from_sds.py

Compute RSAM/SAM-style amplitude metrics from a nodal SDS archive using FLOVOpy.

Designed for the Karst Geophysics SmartSolo nodal archive, e.g.

    /Volumes/tachyon/LBSSP_DATA/nodal_sds

The script:
  - uses flovopy.enhanced.sdsclient.EnhancedSDSClient
  - reads data in hourly or daily chunks
  - optionally reads a buffer around each chunk to reduce filter edge effects
  - performs conservative preprocessing
  - computes RSAM using flovopy.processing.sam.RSAM
  - supports custom primary filter and named frequency bands via command-line args
  - writes output ONLY through RSAM.write(), allowing the RSAM class to manage
    yearly CSV/pickle files and merging across chunks

Example: vertical channels, one-hour chunks, nodal bands

    python compute_nodal_rsam_from_sds.py \\
      --sds-root /Volumes/tachyon/LBSSP_DATA/nodal_sds \\
      --out-dir /Volumes/tachyon/LBSSP_DATA/nodal_rsam/T1_N2_Z \\
      --network T1 \\
      --station '*' \\
      --location N2 \\
      --channel DPZ \\
      --start 2026-05-17T16:00:00 \\
      --end 2026-05-18T16:00:00 \\
      --chunk-hours 1 \\
      --sampling-interval 60 \\
      --primary-filter 5 240 \\
      --band LOW:5-20 \\
      --band MID:20-80 \\
      --band HIGH:80-240 \\
      --ext csv

Notes:
  - For 500 Hz data, keep fmax safely below Nyquist. 240 Hz is safer than 250 Hz.
  - For 1000 Hz data, 240 Hz is still fine and allows comparison with 500 Hz nodes.
  - RSAM.write(..., overwrite=False) is used by default, so each chunk is merged
    into the existing yearly output files managed by the RSAM class.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from obspy import Stream, UTCDateTime

from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.processing.sam import RSAM


DEFAULT_BANDS = {
    "LOW_5_20": [5.0, 20.0],
    "MID_20_80": [20.0, 80.0],
    "HIGH_80_240": [80.0, 240.0],
}


def parse_band_arg(text: str) -> Tuple[str, list[float]]:
    """
    Parse one --band argument.

    Format:
        NAME:FMIN-FMAX

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
    return name, [float(fmin), float(fmax)]


def utc(text: str) -> UTCDateTime:
    """Parse UTCDateTime from command-line text."""
    return UTCDateTime(text)


def iter_chunks(
    start: UTCDateTime,
    end: UTCDateTime,
    chunk_seconds: float,
) -> Iterable[Tuple[UTCDateTime, UTCDateTime]]:
    """Yield target processing windows."""
    t0 = UTCDateTime(start)
    while t0 < end:
        t1 = min(t0 + chunk_seconds, end)
        yield t0, t1
        t0 = t1


def trace_is_vertical(tr) -> bool:
    """Return True if channel code ends in Z."""
    return tr.stats.channel.upper().endswith("Z")


def clip_trace_percentile(tr, pct=99.9, multiplier=2.0):
    """
    Clip extreme samples using a robust percentile threshold.

    This is mainly intended to prevent one or two pathological spikes from
    dominating RSAM.
    """
    tr = tr.copy()
    data = np.asarray(tr.data, dtype=float)

    if data.size == 0:
        return tr

    finite = np.isfinite(data)
    if finite.sum() == 0:
        return tr

    threshold = np.nanpercentile(np.abs(data[finite]), pct) * multiplier

    if not np.isfinite(threshold) or threshold <= 0:
        return tr

    tr.data = np.clip(data, -threshold, threshold).astype(float)
    return tr


def remove_trace_baseline_approximately(tr):
    """
    Conservative baseline stabilization.

    Median removal is safer than mean removal when there are spikes.
    """
    tr = tr.copy()
    data = np.asarray(tr.data, dtype=float)

    if data.size == 0:
        return tr

    finite = np.isfinite(data)
    if finite.sum() == 0:
        return tr

    tr.data = (data - np.nanmedian(data[finite])).astype(float)
    return tr


def preprocess_stream(
    st: Stream,
    use_vertical_only: bool = False,
    clip: bool = True,
    clip_pct: float = 99.9,
    clip_multiplier: float = 2.0,
    detrend: bool = True,
    taper: bool = True,
    taper_percentage: float = 0.01,
    verbose: bool = False,
) -> Stream:
    """
    Conservative preprocessing prior to RSAM computation.

    The RSAM class itself applies the primary bandpass and band-specific filters.
    This function only:
      - optionally selects vertical traces
      - merges traces
      - removes approximate median baseline
      - optionally clips extreme spikes
      - detrends
      - tapers
    """
    st = st.copy()

    if use_vertical_only:
        st = Stream([tr for tr in st if trace_is_vertical(tr)])

    if len(st) == 0:
        return st

    try:
        st.merge(method=1, fill_value="latest")
    except Exception as exc:
        print(f"  merge warning: {exc}")

    processed = Stream()

    for tr in st:
        if tr.stats.npts == 0:
            continue

        tr = remove_trace_baseline_approximately(tr)

        if clip:
            tr = clip_trace_percentile(
                tr,
                pct=clip_pct,
                multiplier=clip_multiplier,
            )

        if detrend:
            try:
                tr.detrend("linear")
                tr.detrend("demean")
            except Exception as exc:
                if verbose:
                    print(f"  detrend warning for {tr.id}: {exc}")

        if taper:
            try:
                tr.taper(max_percentage=taper_percentage, type="hann")
            except Exception as exc:
                if verbose:
                    print(f"  taper warning for {tr.id}: {exc}")

        processed.append(tr)

    return processed


def safe_read_waveforms(
    client: EnhancedSDSClient,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
) -> Stream:
    """Read waveforms and return empty Stream on failure."""
    try:
        return client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
    except Exception as exc:
        print(f"  read failed: {exc}")
        return Stream()


def compute_rsam_archive(
    sds_root: Path,
    out_dir: Path,
    start: UTCDateTime,
    end: UTCDateTime,
    network: str,
    station: str,
    location: str,
    channel: str,
    chunk_seconds: float,
    read_buffer_seconds: float,
    sampling_interval: float,
    primary_filter: Optional[list[float]],
    bands: Dict[str, list[float]],
    corners: int,
    despike: bool,
    clip: bool,
    clip_pct: float,
    clip_multiplier: float,
    max_channels: int,
    vertical_only: bool,
    auto_vertical_if_many: bool,
    ext: str,
    overwrite: bool,
    verbose: bool,
):
    """Main processing loop."""
    client = EnhancedSDSClient(sds_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Nodal RSAM computation")
    print(f"SDS root: {sds_root}")
    print(f"Output directory: {out_dir}")
    print(f"Time range: {start} to {end}")
    print(f"Selectors: net={network} sta={station} loc={location} cha={channel}")
    print(f"Chunk length: {chunk_seconds} s")
    print(f"Read buffer: {read_buffer_seconds} s")
    print(f"Sampling interval: {sampling_interval} s")
    print(f"Primary filter: {primary_filter}")
    print(f"Bands: {bands}")
    print(f"Output extension: {ext}")
    print(f"RSAM.write overwrite: {overwrite}")
    print("=" * 80)

    for target_start, target_end in iter_chunks(start, end, chunk_seconds):
        read_start = target_start - read_buffer_seconds
        read_end = target_end + read_buffer_seconds

        print("\n" + "-" * 80)
        print(f"Target window: {target_start} to {target_end}")
        print(f"Read window:   {read_start} to {read_end}")

        st = safe_read_waveforms(
            client=client,
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=read_start,
            endtime=read_end,
        )
        

        if len(st) == 0:
            print("  no data")
            continue
        
    
        print("Loaded:")
        print(st.__str__(extended=True))   
        if verbose:
            st.plot(equal_scale=False, size=(800, 600), title=f"Loaded waveforms from {target_start} to {target_end} from SDS", outfile=str(out_dir / f"waveforms_{target_start.strftime('%Y%m%dT%H%M%S')}_{target_end.strftime('%Y%m%dT%H%M%S') }_loaded.png"))

        seed_ids = sorted({tr.id for tr in st})
        print(f"  read {len(st)} traces, {len(seed_ids)} unique SEED ids")

        use_vertical = vertical_only
        if auto_vertical_if_many and len(seed_ids) > max_channels:
            use_vertical = True
            print("  more than 10 SEED ids: selecting vertical components only")

        st = preprocess_stream(
            st,
            use_vertical_only=use_vertical,
            clip=clip,
            clip_pct=clip_pct,
            clip_multiplier=clip_multiplier,
            detrend=True,
            taper=True,
            taper_percentage=0.01,
            verbose=verbose,
        )

        if len(st) == 0:
            print("  no traces remain after preprocessing")
            continue

        # Trim back to target after buffered preprocessing.
        st.trim(target_start, target_end)
        st = Stream([tr for tr in st if tr.stats.npts > 0])

        if len(st) == 0:
            print("  no traces remain after target trim")
            continue

        print(f"  computing RSAM for {len(st)} traces")
        if verbose:
            st.plot(equal_scale=False, size=(800, 600), title=f"Sending waveforms from {target_start} to {target_end} to RSAM", outfile=str(out_dir / f"waveforms_{target_start.strftime('%Y%m%dT%H%M%S')}_{target_end.strftime('%Y%m%dT%H%M%S')    }_sent.png"))

    
        try:
            verbose = True
            rsam = RSAM(
                stream=st,
                sampling_interval=sampling_interval,
                filter=primary_filter,
                bands=bands,
                corners=corners,
                despike=despike,
                verbose=verbose,
            )
            print(rsam)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  RSAM failed: {exc}")
            continue

        
        if not getattr(rsam, "dataframes", None):
            print("  RSAM returned no dataframes")
            continue

        try:
            rsam.write(
                SAM_DIR=str(out_dir),
                ext=ext,
                overwrite=overwrite,
                verbose=verbose,
            )
            print(f"  wrote/merged RSAM output via RSAM.write() in {out_dir}")
        except Exception as exc:
            print(f"  RSAM.write() failed: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compute FLOVOpy RSAM/SAM metrics from an SDS archive in chunks."
    )

    parser.add_argument("--sds-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument("--network", default="*")
    parser.add_argument("--station", default="*")
    parser.add_argument("--location", default="*")
    parser.add_argument("--channel", default="*Z")

    parser.add_argument("--start", required=True, type=utc)
    parser.add_argument("--end", required=True, type=utc)

    chunk = parser.add_mutually_exclusive_group()
    chunk.add_argument("--chunk-hours", type=float, default=None)
    chunk.add_argument("--chunk-days", type=float, default=None)

    parser.add_argument("--read-buffer-seconds", type=float, default=300.0)
    parser.add_argument("--sampling-interval", type=float, default=60.0)

    parser.add_argument(
        "--primary-filter",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=[5.0, 240.0],
        help="Primary RSAM bandpass used for core metrics. Use --no-primary-filter to disable.",
    )
    parser.add_argument(
        "--no-primary-filter",
        action="store_true",
        help="Disable primary filter and compute core metrics on preprocessed waveform.",
    )

    parser.add_argument(
        "--band",
        action="append",
        type=parse_band_arg,
        default=[],
        help="Named band as NAME:FMIN-FMAX. Repeatable. Defaults to LOW_5_20, MID_20_80, HIGH_80_240.",
    )
    parser.add_argument(
        "--no-default-bands",
        action="store_true",
        help="Do not include default nodal bands.",
    )

    parser.add_argument("--corners", type=int, default=4)
    parser.add_argument("--despike", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--clip-pct", type=float, default=99.9)
    parser.add_argument("--clip-multiplier", type=float, default=2.0)
    parser.add_argument("--max-channels", type=int, default=100, help="Threshold for Trace selection when many SEED ids are read.")
    parser.add_argument("--vertical-only", action="store_true")
    parser.add_argument(
        "--no-auto-vertical-if-many",
        action="store_true",
        help="Do not automatically restrict to vertical channels when many SEED ids are read.",
    )

    parser.add_argument(
        "--ext",
        choices=["csv", "pickle"],
        default="csv",
        help="Output format passed to RSAM.write().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Pass overwrite=True to RSAM.write(). "
            "Default is False so chunks are merged into existing yearly files."
        ),
    )

    parser.add_argument("--verbose", action="store_true")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.chunk_days is not None:
        chunk_seconds = args.chunk_days * 86400.0
    elif args.chunk_hours is not None:
        chunk_seconds = args.chunk_hours * 3600.0
    else:
        chunk_seconds = 3600.0

    if args.no_primary_filter:
        primary_filter = None
    else:
        primary_filter = [float(args.primary_filter[0]), float(args.primary_filter[1])]

    bands: Dict[str, list[float]] = {}
    if not args.no_default_bands:
        bands.update(DEFAULT_BANDS)

    for name, limits in args.band:
        bands[name] = limits

    compute_rsam_archive(
        sds_root=args.sds_root,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        network=args.network,
        station=args.station,
        location=args.location,
        channel=args.channel,
        chunk_seconds=chunk_seconds,
        read_buffer_seconds=args.read_buffer_seconds,
        sampling_interval=args.sampling_interval,
        primary_filter=primary_filter,
        bands=bands,
        corners=args.corners,
        despike=args.despike,
        clip=not args.no_clip,
        clip_pct=args.clip_pct,
        clip_multiplier=args.clip_multiplier,
        max_channels=args.max_channels,
        vertical_only=args.vertical_only,
        auto_vertical_if_many=not args.no_auto_vertical_if_many,
        ext=args.ext,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
