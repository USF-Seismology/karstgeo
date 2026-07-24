#!/usr/bin/env python3
"""
101_plot_virtual_T1_shot_gathers_v2.py

Create animation-ready PNGs from the per-shot SEG-Y files produced by notebook
100. The output folder structure deliberately follows the comparison pipeline:

    001_Source_0001_x0010p000m/
        virtual_T1_gather_Z.png
        virtual_T1_wiggle_Z.png

The leading folder number records creation/processing order, but movie ordering
must use the source coordinate token at the end of the folder name.

Version 2 fixes both animation axes: receiver x is fixed by command-line limits,
and time is cropped relative to an automatically detected coherent shot onset.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read


SOURCE_RE = re.compile(
    r"T1_VIRTUAL_SHOT_(?P<shot>\d+)_x(?P<x>\d+(?:\.\d+)?)m_(?P<component>[ZNE])\.segy$",
    re.I,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-shot T1 virtual SEG-Y files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for shot folders and PNG products.",
    )
    parser.add_argument("--component", default="Z")
    parser.add_argument(
        "--x-min-m",
        type=float,
        default=0.0,
        help="Fixed minimum receiver coordinate shown in every frame.",
    )
    parser.add_argument(
        "--x-max-m",
        type=float,
        default=300.0,
        help="Fixed maximum receiver coordinate shown in every frame.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("auto_energy", "file_start"),
        default="auto_energy",
        help=(
            "auto_energy detects the first coherent shot energy and resets it "
            "to t=0; file_start uses the original SEG-Y start."
        ),
    )
    parser.add_argument(
        "--pre-onset-s",
        type=float,
        default=0.025,
        help="Time retained before the detected shot onset.",
    )
    parser.add_argument(
        "--post-onset-s",
        type=float,
        default=0.600,
        help="Time retained after the detected shot onset.",
    )
    parser.add_argument(
        "--onset-smooth-s",
        type=float,
        default=0.006,
        help="Smoothing length for the robust energy detector.",
    )
    parser.add_argument(
        "--onset-threshold-mad",
        type=float,
        default=8.0,
        help="Detector threshold above baseline in robust MAD units.",
    )
    parser.add_argument(
        "--onset-min-fraction-of-peak",
        type=float,
        default=0.08,
        help="Minimum detector threshold as a fraction of peak energy.",
    )
    parser.add_argument(
        "--onset-min-duration-s",
        type=float,
        default=0.004,
        help="Minimum sustained threshold crossing duration.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Symmetric image clipping percentile.",
    )
    parser.add_argument(
        "--trace-normalize",
        action="store_true",
        help="Normalize each receiver trace independently for the gather image.",
    )
    parser.add_argument(
        "--wiggle-scale",
        type=float,
        default=0.45,
        help="Fraction of median receiver spacing used for wiggle amplitude.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate PNGs that already exist.",
    )
    return parser.parse_args()


def decode_scaled_coordinate(raw_value, scalar):
    if scalar is None or scalar == 0:
        return float(raw_value)
    scalar = int(scalar)
    if scalar > 0:
        return float(raw_value) * scalar
    return float(raw_value) / abs(scalar)


def receiver_x_from_trace(trace):
    header = getattr(getattr(trace.stats, "segy", None), "trace_header", None)
    if header is None:
        return np.nan

    raw_x = getattr(header, "group_coordinate_x", 0)
    scalar = getattr(header, "scalar_to_be_applied_to_all_coordinates", 0)
    return decode_scaled_coordinate(raw_x, scalar)


def source_token(x_m):
    millimetres = int(round(float(x_m) * 1000.0))
    whole = millimetres // 1000
    fraction = millimetres % 1000
    return f"x{whole:04d}p{fraction:03d}m"



def read_gather(path):
    stream = read(str(path))
    if not stream:
        raise ValueError(f"No traces in {path}")

    traces = []
    receiver_x = []

    for trace in stream:
        data = np.asarray(trace.data, dtype=float)
        if data.size == 0:
            continue
        traces.append(data)
        receiver_x.append(receiver_x_from_trace(trace))

    if not traces:
        raise ValueError(f"No usable traces in {path}")

    n_samples = min(len(trace) for trace in traces)
    data = np.vstack([trace[:n_samples] for trace in traces])
    receiver_x = np.asarray(receiver_x, dtype=float)

    finite_x = np.isfinite(receiver_x)
    if not np.any(finite_x):
        raise ValueError(f"No finite receiver coordinates in {path}")

    data = data[finite_x]
    receiver_x = receiver_x[finite_x]

    order = np.argsort(receiver_x, kind="stable")
    data = data[order]
    receiver_x = receiver_x[order]

    dt = float(stream[0].stats.delta)
    file_time_s = np.arange(n_samples, dtype=float) * dt

    return file_time_s, receiver_x, data, dt


def moving_average_1d(values, n_samples):
    values = np.asarray(values, dtype=float)
    n_samples = max(1, int(n_samples))
    if n_samples == 1:
        return values.copy()
    kernel = np.ones(n_samples, dtype=float) / n_samples
    return np.convolve(values, kernel, mode="same")


def robust_scale_rows(data):
    centered = data - np.nanmedian(data, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(centered), axis=1)
    scale = 1.4826 * mad

    fallback = np.nanmax(np.abs(centered), axis=1)
    invalid = ~np.isfinite(scale) | (scale <= 0)
    scale[invalid] = fallback[invalid]
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0

    return centered / scale[:, None]


def first_sustained_crossing(mask, minimum_samples):
    minimum_samples = max(1, int(minimum_samples))
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if bool(value) else 0
        if run >= minimum_samples:
            return index - minimum_samples + 1
    return None


def detect_shot_onset(
    data,
    dt,
    *,
    smooth_s,
    threshold_mad,
    min_fraction_of_peak,
    min_duration_s,
):
    """
    Detect the earliest coherent shot energy.

    Each trace is robustly standardized. The detector is the 90th percentile
    of smoothed absolute amplitude across receivers, so an early arrival need
    not be visible on half of the array before it can be detected.
    """
    standardized = robust_scale_rows(np.asarray(data, dtype=float))
    absolute_data = np.abs(standardized)

    smooth_n = max(1, int(round(float(smooth_s) / dt)))
    smoothed = np.vstack(
        [moving_average_1d(trace, smooth_n) for trace in absolute_data]
    )
    detector = np.nanpercentile(smoothed, 90.0, axis=0)

    npts = detector.size
    baseline_end = max(
        smooth_n * 4,
        min(npts, int(round(0.20 * npts))),
    )
    baseline = detector[:baseline_end]

    baseline_median = float(np.nanmedian(baseline))
    baseline_mad = float(
        1.4826 * np.nanmedian(np.abs(baseline - baseline_median))
    )
    if not np.isfinite(baseline_mad) or baseline_mad <= 0:
        baseline_mad = float(np.nanstd(baseline))
    if not np.isfinite(baseline_mad) or baseline_mad <= 0:
        baseline_mad = 1e-12

    peak = float(np.nanmax(detector))
    threshold = max(
        baseline_median + float(threshold_mad) * baseline_mad,
        float(min_fraction_of_peak) * peak,
    )

    minimum_samples = max(
        1,
        int(round(float(min_duration_s) / dt)),
    )
    crossing = first_sustained_crossing(
        detector >= threshold,
        minimum_samples,
    )

    if crossing is None:
        # Conservative fallback: first time detector reaches 20% of its peak.
        fallback_threshold = max(
            baseline_median + 3.0 * baseline_mad,
            0.20 * peak,
        )
        crossing = first_sustained_crossing(
            detector >= fallback_threshold,
            minimum_samples,
        )

    if crossing is None:
        crossing = int(np.nanargmax(detector))

    return {
        "onset_index": int(crossing),
        "onset_s": float(crossing * dt),
        "threshold": float(threshold),
        "peak": peak,
        "baseline_median": baseline_median,
        "baseline_mad": baseline_mad,
        "detector": detector,
    }


def crop_relative_to_onset(
    file_time_s,
    data,
    onset_s,
    *,
    pre_onset_s,
    post_onset_s,
):
    crop_start_s = max(
        float(file_time_s[0]),
        float(onset_s) - float(pre_onset_s),
    )
    crop_end_s = min(
        float(file_time_s[-1]),
        float(onset_s) + float(post_onset_s),
    )

    mask = (
        (file_time_s >= crop_start_s)
        & (file_time_s <= crop_end_s)
    )
    if not np.any(mask):
        raise ValueError(
            f"Empty crop: onset={onset_s}, "
            f"window={crop_start_s}..{crop_end_s}"
        )

    relative_time_s = file_time_s[mask] - float(onset_s)
    cropped_data = data[:, mask]

    return (
        relative_time_s,
        cropped_data,
        crop_start_s,
        crop_end_s,
    )

def normalize_rows(data):
    scale = np.nanmax(np.abs(data), axis=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return data / scale[:, None]


def plot_gather(
    time_s,
    receiver_x,
    data,
    source_x_m,
    output_path,
    *,
    clip_percentile,
    trace_normalize,
    x_min_m,
    x_max_m,
    dpi,
):
    image_data = normalize_rows(data) if trace_normalize else data.copy()

    finite = np.abs(image_data[np.isfinite(image_data)])
    clip = (
        np.percentile(finite, clip_percentile)
        if finite.size
        else 1.0
    )
    if not np.isfinite(clip) or clip <= 0:
        clip = 1.0

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(receiver_x) > 1:
        dx = np.median(np.diff(np.unique(receiver_x)))
    else:
        dx = 1.0

    extent = [
        receiver_x.min() - dx / 2,
        receiver_x.max() + dx / 2,
        time_s.max(),
        time_s.min(),
    ]

    im = ax.imshow(
        image_data.T,
        aspect="auto",
        extent=extent,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        interpolation="nearest",
    )

    ax.axvline(source_x_m, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlim(float(x_min_m), float(x_max_m))
    ax.set_ylim(float(time_s.max()), float(time_s.min()))
    ax.set_xlabel("Receiver position along T1 (m)")
    ax.set_ylabel("Time relative to detected shot onset (s)")
    normalization_label = "trace-normalized" if trace_normalize else "physical amplitude"
    ax.set_title(
        f"Virtual T1 shot at x = {source_x_m:.3f} m\n"
        f"Z component, {normalization_label}, {len(receiver_x)} receivers"
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized amplitude" if trace_normalize else "Amplitude")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_wiggle(
    time_s,
    receiver_x,
    data,
    source_x_m,
    output_path,
    *,
    wiggle_scale,
    x_min_m,
    x_max_m,
    dpi,
):
    normalized = normalize_rows(data)

    if len(receiver_x) > 1:
        unique_x = np.unique(receiver_x)
        spacing = np.median(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    else:
        spacing = 1.0

    amplitude_scale = float(wiggle_scale) * spacing

    fig, ax = plt.subplots(figsize=(12, 7))

    for x_m, trace in zip(receiver_x, normalized):
        curve = x_m + amplitude_scale * trace
        ax.plot(curve, time_s, color="black", linewidth=0.45)
        ax.fill_betweenx(
            time_s,
            x_m,
            curve,
            where=curve >= x_m,
            color="black",
            alpha=0.35,
            linewidth=0,
        )

    ax.axvline(source_x_m, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlim(float(x_min_m), float(x_max_m))
    ax.set_ylim(float(time_s.max()), float(time_s.min()))
    ax.set_xlabel("Receiver position along T1 (m)")
    ax.set_ylabel("Time relative to detected shot onset (s)")
    ax.set_title(
        f"Virtual T1 shot at x = {source_x_m:.3f} m\n"
        f"Z-component trace-normalized wiggles, {len(receiver_x)} receivers"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for path in sorted(args.input_dir.glob("*.segy")):
        match = SOURCE_RE.match(path.name)
        if not match:
            continue
        if match.group("component").upper() != args.component.upper():
            continue

        records.append(
            {
                "path": path,
                "shot_number": int(match.group("shot")),
                "source_x_m": float(match.group("x")),
                "component": match.group("component").upper(),
            }
        )

    records.sort(key=lambda row: (row["source_x_m"], row["shot_number"]))

    if args.limit is not None:
        records = records[: args.limit]

    manifest_rows = []

    for creation_index, record in enumerate(records, start=1):
        source_x_m = record["source_x_m"]
        shot_number = record["shot_number"]
        folder_name = (
            f"{creation_index:03d}_Source_{shot_number:04d}_"
            f"{source_token(source_x_m)}"
        )
        shot_dir = args.output_dir / folder_name
        shot_dir.mkdir(parents=True, exist_ok=True)

        gather_png = shot_dir / f"virtual_T1_gather_{args.component.upper()}.png"
        gather_norm_png = shot_dir / (
            f"virtual_T1_gather_trace_normalized_{args.component.upper()}.png"
        )
        wiggle_png = shot_dir / f"virtual_T1_wiggle_{args.component.upper()}.png"

        outputs_exist = all(
            path.exists()
            for path in (gather_png, gather_norm_png, wiggle_png)
        )

        if args.overwrite or not outputs_exist:
            file_time_s, receiver_x, full_data, dt = read_gather(
                record["path"]
            )

            if args.align_mode == "auto_energy":
                onset = detect_shot_onset(
                    full_data,
                    dt,
                    smooth_s=args.onset_smooth_s,
                    threshold_mad=args.onset_threshold_mad,
                    min_fraction_of_peak=(
                        args.onset_min_fraction_of_peak
                    ),
                    min_duration_s=args.onset_min_duration_s,
                )
                onset_s = float(onset["onset_s"])
            else:
                onset = {
                    "threshold": np.nan,
                    "peak": np.nan,
                    "baseline_median": np.nan,
                    "baseline_mad": np.nan,
                }
                onset_s = 0.0

            (
                time_s,
                data,
                crop_start_s,
                crop_end_s,
            ) = crop_relative_to_onset(
                file_time_s,
                full_data,
                onset_s,
                pre_onset_s=args.pre_onset_s,
                post_onset_s=args.post_onset_s,
            )

            plot_gather(
                time_s,
                receiver_x,
                data,
                source_x_m,
                gather_png,
                clip_percentile=args.clip_percentile,
                trace_normalize=False,
                x_min_m=args.x_min_m,
                x_max_m=args.x_max_m,
                dpi=args.dpi,
            )
            plot_gather(
                time_s,
                receiver_x,
                data,
                source_x_m,
                gather_norm_png,
                clip_percentile=args.clip_percentile,
                trace_normalize=True,
                x_min_m=args.x_min_m,
                x_max_m=args.x_max_m,
                dpi=args.dpi,
            )
            plot_wiggle(
                time_s,
                receiver_x,
                data,
                source_x_m,
                wiggle_png,
                wiggle_scale=args.wiggle_scale,
                x_min_m=args.x_min_m,
                x_max_m=args.x_max_m,
                dpi=args.dpi,
            )

            n_receivers = len(receiver_x)
            n_samples = data.shape[1]
        else:
            n_receivers = np.nan
            n_samples = np.nan
            onset_s = np.nan
            crop_start_s = np.nan
            crop_end_s = np.nan
            onset = {
                "threshold": np.nan,
                "peak": np.nan,
                "baseline_median": np.nan,
                "baseline_mad": np.nan,
            }

        manifest_rows.append(
            {
                "creation_index": creation_index,
                "shot_number": shot_number,
                "source_x_m": source_x_m,
                "source_segy": str(record["path"]),
                "shot_folder": str(shot_dir),
                "gather_png": str(gather_png),
                "trace_normalized_png": str(gather_norm_png),
                "wiggle_png": str(wiggle_png),
                "n_receivers": n_receivers,
                "n_samples": n_samples,
                "align_mode": args.align_mode,
                "detected_onset_s_from_file_start": onset_s,
                "crop_start_s_from_file_start": crop_start_s,
                "crop_end_s_from_file_start": crop_end_s,
                "display_time_min_s": -args.pre_onset_s,
                "display_time_max_s": args.post_onset_s,
                "x_min_m": args.x_min_m,
                "x_max_m": args.x_max_m,
                "onset_detector_threshold": onset["threshold"],
                "onset_detector_peak": onset["peak"],
            }
        )

        print(
            f"[{creation_index:03d}/{len(records):03d}] "
            f"x={source_x_m:.3f} m -> {shot_dir.name}"
        )

    manifest_path = args.output_dir / "virtual_T1_animation_frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
                "creation_index",
                "shot_number",
                "source_x_m",
                "source_segy",
                "shot_folder",
                "gather_png",
                "trace_normalized_png",
                "wiggle_png",
                "n_receivers",
                "n_samples",
                "align_mode",
                "detected_onset_s_from_file_start",
                "crop_start_s_from_file_start",
                "crop_end_s_from_file_start",
                "display_time_min_s",
                "display_time_max_s",
                "x_min_m",
                "x_max_m",
                "onset_detector_threshold",
                "onset_detector_peak",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nFrames written below: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
