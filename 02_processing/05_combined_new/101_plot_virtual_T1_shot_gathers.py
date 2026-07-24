#!/usr/bin/env python3
"""
101_plot_virtual_T1_shot_gathers.py

Create animation-ready PNGs from the per-shot SEG-Y files produced by notebook
100. The output folder structure deliberately follows the comparison pipeline:

    001_Source_0001_x0010p000m/
        virtual_T1_gather_Z.png
        virtual_T1_wiggle_Z.png

The leading folder number records creation/processing order, but movie ordering
must use the source coordinate token at the end of the folder name.
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
    parser.add_argument("--tmin-s", type=float, default=0.0)
    parser.add_argument("--tmax-s", type=float, default=1.25)
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


def read_gather(path, tmin_s, tmax_s):
    stream = read(str(path))
    if not stream:
        raise ValueError(f"No traces in {path}")

    traces = []
    receiver_x = []

    for trace in stream:
        data = np.asarray(trace.data, dtype=float)
        dt = float(trace.stats.delta)
        times = np.arange(data.size, dtype=float) * dt
        mask = (times >= tmin_s) & (times <= tmax_s)
        if not np.any(mask):
            continue
        traces.append(data[mask])
        receiver_x.append(receiver_x_from_trace(trace))

    if not traces:
        raise ValueError(f"No samples in requested time window for {path}")

    n_samples = min(len(trace) for trace in traces)
    data = np.vstack([trace[:n_samples] for trace in traces])
    receiver_x = np.asarray(receiver_x, dtype=float)

    order = np.argsort(receiver_x, kind="stable")
    data = data[order]
    receiver_x = receiver_x[order]

    dt = float(stream[0].stats.delta)
    time_s = tmin_s + np.arange(n_samples) * dt

    return time_s, receiver_x, data


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
    ax.set_xlabel("Receiver position along T1 (m)")
    ax.set_ylabel("Time after virtual shot (s)")
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
    ax.invert_yaxis()
    ax.set_xlabel("Receiver position along T1 (m)")
    ax.set_ylabel("Time after virtual shot (s)")
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
            time_s, receiver_x, data = read_gather(
                record["path"],
                args.tmin_s,
                args.tmax_s,
            )

            plot_gather(
                time_s,
                receiver_x,
                data,
                source_x_m,
                gather_png,
                clip_percentile=args.clip_percentile,
                trace_normalize=False,
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
                dpi=args.dpi,
            )
            plot_wiggle(
                time_s,
                receiver_x,
                data,
                source_x_m,
                wiggle_png,
                wiggle_scale=args.wiggle_scale,
                dpi=args.dpi,
            )

            n_receivers = len(receiver_x)
            n_samples = data.shape[1]
        else:
            n_receivers = np.nan
            n_samples = np.nan

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
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nFrames written below: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
