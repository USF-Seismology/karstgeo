#!/usr/bin/env python
"""
plot_hourly_band_median_abs.py

Plot hourly band median-absolute amplitudes from the CSV created by
compute_hourly_band_median_abs.py.

Creates one PNG per SEED id, with all selected band columns shown on the same
axes. The plot is trimmed to the actual time range where at least one selected
band has non-null data for that SEED id.

Example:

python plot_hourly_band_median_abs.py \
  --csv /Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_hourly_band_median_abs.csv \
  --out-dir /Volumes/tachyon/LBSSP_DATA/nodal_qc/T1_Z_hourly_band_plots \
  --bands LOW_5_20_median_abs MID_20_80_median_abs HIGH_80_200_median_abs \
  --logy

"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DEFAULT_BAND_COLUMNS = [
    "LOW_5_20_median_abs",
    "MID_20_80_median_abs",
    "HIGH_80_200_median_abs",
]


def safe_filename(text: str) -> str:
    """Return a filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def infer_band_columns(df: pd.DataFrame) -> list[str]:
    """Infer band median_abs columns, excluding RAW."""
    cols = [
        c for c in df.columns
        if c.endswith("_median_abs") and not c.startswith("RAW_")
    ]
    return cols


def trim_dataframe_to_nonnull_data(
    df: pd.DataFrame,
    band_columns: list[str],
) -> pd.DataFrame:
    """
    Trim to first and last row where at least one selected band is non-null.

    This trims independently for each SEED id.
    """
    present = [c for c in band_columns if c in df.columns]
    if not present:
        return df.iloc[0:0].copy()

    mask = df[present].notna().any(axis=1)

    if not mask.any():
        return df.iloc[0:0].copy()

    valid_times = df.loc[mask, "datetime"]
    tmin = valid_times.min()
    tmax = valid_times.max()

    return df[(df["datetime"] >= tmin) & (df["datetime"] <= tmax)].copy()


def plot_one_seed_id(
    df: pd.DataFrame,
    seed_id: str,
    band_columns: list[str],
    out_dir: Path,
    logy: bool = False,
    ylims: tuple[float, float] | None = None,
    title_prefix: str = "",
    dpi: int = 150,
):
    """Plot one SEED id."""
    df = df.sort_values("datetime").copy()
    df = trim_dataframe_to_nonnull_data(df, band_columns)

    if len(df) == 0:
        print(f"{seed_id}: no non-null data in selected bands")
        return None

    present = [c for c in band_columns if c in df.columns]
    if not present:
        print(f"{seed_id}: selected band columns not found")
        return None

    # Drop columns that are entirely NaN for this seed id.
    present = [c for c in present if df[c].notna().any()]
    if not present:
        print(f"{seed_id}: all selected bands are NaN")
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    for col in present:
        label = col.replace("_median_abs", "")
        ax.plot(df["datetime"], df[col], marker="o", linewidth=1.2, markersize=3, label=label)

    if logy:
        ax.set_yscale("log")

    if ylims is not None:
        ax.set_ylim(*ylims)

    title = seed_id
    if title_prefix:
        title = f"{title_prefix} {seed_id}"

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Hourly median absolute amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Nice date formatting.
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()

    out_file = out_dir / f"{safe_filename(seed_id)}_hourly_band_median_abs.png"
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)

    print(f"Wrote {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Plot hourly band median absolute amplitudes, one PNG per SEED id."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=(
            "Columns to plot. Default tries LOW_5_20_median_abs, "
            "MID_20_80_median_abs, HIGH_80_200_median_abs; if absent, "
            "all non-RAW *_median_abs columns are used."
        ),
    )
    parser.add_argument(
        "--seed-id",
        nargs="*",
        default=None,
        help="Optional one or more specific SEED ids to plot.",
    )
    parser.add_argument("--logy", action="store_true")
    parser.add_argument(
        "--ylims",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.csv}")
    df = pd.read_csv(args.csv)

    if "seed_id" not in df.columns:
        raise ValueError("CSV must contain a seed_id column")

    # Prefer ISO `time` column from the compute script.
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    elif "starttime" in df.columns:
        df["datetime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    else:
        raise ValueError("CSV must contain either time or starttime column")

    df = df.dropna(subset=["datetime"]).copy()

    if args.bands is None:
        band_columns = [c for c in DEFAULT_BAND_COLUMNS if c in df.columns]
        if not band_columns:
            band_columns = infer_band_columns(df)
    else:
        band_columns = args.bands

    missing = [c for c in band_columns if c not in df.columns]
    if missing:
        print("Warning: selected columns missing from CSV:")
        for c in missing:
            print(f"  {c}")

    band_columns = [c for c in band_columns if c in df.columns]
    if not band_columns:
        raise ValueError("No valid band columns available to plot")

    print("Plotting columns:")
    for c in band_columns:
        print(f"  {c}")

    if args.seed_id:
        df = df[df["seed_id"].isin(args.seed_id)].copy()

    seed_ids = sorted(df["seed_id"].dropna().unique())
    print(f"Found {len(seed_ids)} SEED ids to plot")

    written = []
    for seed_id in seed_ids:
        seed_df = df[df["seed_id"] == seed_id].copy()
        out_file = plot_one_seed_id(
            seed_df,
            seed_id=seed_id,
            band_columns=band_columns,
            out_dir=args.out_dir,
            logy=args.logy,
            ylims=tuple(args.ylims) if args.ylims is not None else None,
            title_prefix=args.title_prefix,
            dpi=args.dpi,
        )
        if out_file is not None:
            written.append(out_file)

    print(f"Done. Wrote {len(written)} PNG files to {args.out_dir}")


if __name__ == "__main__":
    main()
