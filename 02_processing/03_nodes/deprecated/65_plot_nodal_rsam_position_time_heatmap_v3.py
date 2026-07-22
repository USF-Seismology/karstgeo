#!/usr/bin/env python3
"""Plot nodal RSAM amplitude as a position-time heatmap.

Legacy RSAM station codes are SmartSolo serial suffixes (last five digits).
The mapping CSV associates each network + location + serial with along-line
position. Deployment/location codes (N1, N2, N3, ...) therefore resolve node
redeployments without requiring explicit start/end times.

Measurements from positions separated by no more than ``--position-tolerance-m``
(default 0.25 m) are assigned to their cluster mean and merged across deployment
codes. Thus a node left in effectively the same position through N1, N2 and N3
appears as one heatmap row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from flovopy.processing.sam import RSAM


def utc(text: str) -> UTCDateTime:
    return UTCDateTime(text)


def clean_seed_station(value: object) -> str:
    """Normalize a legacy five-character RSAM/SDS station code."""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(5) if s.isdigit() else s


def serial_suffix(value: object) -> str:
    """Return the five-digit station suffix used for a full SmartSolo serial."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[-5:].zfill(5) if digits else ""


def load_mapping(path: Optional[Path]) -> pd.DataFrame:
    """Load unique network/location/station-suffix to position mappings."""
    if path is None:
        return pd.DataFrame(columns=["network", "location", "station", "x_m"])

    raw = pd.read_csv(path)
    required = {"network", "location", "x_m"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Mapping CSV lacks columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, r in raw.iterrows():
        if pd.isna(r["network"]) or pd.isna(r["location"]) or pd.isna(r["x_m"]):
            continue

        codes: set[str] = set()
        # Full serial is authoritative; RSAM uses its last five digits.
        for col in ("serial", "serial_clean"):
            if col in raw.columns:
                code = serial_suffix(r.get(col))
                if code:
                    codes.add(code)

        # old_station may already be the legacy five-digit code.
        for col in ("old_station", "old_station_clean"):
            if col in raw.columns and pd.notna(r.get(col)):
                codes.add(clean_seed_station(r[col]))

        for code in codes:
            rows.append(
                {
                    "network": str(r["network"]).strip(),
                    "location": str(r["location"]).strip(),
                    "station": code,
                    "x_m": float(r["x_m"]),
                }
            )

    mapping = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if mapping.empty:
        raise ValueError("No usable station mappings were found in the CSV.")

    # A network/location/station combination must resolve to one position.
    ambiguity = (
        mapping.groupby(["network", "location", "station"])["x_m"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    if not ambiguity.empty:
        raise ValueError(
            "Ambiguous network/location/station mappings remain:\n"
            + ambiguity.to_string()
        )
    return mapping


def parse_seed_id(seed_id: str) -> tuple[str, str, str, str]:
    parts = seed_id.split(".")
    if len(parts) != 4:
        raise ValueError(f"Expected NET.STA.LOC.CHA, got {seed_id!r}")
    return tuple(parts)  # type: ignore[return-value]


def lookup_position(seed_id: str, mapping: pd.DataFrame) -> float:
    net, sta, loc, _ = parse_seed_id(seed_id)
    sta_clean = clean_seed_station(sta)

    candidates = mapping[
        (mapping["network"] == net)
        & (mapping["location"] == loc)
        & (mapping["station"] == sta_clean)
    ]
    if len(candidates) == 1:
        return float(candidates.iloc[0]["x_m"])
    if len(candidates) > 1:
        positions = sorted(candidates["x_m"].unique())
        if len(positions) == 1:
            return float(positions[0])
        raise ValueError(f"Ambiguous position for {seed_id}: {positions}")

    # Position-coded station names written by stage 85 need no CSV lookup.
    if sta_clean.isdigit() and len(sta_clean) == 5:
        possible_x = int(sta_clean) / 100.0
        # Avoid interpreting a legacy serial suffix as a position when a mapping
        # file was supplied but this ID simply failed to match it.
        if mapping.empty:
            return possible_x

    raise KeyError(f"No position mapping for {seed_id}")


def dataframe_time_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        raise KeyError(
            f"Metric {metric!r} not present; available columns: {list(df.columns)}"
        )

    if isinstance(df.index, pd.DatetimeIndex):
        times = pd.to_datetime(df.index, utc=True)
    else:
        time_col = next(
            (c for c in ("time", "datetime", "date", "timestamp") if c in df.columns),
            None,
        )
        if time_col is None:
            numeric = pd.to_numeric(pd.Index(df.index), errors="coerce")
            if np.isfinite(numeric).all():
                times = pd.to_datetime(numeric, unit="s", utc=True)
            else:
                times = pd.to_datetime(df.index, utc=True, errors="coerce")
        else:
            values = df[time_col]
            times = pd.to_datetime(
                values,
                unit="s" if pd.api.types.is_numeric_dtype(values) else None,
                utc=True,
                errors="coerce",
            )

    values = pd.to_numeric(df[metric], errors="coerce").to_numpy()
    series = pd.Series(values, index=times)
    return series[~series.index.isna()].sort_index()


def cluster_positions(values: pd.Series, tolerance_m: float) -> tuple[pd.Series, pd.DataFrame]:
    """Collapse positions into groups whose total span is <= tolerance_m."""
    unique = np.sort(values.dropna().unique().astype(float))
    if not len(unique):
        return values.copy(), pd.DataFrame()

    clusters: list[list[float]] = []
    current = [float(unique[0])]
    for x in unique[1:]:
        x = float(x)
        if x - current[0] <= tolerance_m + 1e-12:
            current.append(x)
        else:
            clusters.append(current)
            current = [x]
    clusters.append(current)

    lookup: dict[float, float] = {}
    summary_rows = []
    for cluster_id, members in enumerate(clusters, start=1):
        mean_x = float(np.mean(members))
        for x in members:
            lookup[x] = mean_x
        summary_rows.append(
            {
                "cluster": cluster_id,
                "mean_x_m": mean_x,
                "min_x_m": min(members),
                "max_x_m": max(members),
                "n_original_positions": len(members),
            }
        )

    clustered = values.map(lambda x: lookup.get(float(x), np.nan) if pd.notna(x) else np.nan)
    return clustered, pd.DataFrame(summary_rows)


def build_long_table(
    rsam: RSAM,
    mapping: pd.DataFrame,
    metric: str,
    channel: Optional[str],
    position_tolerance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []

    for seed_id, df in rsam.dataframes.items():
        try:
            net, sta, loc, cha = parse_seed_id(seed_id)
            if channel and channel != "*" and cha != channel:
                continue
            position = lookup_position(seed_id, mapping)
            series = dataframe_time_series(df, metric)
        except Exception as exc:
            skipped.append((seed_id, str(exc)))
            continue

        good = np.isfinite(series.to_numpy(dtype=float))
        if not good.any():
            continue
        rows.append(
            pd.DataFrame(
                {
                    "time": series.index[good],
                    "value": series.to_numpy(dtype=float)[good],
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": cha,
                    "x_m_original": position,
                    "seed_id": seed_id,
                }
            )
        )

    if skipped:
        print(f"Skipped {len(skipped)} trace IDs. First examples:")
        for seed_id, reason in skipped[:15]:
            print(f"  {seed_id}: {reason}")

    if not rows:
        raise RuntimeError("No mapped RSAM samples remained after filtering.")

    table = pd.concat(rows, ignore_index=True)
    table["x_m"], clusters = cluster_positions(
        table["x_m_original"], position_tolerance_m
    )
    return table, clusters


def make_grid(
    table: pd.DataFrame,
    cadence: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Make one time series per merged physical node position.

    Returns
    -------
    times
        Resampled time-bin labels.
    positions
        Mean physical positions after the 0.25-m clustering.
    grid
        Array with shape (n_positions, n_times).
    """
    positions = np.sort(table["x_m"].dropna().unique())

    t0 = table["time"].min().floor(cadence)
    t1 = table["time"].max().ceil(cadence)
    times = pd.date_range(t0, t1, freq=cadence, tz="UTC")

    grid = np.full((len(positions), len(times)), np.nan)

    for i, x_m in enumerate(positions):
        series = (
            table.loc[table["x_m"] == x_m]
            .set_index("time")["value"]
            .sort_index()
            .resample(cadence)
            .median()
            .reindex(times)
        )
        grid[i, :] = series.to_numpy(dtype=float)

    return times, positions, grid


def make_fixed_height_position_grid(
    positions: np.ndarray,
    grid: np.ndarray,
    *,
    node_height_m: float = 2.0,
    y_resolution_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand each node time series into a fixed-height band.

    The clustered mean position is rounded to the nearest metre. A node at
    150 m with node_height_m=2 occupies the interval 149–151 m.

    Empty space between node bands remains NaN and therefore plots as white.
    """
    if node_height_m <= 0:
        raise ValueError("node_height_m must be positive.")
    if y_resolution_m <= 0:
        raise ValueError("y_resolution_m must be positive.")

    display_positions = np.rint(positions).astype(int)

    # Make sure rounding has not accidentally collapsed distinct nodes.
    duplicates = pd.Series(display_positions).value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        details = []
        for rounded_position in duplicates.index:
            original = positions[display_positions == rounded_position]
            details.append(
                f"{rounded_position} m <- "
                + ", ".join(f"{x:.3f} m" for x in original)
            )
        raise ValueError(
            "Rounding clustered positions to whole metres merged distinct nodes:\n"
            + "\n".join(details)
        )

    half_height = node_height_m / 2.0

    y_min = np.floor(display_positions.min() - half_height)
    y_max = np.ceil(display_positions.max() + half_height)

    y_edges = np.arange(
        y_min,
        y_max + y_resolution_m,
        y_resolution_m,
        dtype=float,
    )

    y_lower = y_edges[:-1]
    y_upper = y_edges[1:]

    expanded = np.full(
        (len(y_edges) - 1, grid.shape[1]),
        np.nan,
        dtype=float,
    )

    for source_row, position in enumerate(display_positions):
        band_min = position - half_height
        band_max = position + half_height

        target_rows = (
            (y_lower >= band_min - 1e-9)
            & (y_upper <= band_max + 1e-9)
        )

        expanded[target_rows, :] = grid[source_row, :]

    return display_positions, y_edges, expanded


def datetime_bin_edges(
    times: pd.DatetimeIndex,
    cadence: str,
) -> np.ndarray:
    """
    Construct explicit time-bin edges for pcolormesh.

    Each RSAM timestamp is treated as the left edge of its cadence interval.
    """
    if len(times) == 0:
        raise ValueError("No times supplied.")

    delta = pd.to_timedelta(cadence)

    edges = times.append(
        pd.DatetimeIndex([times[-1] + delta])
    )

    return mdates.date2num(edges.to_pydatetime())


def plot_heatmap(
    table: pd.DataFrame,
    outfile: Path,
    metric: str,
    cadence: str,
    log10: bool,
    percentile_range: tuple[float, float],
    title: Optional[str],
    node_height_m: float,
    y_resolution_m: float,
) -> None:
    times, physical_positions, node_grid = make_grid(table, cadence)

    display_positions, y_edges, display_grid = (
        make_fixed_height_position_grid(
            physical_positions,
            node_grid,
            node_height_m=node_height_m,
            y_resolution_m=y_resolution_m,
        )
    )

    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log10(display_grid)
    else:
        z = display_grid.copy()

    z[~np.isfinite(z)] = np.nan

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise RuntimeError("No finite values available for plotting.")

    vmin, vmax = np.nanpercentile(finite, percentile_range)

    x_edges = datetime_bin_edges(times, cadence)

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_ylabel("Position along profile (m)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylim(y_edges[0], y_edges[-1])

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    # Label the actual rounded node positions, but avoid overcrowding.
    if len(display_positions) <= 50:
        ax.set_yticks(display_positions)

    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label(
        f"log10({metric})" if log10 else metric
    )

    ax.set_title(
        title
        or f"Nodal {metric} amplitude through time and along-line position"
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"Wrote {outfile}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sam-dir", required=True, type=Path)
    p.add_argument("--mapping-csv", required=True, type=Path)
    p.add_argument("--network", required=True)
    p.add_argument("--start", required=True, type=utc)
    p.add_argument("--end", required=True, type=utc)
    p.add_argument("--sampling-interval", type=int, default=60)
    p.add_argument("--ext", choices=["csv", "pickle"], default="csv")
    p.add_argument("--metric", default="median")
    p.add_argument("--channel", default="DPZ")
    p.add_argument("--cadence", default="1min")
    p.add_argument(
        "--position-tolerance-m",
        type=float,
        default=0.25,
        help="Maximum span of positions merged into one row (default: 0.25 m).",
    )
    p.add_argument(
        "--node-height-m",
        type=float,
        default=2.0,
        help=(
            "Displayed vertical height of each node band in metres "
            "(default: 2.0)."
        ),
    )
    p.add_argument(
        "--y-resolution-m",
        type=float,
        default=1.0,
        help=(
            "Vertical raster-cell height used to construct node bands "
            "(default: 1.0 m)."
        ),
    )
    p.add_argument("--linear", action="store_true")
    p.add_argument("--percentiles", nargs=2, type=float, default=(2.0, 98.0))
    p.add_argument("--title", default=None)
    p.add_argument("--outfile", required=True, type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mapping = load_mapping(args.mapping_csv)
    print(f"Loaded {len(mapping)} unique network/location/station mappings.")

    rsam = RSAM.read(
        args.start,
        args.end,
        SAM_DIR=str(args.sam_dir),
        network=args.network,
        sampling_interval=args.sampling_interval,
        ext=args.ext,
        verbose=args.verbose,
    )

    table, clusters = build_long_table(
        rsam=rsam,
        mapping=mapping,
        metric=args.metric,
        channel=args.channel,
        position_tolerance_m=args.position_tolerance_m,
    )

    print("\nPosition clusters:")
    print(clusters.to_string(index=False))
    print("\nSamples by merged position:")
    print(table.groupby("x_m").size().rename("samples").to_string())

    plot_heatmap(
        table=table,
        outfile=args.outfile,
        metric=args.metric,
        cadence=args.cadence,
        log10=not args.linear,
        percentile_range=tuple(args.percentiles),
        title=args.title,
        node_height_m=args.node_height_m,
        y_resolution_m=args.y_resolution_m,
    )


if __name__ == "__main__":
    main()
