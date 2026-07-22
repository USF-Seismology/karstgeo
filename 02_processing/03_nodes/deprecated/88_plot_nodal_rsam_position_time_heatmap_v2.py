#!/usr/bin/env python3
"""
Plot nodal RSAM amplitude as a position-time heatmap.

The script reads FLOVOpy RSAM products and places each trace at its survey-line
position. It supports either:

1. legacy RSAM files whose station code is the SmartSolo serial suffix, using
   a time-aware station-position mapping CSV; or
2. RSAM recomputed from the rewritten SDS archive, where the 5-character station
   code is position in centimetres (e.g. 04050 = 40.50 m).

A separate panel is made for each location/deployment code by default, avoiding
accidental averaging where the same line position was occupied in different
windows.
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


def clean_code(value: object) -> str:
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return s.zfill(5)
    return s


def load_mapping(path: Optional[Path]) -> pd.DataFrame:
    """Load serial-to-position mappings, optionally bounded by time.

    Recommended columns are::

        network,location,old_station,x_m,start_time,end_time

    ``start_time`` is inclusive and ``end_time`` is exclusive. Empty bounds
    mean open-ended. A mapping without bounds is accepted only when the
    network/location/station combination maps uniquely to one position.
    """
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {'network', 'location', 'x_m'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Mapping CSV lacks columns: {sorted(missing)}')

    rows = []
    for _, r in df.iterrows():
        start_time = pd.to_datetime(r.get('start_time', pd.NaT), utc=True, errors='coerce')
        end_time = pd.to_datetime(r.get('end_time', pd.NaT), utc=True, errors='coerce')
        for source_column in ('old_station', 'serial', 'new_station'):
            if source_column not in df.columns or pd.isna(r.get(source_column)):
                continue
            code = clean_code(r[source_column])
            if not code:
                continue
            rows.append({
                'network': str(r['network']).strip(),
                'location': str(r['location']).strip(),
                'station': code,
                'x_m': float(r['x_m']),
                'start_time': start_time,
                'end_time': end_time,
                'window_id': r.get('window_id', ''),
            })
    if not rows:
        return pd.DataFrame(columns=[
            'network', 'location', 'station', 'x_m',
            'start_time', 'end_time', 'window_id'
        ])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def parse_seed_id(seed_id: str) -> tuple[str, str, str, str]:
    parts = seed_id.split('.')
    if len(parts) != 4:
        raise ValueError(f'Expected NET.STA.LOC.CHA, got {seed_id!r}')
    return tuple(parts)


def mapping_candidates(
    network: str, station: str, location: str, mapping: pd.DataFrame
) -> pd.DataFrame:
    if mapping.empty:
        return mapping
    return mapping[
        (mapping['network'] == network)
        & (mapping['location'] == location)
        & (mapping['station'] == clean_code(station))
    ].copy()


def assign_positions_by_time(
    times: pd.DatetimeIndex,
    seed_id: str,
    mapping: pd.DataFrame,
) -> np.ndarray:
    """Return one position per timestamp, using half-open deployment windows.

    Samples matching zero or more than one position are returned as NaN. This
    deliberately prevents a serial number reused at different positions from
    being silently assigned to the wrong location.
    """
    net, sta, loc, _ = parse_seed_id(seed_id)
    sta_clean = clean_code(sta)
    candidates = mapping_candidates(net, sta_clean, loc, mapping)

    # Position-coded station names introduced by stage 85 need no lookup.
    if candidates.empty and sta_clean.isdigit() and len(sta_clean) == 5:
        return np.full(len(times), int(sta_clean) / 100.0, dtype=float)
    if candidates.empty:
        raise KeyError(f'No position mapping for {seed_id}')

    unique_positions = np.sort(candidates['x_m'].unique())
    has_any_bounds = candidates['start_time'].notna().any() or candidates['end_time'].notna().any()
    if len(unique_positions) == 1 and not has_any_bounds:
        return np.full(len(times), unique_positions[0], dtype=float)
    if len(unique_positions) > 1 and not has_any_bounds:
        raise ValueError(
            f'Ambiguous mapping for {seed_id}: positions {unique_positions.tolist()} '
            'but mapping CSV has no start_time/end_time bounds.'
        )

    out = np.full(len(times), np.nan, dtype=float)
    match_count = np.zeros(len(times), dtype=np.uint8)
    for _, row in candidates.iterrows():
        mask = np.ones(len(times), dtype=bool)
        if pd.notna(row['start_time']):
            mask &= times >= row['start_time']
        if pd.notna(row['end_time']):
            mask &= times < row['end_time']
        out[mask] = float(row['x_m'])
        match_count[mask] += 1

    out[match_count != 1] = np.nan
    return out


def dataframe_time_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        raise KeyError(f'Metric {metric!r} not present; available columns: {list(df.columns)}')

    if isinstance(df.index, pd.DatetimeIndex):
        times = pd.to_datetime(df.index, utc=True)
    else:
        time_col = next((c for c in ('time', 'datetime', 'date', 'timestamp') if c in df.columns), None)
        if time_col is None:
            # FLOVOpy SAM dataframes often use epoch seconds as index.
            numeric = pd.to_numeric(pd.Index(df.index), errors='coerce')
            if np.isfinite(numeric).all():
                times = pd.to_datetime(numeric, unit='s', utc=True)
            else:
                times = pd.to_datetime(df.index, utc=True, errors='coerce')
        else:
            values = df[time_col]
            if pd.api.types.is_numeric_dtype(values):
                times = pd.to_datetime(values, unit='s', utc=True)
            else:
                times = pd.to_datetime(values, utc=True, errors='coerce')

    values = pd.to_numeric(df[metric], errors='coerce').to_numpy()
    s = pd.Series(values, index=times)
    return s[~s.index.isna()].sort_index()


def build_long_table(rsam: RSAM, mapping: pd.DataFrame, metric: str, channel: Optional[str]) -> pd.DataFrame:
    rows = []
    skipped = []
    unmapped_samples = 0

    for seed_id, df in rsam.dataframes.items():
        try:
            net, sta, loc, cha = parse_seed_id(seed_id)
        except Exception as exc:
            skipped.append((seed_id, str(exc)))
            continue

        if channel and channel != '*' and cha != channel:
            continue

        try:
            s = dataframe_time_series(df, metric)
            positions = assign_positions_by_time(s.index, seed_id, mapping)
        except Exception as exc:
            skipped.append((seed_id, str(exc)))
            continue

        values = s.to_numpy(dtype=float)
        good = np.isfinite(values) & np.isfinite(positions)
        unmapped_samples += int((np.isfinite(values) & ~np.isfinite(positions)).sum())
        if not good.any():
            continue
        rows.append(pd.DataFrame({
            'time': s.index[good],
            'value': values[good],
            'network': net,
            'station': sta,
            'location': loc,
            'channel': cha,
            'x_m': positions[good],
            'seed_id': seed_id,
        }))

    if skipped:
        print(f'Skipped {len(skipped)} trace IDs. First examples:')
        for item in skipped[:10]:
            print('  ', item)
    if unmapped_samples:
        print(
            f'Dropped {unmapped_samples} RSAM samples that matched zero or multiple '
            'deployment windows.'
        )

    if not rows:
        raise RuntimeError('No unambiguously mapped RSAM samples remained after filtering.')
    return pd.concat(rows, ignore_index=True)


def make_grid(group: pd.DataFrame, cadence: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    positions = np.sort(group['x_m'].unique())
    t0 = group['time'].min().floor(cadence)
    t1 = group['time'].max().ceil(cadence)
    times = pd.date_range(t0, t1, freq=cadence, tz='UTC')
    grid = np.full((len(positions), len(times)), np.nan)

    for i, x_m in enumerate(positions):
        g = group[group['x_m'] == x_m].set_index('time')['value'].sort_index()
        # Median protects against duplicated records at a position/time.
        g = g.resample(cadence).median().reindex(times)
        grid[i, :] = g.to_numpy(dtype=float)
    return times, positions, grid


def plot_heatmap(
    table: pd.DataFrame,
    outfile: Path,
    metric: str,
    cadence: str,
    log10: bool,
    percentile_range: tuple[float, float],
    combine_locations: bool,
    title: Optional[str],
):
    groups = [('all', table)] if combine_locations else list(table.groupby('location', sort=True))
    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(12, max(3.2, 2.7 * len(groups))),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    arrays = []
    prepared = []
    for label, group in groups:
        times, positions, grid = make_grid(group, cadence)
        z = np.log10(grid) if log10 else grid.copy()
        z[~np.isfinite(z)] = np.nan
        prepared.append((label, times, positions, z))
        arrays.append(z[np.isfinite(z)])

    finite = np.concatenate([a for a in arrays if len(a)])
    vmin, vmax = np.nanpercentile(finite, percentile_range)

    mesh = None
    for ax, (label, times, positions, z) in zip(axes[:, 0], prepared):
        x = mdates.date2num(times.to_pydatetime())
        mesh = ax.pcolormesh(x, positions, z, shading='nearest', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Position (m)')
        if not combine_locations:
            ax.set_title(f'Location/deployment {label}', loc='left', fontsize=10)
        ax.grid(False)

    axes[-1, 0].xaxis_date()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1, 0].set_xlabel('Time (UTC)')

    label = f'log10({metric})' if log10 else metric
    cbar = fig.colorbar(mesh, ax=axes[:, 0], pad=0.015)
    cbar.set_label(label)
    fig.suptitle(title or f'Nodal {metric} amplitude through time and along-line position')

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f'Wrote {outfile}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sam-dir', required=True, type=Path)
    p.add_argument('--mapping-csv', type=Path, default=None)
    p.add_argument('--network', required=True)
    p.add_argument('--start', required=True, type=utc)
    p.add_argument('--end', required=True, type=utc)
    p.add_argument('--sampling-interval', type=int, default=60)
    p.add_argument('--ext', choices=['csv', 'pickle'], default='csv')
    p.add_argument('--metric', default='median')
    p.add_argument('--channel', default='DPZ')
    p.add_argument('--cadence', default='1min', help='Pandas offset alias, e.g. 1min or 10s')
    p.add_argument('--linear', action='store_true', help='Plot linear rather than log10 amplitude')
    p.add_argument('--percentiles', nargs=2, type=float, default=(2.0, 98.0))
    p.add_argument('--combine-locations', action='store_true')
    p.add_argument('--title', default=None)
    p.add_argument('--outfile', required=True, type=Path)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    mapping = load_mapping(args.mapping_csv)
    rsam = RSAM.read(
        args.start, args.end,
        SAM_DIR=str(args.sam_dir),
        network=args.network,
        sampling_interval=args.sampling_interval,
        ext=args.ext,
        verbose=args.verbose,
    )
    table = build_long_table(rsam, mapping, args.metric, args.channel)
    print(table.groupby(['location', 'x_m']).size().rename('samples').to_string())

    plot_heatmap(
        table=table,
        outfile=args.outfile,
        metric=args.metric,
        cadence=args.cadence,
        log10=not args.linear,
        percentile_range=tuple(args.percentiles),
        combine_locations=args.combine_locations,
        title=args.title,
    )


if __name__ == '__main__':
    main()
