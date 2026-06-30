#!/usr/bin/env python3
"""
Generate oversampled SPECFEM2D geometry for T1.

Purpose
-------
Create:
  1. A single STATIONS file with receivers from 0 to 300 m at 0.5 m spacing.
  2. A single SOURCES_LIST file containing all unique real T1 source_x_m positions.

Topography
----------
Preferred input is Mel's LiDAR workbook, e.g. Profile_LIDAR_All_Transects.xlsx,
with one sheet per transect and distance/elevation columns. The script converts
raw LiDAR elevation to the SPECFEM z-coordinate convention used in our other
scripts:

    z_specfem = z_origin + (elevation(x) - elevation(first_lidar_point))

If a LiDAR workbook is not available, the script can instead use an existing
SOURCES_LIST-style file as a topographic reference. That file should contain:

    Source_ID   X_coord   Z_coord

and z values are interpolated directly from it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    # loose fallback by substring
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for key, original in lowered.items():
        for name in candidates:
            n = name.lower()
            if n in key or key in n:
                return original
    return None


def load_lidar_topography(path: Path, sheet: str, z_origin: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Load LiDAR distance/elevation profile and return x, shifted SPECFEM z."""
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    xcol = find_col(df, [
        "Dist along profile (N=0)",
        "Dist along profile",
        "Distance along profile",
        "distance_m",
        "x_m",
        "station_m",
        "position_m",
    ])
    zcol = find_col(df, [
        "Elevation (m)",
        "Elevation",
        "elevation_m",
        "elev_m",
        "lidar_elevation_m",
        "z_m",
    ])
    if xcol is None or zcol is None:
        raise ValueError(f"Could not identify distance/elevation columns in {path} sheet {sheet}. Columns: {list(df.columns)}")

    x = pd.to_numeric(df[xcol], errors="coerce")
    elev = pd.to_numeric(df[zcol], errors="coerce")
    ok = x.notna() & elev.notna()
    prof = pd.DataFrame({"x": x[ok], "elev": elev[ok]}).sort_values("x")
    prof = prof.drop_duplicates("x", keep="first")
    if prof.empty:
        raise ValueError(f"No numeric topography rows found in {path} sheet {sheet}")

    xp = prof["x"].to_numpy(float)
    ep = prof["elev"].to_numpy(float)
    zp = z_origin + (ep - ep[0])
    return xp, zp


def load_sources_list_topography(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Source_ID X_coord Z_coord file and return x,z reference arrays."""
    xs: list[float] = []
    zs: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            xs.append(float(parts[1]))
            zs.append(float(parts[2]))
        except ValueError:
            continue
    if not xs:
        raise ValueError(f"No numeric x/z rows found in {path}")
    df = pd.DataFrame({"x": xs, "z": zs}).sort_values("x").drop_duplicates("x", keep="first")
    return df["x"].to_numpy(float), df["z"].to_numpy(float)


def interp_z(xs: np.ndarray, topo_x: np.ndarray, topo_z: np.ndarray) -> np.ndarray:
    """Interpolate z. Outside topo range, numpy.interp uses endpoint values."""
    return np.interp(xs, topo_x, topo_z)


def read_source_positions(source_csvs: list[Path], line: str = "T1") -> pd.DataFrame:
    rows = []
    for path in source_csvs:
        df = pd.read_csv(path)
        if "source_x_m" not in df.columns:
            continue
        if "line" in df.columns:
            df = df[df["line"].astype(str).str.strip().str.upper() == line.upper()]
        for _, r in df.iterrows():
            try:
                x = float(r["source_x_m"])
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            rows.append({
                "source_x_m": round(x, 7),
                "source_family": str(r.get("source_family", "")),
                "survey": str(r.get("survey", "")),
                "file": path.name,
            })
    if not rows:
        raise ValueError("No source_x_m positions found in supplied CSV files")

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby("source_x_m", as_index=False)
        .agg({
            "survey": lambda s: "; ".join(sorted(set(v for v in s if v and v != "nan"))),
            "source_family": lambda s: "; ".join(sorted(set(v for v in s if v and v != "nan"))),
            "file": lambda s: "; ".join(sorted(set(s))),
        })
        .sort_values("source_x_m")
        .reset_index(drop=True)
    )
    return grouped


def write_stations(path: Path, xs: np.ndarray, zs: np.ndarray, network: str = "T1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, (x, z) in enumerate(zip(xs, zs), start=1):
            # Position-coded, 0.5 m receivers produce unique cm-based station names.
            station = f"R{int(round(x * 100)):06d}"
            f.write(f"{station:<10s}{network:<6s}{x:16.7f}{z:16.7f} 0.0 0.0\n")


def write_sources_list(path: Path, xs: np.ndarray, zs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{'# Source_ID':<16s}{'X_coord':>14s}{'Z_coord':>16s}\n")
        for i, (x, z) in enumerate(zip(xs, zs), start=1):
            f.write(f"Source_{i:04d} {x:14.7f} {z:15.7f}\n")


def float_range(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    xs = start + step * np.arange(n, dtype=float)
    # avoid -0.0000000 and round-off tails
    return np.round(xs, 7)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate T1 oversampled STATIONS and actual-shot SOURCES_LIST files for SPECFEM2D.")
    ap.add_argument("--line", default="T1")
    ap.add_argument("--receiver-min", type=float, default=0.0)
    ap.add_argument("--receiver-max", type=float, default=300.0)
    ap.add_argument("--receiver-spacing", type=float, default=0.5)
    ap.add_argument("--source-csv", nargs="*", default=None,
                    help="CSV files containing source_x_m. If omitted, uses T1*.csv in current directory, excluding summary/output CSVs.")
    ap.add_argument("--lidar-xlsx", type=Path, default=None,
                    help="Optional LiDAR Excel workbook, e.g. Profile_LIDAR_All_Transects.xlsx")
    ap.add_argument("--lidar-sheet", default="T1")
    ap.add_argument("--topography-sources-list", type=Path, default=Path("SOURCES_LIST.txt"),
                    help="Fallback source-list style x/z topography reference.")
    ap.add_argument("--z-origin", type=float, default=50.0)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--stations-name", default="STATIONS_T1_0p5m_0_300m")
    ap.add_argument("--sources-name", default="SOURCES_LIST_T1_114_ACTUAL_SHOTS")
    args = ap.parse_args()

    if args.lidar_xlsx is not None and args.lidar_xlsx.exists():
        topo_x, topo_z = load_lidar_topography(args.lidar_xlsx, args.lidar_sheet, args.z_origin)
        topo_source = str(args.lidar_xlsx)
    elif args.topography_sources_list is not None and args.topography_sources_list.exists():
        topo_x, topo_z = load_sources_list_topography(args.topography_sources_list)
        topo_source = str(args.topography_sources_list)
    else:
        raise SystemExit("No usable topography input. Provide --lidar-xlsx or --topography-sources-list.")

    if args.source_csv:
        source_csvs = [Path(p) for p in args.source_csv]
    else:
        source_csvs = sorted(
            p for p in Path(".").glob(f"{args.line}_*.csv")
            if not re.search(r"minimal|unique|provenance|source_runs", p.name, re.I)
        )
    src = read_source_positions(source_csvs, args.line)

    station_xs = float_range(args.receiver_min, args.receiver_max, args.receiver_spacing)
    station_zs = interp_z(station_xs, topo_x, topo_z)

    source_xs = src["source_x_m"].to_numpy(float)
    source_zs = interp_z(source_xs, topo_x, topo_z)
    src.insert(0, "source_id", [f"Source_{i:04d}" for i in range(1, len(src) + 1)])
    src["z_coord"] = source_zs

    stations_path = args.outdir / args.stations_name
    sources_path = args.outdir / args.sources_name
    provenance_path = args.outdir / f"{args.sources_name}_with_provenance.csv"
    station_csv_path = args.outdir / f"{args.stations_name}.csv"

    write_stations(stations_path, station_xs, station_zs, args.line)
    write_sources_list(sources_path, source_xs, source_zs)

    pd.DataFrame({
        "station": [f"R{int(round(x * 100)):06d}" for x in station_xs],
        "network": args.line,
        "x_m": station_xs,
        "z_m": station_zs,
        "topography_source": topo_source,
    }).to_csv(station_csv_path, index=False)
    src.to_csv(provenance_path, index=False)

    print(f"Topography source: {topo_source}")
    print(f"Receivers: {len(station_xs)} from {station_xs[0]} to {station_xs[-1]} m at {args.receiver_spacing} m spacing")
    print(f"Unique sources: {len(source_xs)} from {source_xs.min()} to {source_xs.max()} m")
    print(f"Wrote {stations_path}")
    print(f"Wrote {sources_path}")
    print(f"Wrote {station_csv_path}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
