#!/usr/bin/env python3
"""Generate SPECFEM2D run directories using Felix-style SOURCE replacement.

Updated naming convention:
- T1_N1_Geometry -> inputs/receiver_components/T1_N1_geometry.csv
- T1_N2_Geometry -> inputs/receiver_components/T1_N2_geometry.csv
- T1_N1_Refraction -> inputs/surveys/T1_N1_refraction.csv
- T1_N2_Refraction -> inputs/surveys/T1_N2_refraction.csv


This script creates one SPECFEM-ready run directory per shot/source case. It does
not submit jobs.

Receiver geometry can be assembled on the fly from optional components:

  --refraction-geometry  fixed geophone/refraction receivers
  --nodal-geometry       fixed nodal receivers
  --streamer-geometry    moving streamer receivers, defined by offset_m from source

If a component is omitted, it contributes no receivers.

For streamer surveys, STATIONS is regenerated for every shot because the streamer
receiver positions move with the PEG shot position. Fixed receiver components are
reused for every shot.

The physical model is controlled by:
  --par-template
  --interfaces-file
  --cave-regions-file / --no-cave-regions-file

For T1, use the LiDAR/topography interface and the SunFISH cave/no-cave region files.
For T2/T3/T4, pass placeholder flat-interface and cave-region files until real
topography/cave products are available.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def relpath(pathlike: str | Path) -> Path:
    """Return path relative to repo root unless already absolute."""
    p = Path(pathlike)
    return p if p.is_absolute() else ROOT / p


def load_source_table(path: str = "inputs/sources/source_families.csv") -> pd.DataFrame:
    return pd.read_csv(relpath(path))


def load_topography(path: str | None, line: str | None = None):
    """Load optional topography from CSV or Mel's all-transect Excel workbook.

    CSV format:
      distance_m,elevation_m

    Excel format:
      one sheet per line: T1, T2, T3, T4
      row 1 title, row 2 headers, row 3+ data
      columns: distance, easting, northing, elevation
    """
    if not path:
        return None

    p = relpath(path)
    if not p.exists():
        raise FileNotFoundError(f"Topography file not found: {p}")

    suffix = p.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        if not line:
            raise ValueError("line is required when loading topography from an Excel workbook")
        sheet = str(line).strip()
        df = pd.read_excel(p, sheet_name=sheet, header=1)
        if df.empty or len(df.columns) < 4:
            raise ValueError(f"Topography sheet {sheet!r} in {p} does not have at least four columns")

        distance = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        elevation = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        ok = distance.notna() & elevation.notna()

        if not ok.any():
            raise ValueError(f"Topography sheet {sheet!r} in {p} has no usable distance/elevation rows")

        return distance[ok].to_numpy(float), elevation[ok].to_numpy(float)

    df = pd.read_csv(p)
    if not {"distance_m", "elevation_m"}.issubset(df.columns):
        raise ValueError(f"{p} must contain distance_m and elevation_m columns")
    return df["distance_m"].to_numpy(float), df["elevation_m"].to_numpy(float)


def surface_z(xs, topo, z0: float = 50.0):
    """Interpolate SPECFEM z coordinate from topography, or return flat z0."""
    xs = np.asarray(xs, dtype=float)
    if topo is None:
        return np.full(xs.shape, float(z0))
    tx, elev = topo
    e = np.interp(xs, tx, elev)
    return z0 + (e - elev[0])


def load_receiver_component(path: str | None) -> pd.DataFrame:
    """Load fixed receiver component CSV.

    Required columns:
      station, x_m

    Optional:
      network, z_m, receiver_type, source_geometry
    """
    if path is None:
        return pd.DataFrame(columns=["station", "network", "x_m", "z_m", "receiver_type"])

    p = relpath(path)
    df = pd.read_csv(p)

    required = {"station", "x_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")

    if "network" not in df.columns:
        df["network"] = "XX"
    if "z_m" not in df.columns:
        df["z_m"] = 50.0
    if "receiver_type" not in df.columns:
        df["receiver_type"] = "fixed"

    return df[["station", "network", "x_m", "z_m", "receiver_type"]].copy()


def make_streamer_receivers(streamer_df: pd.DataFrame | None, shot_x: float, topo=None, z0: float = 50.0) -> pd.DataFrame:
    """Generate moving streamer receivers for one shot.

    streamer_df required columns:
      station, offset_m

    Sign convention:
      receiver_x = shot_x + offset_m

    If z_m is not supplied, receiver z is interpolated from topography if available,
    otherwise flat z0 is used.
    """
    if streamer_df is None or streamer_df.empty:
        return pd.DataFrame(columns=["station", "network", "x_m", "z_m", "receiver_type"])

    out = streamer_df.copy()

    if "offset_m" not in out.columns:
        raise ValueError("streamer geometry must contain offset_m column")

    out["x_m"] = float(shot_x) + out["offset_m"].astype(float)

    if "network" not in out.columns:
        out["network"] = "STMR"

    if "z_m" not in out.columns:
        out["z_m"] = surface_z(out["x_m"].to_numpy(float), topo, z0=z0)

    if "receiver_type" not in out.columns:
        out["receiver_type"] = "moving_streamer"

    return out[["station", "network", "x_m", "z_m", "receiver_type"]].copy()


def write_stations(path: Path, stations: pd.DataFrame):
    """Write SPECFEM STATIONS file."""
    stations = stations.copy()

    # Avoid exact duplicates while preserving mixed instrument types.
    stations = stations.drop_duplicates(subset=["station", "network", "x_m", "z_m"])

    with path.open("w") as f:
        for _, r in stations.iterrows():
            f.write(
                f"{str(r.station):<10s} {str(r.network):<4s} "
                f"{float(r.x_m):16.7f} {float(r.z_m):16.7f} "
                f"0.0 0.0\n"
            )


def read_region_block(path: str | None) -> str:
    if not path:
        return ""
    p = relpath(path)
    if not p.exists():
        raise FileNotFoundError(f"Region file not found: {p}")
    return p.read_text().strip()


def count_region_lines(region_text: str) -> int:
    return len([
        ln for ln in region_text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ])


def fill_par(cave_state: str, par_template: str, cave_regions_file: str | None, no_cave_regions_file: str | None) -> str:
    """Fill Par_file template.

    The template should contain:
      {nbregions}
      {cave_regions}

    Base geology is assumed to use 3 region lines. Cave/no-cave include files
    should contain only the additional material-4 cave region lines, or nothing
    for no cave.
    """
    text = relpath(par_template).read_text()

    if cave_state == "with_cave":
        region_text = read_region_block(cave_regions_file)
    else:
        region_text = read_region_block(no_cave_regions_file)

    n_extra = count_region_lines(region_text)
    return text.format(nbregions=3 + n_extra, cave_regions=region_text)


def source_content(source_row: pd.Series, xs: float, source_template: str) -> str:
    template = relpath(source_template).read_text()
    f0 = float(source_row.f0_hz)
    tshift = 2.0 / f0

    def get(k, default):
        v = source_row.get(k, default)
        if pd.isna(v) or str(v) == "":
            return default
        return v

    return template.format(
        xs=f"{float(xs):.6f}",
        source_type=int(source_row.source_type),
        f0=f"{f0:.6f}",
        tshift=f"{tshift:.6f}",
        anglesource=get("anglesource_deg", "180.0"),
        Mxx=get("Mxx", "1."),
        Mzz=get("Mzz", "1."),
        Mxz=get("Mxz", "0."),
        factor=get("factor", "1.d10"),
    )


def normalize_source_family(value, default="hammer") -> str:
    fam = str(value).lower() if pd.notna(value) else default
    if "betsy" in fam:
        return "betsy"
    if "peg" in fam:
        return "peg"
    if "hammer" in fam:
        return "hammer"
    return default


def make_run(
    out_root: str,
    line: str,
    survey: str,
    cave_state: str,
    stations_df: pd.DataFrame,
    shot_id,
    xs: float,
    source_row: pd.Series,
    par_template: str,
    source_template: str,
    interfaces_file: str,
    cave_regions_file: str | None,
    no_cave_regions_file: str | None,
):
    case = f"shot_{int(shot_id):04d}_x{float(xs):07.2f}_{source_row.source_name}"
    run_dir = relpath(out_root) / line / survey / cave_state / case
    data = run_dir / "DATA"
    data.mkdir(parents=True, exist_ok=True)

    (data / "Par_file").write_text(
        fill_par(cave_state, par_template, cave_regions_file, no_cave_regions_file)
    )
    (data / "SOURCE").write_text(
        source_content(source_row, xs, source_template)
    )
    write_stations(data / "STATIONS", stations_df)
    shutil.copy(relpath(interfaces_file), data / "interfaces.dat")

    metadata = {
        "line": line,
        "survey": survey,
        "cave_state": cave_state,
        "shot_id": int(shot_id),
        "source_x_m": float(xs),
        "source_name": str(source_row.source_name),
        "source_family": str(source_row.source_family),
        "n_stations": int(len(stations_df)),
        "par_template": str(par_template),
        "source_template": str(source_template),
        "interfaces_file": str(interfaces_file),
        "cave_regions_file": str(cave_regions_file) if cave_state == "with_cave" else None,
        "no_cave_regions_file": str(no_cave_regions_file) if cave_state == "no_cave" else None,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def main():
    ap = argparse.ArgumentParser()

    # Shot/source list.
    ap.add_argument("--survey-csv", required=True, help="CSV with shot_no, source_x_m, source_family columns")
    ap.add_argument("--survey-name", required=True)
    ap.add_argument("--line", default="T1")
    ap.add_argument("--cave-state", choices=["with_cave", "no_cave"], default="with_cave")
    ap.add_argument("--out-root", default="runs")

    # Receiver components.
    ap.add_argument("--refraction-geometry", default=None, help="Fixed receiver component CSV")
    ap.add_argument("--nodal-geometry", default=None, help="Fixed nodal receiver component CSV")
    ap.add_argument("--streamer-geometry", default=None, help="Moving streamer receiver component CSV with offset_m column")

    # Topography for moving streamer z-coordinates only. Fixed components already store z_m.
    ap.add_argument("--topography", default=None, help="Optional line topography CSV for moving streamer receivers")
    ap.add_argument("--flat-z", type=float, default=50.0, help="Flat receiver z used when no topography is supplied")

    # Physical model files.
    ap.add_argument("--par-template", default="templates/Par_file_T1_0_300.template")
    ap.add_argument("--source-template", default="templates/SOURCE_ricker.template")
    ap.add_argument("--interfaces-file", default="inputs/topography/T1_interfaces_0_300m_flat_subsurface.dat")
    ap.add_argument("--cave-regions-file", default="inputs/caves/T1_sunfish_regions_0p5m.inc")
    ap.add_argument("--no-cave-regions-file", default="inputs/caves/T1_no_cave_regions.inc")

    # Source selection.
    ap.add_argument("--source-table", default="inputs/sources/source_families.csv")
    ap.add_argument("--limit-source-family", default=None, help="e.g. hammer, betsy, peg")
    ap.add_argument("--single-source-name", default=None, help="Use exactly one source_name instead of whole family")

    args = ap.parse_args()

    shots = pd.read_csv(relpath(args.survey_csv))
    if "source_x_m" not in shots.columns:
        raise ValueError(f"{args.survey_csv} must contain source_x_m")
    if "shot_no" not in shots.columns:
        shots["shot_no"] = range(1, len(shots) + 1)
    if "source_family" not in shots.columns:
        shots["source_family"] = "hammer"

    shots = shots[shots.source_x_m.notna()].copy()

    srcs = load_source_table(args.source_table)

    refraction = load_receiver_component(args.refraction_geometry)
    nodal = load_receiver_component(args.nodal_geometry)

    streamer_def = None
    if args.streamer_geometry:
        streamer_def = pd.read_csv(relpath(args.streamer_geometry))

    topo = load_topography(args.topography, args.line)

    made = []

    for _, shot in shots.iterrows():
        streamer = make_streamer_receivers(
            streamer_def,
            shot.source_x_m,
            topo=topo,
            z0=args.flat_z,
        )

        #stations_df = pd.concat([refraction, nodal, streamer], ignore_index=True)
        station_parts = [
            df for df in (refraction, nodal, streamer)
            if df is not None and not df.empty
        ]
        stations_df = pd.concat(station_parts, ignore_index=True)


        fam = normalize_source_family(shot.get("source_family", "hammer"))

        if args.limit_source_family and fam != args.limit_source_family:
            continue

        if args.single_source_name:
            subset = srcs[srcs.source_name == args.single_source_name]
        else:
            subset = srcs[srcs.source_family == fam]

        if subset.empty:
            raise ValueError(f"No source definitions found for source family/name: {fam}")

        for _, srow in subset.iterrows():
            made.append(
                make_run(
                    args.out_root,
                    args.line,
                    args.survey_name,
                    args.cave_state,
                    stations_df,
                    shot.shot_no,
                    shot.source_x_m,
                    srow,
                    args.par_template,
                    args.source_template,
                    args.interfaces_file,
                    args.cave_regions_file,
                    args.no_cave_regions_file,
                )
            )

    print(f"Generated {len(made)} run directories under {args.out_root}")


if __name__ == "__main__":
    main()
