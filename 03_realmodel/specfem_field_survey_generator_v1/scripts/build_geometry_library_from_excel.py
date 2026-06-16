#!/usr/bin/env python3
"""
Build SPECFEM survey/receiver geometry library from field metadata Excel files.

This script is intentionally conservative and Git-friendly:

- Excel workbooks are treated as raw metadata.
- Outputs are simple CSV files and SPECFEM STATIONS files under inputs/.
- No combined receiver components are generated. Combination happens later in
  generate_specfem_runs.py via --refraction-geometry, --nodal-geometry, and
  --streamer-geometry.

Key output conventions
----------------------
surveys/
  T1_1m_refraction.csv
  T1_2m_refraction_hammer.csv
  T1_2m_betsy.csv
  T1_streamer_MASW.csv or T1_Streamer_MASW_main_transect.csv depending workbook survey label
  T2_streamer_MASW.csv or T2_Streamer_MASW_western_transect.csv depending workbook survey label
  T1_N1_refraction.csv
  T1_N2_refraction.csv
  T3_1m_refraction.csv
  T3_nodal_only.csv
  T4_1m_refraction.csv

receiver_components/
  T1_1m_refraction_receivers.csv
  T1_2m_refraction_receivers.csv
  T1_N1_geometry.csv
  T1_N2_geometry.csv
  T3_1m_refraction_receivers.csv
  T3_nodal.csv
  T4_1m_refraction_receivers.csv
  streamer_24ch_5ft_relative.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Optional, Iterable

import numpy as np
import pandas as pd


def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_topography(path: Optional[Path]):
    """Load topography from either a CSV file or Mel's all-transect Excel workbook.

    CSV format:
      distance_m,elevation_m

    Excel format:
      one sheet per line: T1, T2, T3, T4
      row 1 title, row 2 headers, row 3+ data
      columns: distance, easting, northing, elevation

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]] | None
        Mapping from line name to (distance_m, elevation_m).
    """
    if path is None or not path.exists():
        return None

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        topo = {}
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            line = str(sheet).strip()
            df = pd.read_excel(path, sheet_name=sheet, header=1)
            if df.empty or len(df.columns) < 4:
                continue

            distance = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            elevation = pd.to_numeric(df.iloc[:, 3], errors="coerce")
            ok = distance.notna() & elevation.notna()

            if ok.any():
                topo[line] = (
                    distance[ok].to_numpy(float),
                    elevation[ok].to_numpy(float),
                )

        if not topo:
            raise ValueError(f"No usable topography sheets found in {path}")
        return topo

    df = pd.read_csv(path)
    if not {"distance_m", "elevation_m"}.issubset(df.columns):
        raise ValueError(f"Topography CSV {path} must contain distance_m and elevation_m")

    return {
        "T1": (
            df["distance_m"].to_numpy(float),
            df["elevation_m"].to_numpy(float),
        )
    }


def surface_z(xs: Iterable[float], line: str, topo):
    """Return SPECFEM z-coordinates for receiver/source positions.

    If topography exists for the requested line, z is shifted so that the first
    profile elevation maps to z=50 m. Otherwise return flat z=50 m.
    """
    xs = np.asarray(list(xs), dtype=float)
    if topo is None:
        return np.full(xs.shape, 50.0, dtype=float)

    line_key = str(line).strip()
    if isinstance(topo, dict) and line_key in topo:
        tx, elev = topo[line_key]
        e = np.interp(xs, tx, elev)
        return 50.0 + (e - elev[0])

    return np.full(xs.shape, 50.0, dtype=float)


def write_stations(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for _, r in df.iterrows():
            f.write(
                f"{str(r.station):<10s} {str(r.network):<4s} "
                f"{float(r.x_m):16.7f} {float(r.z_m):16.7f} "
                f"0.0 0.0\n"
            )


def write_survey(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["shot_no", "file_no", "line", "survey", "source_x_m", "source_family",
            "confidence", "review_status"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols].to_csv(out_path, index=False)


def make_receiver_range(line: str, first: float, last: float, spacing: float, prefix: str,
                        topo, receiver_type: str, source_geometry: str) -> pd.DataFrame:
    xs = np.arange(float(first), float(last) + 0.5 * float(spacing), float(spacing))
    zs = surface_z(xs, line, topo)
    return pd.DataFrame({
        "station": [f"{prefix}{i:04d}" for i in range(1, len(xs) + 1)],
        "network": line,
        "x_m": xs.astype(float),
        "z_m": zs.astype(float),
        "receiver_type": receiver_type,
        "source_geometry": source_geometry,
    })


def normalize_source_family(value, default="hammer") -> str:
    s = str(value).lower() if pd.notna(value) else default
    if "betsy" in s:
        return "betsy"
    if "peg" in s:
        return "peg"
    if "hammer" in s or "sledge" in s:
        return "hammer"
    return default


def read_refraction_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    out = pd.DataFrame()
    out["shot_no"] = df.get("shot_no")
    out["file_no"] = df.get("file_no")
    out["line"] = df.get("transect")
    out["survey"] = df.get("survey", sheet)
    out["source_x_m"] = df.get("source_position_m")
    out["source_family"] = "hammer"
    out["receiver_first_m"] = df.get("receiver_first_m")
    out["receiver_last_m"] = df.get("receiver_last_m")
    out["receiver_spacing_m"] = df.get("receiver_spacing_m")
    out["operator"] = df.get("operator")
    out["plate_type"] = df.get("plate_type")
    out["confidence"] = df.get("confidence")
    out["review_status"] = df.get("review_status")
    out = out[out["shot_no"].notna()].copy()
    return out


def build_refraction_products(jochen_xlsx: Path, out_root: Path, topo):
    xl = pd.ExcelFile(jochen_xlsx)
    summary = {"refraction_sheets": []}

    for sheet in xl.sheet_names:
        if not sheet.endswith("_Refraction"):
            continue

        df = read_refraction_sheet(jochen_xlsx, sheet)
        if df.empty:
            continue

        survey_name = safe_name(str(df["survey"].dropna().iloc[0])) if df["survey"].notna().any() else safe_name(sheet)
        line = safe_name(str(df["line"].dropna().iloc[0])) if df["line"].notna().any() else survey_name.split("_")[0]

        valid_geom = df[
            df["receiver_first_m"].notna()
            & df["receiver_last_m"].notna()
            & df["receiver_spacing_m"].notna()
        ]

        if not valid_geom.empty:
            g = valid_geom.iloc[0]
            first = float(g["receiver_first_m"])
            last = float(g["receiver_last_m"])
            spacing = float(g["receiver_spacing_m"])
            comp = make_receiver_range(
                line=line,
                first=first,
                last=last,
                spacing=spacing,
                prefix=f"G{line}",
                topo=topo,
                receiver_type="geode_refraction",
                source_geometry=survey_name,
            )
            comp.to_csv(out_root / "receiver_components" / f"{survey_name}_receivers.csv", index=False)
            write_stations(comp, out_root / "receivers" / f"STATIONS_{survey_name}_receivers.dat")

        # T1 2-m gets split into hammer and Betsy source lists using source metadata,
        # not source position alone.
        is_t1_2m = (
            line == "T1"
            and "2m" in survey_name.lower()
            and "refraction" in survey_name.lower()
        )

        if is_t1_2m:
            operator = df["operator"].astype(str).str.lower() if "operator" in df.columns else pd.Series("", index=df.index)
            plate_type = df["plate_type"].astype(str).str.lower() if "plate_type" in df.columns else pd.Series("", index=df.index)

            is_betsy = (
                operator.str.contains("betsy", na=False)
                | plate_type.str.contains("betsy", na=False)
            )

            betsy = df[is_betsy].copy()
            betsy["source_family"] = "betsy"
            write_survey(betsy, out_root / "surveys" / "T1_2m_betsy.csv")

            hammer = df[~is_betsy].copy()
            hammer["source_family"] = "hammer"
            write_survey(hammer, out_root / "surveys" / "T1_2m_refraction_hammer.csv")
        else:
            df["source_family"] = "hammer"
            write_survey(df, out_root / "surveys" / f"{survey_name}.csv")

        summary["refraction_sheets"].append(sheet)

    return summary


def read_streamer_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    out = pd.DataFrame()
    out["shot_no"] = df.get("shot_no")
    out["file_no"] = df.get("file_no")
    out["line"] = df.get("transect")
    out["survey"] = df.get("survey", sheet)
    out["source_x_m"] = df.get("shot_location_m")
    out["source_family"] = df.get("source_type", "peg").apply(lambda v: normalize_source_family(v, "peg"))
    out["confidence"] = df.get("confidence")
    out["review_status"] = df.get("review_status")
    out = out[out["shot_no"].notna()].copy()
    out = out[out["source_x_m"].notna()].copy()
    return out


def build_streamer_products(jochen_xlsx: Path, out_root: Path):
    xl = pd.ExcelFile(jochen_xlsx)
    summary = {"streamer_sheets": []}

    for sheet in xl.sheet_names:
        if "Streamer" not in sheet:
            continue

        df = read_streamer_sheet(jochen_xlsx, sheet)
        if df.empty:
            continue

        line = safe_name(str(df["line"].dropna().iloc[0])) if df["line"].notna().any() else safe_name(sheet).split("_")[0]
        survey_name = safe_name(str(df["survey"].dropna().iloc[0])) if df["survey"].notna().any() else safe_name(sheet)

        # Preserve descriptive survey label if present, but make line prefix consistent.
        if "streamer" in survey_name.lower():
            out_name = f"{line}_{survey_name}"
        else:
            out_name = f"{line}_streamer_MASW"

        write_survey(df, out_root / "surveys" / f"{out_name}.csv")
        summary["streamer_sheets"].append(sheet)

    # Default moving streamer geometry: 24 channels, 5 ft spacing.
    # Sign convention: receiver_x = shot_x + offset_m.
    spacing_m = 5.0 * 0.3048
    offsets = np.arange(1, 25, dtype=float) * spacing_m
    streamer = pd.DataFrame({
        "station": [f"STR{i:04d}" for i in range(1, 25)],
        "network": "STMR",
        "offset_m": offsets,
        "receiver_type": "moving_streamer",
        "source_geometry": "streamer_24ch_5ft_relative",
    })
    streamer.to_csv(out_root / "receiver_components" / "streamer_24ch_5ft_relative.csv", index=False)

    return summary


def read_nodal_geometry_sheet(nodal_xlsx: Path, sheet: str, line: str, geom_name: str, topo) -> pd.DataFrame:
    # Some nodal sheets have the real header on the second row; try both.
    for header in (0, 1):
        df = pd.read_excel(nodal_xlsx, sheet_name=sheet, header=header)
        cols = set(df.columns.astype(str))
        if {"adopted_position_m"}.issubset(cols) or {"position_m"}.issubset(cols) or {"node_position_m"}.issubset(cols):
            break
    else:
        raise ValueError(f"Could not identify position column in {sheet}")

    if "adopted_position_m" in df.columns:
        pos_col = "adopted_position_m"
    elif "position_m" in df.columns:
        pos_col = "position_m"
    elif "node_position_m" in df.columns:
        pos_col = "node_position_m"
    else:
        raise ValueError(f"{sheet} missing position column")

    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")
    df = df[df[pos_col].notna()].copy()

    xs = df[pos_col].astype(float).to_numpy()
    zs = surface_z(xs, line, topo)

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        node_val = pd.to_numeric(pd.Series([r.get("node_from_north", i-1)]), errors="coerce").iloc[0]
        node_id = int(node_val) if pd.notna(node_val) else i - 1
        serial = r.get("normalized_serial_number", r.get("serial_number", r.get("raw_serial_number", "")))
        rows.append({
            "station": f"N{node_id:03d}",
            "network": line,
            "x_m": float(r[pos_col]),
            "z_m": float(zs[i-1]),
            "receiver_type": "smartsolo_node",
            "source_geometry": geom_name,
            "serial_number": "" if pd.isna(serial) else str(serial),
        })

    return pd.DataFrame(rows)


def read_nodal_refraction_sheet(nodal_xlsx: Path, sheet: str, line: str, survey_name: str) -> pd.DataFrame:
    df = pd.read_excel(nodal_xlsx, sheet_name=sheet)
    source_col = None
    for c in ["source_x_m", "source_position_m", "shot_position_m"]:
        if c in df.columns:
            source_col = c
            break
    if source_col is None:
        raise ValueError(f"{sheet} must contain source_x_m, source_position_m, or shot_position_m")

    df[source_col] = pd.to_numeric(df[source_col], errors="coerce")
    df = df[df[source_col].notna()].copy()

    out = pd.DataFrame()
    out["shot_no"] = df["shot_no"] if "shot_no" in df.columns else range(1, len(df) + 1)
    out["file_no"] = df["file_no"] if "file_no" in df.columns else np.nan
    out["line"] = line
    out["survey"] = survey_name
    out["source_x_m"] = df[source_col]
    out["source_family"] = df["source_family"].apply(lambda v: normalize_source_family(v, "hammer")) if "source_family" in df.columns else "hammer"
    out["confidence"] = df["confidence"] if "confidence" in df.columns else "from_nodal_metadata"
    out["review_status"] = df["review_status"] if "review_status" in df.columns else "OK"
    return out


def build_nodal_products(nodal_xlsx: Path, out_root: Path, topo):
    xl = pd.ExcelFile(nodal_xlsx)
    summary = {"nodal_geometries": []}

    # Support both updated and older sheet names.
    geom_candidates = [
        ("T1_N1_Geometry", "T1_N1_geometry"),
        ("T1_Nodal_Geometry_Orig", "T1_N1_geometry"),
        ("T1_N2_Geometry", "T1_N2_geometry"),
        ("T1_Nodal_Geometry_DenseConfig", "T1_N2_geometry"),
        ("T3_N4_Geometry", "T3_N4_geometry"),
        ("T3_Nodal_Geometry", "T3_N4_geometry"),
    ]

    seen_geoms = set()
    for sheet, geom_name in geom_candidates:
        if sheet not in xl.sheet_names or geom_name in seen_geoms:
            continue
        line = "T3" if geom_name.startswith("T3") else "T1"
        comp = read_nodal_geometry_sheet(nodal_xlsx, sheet, line, geom_name, topo)
        comp.to_csv(out_root / "receiver_components" / f"{geom_name}.csv", index=False)
        write_stations(comp, out_root / "receivers" / f"STATIONS_{geom_name}.dat")
        summary["nodal_geometries"].append(geom_name)
        seen_geoms.add(geom_name)

    # Updated explicit nodal shot survey sheets.
    shot_candidates = [
        ("T1_N1_Refraction", "T1", "T1_N1_refraction"),
        ("T1_N2_Refraction", "T1", "T1_N2_refraction"),
    ]
    for sheet, line, survey_name in shot_candidates:
        if sheet in xl.sheet_names:
            sdf = read_nodal_refraction_sheet(nodal_xlsx, sheet, line, survey_name)
            write_survey(sdf, out_root / "surveys" / f"{survey_name}.csv")

    # Fallback old T3 nodal-only shots from T3_Nodal_Geometry before/after columns.
    if "T3_Nodal_Geometry" in xl.sheet_names:
        try:
            df = pd.read_excel(nodal_xlsx, sheet_name="T3_Nodal_Geometry")
            shots = []
            for _, r in df.iterrows():
                for col in ["shot_position_before_m", "shot_position_after_m"]:
                    if col in df.columns and pd.notna(r.get(col)):
                        try:
                            shots.append(float(r[col]))
                        except Exception:
                            pass
            if shots:
                shots = sorted(set(round(s, 6) for s in shots))
                sdf = pd.DataFrame({
                    "shot_no": range(1, len(shots) + 1),
                    "file_no": np.nan,
                    "line": "T3",
                    "survey": "T3_nodal_only",
                    "source_x_m": shots,
                    "source_family": "hammer",
                    "confidence": "from_nodal_metadata",
                    "review_status": "CHECK",
                })
                write_survey(sdf, out_root / "surveys" / "T3_nodal_only.csv")
        except Exception:
            pass

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jochen-xlsx", required=True)
    ap.add_argument("--nodal-xlsx", required=True)
    ap.add_argument("--topography", default=None)
    ap.add_argument("--out-root", default="inputs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    for sub in ["surveys", "receiver_components", "receivers"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    topo = load_topography(Path(args.topography)) if args.topography else None

    summary = {
        "jochen_xlsx": str(args.jochen_xlsx),
        "nodal_xlsx": str(args.nodal_xlsx),
        "topography": str(args.topography),
    }

    summary.update(build_refraction_products(Path(args.jochen_xlsx), out_root, topo))
    summary.update(build_streamer_products(Path(args.jochen_xlsx), out_root))
    summary.update(build_nodal_products(Path(args.nodal_xlsx), out_root, topo))
    summary["combined_components"] = []

    summary["n_surveys"] = len(list((out_root / "surveys").glob("*.csv")))
    summary["n_receiver_components"] = len(list((out_root / "receiver_components").glob("*.csv")))
    summary["n_stations_files"] = len(list((out_root / "receivers").glob("STATIONS_*.dat")))

    (out_root / "geometry_build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
