#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import pandas as pd


INPUT_DIR = Path("../inputs/receivers")
LIDAR_DIR = Path("/Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/05_Analysis/Elevation and DEMs")
SURVEY_DIR = Path("/Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/04_FieldData")

JOCHEN_XLSX = SURVEY_DIR / "jochen_field_notes_metadata_tables_with_geode_times.xlsx"
GLENN_XLSX = SURVEY_DIR / "glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx"
LIDAR_XLSX = LIDAR_DIR / "Profile_LIDAR_All_Transects.xlsx"

LINES = ["T1", "T2", "T3", "T4"]

SHOT_SHEETS = {
    "T1": {
        "jochen": ["T1_1m_Refraction", "T1_2m_Refraction", "T1_Streamer_MASW"],
        "glenn": ["T1_N2_Refraction", "T1_N3_Refraction"],
    },
    "T2": {
        "jochen": ["T2_Streamer_MASW"],
        "glenn": [],
    },
    "T3": {
        "jochen": ["T3_1m_Refraction"],
        "glenn": [],
    },
    "T4": {
        "jochen": ["T4_1m_Refraction"],
        "glenn": [],
    },
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def load_lidar_profile(line: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(LIDAR_XLSX, sheet_name=line)

    xcol = find_col(df, ["Dist along profile (N=0)"])
    zcol = find_col(df, ["Elevation (m)"])

    if xcol is None or zcol is None:
        raise ValueError(f"{line}: could not identify LiDAR x/elevation columns: {list(df.columns)}")

    prof = df[[xcol, zcol]].dropna().sort_values(xcol)
    return prof[xcol].to_numpy(float), prof[zcol].to_numpy(float)


def interp_elevation(line: str, x: float, cache: dict) -> float:
    if line not in cache:
        cache[line] = load_lidar_profile(line)

    xp, zp = cache[line]
    return float(np.interp(x, xp, zp))


def read_station_rows(path: Path) -> list[dict]:
    rows = []

    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) < 6:
            continue

        try:
            x = float(parts[2])
            z = float(parts[3])
            c5 = float(parts[4])
            c6 = float(parts[5])
        except ValueError:
            continue

        rows.append(
            {
                "old_sta": parts[0],
                "net": parts[1],
                "x": x,
                "z": z,
                "c5": c5,
                "c6": c6,
                "source_file": path.name,
            }
        )

    return rows


def instrument_prefix(old_sta: str, source_file: str) -> str:
    old = old_sta.upper()
    src = source_file.upper()

    if old.startswith("N") or "_N" in src or "NOD" in src:
        return "N"
    if old.startswith("GT") or old.startswith("G") or "GEODE" in src or "REFRACTION" in src:
        return "G"
    return "S"


def make_station_name(row: dict) -> str:
    prefix = instrument_prefix(row["old_sta"], row["source_file"])
    cm = int(round(row["x"] * 100))
    return f"{prefix}{cm:06d}"


def format_station_line(sta: str, net: str, x: float, z: float, c5: float, c6: float) -> str:
    return f"{sta:<10s}{net:<6s}{x:16.7f}{z:16.7f} {c5:.1f} {c6:.1f}"


def find_station_files(line: str) -> list[Path]:
    files = []
    for pattern in [
        f"STATIONS_{line}_*receivers*.dat",
        f"STATIONS_{line}_*geometry*.dat",
    ]:
        files.extend(INPUT_DIR.glob(pattern))
    return sorted(set(files))


def process_stations(line: str) -> None:
    files = find_station_files(line)

    if not files:
        print(f"{line}: no station files found")
        return

    raw_rows = []
    for path in files:
        raw_rows.extend(read_station_rows(path))

    seen = {}
    for row in raw_rows:
        if row["x"] not in seen:
            seen[row["x"]] = row

    rows = [seen[x] for x in sorted(seen)]

    for row in rows:
        row["new_sta"] = make_station_name(row)
        row["net"] = line

    out = INPUT_DIR / f"STATIONS_{line}_ALL_SORTED_UNIQUE_FORMATTED_RENAMED"
    out_map = INPUT_DIR / f"STATIONS_{line}_ALL_RENAME_MAP.csv"

    out.write_text(
        "\n".join(
            format_station_line(r["new_sta"], r["net"], r["x"], r["z"], r["c5"], r["c6"])
            for r in rows
        )
        + "\n"
    )

    with out_map.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["old_station", "new_station", "network", "x_m", "z_m", "source_file"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "old_station": r["old_sta"],
                    "new_station": r["new_sta"],
                    "network": r["net"],
                    "x_m": f"{r['x']:.7f}",
                    "z_m": f"{r['z']:.7f}",
                    "source_file": r["source_file"],
                }
            )

    print(f"{line}: wrote {out.name} with {len(rows)} unique stations")


def read_shots_from_sheet(xlsx: Path, sheet: str, line: str, source_workbook: str) -> list[dict]:
    df = pd.read_excel(xlsx, sheet_name=sheet)

    shot_col = find_col(df, ["source_position_m", "shot_location_m", "shot_position_m"])
    if shot_col is None:
        print(f"{line} {sheet}: no shot-position column found; skipping")
        return []

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get(shot_col)):
            continue

        try:
            x = float(r[shot_col])
        except Exception:
            continue

        row = {
            "line": line,
            "shot_position_m": x,
            "source_sheet": sheet,
            "source_workbook": source_workbook,
        }

        for col in df.columns:
            val = r[col]
            if pd.isna(val):
                continue
            row[str(col)] = val

        rows.append(row)

    return rows


def process_shots(line: str, lidar_cache: dict) -> None:
    all_rows = []

    config = SHOT_SHEETS.get(line, {})

    for sheet in config.get("jochen", []):
        if JOCHEN_XLSX.exists():
            all_rows.extend(read_shots_from_sheet(JOCHEN_XLSX, sheet, line, "jochen"))

    for sheet in config.get("glenn", []):
        if GLENN_XLSX.exists():
            all_rows.extend(read_shots_from_sheet(GLENN_XLSX, sheet, line, "glenn"))

    if not all_rows:
        print(f"{line}: no shot rows found")
        return

    for row in all_rows:
        row["elevation_m"] = interp_elevation(line, row["shot_position_m"], lidar_cache)
        row["shot_position_cm"] = int(round(row["shot_position_m"] * 100))
        row["shot_name"] = f"S{row['shot_position_cm']:06d}"

    all_rows = sorted(all_rows, key=lambda r: r["shot_position_m"])

    out_all = INPUT_DIR / f"SHOTS_{line}_ALL_METADATA.csv"
    out_unique = INPUT_DIR / f"SHOTS_{line}_UNIQUE_POSITIONS.csv"

    # Union of metadata columns.
    preferred = [
        "line",
        "shot_name",
        "shot_position_m",
        "shot_position_cm",
        "elevation_m",
        "source_sheet",
        "source_workbook",
    ]
    all_keys = []
    for row in all_rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    fieldnames = preferred + [k for k in all_keys if k not in preferred]

    with out_all.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["shot_position_cm"]].append(row)

    unique_rows = []
    for cm in sorted(grouped):
        group = grouped[cm]
        first = group[0]
        unique_rows.append(
            {
                "line": line,
                "shot_name": f"S{cm:06d}",
                "shot_position_m": f"{first['shot_position_m']:.7f}",
                "shot_position_cm": cm,
                "elevation_m": f"{first['elevation_m']:.7f}",
                "n_records_at_position": len(group),
                "source_sheets": ";".join(sorted(set(str(g["source_sheet"]) for g in group))),
                "source_workbooks": ";".join(sorted(set(str(g["source_workbook"]) for g in group))),
            }
        )

    with out_unique.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line",
                "shot_name",
                "shot_position_m",
                "shot_position_cm",
                "elevation_m",
                "n_records_at_position",
                "source_sheets",
                "source_workbooks",
            ],
        )
        writer.writeheader()
        writer.writerows(unique_rows)

    print(
        f"{line}: wrote {out_all.name} with {len(all_rows)} shot rows; "
        f"{out_unique.name} with {len(unique_rows)} unique positions"
    )


def main() -> None:
    lidar_cache = {}

    for line in LINES:
        process_stations(line)
        process_shots(line, lidar_cache)


if __name__ == "__main__":
    main()