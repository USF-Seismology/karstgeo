#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv


INPUT_DIR = Path("inputs/receivers")
OUTPUT_DIR = INPUT_DIR

LINES = ["T1", "T2", "T3", "T4"]


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


def make_new_station_name(row: dict) -> str:
    prefix = instrument_prefix(row["old_sta"], row["source_file"])
    cm = int(round(row["x"] * 100))
    return f"{prefix}{cm:06d}"


def format_station_line(sta: str, net: str, x: float, z: float, c5: float, c6: float) -> str:
    return f"{sta:<10s}{net:<6s}{x:16.7f}{z:16.7f} {c5:.1f} {c6:.1f}"


def find_input_files(line_name: str) -> list[Path]:
    patterns = [
        f"STATIONS_{line_name}_*receivers*.dat",
        f"STATIONS_{line_name}_*geometry*.dat",
    ]

    files = []
    for pattern in patterns:
        files.extend(sorted(INPUT_DIR.glob(pattern)))

    return sorted(set(files))


def process_line(line_name: str) -> None:
    input_files = find_input_files(line_name)

    if not input_files:
        print(f"{line_name}: no geometry files found; skipping.")
        return

    raw_rows = []
    for path in input_files:
        raw_rows.extend(read_station_rows(path))

    if not raw_rows:
        print(f"{line_name}: files found but no valid station rows; skipping.")
        return

    # De-duplicate by exact x-position, keeping first occurrence.
    seen = {}
    for row in raw_rows:
        if row["x"] not in seen:
            seen[row["x"]] = row

    rows = [seen[x] for x in sorted(seen)]

    for row in rows:
        row["new_sta"] = make_new_station_name(row)
        row["net"] = line_name

    name_counts = defaultdict(int)
    for row in rows:
        name_counts[row["new_sta"]] += 1

    collisions = {name: n for name, n in name_counts.items() if n > 1}
    if collisions:
        raise ValueError(f"{line_name}: station-name collisions detected: {collisions}")

    out_renamed = OUTPUT_DIR / f"STATIONS_{line_name}_ALL_SORTED_UNIQUE_FORMATTED_RENAMED"
    out_original = OUTPUT_DIR / f"STATIONS_{line_name}_ALL_SORTED_UNIQUE_FORMATTED_ORIGINAL_NAMES"
    out_map = OUTPUT_DIR / f"STATIONS_{line_name}_ALL_RENAME_MAP.csv"

    original_lines = [
        format_station_line(row["old_sta"], row["net"], row["x"], row["z"], row["c5"], row["c6"])
        for row in rows
    ]

    renamed_lines = [
        format_station_line(row["new_sta"], row["net"], row["x"], row["z"], row["c5"], row["c6"])
        for row in rows
    ]

    out_original.write_text("\n".join(original_lines) + "\n")
    out_renamed.write_text("\n".join(renamed_lines) + "\n")

    with out_map.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "old_station",
                "new_station",
                "network",
                "x_m",
                "z_m",
                "col5",
                "col6",
                "source_file",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "old_station": row["old_sta"],
                    "new_station": row["new_sta"],
                    "network": row["net"],
                    "x_m": f"{row['x']:.7f}",
                    "z_m": f"{row['z']:.7f}",
                    "col5": f"{row['c5']:.1f}",
                    "col6": f"{row['c6']:.1f}",
                    "source_file": row["source_file"],
                }
            )

    print(f"\n{line_name}")
    print(f"  Files used:")
    for p in input_files:
        print(f"    {p.name}")
    print(f"  Raw rows read:       {len(raw_rows)}")
    print(f"  Unique x positions: {len(rows)}")
    print(f"  Wrote: {out_renamed.name}")
    print(f"  Wrote: {out_original.name}")
    print(f"  Wrote: {out_map.name}")


def main() -> None:
    for line_name in LINES:
        process_line(line_name)


if __name__ == "__main__":
    main()