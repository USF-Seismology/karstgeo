#!/usr/bin/env python3
"""
85_rewrite_sds_station_codes_by_position_v3.py

Create a derived SDS archive in which station codes are 5-character
position-in-centimetres codes.

Examples:
    122.00 m -> 12200
      4.00 m -> 00400
     40.50 m -> 04050

This version:
- uses Combined_By_Window for T1/N1,N2,N3 mappings;
- preserves both DP* 500 Hz and GP* 1000 Hz channels;
- allows an extra manual mapping CSV for stations not represented in the workbook;
- includes the known T3 note that station ending 05764 was at 40.5 m, if enabled;
- writes a mapping CSV and a full rewrite report.

Manual mapping CSV format
-------------------------
Optional CSV columns:

    network,location,old_station,x_m

Example:

    T3,N4,05764,40.5
    T1,N1,12928,123.4

Use this for T3/N4 once we identify the missing serial-position mapping.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import read


DEFAULT_WINDOW_TO_LOCATIONS = {
    1: ["N1", "N3"],
    2: ["N2"],
    3: ["N3"],
    4: ["N4"],
}


# From T3_Nodal_Geometry notes in the uploaded workbook:
# "node serial number ending in 764 located at 40.5 m"
# The SDS station code uses last 5 digits, so 05764 -> 40.5 m.
KNOWN_EXTRA_MAPPINGS = [
    {"metadata_sheet": "known_extra", "window_id": "", "network": "T3", "location": "N4",
     "old_station": "05764", "serial": "", "x_m": 40.5, "new_station": "04050",
     "mapping_note": "T3 note: serial ending 764 located at 40.5 m"},
]


def clean_station_like_value(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "nat"}:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def station_variants(s: object) -> set[str]:
    base = clean_station_like_value(s)
    if not base:
        return set()

    variants = {base}

    if re.fullmatch(r"\d+", base):
        nozero = str(int(base))
        variants.add(nozero)
        for width in [4, 5, 6, 9]:
            variants.add(nozero.zfill(width))

        # SmartSolo serial -> SDS station code is usually last 5 digits.
        if len(nozero) > 5:
            last5 = nozero[-5:]
            variants.add(last5)
            variants.add(str(int(last5)))
            variants.add(last5.zfill(5))

    return variants


def station_code_from_position_cm(x_m: float) -> str:
    cm = int(round(float(x_m) * 100.0))
    if cm < 0 or cm > 99999:
        raise ValueError(f"Position {x_m} m -> {cm} cm outside 00000..99999")
    return f"{cm:05d}"


def parse_window_location_map(text: Optional[str]) -> dict[int, list[str]]:
    if not text:
        return DEFAULT_WINDOW_TO_LOCATIONS.copy()

    out = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        out[int(k)] = [x.strip() for x in v.split(",") if x.strip()]
    return out


def load_combined_by_window_mapping(
    metadata_xlsx: Path,
    window_to_locations: dict[int, list[str]],
) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx, sheet_name="Combined_By_Window")

    required = [
        "window_id",
        "line_or_transect",
        "normalized_serial_number",
        "position_m",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Combined_By_Window is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []
    for _, r in df.iterrows():
        serial = clean_station_like_value(r["normalized_serial_number"])
        if not serial:
            continue

        x_m = pd.to_numeric(r["position_m"], errors="coerce")
        if pd.isna(x_m):
            continue

        net = str(r["line_or_transect"]).strip()
        if not net or net.lower() == "nan":
            continue

        try:
            window_id = int(r["window_id"])
        except Exception:
            continue

        locs = window_to_locations.get(window_id, [])
        for loc in locs:
            rows.append({
                "metadata_sheet": "Combined_By_Window",
                "window_id": window_id,
                "network": net,
                "location": loc,
                "old_station": "",
                "serial": serial,
                "x_m": float(x_m),
                "new_station": station_code_from_position_cm(float(x_m)),
                "mapping_note": "",
            })

    return standardize_mapping_df(pd.DataFrame(rows))


def load_manual_mapping_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = ["network", "location", "old_station", "x_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manual mapping CSV missing columns: {missing}")

    rows = []
    for _, r in df.iterrows():
        x_m = pd.to_numeric(r["x_m"], errors="coerce")
        if pd.isna(x_m):
            continue
        rows.append({
            "metadata_sheet": "manual_csv",
            "window_id": "",
            "network": str(r["network"]).strip(),
            "location": str(r["location"]).strip(),
            "old_station": clean_station_like_value(r["old_station"]),
            "serial": clean_station_like_value(r.get("serial", "")),
            "x_m": float(x_m),
            "new_station": station_code_from_position_cm(float(x_m)),
            "mapping_note": str(r.get("mapping_note", "")),
        })

    return standardize_mapping_df(pd.DataFrame(rows))


def standardize_mapping_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "metadata_sheet", "window_id", "network", "location", "old_station",
        "serial", "x_m", "new_station", "mapping_note"
    ]
    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=cols)
    else:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        out = out[cols]

    if len(out):
        out["network"] = out["network"].astype(str).str.strip()
        out["location"] = out["location"].astype(str).str.strip()
        out["old_station"] = out["old_station"].apply(clean_station_like_value)
        out["serial"] = out["serial"].apply(clean_station_like_value)
        out["x_m"] = pd.to_numeric(out["x_m"], errors="coerce")
        out = out.dropna(subset=["x_m"]).copy()
        out["new_station"] = out["x_m"].apply(station_code_from_position_cm)
        out["old_station_clean"] = out["old_station"].apply(clean_station_like_value)
        out["serial_clean"] = out["serial"].apply(clean_station_like_value)
        out = out.drop_duplicates(
            subset=["network", "location", "old_station_clean", "serial_clean", "new_station"],
            keep="first"
        ).reset_index(drop=True)
    else:
        out["old_station_clean"] = []
        out["serial_clean"] = []

    return out


def build_mapping_df(
    metadata_xlsx: Path,
    window_to_locations: dict[int, list[str]],
    manual_mapping_csv: Optional[Path],
    include_known_extras: bool,
) -> pd.DataFrame:
    parts = [load_combined_by_window_mapping(metadata_xlsx, window_to_locations)]

    if include_known_extras:
        parts.append(standardize_mapping_df(pd.DataFrame(KNOWN_EXTRA_MAPPINGS)))

    if manual_mapping_csv is not None:
        parts.append(load_manual_mapping_csv(manual_mapping_csv))

    out = pd.concat([p for p in parts if p is not None and len(p) > 0], ignore_index=True)
    return standardize_mapping_df(out)


def build_mapping_dict(mapping_df: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    mapping = {}
    conflicts = []

    for _, row in mapping_df.iterrows():
        net = str(row.get("network", "")).strip()
        loc = str(row.get("location", "")).strip()
        new_sta = str(row["new_station"]).strip()

        old_values = set()
        old_values |= station_variants(row.get("old_station_clean", ""))
        old_values |= station_variants(row.get("serial_clean", ""))

        for old in old_values:
            key = (net, loc, old)
            if key in mapping and mapping[key] != new_sta:
                conflicts.append((key, mapping[key], new_sta))
            mapping[key] = new_sta

    if conflicts:
        print("WARNING: mapping conflicts detected. First 20:")
        for c in conflicts[:20]:
            print("  ", c)

    return mapping


def lookup_new_station(mapping: dict[tuple[str, str, str], str], net: str, loc: str, sta: str) -> Optional[str]:
    for sv in station_variants(sta):
        key = (net, loc, sv)
        if key in mapping:
            return mapping[key]
    return None


@dataclass
class SdsFileInfo:
    path: Path
    year: str
    network: str
    station: str
    channel_dir: str
    filename: str
    location: str
    channel: str
    dtype: str
    jday: str


def parse_sds_path(path: Path, src_sds: Path) -> Optional[SdsFileInfo]:
    try:
        rel = path.relative_to(src_sds)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 5:
        return None

    fname = parts[-1]
    if fname.startswith("._"):
        return None

    bits = fname.split(".")
    if len(bits) < 7:
        return None

    fnet, fsta, floc, fchan, fdtype, fyear, fjday = bits[:7]

    return SdsFileInfo(
        path=path,
        year=fyear,
        network=fnet,
        station=fsta,
        channel_dir=parts[3],
        filename=fname,
        location=floc,
        channel=fchan,
        dtype=fdtype,
        jday=fjday,
    )


def iter_sds_files(src_sds: Path):
    for p in sorted(src_sds.rglob("*")):
        if not p.is_file():
            continue
        info = parse_sds_path(p, src_sds)
        if info is not None:
            yield info


def make_output_sds_path(info: SdsFileInfo, dst_sds: Path, new_station: str) -> Path:
    chan_dir = f"{info.channel}.{info.dtype}"
    fname = ".".join([
        info.network,
        new_station,
        info.location,
        info.channel,
        info.dtype,
        info.year,
        info.jday,
    ])
    return dst_sds / info.year / info.network / new_station / chan_dir / fname


def rewrite_mseed_station(src: Path, dst: Path, new_station: str, overwrite: bool = False) -> dict:
    if dst.exists() and not overwrite:
        return {"status": "exists", "src": str(src), "dst": str(dst)}

    dst.parent.mkdir(parents=True, exist_ok=True)

    st = read(str(src))
    for tr in st:
        tr.stats.station = new_station

    st.write(str(dst), format="MSEED")
    return {"status": "written", "src": str(src), "dst": str(dst), "ntraces": len(st)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rewrite SDS station codes to 5-character position-cm station codes."
    )
    parser.add_argument("--src-sds", required=True, type=Path)
    parser.add_argument("--dst-sds", required=True, type=Path)
    parser.add_argument("--metadata-xlsx", required=True, type=Path)
    parser.add_argument(
        "--window-location-map",
        default=None,
        help="Override geometry-window to SDS-location mapping, e.g. '1:N1,N3;2:N2;4:N4'",
    )
    parser.add_argument(
        "--manual-mapping-csv",
        default=None,
        type=Path,
        help="Optional CSV with columns network,location,old_station,x_m for unmapped stations.",
    )
    parser.add_argument(
        "--no-known-extras",
        action="store_true",
        help="Disable embedded known extra mappings, currently T3/N4 station 05764 -> 04050.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.src_sds.exists():
        raise FileNotFoundError(args.src_sds)
    if not args.metadata_xlsx.exists():
        raise FileNotFoundError(args.metadata_xlsx)

    window_to_locations = parse_window_location_map(args.window_location_map)

    print("Loading metadata mapping...")
    mapping_df = build_mapping_df(
        metadata_xlsx=args.metadata_xlsx,
        window_to_locations=window_to_locations,
        manual_mapping_csv=args.manual_mapping_csv,
        include_known_extras=not args.no_known_extras,
    )
    mapping = build_mapping_dict(mapping_df)

    print(f"Loaded {len(mapping_df)} metadata mapping rows")
    print(f"Built {len(mapping)} lookup keys")
    print("\nMapping preview:")
    if len(mapping_df):
        print(mapping_df.head(35).to_string(index=False))
    else:
        print(mapping_df)

    args.dst_sds.mkdir(parents=True, exist_ok=True)

    if args.mapping_csv is None:
        args.mapping_csv = args.dst_sds / "station_position_mapping.csv"
    args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(args.mapping_csv, index=False)
    print(f"\nWrote mapping CSV: {args.mapping_csv}")

    rows = []
    n_seen = n_matched = n_written = n_exists = n_unmatched = n_failed = 0

    for info in iter_sds_files(args.src_sds):
        n_seen += 1
        if args.max_files is not None and n_seen > args.max_files:
            break

        new_sta = lookup_new_station(mapping, info.network, info.location, info.station)

        row = {
            "src": str(info.path),
            "network": info.network,
            "old_station": info.station,
            "location": info.location,
            "channel": info.channel,
            "year": info.year,
            "jday": info.jday,
            "new_station": new_sta or "",
            "dst": "",
            "status": "",
            "error": "",
        }

        if new_sta is None:
            row["status"] = "unmatched"
            n_unmatched += 1
            rows.append(row)
            continue

        n_matched += 1
        dst = make_output_sds_path(info, args.dst_sds, new_sta)
        row["dst"] = str(dst)

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        try:
            result = rewrite_mseed_station(info.path, dst, new_sta, overwrite=args.overwrite)
            row["status"] = result["status"]
            if result["status"] == "written":
                n_written += 1
            elif result["status"] == "exists":
                n_exists += 1
        except Exception as e:
            row["status"] = "failed"
            row["error"] = repr(e)
            n_failed += 1

        rows.append(row)

        if n_seen % 100 == 0:
            print(
                f"Processed {n_seen} files: "
                f"matched={n_matched}, written={n_written}, "
                f"unmatched={n_unmatched}, failed={n_failed}"
            )

    report = pd.DataFrame(rows)
    report_csv = args.report_csv or (args.dst_sds / "rewrite_sds_station_codes_report.csv")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)

    summary = {
        "src_sds": str(args.src_sds),
        "dst_sds": str(args.dst_sds),
        "metadata_xlsx": str(args.metadata_xlsx),
        "manual_mapping_csv": str(args.manual_mapping_csv) if args.manual_mapping_csv else "",
        "window_to_locations": window_to_locations,
        "include_known_extras": not args.no_known_extras,
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "n_seen": int(n_seen),
        "n_matched": int(n_matched),
        "n_written": int(n_written),
        "n_exists": int(n_exists),
        "n_unmatched": int(n_unmatched),
        "n_failed": int(n_failed),
        "report_csv": str(report_csv),
        "mapping_csv": str(args.mapping_csv),
    }

    summary_json = report_csv.with_suffix(".summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary")
    print(json.dumps(summary, indent=2))

    if n_unmatched:
        print("\nUnmatched examples:")
        cols = ["src", "network", "old_station", "location", "channel", "year", "jday", "status"]
        print(report.loc[report["status"] == "unmatched", cols].head(30).to_string(index=False))

        unmatched_unique = (
            report.loc[report["status"] == "unmatched", ["network", "location", "old_station"]]
            .drop_duplicates()
            .sort_values(["network", "location", "old_station"])
        )
        unmatched_csv = report_csv.with_name("unmatched_unique_stations.csv")
        unmatched_unique.to_csv(unmatched_csv, index=False)
        print(f"\nWrote unique unmatched stations: {unmatched_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
