#!/usr/bin/env python
"""
make_nodal_stationxml.py

Create StationXML inventories for SmartSolo nodal deployments from:
  1. a nodal metadata Excel workbook with serial numbers and coordinates
  2. one SmartSolo StationXML file containing the instrument response

This script assumes:
  - network codes identify transects: T1, T3
  - location codes identify nodal deployments: N1, N2, N3, N4
  - station codes are the last 5 digits of the SmartSolo serial number
  - SmartSolo channel codes are preserved:
        500 Hz  -> DPE, DPN, DPZ
        1000 Hz -> GPE, GPN, GPZ
  - the instrument response is the same for all SmartSolo nodes/components
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


# ----------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------

WORKBOOK = Path("/Users/glennthompson/Library/CloudStorage/Box-Box/thompsong/2026KarstGeophysicsDEP/04_FieldData/glenn_smartsolo_nodal_metadata_with_estimated_coords.xlsx")
RESPONSE_XML = Path("/Volumes/tachyon/LBSSP_DATA/SOLODATA/prospect_geokarst/Karst_Geophysics/Transect1_Nodal2_500Hz/FDSN Information/FDSN_Information_453012795_1.xml")
OUTDIR = Path("/Volumes/tachyon/LBSSP_DATA/nodal_sds")

# One StationXML file will be created for each entry below.
# geometry_sheet is the sheet containing serial numbers and coordinates.
# N3 reuses the original T1 geometry, because you moved the nodes back.
DEPLOYMENTS = [
    {
        "name": "Transect1_Nodal1_500Hz",
        "network": "T1",
        "location": "N1",
        "sample_rate": 500.0,
        "channel_prefix": "DP",
        "geometry_sheet": "T1_Nodal_Geometry_Orig",
        "starttime": "2026-05-16T00:00:00",
        "endtime": "2026-05-17T16:00:00",
    },
    {
        "name": "Transect1_Nodal1_1000Hz",
        "network": "T1",
        "location": "N1",
        "sample_rate": 1000.0,
        "channel_prefix": "GP",
        "geometry_sheet": "T1_Nodal_Geometry_Orig",
        "starttime": "2026-05-16T00:00:00",
        "endtime": "2026-05-17T16:00:00",
    },
    {
        "name": "Transect1_Nodal2_500Hz",
        "network": "T1",
        "location": "N2",
        "sample_rate": 500.0,
        "channel_prefix": "DP",
        "geometry_sheet": "T1_Nodal_Geometry_DenseConfig",
        "starttime": "2026-05-17T16:00:00",
        "endtime": "2026-05-19T13:20:00",
    },
    {
        "name": "Transect1_Nodal2_1000Hz",
        "network": "T1",
        "location": "N2",
        "sample_rate": 1000.0,
        "channel_prefix": "GP",
        "geometry_sheet": "T1_Nodal_Geometry_DenseConfig",
        "starttime": "2026-05-17T16:00:00",
        "endtime": "2026-05-19T13:20:00",
    },
    {
        "name": "Transect1_Nodal3_500Hz",
        "network": "T1",
        "location": "N3",
        "sample_rate": 500.0,
        "channel_prefix": "DP",
        "geometry_sheet": "T1_Nodal_Geometry_Orig",
        "starttime": "2026-05-19T13:20:00",
        "endtime": "2026-05-19T15:00:00",
    },
    {
        "name": "Transect1_Nodal3_1000Hz",
        "network": "T1",
        "location": "N3",
        "sample_rate": 1000.0,
        "channel_prefix": "GP",
        "geometry_sheet": "T1_Nodal_Geometry_Orig",
        "starttime": "2026-05-19T13:20:00",
        "endtime": "2026-05-19T15:00:00",
    },
    {
        "name": "TransectE_Nodal4_500Hz",
        "network": "T3",
        "location": "N4",
        "sample_rate": 500.0,
        "channel_prefix": "DP",
        "geometry_sheet": "T3_Nodal_Geometry",
        "starttime": "2026-05-19T15:00:00",
        "endtime": "2026-05-19T18:25:00",
    },
    {
        "name": "TransectE_Nodal4_1000Hz",
        "network": "T3",
        "location": "N4",
        "sample_rate": 1000.0,
        "channel_prefix": "GP",
        "geometry_sheet": "T3_Nodal_Geometry",
        "starttime": "2026-05-19T15:00:00",
        "endtime": "2026-05-19T18:25:00",
    },
]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def station_code_from_serial(serial) -> Optional[str]:
    """Return the 5-digit SmartSolo station code from a serial number."""
    if pd.isna(serial):
        return None
    s = str(serial).strip()
    # Excel sometimes stores serials as floats, e.g. 453020358.0
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) < 5:
        return None
    return s[-5:]


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column from a list of possible names."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower:
            return lower[key]
    return None


def load_geometry_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load a geometry sheet.

    The T1 sheets in Glenn's workbook have a title row and then a header row,
    so this routine first tries header=1 and falls back to header=0.
    """
    for header_row in (1, 0):
        df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
        serial_col = find_column(df, [
            "normalized_serial_number",
            "serial_number",
            "raw_serial_number",
        ])
        lat_col = find_column(df, ["latitude", "lat"])
        lon_col = find_column(df, ["longitude", "lon", "long"])
        if serial_col and lat_col and lon_col:
            return df

    raise ValueError(
        f"Could not find serial_number/latitude/longitude columns in sheet {sheet_name!r}"
    )


def get_template_response(response_xml: Path):
    """
    Read one SmartSolo StationXML file and return a deep-copyable response.

    The response is taken from the first channel found in the file.
    """
    inv = read_inventory(str(response_xml))
    for net in inv:
        for sta in net:
            for cha in sta:
                if cha.response is not None:
                    return deepcopy(cha.response)
    raise ValueError(f"No channel response found in {response_xml}")


def make_channel(
    code: str,
    location_code: str,
    latitude: float,
    longitude: float,
    elevation: float,
    sample_rate: float,
    start_date: UTCDateTime,
    end_date: Optional[UTCDateTime],
    response,
) -> Channel:
    """Create one SEED channel with orientation and response."""
    orientation = code[-1].upper()

    azimuth = {
        "N": 0.0,
        "E": 90.0,
        "Z": 0.0,
    }[orientation]

    dip = {
        "N": 0.0,
        "E": 0.0,
        "Z": -90.0,
    }[orientation]

    ch = Channel(
        code=code,
        location_code=location_code,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0.0,
        azimuth=azimuth,
        dip=dip,
        sample_rate=sample_rate,
        start_date=start_date,
        end_date=end_date,
    )
    ch.response = deepcopy(response)
    return ch


def build_inventory_for_deployment(deployment: dict, template_response) -> Inventory:
    """Create an ObsPy Inventory for one deployment."""
    df = load_geometry_sheet(WORKBOOK, deployment["geometry_sheet"])

    serial_col = find_column(df, [
        "normalized_serial_number",
        "serial_number",
        "raw_serial_number",
    ])
    lat_col = find_column(df, ["latitude", "lat"])
    lon_col = find_column(df, ["longitude", "lon", "long"])
    elev_col = find_column(df, ["elevation", "elevation_m", "altitude", "altitude_m"])
    pos_col = find_column(df, ["adopted_position_m", "node_position_m", "position_m"])

    start = UTCDateTime(deployment["starttime"])
    end = UTCDateTime(deployment["endtime"]) if deployment.get("endtime") else None

    net = Network(
        code=deployment["network"],
        description=f"{deployment['name']} SmartSolo nodal deployment",
        start_date=start,
        end_date=end,
        stations=[],
    )

    channel_codes = [
        f"{deployment['channel_prefix']}E",
        f"{deployment['channel_prefix']}N",
        f"{deployment['channel_prefix']}Z",
    ]

    seen = set()

    for _, row in df.iterrows():
        sta_code = station_code_from_serial(row[serial_col])
        if not sta_code:
            continue

        if sta_code in seen:
            continue
        seen.add(sta_code)

        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
            print(f"Skipping {deployment['name']} station {sta_code}: missing lat/lon")
            continue

        latitude = float(row[lat_col])
        longitude = float(row[lon_col])
        elevation = 0.0 if elev_col is None or pd.isna(row[elev_col]) else float(row[elev_col])

        site_name = sta_code
        if pos_col and not pd.isna(row[pos_col]):
            site_name = f"{sta_code} x={float(row[pos_col]):.3f} m"

        sta = Station(
            code=sta_code,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            creation_date=start,
            start_date=start,
            end_date=end,
            site=Site(name=site_name),
        )

        for cha_code in channel_codes:
            sta.channels.append(
                make_channel(
                    code=cha_code,
                    location_code=deployment["location"],
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    sample_rate=deployment["sample_rate"],
                    start_date=start,
                    end_date=end,
                    response=template_response,
                )
            )

        net.stations.append(sta)

    inv = Inventory(
        networks=[net],
        source="USF/FLO-VO KarstGeo SmartSolo nodal metadata generated with ObsPy",
    )
    return inv


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    template_response = get_template_response(RESPONSE_XML)

    for dep in DEPLOYMENTS:
        inv = build_inventory_for_deployment(dep, template_response)
        outfile = OUTDIR / f"{dep['name']}_stationxml.xml"
        inv.write(str(outfile), format="STATIONXML")
        nstations = sum(len(net.stations) for net in inv)
        nchannels = sum(len(sta.channels) for net in inv for sta in net)
        print(f"Wrote {outfile}  ({nstations} stations, {nchannels} channels)")


if __name__ == "__main__":
    main()
