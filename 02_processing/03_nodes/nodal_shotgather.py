"""
nodal_shotgather.py

Reusable functions for converting continuous nodal SDS data into detected
active-source event windows / shot gathers.

This module generalizes the GeoPark segmentation workflow to the Karst
Geophysics nodal SDS archive.

Typical workflow
----------------
1. Read one deployment from SDS with EnhancedSDSClient.
2. Attach station geometry from a metadata table.
3. Filter for network coincidence detection.
4. Detect impulsive events across adjacent/multiple nodes.
5. Filter/pick each detected event by station/component.
6. Build station consensus first-break picks.
7. Trim fixed-length event windows from earliest station pick.
8. Save event MiniSEED files, QC figures, and CSV tables.

The module deliberately keeps raw data, detection data, and picking data
separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
from obspy import Stream, UTCDateTime
from obspy.signal.trigger import pk_baer, aic_simple, ar_pick

from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.processing.detection import (
    run_coincidence_trigger_dataframe,
    plot_stream_with_event_markers,
    filter_events_df,
)


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    freqmin: float = 5.0
    freqmax: float = 200.0
    corners: int = 4
    zerophase: bool = True
    sta_seconds: float = 0.02
    lta_seconds: float = 0.30
    threshold_on: float = 4.0
    threshold_off: float = 1.5
    min_channels: int = 12
    min_snr: float = 10.0
    pretrigger_seconds: float = 0.25
    posttrigger_seconds: float = 0.25
    min_event_peak_amplitude: float = 0.0


@dataclass
class PickingConfig:
    freqmin: float = 10.0
    freqmax: float = 150.0
    corners: int = 4
    zerophase: bool = False
    pick_tolerance_s: float = 0.02
    min_votes: int = 2
    min_weight: Optional[float] = None
    include_ar_s: bool = False
    baer_weight: float = 1.0 / 3.0
    mute_seconds: float = 0.20
    event_pre_pick_s: float = 0.10
    event_length_s: float = 0.50


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def safe_max_abs(st: Stream) -> float:
    """Return maximum absolute amplitude across a Stream."""
    if len(st) == 0:
        return np.nan

    vals = []
    for tr in st:
        if tr.stats.npts:
            data = np.asarray(tr.data, dtype=float)
            if np.isfinite(data).any():
                vals.append(np.nanmax(np.abs(data)))

    return max(vals) if vals else np.nan


def normalize_trace_data(y) -> np.ndarray:
    """Demean and normalize a 1-D array for plotting/picking diagnostics."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y

    y = y - np.nanmedian(y)
    ymax = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 0.0

    if ymax > 0:
        y = y / ymax

    return y


def seed_id_parts(seed_id: str) -> Tuple[str, str, str, str]:
    """Split NET.STA.LOC.CHA, padding with empty strings if needed."""
    parts = str(seed_id).split(".")
    if len(parts) == 4:
        return tuple(parts)
    return "", "", "", ""


def ensure_dir(path: Path | str) -> Path:
    """Create and return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def component_selector(component: str) -> str:
    """Return wildcard channel selector for a component, e.g. Z -> *Z."""
    component = component.upper()
    if len(component) == 1:
        return f"*{component}"
    return component


# -----------------------------------------------------------------------------
# SDS reading and geometry
# -----------------------------------------------------------------------------

def read_deployment_from_sds(
    sds_root: Path | str,
    network: str,
    location: str,
    starttime: UTCDateTime | str,
    endtime: UTCDateTime | str,
    station: str = "*",
    channel: str = "*",
    merge: bool = True,
    attach_response: bool = False,
    inventory=None,
    verbose: bool = True,
) -> Stream:
    """
    Read one nodal deployment from an SDS archive.

    Parameters
    ----------
    sds_root
        Root SDS directory.
    network
        Network code, e.g. T1 or T3.
    location
        Deployment/location code, e.g. N1, N2, N3, N4.
    starttime, endtime
        UTC time range.
    station, channel
        SDS selectors. Use channel='DP*' for 500 Hz or 'GP*' for 1000 Hz.
    merge
        Merge traces after reading.
    attach_response
        If True and inventory is provided, attach response to traces.
    inventory
        ObsPy Inventory.
    """
    client = EnhancedSDSClient(Path(sds_root))
    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)

    st = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
    )

    if merge and len(st):
        st.merge(method=1, fill_value="interpolate")

    if attach_response and inventory is not None:
        st.attach_response(inventory)

    if verbose:
        print(st)
        print("Unique IDs:", len(sorted({tr.id for tr in st})))

    return st


def load_station_geometry_table(
    path: Path | str,
    sheet_name: Optional[str] = None,
    station_col: str = "station",
    serial_col: Optional[str] = None,
    x_col: str = "adopted_position_m",
    y_col: Optional[str] = None,
    elevation_col: Optional[str] = None,
    latitude_col: Optional[str] = "latitude",
    longitude_col: Optional[str] = "longitude",
) -> pd.DataFrame:
    """
    Load a station geometry table from CSV or Excel.

    Expected columns can be mapped using arguments. The returned table contains
    standard columns where available:
        station, serial, x_m, y_m, elevation_m, latitude, longitude
    """
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        if sheet_name is None:
            raise ValueError("sheet_name is required for Excel geometry files")
        # Try header row 1 first, because Glenn's metadata sheets often use a
        # title row followed by real headers.
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=1)
        except Exception:
            df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    out = pd.DataFrame()

    if station_col in df.columns:
        out["station"] = df[station_col].astype(str).str.strip()
    elif serial_col and serial_col in df.columns:
        out["station"] = df[serial_col].apply(lambda x: str(int(float(x)))[-5:] if pd.notna(x) else None)
    else:
        # Try common serial column names.
        for c in ["normalized_serial_number", "serial_number", "raw_serial_number"]:
            if c in df.columns:
                out["station"] = df[c].apply(lambda x: str(int(float(x)))[-5:] if pd.notna(x) else None)
                serial_col = c
                break

    if "station" not in out.columns:
        raise ValueError("Could not identify station or serial column")

    if serial_col and serial_col in df.columns:
        out["serial"] = df[serial_col].astype(str)

    if x_col in df.columns:
        out["x_m"] = pd.to_numeric(df[x_col], errors="coerce")
    else:
        out["x_m"] = np.nan

    if y_col and y_col in df.columns:
        out["y_m"] = pd.to_numeric(df[y_col], errors="coerce")
    else:
        out["y_m"] = 0.0

    if elevation_col and elevation_col in df.columns:
        out["elevation_m"] = pd.to_numeric(df[elevation_col], errors="coerce")
    else:
        out["elevation_m"] = 0.0

    if latitude_col and latitude_col in df.columns:
        out["latitude"] = pd.to_numeric(df[latitude_col], errors="coerce")

    if longitude_col and longitude_col in df.columns:
        out["longitude"] = pd.to_numeric(df[longitude_col], errors="coerce")

    out = out.dropna(subset=["station"]).copy()
    out["station"] = out["station"].astype(str).str.zfill(5)

    return out


def geometry_dataframe_to_dict(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert standardized geometry dataframe to dict keyed by station."""
    geometry = {}

    for _, row in df.iterrows():
        station = str(row["station"]).zfill(5)
        geometry[station] = {
            "line_x_m": float(row.get("x_m", np.nan)),
            "line_y_m": float(row.get("y_m", 0.0)),
            "elevation_m": float(row.get("elevation_m", 0.0)),
        }

        if "serial" in row:
            geometry[station]["serial"] = row["serial"]
        if "latitude" in row and pd.notna(row["latitude"]):
            geometry[station]["latitude"] = float(row["latitude"])
        if "longitude" in row and pd.notna(row["longitude"]):
            geometry[station]["longitude"] = float(row["longitude"])

    return geometry


def attach_station_geometry(st: Stream, geometry: Dict[str, dict]) -> Stream:
    """
    Attach coordinates to traces using station geometry.

    The function uses `Trace.stats.coordinates`, which is commonly recognized
    by downstream ObsPy workflows.
    """
    for tr in st:
        g = geometry.get(str(tr.stats.station).zfill(5))
        if g is None:
            continue

        tr.stats.coordinates = {
            "x": float(g.get("line_x_m", np.nan)),
            "y": float(g.get("line_y_m", 0.0)),
            "elevation": float(g.get("elevation_m", 0.0)),
        }

        if "latitude" in g:
            tr.stats.coordinates["latitude"] = float(g["latitude"])
        if "longitude" in g:
            tr.stats.coordinates["longitude"] = float(g["longitude"])

    return st


def save_station_geometry_csv(geometry: Dict[str, dict], outfile: Path | str) -> pd.DataFrame:
    """Save station geometry dictionary to CSV and return dataframe."""
    rows = []
    for station, g in sorted(geometry.items()):
        row = {"station": station}
        row.update(g)
        rows.append(row)

    df = pd.DataFrame(rows)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    return df


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def preprocess_for_detection(st: Stream, cfg: DetectionConfig) -> Stream:
    """Return a filtered copy of a Stream for network event detection."""
    st_detect = st.copy()
    st_detect.detrend("demean")
    st_detect.detrend("linear")
    st_detect.taper(max_percentage=0.001)
    st_detect.filter(
        "bandpass",
        freqmin=cfg.freqmin,
        freqmax=cfg.freqmax,
        corners=cfg.corners,
        zerophase=cfg.zerophase,
    )
    return st_detect


def detect_network_events(
    st_detect: Stream,
    cfg: DetectionConfig,
    write_mseed: bool = False,
    make_plots: bool = False,
) -> pd.DataFrame:
    """Run coincidence trigger and SNR filter, returning an event dataframe."""
    df_events = run_coincidence_trigger_dataframe(
        st_detect,
        trigger_type="recstalta",
        sta_seconds=cfg.sta_seconds,
        lta_seconds=cfg.lta_seconds,
        threshold_on=cfg.threshold_on,
        threshold_off=cfg.threshold_off,
        min_channels=cfg.min_channels,
        pretrigger_seconds=cfg.pretrigger_seconds,
        posttrigger_seconds=cfg.posttrigger_seconds,
        write_mseed=write_mseed,
        make_plots=make_plots,
    )

    if len(df_events):
        df_events = filter_events_df(df_events, min_snr=cfg.min_snr)

    return df_events


# -----------------------------------------------------------------------------
# Picking
# -----------------------------------------------------------------------------

def preprocess_for_picking(st: Stream, cfg: PickingConfig) -> Stream:
    """Return a filtered copy of a Stream for first-break picking."""
    st_pick = st.copy()
    st_pick.detrend("simple")
    st_pick.taper(max_percentage=0.001)
    st_pick.filter(
        "bandpass",
        freqmin=cfg.freqmin,
        freqmax=cfg.freqmax,
        corners=cfg.corners,
        zerophase=cfg.zerophase,
    )
    return st_pick


def pick_baer_aic_on_trace(tr) -> dict:
    """
    Apply AIC and Baer pickers to one Trace.

    Returns
    -------
    dict
        Pick status, absolute pick times, and sample indices.
    """
    sr = float(tr.stats.sampling_rate)
    y = normalize_trace_data(tr.data)

    out = {
        "baer_time": None,
        "baer_sample": None,
        "baer_ok": False,
        "baer_error": None,
        "aic_time": None,
        "aic_sample": None,
        "aic_ok": False,
        "aic_error": None,
    }

    try:
        aic = aic_simple(y)
        iaic = int(np.argmin(aic))
        out["aic_sample"] = iaic
        out["aic_time"] = tr.stats.starttime + iaic / sr
        out["aic_ok"] = True
    except Exception as e:
        out["aic_error"] = repr(e)

    try:
        p_sample, phase_info = pk_baer(
            y,
            sr,
            max(3, int(0.006 * sr)),
            max(8, int(0.025 * sr)),
            3.0,
            6.0,
            max(5, int(0.010 * sr)),
            max(5, int(0.015 * sr)),
        )
        if p_sample is not None and 0 <= p_sample < len(y):
            out["baer_sample"] = int(p_sample)
            out["baer_time"] = tr.stats.starttime + p_sample / sr
            out["baer_ok"] = True
    except Exception as e:
        out["baer_error"] = repr(e)

    return out


def try_ar_pick_short_station(tr_z, tr_n, tr_e, f1=10.0, f2=150.0):
    """
    Apply ObsPy ar_pick to a 3-component station window.

    Returns AR P and S pick times in seconds relative to trace start.
    """
    sr = float(tr_z.stats.sampling_rate)
    npts = min(tr_z.stats.npts, tr_n.stats.npts, tr_e.stats.npts)

    p_ar, s_ar = ar_pick(
        tr_z.data[:npts].astype(float),
        tr_n.data[:npts].astype(float),
        tr_e.data[:npts].astype(float),
        sr,
        f1,
        f2,
        lta_p=0.08,
        sta_p=0.01,
        lta_s=0.10,
        sta_s=0.02,
        m_p=2,
        m_s=3,
        l_p=0.04,
        l_s=0.06,
        s_pick=True,
    )
    return p_ar, s_ar


def consensus_pick_for_station(
    picks_by_comp: dict,
    tr_z,
    p_ar=None,
    s_ar=None,
    pick_tolerance_s: float = 0.02,
    min_votes: int = 2,
    min_weight: Optional[float] = None,
    include_ar_s: bool = False,
    baer_weight: float = 1.0 / 3.0,
    mute_seconds: float = 0.20,
) -> dict:
    """
    Declare a station pick if enough weighted candidate picks cluster in time.
    """
    if min_weight is None:
        min_weight = float(min_votes)

    candidates = []

    def add_candidate(method, component, time, weight):
        time = obspy.UTCDateTime(time)
        relative_s = time - tr_z.stats.starttime
        if relative_s < mute_seconds:
            weight = 0.0
        candidates.append({
            "method": method,
            "component": component,
            "time": time,
            "relative_s": relative_s,
            "weight": float(weight),
        })

    for comp, picks in picks_by_comp.items():
        if picks.get("aic_ok") and picks.get("aic_time") is not None:
            add_candidate("aic", comp, picks["aic_time"], 1.0)
        if picks.get("baer_ok") and picks.get("baer_time") is not None:
            add_candidate("baer", comp, picks["baer_time"], baer_weight)

    if p_ar is not None:
        add_candidate("ar_p", "ZNE", tr_z.stats.starttime + p_ar, 1.0)

    if include_ar_s and s_ar is not None:
        add_candidate("ar_s", "ZNE", tr_z.stats.starttime + s_ar, 1.0)

    if len(candidates) == 0:
        return {
            "ok": False,
            "time": None,
            "relative_s": None,
            "n_votes": 0,
            "weight": 0.0,
            "methods": "",
            "components": "",
            "candidates": candidates,
        }

    best = None

    for cand in candidates:
        t0 = cand["time"]
        cluster = [
            c for c in candidates
            if abs(c["time"] - t0) <= pick_tolerance_s
        ]
        weight = sum(c["weight"] for c in cluster)
        n_votes = len([c for c in cluster if c["weight"] > 0])

        if best is None or weight > best["weight"]:
            pick_time = min(c["time"] for c in cluster)
            best = {
                "ok": weight >= min_weight,
                "time": pick_time,
                "relative_s": pick_time - tr_z.stats.starttime,
                "n_votes": n_votes,
                "weight": weight,
                "methods": ",".join(sorted(set(c["method"] for c in cluster))),
                "components": ",".join(sorted(set(c["component"] for c in cluster))),
                "candidates": candidates,
            }

    return best


def run_station_picking(
    st_pick: Stream,
    df_events: pd.DataFrame,
    station_names: Sequence[str],
    cfg: PickingConfig,
    min_event_peak_amplitude: float = 0.0,
    diagnostics_dir: Optional[Path] = None,
    make_diagnostic_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run per-station, per-event picking.

    Returns
    -------
    picks_df
        Individual method/component picks.
    station_picks_df
        One consensus row per event/station.
    """
    all_picks = []
    all_station_picks = []

    if diagnostics_dir is not None:
        diagnostics_dir = ensure_dir(diagnostics_dir)

    for event_idx, row in df_events.iterrows():
        event_start = obspy.UTCDateTime(row["on_time"]) - float(row["pretrigger_s"])
        event_end = obspy.UTCDateTime(row["off_time"]) + float(row["posttrigger_s"])

        st_event = st_pick.copy().trim(starttime=event_start, endtime=event_end, pad=False)

        datamax = safe_max_abs(st_event)
        if not np.isfinite(datamax) or datamax <= min_event_peak_amplitude:
            continue

        print(
            f"Event {event_idx}: "
            f"SNR={row.get('snr_rms', np.nan):.1f}, "
            f"on_time={row['on_time']}, off_time={row['off_time']}, "
            f"peak={datamax:.3g}"
        )

        for station in station_names:
            st_sta = st_event.select(station=station)

            if len(st_sta) < 3:
                continue

            try:
                tr_z = st_sta.select(component="Z")[0]
                tr_n = st_sta.select(component="N")[0]
                tr_e = st_sta.select(component="E")[0]
            except IndexError:
                continue

            picks_by_comp = {
                "Z": pick_baer_aic_on_trace(tr_z),
                "N": pick_baer_aic_on_trace(tr_n),
                "E": pick_baer_aic_on_trace(tr_e),
            }

            try:
                p_ar, s_ar = try_ar_pick_short_station(
                    tr_z, tr_n, tr_e, f1=cfg.freqmin, f2=cfg.freqmax
                )
            except Exception as e:
                p_ar, s_ar = None, None
                print(f"  AR picker failed for {station}: {e}")

            consensus = consensus_pick_for_station(
                picks_by_comp,
                tr_z,
                p_ar=p_ar,
                s_ar=s_ar,
                pick_tolerance_s=cfg.pick_tolerance_s,
                min_votes=cfg.min_votes,
                min_weight=cfg.min_weight,
                include_ar_s=cfg.include_ar_s,
                baer_weight=cfg.baer_weight,
                mute_seconds=cfg.mute_seconds,
            )

            for comp, picks in picks_by_comp.items():
                for method in ["aic", "baer"]:
                    all_picks.append({
                        "event_idx": event_idx,
                        "station": station,
                        "component": comp,
                        "method": method,
                        "ok": bool(picks.get(f"{method}_ok")),
                        "time": picks.get(f"{method}_time"),
                        "sample": picks.get(f"{method}_sample"),
                        "error": picks.get(f"{method}_error"),
                    })

            all_station_picks.append({
                "event_idx": event_idx,
                "station": station,
                "consensus_ok": consensus["ok"],
                "consensus_time": consensus["time"],
                "consensus_relative_s": consensus["relative_s"],
                "consensus_votes": consensus["n_votes"],
                "consensus_weight": consensus["weight"],
                "consensus_methods": consensus["methods"],
                "consensus_components": consensus["components"],
                "p_ar_s": p_ar,
                "s_ar_s": s_ar,
                "event_on_time": row["on_time"],
                "event_off_time": row["off_time"],
            })

            if make_diagnostic_plots and diagnostics_dir is not None:
                outfile = diagnostics_dir / f"event_{int(event_idx):03d}_{station}_pick_diagnostics.png"
                plot_station_pick_diagnostics(
                    station,
                    tr_z,
                    tr_n,
                    tr_e,
                    picks_by_comp,
                    p_ar=p_ar,
                    s_ar=s_ar,
                    consensus=consensus,
                    event_idx=event_idx,
                    outfile=outfile,
                )

    picks_df = pd.DataFrame(all_picks)
    station_picks_df = pd.DataFrame(all_station_picks)

    return picks_df, station_picks_df


# -----------------------------------------------------------------------------
# Event windows and output
# -----------------------------------------------------------------------------

def trim_events_from_consensus_picks(
    st: Stream,
    station_picks_df: pd.DataFrame,
    pre_pick_s: float = 0.10,
    event_length_s: float = 0.50,
    only_consensus_ok: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    Trim fixed-length event Streams using earliest station consensus pick per event.
    """
    event_streams = {}
    event_windows = []

    dfp = station_picks_df.copy()
    if only_consensus_ok and "consensus_ok" in dfp.columns:
        dfp = dfp[dfp["consensus_ok"] == True]

    for event_idx, group in dfp.groupby("event_idx"):
        pick_times = []
        for t in group["consensus_time"]:
            if t is None or str(t) in ("NaT", "None", "nan"):
                continue
            pick_times.append(obspy.UTCDateTime(t))

        if len(pick_times) == 0:
            continue

        reference_time = min(pick_times)
        event_start = reference_time - pre_pick_s
        event_end = event_start + event_length_s

        st_event = st.copy().trim(
            starttime=event_start,
            endtime=event_end,
            pad=True,
            fill_value=0,
        )

        event_streams[event_idx] = st_event

        event_windows.append({
            "event_idx": event_idx,
            "reference_time": reference_time,
            "event_start": event_start,
            "event_end": event_end,
            "pre_pick_s": pre_pick_s,
            "event_length_s": event_length_s,
            "n_station_picks": len(group),
            "stations": ",".join(sorted(group["station"].astype(str).unique())),
        })

    return event_streams, pd.DataFrame(event_windows)


def save_event_mseed_files(
    event_streams: dict,
    outdir: Path | str,
    prefix: str = "event",
) -> list[Path]:
    """Write event streams to MiniSEED."""
    outdir = ensure_dir(outdir)
    written = []

    for event_idx, st_event in event_streams.items():
        outfile = outdir / f"{prefix}_{int(event_idx):03d}.mseed"
        st_event.write(str(outfile), format="MSEED")
        written.append(outfile)

    return written


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_station_pick_diagnostics(
    station,
    tr_z,
    tr_n,
    tr_e,
    picks_by_comp,
    p_ar=None,
    s_ar=None,
    consensus=None,
    event_idx=None,
    outfile=None,
    dpi=160,
):
    """Plot Z/N/E traces for one station/event with picker diagnostics."""
    traces = {"Z": tr_z, "N": tr_n, "E": tr_e}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    for ax, comp in zip(axes, ["Z", "N", "E"]):
        tr = traces[comp]
        sr = float(tr.stats.sampling_rate)
        t = np.arange(tr.stats.npts) / sr
        y = normalize_trace_data(tr.data)

        ax.plot(t, y, color="black", linewidth=0.8, label=comp)
        picks = picks_by_comp[comp]

        if picks.get("aic_ok"):
            ax.axvline(picks["aic_sample"] / sr, color="blue", linestyle="--", linewidth=1.2, label="AIC")
        if picks.get("baer_ok"):
            ax.axvline(picks["baer_sample"] / sr, color="red", linestyle="--", linewidth=1.2, label="Baer")
        if comp == "Z" and p_ar is not None:
            ax.axvline(p_ar, color="green", linestyle="-.", linewidth=1.5, label="AR P")
        if comp == "Z" and s_ar is not None:
            ax.axvline(s_ar, color="purple", linestyle=":", linewidth=1.5, label="AR S")
        if consensus is not None and consensus.get("ok"):
            ax.axvline(
                consensus["relative_s"],
                color="orange",
                linestyle="-",
                linewidth=2.0,
                label=f"consensus ({consensus['n_votes']} votes)",
            )

        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time from station event-window start [s]")
    title = f"{station}"
    if event_idx is not None:
        title = f"Event {event_idx}: {station}"
    fig.suptitle(title)
    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)

    return fig


def plot_event_stream_with_station_consensus(
    st_event: Stream,
    station_picks_df: pd.DataFrame,
    event_idx,
    component: str = "Z",
    outfile=None,
    normalize: bool = True,
    scale: float = 1.0,
    dpi: int = 160,
):
    """
    Plot one component for all stations in an event Stream, with station picks.
    """
    st_comp = st_event.select(component=component)

    if len(st_comp) == 0:
        print(f"No {component} traces for event {event_idx}")
        return None

    picks = station_picks_df[
        (station_picks_df["event_idx"] == event_idx)
        & (station_picks_df["consensus_ok"] == True)
    ].copy()

    t0 = min(tr.stats.starttime for tr in st_event)
    traces = sorted(st_comp, key=lambda tr: tr.stats.station)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, tr in enumerate(traces):
        station = tr.stats.station
        y = normalize_trace_data(tr.data) if normalize else tr.data.astype(float)
        t = np.arange(tr.stats.npts) * tr.stats.delta + (tr.stats.starttime - t0)

        ax.plot(t, y * scale + i, color="black", linewidth=0.7)
        ax.text(t[-1], i, f" {tr.id}", va="center", fontsize=8)

        station_pick = picks[picks["station"] == station]
        if len(station_pick) > 0:
            pick_time = obspy.UTCDateTime(station_pick.iloc[0]["consensus_time"])
            pick_rel = pick_time - t0

            ax.plot(
                pick_rel,
                i,
                marker="o",
                markersize=8,
                color="orange",
                markeredgecolor="black",
                linestyle="None",
                label="station consensus pick" if i == 0 else None,
            )
            ax.plot(
                [pick_rel, pick_rel],
                [i - 0.32, i + 0.32],
                color="orange",
                linewidth=2.5,
            )

    ax.set_title(f"Event {int(event_idx):03d} {component}-component station consensus")
    ax.set_xlabel("Time from event window start [s]")
    ax.set_ylabel("Station / trace")
    ax.grid(True, alpha=0.3)

    if len(picks):
        ax.legend(loc="upper right")

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)

    return fig


def save_event_qc_figures(
    event_streams: dict,
    station_picks_df: pd.DataFrame,
    outdir: Path | str,
    components: Sequence[str] = ("Z", "N", "E"),
    prefix: str = "event",
) -> list[Path]:
    """Save event wiggle/QC plots for selected components."""
    outdir = ensure_dir(outdir)
    written = []

    for event_idx, st_event in event_streams.items():
        for component in components:
            outfile = outdir / f"{prefix}_{int(event_idx):03d}_{component}_station_consensus.png"
            fig = plot_event_stream_with_station_consensus(
                st_event,
                station_picks_df,
                event_idx=event_idx,
                component=component,
                outfile=outfile,
                normalize=True,
                scale=0.8,
            )
            if fig is not None:
                written.append(outfile)

    return written
