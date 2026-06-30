#!/usr/bin/env python3
"""
Generic Shot Gather Wiggle Picker standalone app.

A lightweight Tk/matplotlib application for reviewing and picking active-source
shot gathers from MiniSEED, SEG-Y, SU, or SEG-2 files.

Features
--------
- Loads arbitrary MiniSEED (*.mseed, *.miniseed, *.ms, etc.), SEG-Y (*.sgy, *.segy), SU (*.su), and SEG-2 (*.dat, *.seg2) files.
- Auto-detects file format from extension, with ObsPy fallback.
- Uses SEG-Y source/receiver x coordinates when available.
- For MiniSEED, attempts receiver x from trace metadata, numeric station codes,
  numeric channel/location codes, or finally trace index.
- Treats time as seconds relative to the earliest trace start in the loaded file,
  which is usually what you want for shot gather, stacked gather, and super-gather files.
- Optional detrend, taper, and bandpass/highpass filtering.
- Wiggle gain, trace scale, clipping, time-window controls.
- Red/blue positive/negative lobe shading.
- Single-gather and two-gather overlay display modes.
- Pick mode: left click adds pick, right click deletes nearest pick.
- Line mode: two left clicks define a line; velocity is dx/dt.
- Saves picks and velocity lines to CSV.

Dependencies
------------
conda install obspy scipy numpy pandas matplotlib
Python's tkinter is also required. On some systems it is a separate package.
"""

from __future__ import annotations

import re
import traceback
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from obspy import read, Stream

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



APP_VERSION = "2026-06-23 fixed-ui-db-component-su-seg2"
DEFAULT_BASE_DIR = Path("/Volumes/tachyon/LBSSP_DATA")
DATASET_KEYS = ["gather_1", "gather_2"]

# -----------------------------------------------------------------------------
# Project import path
# -----------------------------------------------------------------------------
# This app is expected to live in <repo>/apps and the reusable package in
# <repo>/lib/segy_tools.  We add exactly that one lib directory to sys.path.
# If that layout is wrong, imports should fail clearly rather than searching
# multiple possible locations.
REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = REPO_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


@dataclass
class TraceRow:
    dataset: str
    x: float
    t: np.ndarray
    y: np.ndarray
    trace_index: int
    trace_id: str
    source_x: float


@dataclass
class Pick:
    dataset: str
    receiver_x_m: float
    pick_time_s: float
    trace_index: int
    trace_id: str
    source_x_m: float
    display_mode: str


@dataclass
class VelocityLine:
    dataset: str
    x1_m: float
    t1_s: float
    x2_m: float
    t2_s: float
    dx_m: float
    dt_s: float
    velocity_mps: float
    abs_velocity_mps: float
    source_x_m: float
    display_mode: str




# -----------------------------------------------------------------------------
# segy_tools-backed helper layer
# -----------------------------------------------------------------------------
#
# This app is intentionally kept as a GUI/front-end.  Generic waveform I/O,
# SEG-Y/header geometry, preprocessing, database geometry lookup, and pick CSV
# serialization should live in segy_tools where available.  The fallback
# implementations below keep this file runnable as a standalone script while the
# package refactor settles.

try:
    import segy_tools.io as _st_io
except Exception:
    _st_io = None

try:
    import segy_tools.processing as _st_processing
except Exception:
    _st_processing = None

# Database lookup is intentionally optional and project-specific, but the import
# path is intentionally NOT optional/fuzzy.  From <repo>/apps/wiggle_picker.py,
# we import exactly <repo>/lib/segy_tools/db.py.  If that fails, report the
# single expected path instead of trying fallback locations.
_st_db = None
_db_import_error = None
try:
    from segy_tools import db as _st_db
except Exception as exc:
    _db_import_error = exc
    _st_db = None

try:
    import segy_tools.picking as _st_picking
except Exception:
    _st_picking = None


def _header_value(header, name: str, default=None):
    if _st_io is not None and hasattr(_st_io, "header_value"):
        try:
            return _st_io.header_value(header, [name], default=default)
        except TypeError:
            try:
                return _st_io.header_value(header, name, default=default)
            except Exception:
                pass
        except Exception:
            pass
    try:
        value = getattr(header, name)
    except Exception:
        return default
    return default if value is None else value


def _coord_scale_from_header(header) -> float:
    scalar = _header_value(header, "scalar_to_be_applied_to_all_coordinates", 1)
    if scalar in (None, 0):
        return 1.0
    scalar = int(scalar)
    if scalar > 0:
        return float(scalar)
    return 1.0 / abs(float(scalar))


def _read_header_geometry(header, i: int) -> Tuple[float, float]:
    """Return receiver_x_m, source_x_m from SEG-Y/SU-style trace header."""
    if header is None:
        return float(i), np.nan

    scale = _coord_scale_from_header(header)

    gx = None
    for name in ("group_coordinate_x", "gx", "receiver_coordinate_x", "geophone_coordinate_x"):
        gx = _header_value(header, name, None)
        if gx not in (None, 0):
            break

    sx = None
    for name in ("source_coordinate_x", "sx", "shot_coordinate_x"):
        sx = _header_value(header, name, None)
        if sx not in (None, 0):
            break

    rx = float(i) if gx in (None, 0) else float(gx) * scale
    src = np.nan if sx in (None, 0) else float(sx) * scale
    return rx, src


def _first_float_from_string(text: str) -> Optional[float]:
    """Extract the first numeric token from a string, if any."""
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _read_stats_geometry(tr, i: int, station_scale: float = 1.0) -> Tuple[float, float]:
    """Best-effort receiver/source geometry from Trace.stats fields."""
    rx0 = getattr(tr.stats, "receiver_x_m", None)
    sx0 = getattr(tr.stats, "source_x_m", None)
    if rx0 is not None:
        try:
            rx = float(rx0)
            sx = float(sx0) if sx0 is not None else np.nan
            return rx, sx
        except Exception:
            pass

    for attr in ("distance", "dist"):
        value = getattr(tr.stats, attr, None)
        if value is not None:
            try:
                sx = getattr(tr.stats, "source_x_m", np.nan)
                return float(value), float(sx) if sx is not None else np.nan
            except Exception:
                pass

    coords = getattr(tr.stats, "coordinates", None)
    if coords is not None:
        for key in ("x", "longitude", "lon"):
            try:
                value = getattr(coords, key)
            except Exception:
                value = None
            if value is None and isinstance(coords, dict):
                value = coords.get(key)
            if value is not None:
                try:
                    return float(value), np.nan
                except Exception:
                    pass

    for code in (
        getattr(tr.stats, "station", ""),
        getattr(tr.stats, "location", ""),
        getattr(tr.stats, "channel", ""),
    ):
        value = _first_float_from_string(code)
        if value is not None:
            return float(value) * float(station_scale), np.nan

    return float(i), np.nan


def _read_trace_geometry(tr, i: int, station_scale: float = 1.0) -> Tuple[float, float]:
    """Return receiver/source x, preferring stats and SEG-Y/SU headers."""
    # Database/project geometry attached to stats is always highest priority.
    rx0 = getattr(tr.stats, "receiver_x_m", None)
    if rx0 is not None:
        return _read_stats_geometry(tr, i, station_scale=station_scale)

    # Let segy_tools do canonical SEG-Y header extraction when available.
    if _st_io is not None and hasattr(_st_io, "extract_geometry_from_stream"):
        try:
            g = _st_io.extract_geometry_from_stream(Stream([tr.copy()]))
            rx = np.asarray(g.get("receiver_x_m", []), dtype=float)
            sx = np.asarray(g.get("source_x_m", []), dtype=float)
            if len(rx) and np.isfinite(rx[0]) and abs(rx[0] - 0.0) > 0:
                return float(rx[0]), float(sx[0]) if len(sx) and np.isfinite(sx[0]) else np.nan
        except Exception:
            pass

    if hasattr(tr.stats, "segy"):
        h = getattr(getattr(tr.stats, "segy", None), "trace_header", None)
        return _read_header_geometry(h, i)
    if hasattr(tr.stats, "su"):
        h = getattr(getattr(tr.stats, "su", None), "trace_header", None)
        return _read_header_geometry(h, i)

    return _read_stats_geometry(tr, i, station_scale=station_scale)


def guess_obspy_format(path: Path) -> Optional[str]:
    if _st_io is not None and hasattr(_st_io, "guess_obspy_format"):
        return _st_io.guess_obspy_format(Path(path))

    ext = Path(path).suffix.lower()
    if ext in {".sgy", ".segy", ".seg"}:
        return "SEGY"
    if ext in {".su"}:
        return "SU"
    if ext in {".dat", ".seg2"}:
        return "SEG2"
    if ext in {".mseed", ".miniseed", ".ms", ".seed"}:
        return "MSEED"
    return None


def read_waveform_file(path: Path) -> Stream:
    """Read MiniSEED, SEG-Y, SU, or SEG-2, using segy_tools.io when available."""
    if _st_io is not None and hasattr(_st_io, "read_waveform_file"):
        return _st_io.read_waveform_file(Path(path))

    fmt = guess_obspy_format(Path(path))
    if fmt is not None:
        try:
            return read(str(path), format=fmt)
        except Exception:
            pass
    return read(str(path))


def debug_database_schema(db_path: Path) -> None:
    """Print DB schema using segy_tools.db/db.py when available."""
    if _st_db is not None and hasattr(_st_db, "debug_database_schema"):
        return _st_db.debug_database_schema(Path(db_path))

    import sqlite3

    print("\n" + "=" * 80)
    print(f"DATABASE DEBUG: {db_path}")
    print("=" * 80)
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print("Tables:", tables)
        for t in tables:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({t})")]
            try:
                n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                n = "?"
            print(f"\n{t} ({n} rows)")
            print("  columns:", cols)
            try:
                sample = conn.execute(f'SELECT * FROM "{t}" LIMIT 3').fetchall()
                print("  sample:", sample)
            except Exception as e:
                print("  sample failed:", e)
        conn.close()
    except Exception as e:
        print("Database schema debug failed:", e)


def load_db_geometry_for_file(db_path: Path, waveform_path: Path, ntraces: int) -> Tuple[Dict[int, Tuple[float, float]], str]:
    """Project DB geometry lookup.  Prefer segy_tools.db/db.py; fallback is none."""
    if _st_db is not None and hasattr(_st_db, "load_db_geometry_for_file"):
        return _st_db.load_db_geometry_for_file(Path(db_path), Path(waveform_path), int(ntraces))
    detail = f": {_db_import_error}" if _db_import_error is not None else ""
    return {}, "database helper module not available" + detail


def apply_geometry_map(st: Stream, geom: Dict[int, Tuple[float, float]]) -> Stream:
    """Attach pre-loaded geometry mapping to trace stats before plotting."""
    if _st_io is not None and hasattr(_st_io, "apply_geometry_map"):
        return _st_io.apply_geometry_map(st, geom)

    out = st.copy()
    for i, tr in enumerate(out):
        if i not in geom:
            continue
        rx, sx = geom[i]
        tr.stats.distance = float(rx)
        tr.stats.receiver_x_m = float(rx)
        if np.isfinite(sx):
            tr.stats.source_x_m = float(sx)
    return out


def _component_code(tr) -> str:
    chan = str(getattr(tr.stats, "channel", "") or "").strip()
    if chan:
        return chan[-1].upper()
    loc = str(getattr(tr.stats, "location", "") or "").strip()
    if loc and loc[-1].upper() in {"Z", "N", "E", "1", "2", "3"}:
        return loc[-1].upper()
    return ""


def filter_stream_by_component(st: Stream, component: str) -> Stream:
    """Return only traces matching selected component/channel suffix."""
    if _st_io is not None and hasattr(_st_io, "filter_stream_by_component"):
        return _st_io.filter_stream_by_component(st, component)

    comp = (component or "All").strip().upper()
    if comp == "ALL":
        return st.copy()
    out = Stream()
    for tr in st:
        if _component_code(tr) == comp:
            out += tr.copy()
    return out


def stream_starttime_string(st: Stream) -> str:
    if _st_io is not None and hasattr(_st_io, "stream_starttime_string"):
        return _st_io.stream_starttime_string(st)
    if not st:
        return "UTC ?"
    try:
        return str(min(tr.stats.starttime for tr in st))
    except Exception:
        return "UTC ?"


def preprocess_stream(
    st: Stream,
    *,
    detrend: bool = True,
    taper_pct: float = 0.02,
    filter_on: bool = True,
    freqmin: float = 5.0,
    freqmax: float = 150.0,
    corners: int = 4,
) -> Stream:
    """Preprocess Stream for display, preferring segy_tools.processing/io."""
    for module in (_st_processing, _st_io):
        if module is not None and hasattr(module, "preprocess_stream"):
            try:
                return module.preprocess_stream(
                    st,
                    detrend=detrend,
                    taper_pct=taper_pct,
                    filter_on=filter_on,
                    freqmin=freqmin,
                    freqmax=freqmax,
                    corners=corners,
                )
            except TypeError:
                pass

    out = st.copy()
    for tr in out:
        tr.data = tr.data.astype(np.float64)

        if detrend:
            try:
                tr.detrend("linear")
            except Exception:
                tr.detrend("demean")

        if taper_pct and taper_pct > 0:
            tr.taper(max_percentage=float(taper_pct))

        if filter_on and freqmin and freqmin > 0:
            nyq = 0.5 / float(tr.stats.delta)
            if freqmax and freqmax > freqmin and freqmax < 0.95 * nyq:
                tr.filter("bandpass", freqmin=float(freqmin), freqmax=float(freqmax), corners=int(corners), zerophase=True)
            else:
                tr.filter("highpass", freq=float(freqmin), corners=int(corners), zerophase=True)
    return out


def stream_to_rows(
    st: Stream,
    *,
    dataset: str,
    tmin: float,
    tmax: float,
    display_time_shift_s: float = 0.0,
    station_scale: float = 1.0,
) -> Tuple[List[TraceRow], Optional[float]]:
    """Convert a Stream to GUI trace rows using receiver-position-aware geometry."""
    rows: List[TraceRow] = []
    sources: List[float] = []

    if not st:
        return rows, None

    earliest_start = min(tr.stats.starttime for tr in st)

    for i, tr in enumerate(st):
        rx, sx = _read_trace_geometry(tr, i, station_scale=station_scale)
        if np.isfinite(sx):
            sources.append(float(sx))

        dt = float(tr.stats.delta)
        t0 = float(tr.stats.starttime - earliest_start)
        t_trace = t0 + np.arange(tr.stats.npts, dtype=float) * dt + display_time_shift_s

        grid = np.arange(tmin, tmax + 0.5 * dt, dt, dtype=float)
        y = np.interp(grid, t_trace, tr.data.astype(float), left=np.nan, right=np.nan)

        if np.isfinite(y).sum() < 5:
            continue

        rows.append(
            TraceRow(
                dataset=dataset,
                x=float(rx),
                t=grid,
                y=y,
                trace_index=i,
                trace_id=getattr(tr, "id", f"trace_{i}"),
                source_x=float(sx) if np.isfinite(sx) else np.nan,
            )
        )

    rows.sort(key=lambda r: r.x)
    source_x = float(np.nanmedian(sources)) if sources else None
    return rows, source_x


def normalize_trace(y: np.ndarray, clip_percentile: float) -> np.ndarray:
    if _st_picking is not None and hasattr(_st_picking, "normalize_trace_data"):
        try:
            # normalize_trace_data may not support clipping percentile, so still
            # perform the picker-app clipping below.
            y = np.asarray(_st_picking.normalize_trace_data(y), dtype=float)
            return np.clip(y, -1.0, 1.0)
        except Exception:
            pass

    y = np.asarray(y, dtype=float)
    y = y - np.nanmedian(y)
    scale = np.nanpercentile(np.abs(y), clip_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(y))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(y / scale, -1.0, 1.0)


def picks_to_dataframe(picks: List[Pick]) -> pd.DataFrame:
    if _st_picking is not None and hasattr(_st_picking, "picks_to_dataframe"):
        try:
            return _st_picking.picks_to_dataframe(picks)
        except Exception:
            pass
    return pd.DataFrame([asdict(p) for p in picks])


def velocity_lines_to_dataframe(lines: List[VelocityLine]) -> pd.DataFrame:
    if _st_picking is not None and hasattr(_st_picking, "velocity_lines_to_dataframe"):
        try:
            return _st_picking.velocity_lines_to_dataframe(lines)
        except Exception:
            pass
    return pd.DataFrame([asdict(v) for v in lines])



class ShotGatherPickerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Shot Gather Wiggle Picker — {APP_VERSION}")

        self.paths: Dict[str, Optional[Path]] = {key: None for key in DATASET_KEYS}
        self.raw_streams: Dict[str, Optional[Stream]] = {key: None for key in DATASET_KEYS}
        self.source_x_by_dataset: Dict[str, Optional[float]] = {key: None for key in DATASET_KEYS}
        self.starttime_by_dataset: Dict[str, Optional[object]] = {key: None for key in DATASET_KEYS}
        self.db_path: Optional[Path] = None
        self.db_geometry_by_dataset: Dict[str, Dict[int, Tuple[float, float]]] = {key: {} for key in DATASET_KEYS}
        self.db_geometry_note_by_dataset: Dict[str, str] = {key: "" for key in DATASET_KEYS}

        self.picks: List[Pick] = []
        self.line_pending: List[Tuple[str, float, float, int, str]] = []
        self.lines: List[VelocityLine] = []

        self.active_rows_by_dataset: Dict[str, List[TraceRow]] = {}
        self.current_mode = tk.StringVar(value="pick")
        self.display_mode = tk.StringVar(value="gather_1")
        self.shade_mode = tk.StringVar(value="both")
        self.component = tk.StringVar(value="Z")

        self.tmin = tk.DoubleVar(value=0.0)
        self.tmax = tk.DoubleVar(value=0.8)
        self.gain = tk.DoubleVar(value=1.0)
        self.trace_scale = tk.DoubleVar(value=0.75)
        self.clip_percentile = tk.DoubleVar(value=99.0)
        self.gather_2_shift = tk.DoubleVar(value=0.0)
        self.station_scale = tk.DoubleVar(value=0.01)

        self.filter_on = tk.BooleanVar(value=True)
        self.detrend_on = tk.BooleanVar(value=True)
        self.taper_pct = tk.DoubleVar(value=0.02)
        self.freqmin = tk.DoubleVar(value=5.0)
        self.freqmax = tk.DoubleVar(value=150.0)
        self.corners = tk.IntVar(value=4)

        self.status = tk.StringVar(value="Load a waveform file. Optional: load the project SQLite DB for geometry.")

        # Redraw automatically when changing display/component selectors.
        self.display_mode.trace_add("write", lambda *_: self.redraw())
        self.component.trace_add("write", lambda *_: self.redraw())

        self._build_gui()
        self.redraw()

    def _build_gui(self):
        left = ttk.Frame(self.root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)

        main = ttk.Frame(self.root)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(left, text=f"Shot gather files — {APP_VERSION}", padding=6)
        file_frame.pack(fill=tk.X, pady=4)

        self.file_labels: Dict[str, ttk.Label] = {}
        labels = {"gather_1": "Load gather 1", "gather_2": "Load gather 2 / overlay"}
        for key in DATASET_KEYS:
            row = ttk.Frame(file_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Button(row, text=labels[key], command=lambda k=key: self.load_file_dialog(k)).pack(side=tk.LEFT)
            lab = ttk.Label(row, text="(not loaded)", width=42)
            lab.pack(side=tk.LEFT)
            self.file_labels[key] = lab

        # Duplicate the database button here so it is hard to miss, even on
        # smaller screens where lower controls may be clipped.
        ttk.Button(file_frame, text="LOAD DB / GEOMETRY INDEX", command=self.load_database_dialog).pack(fill=tk.X, pady=(6, 2))

        quick = ttk.Frame(file_frame)
        quick.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(quick, text="Component Z/N/E:").pack(side=tk.LEFT)
        ttk.Combobox(
            quick,
            textvariable=self.component,
            values=["Z", "N", "E", "1", "2", "3", "All"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(quick, text="Station scale:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(quick, textvariable=self.station_scale, width=7).pack(side=tk.LEFT, padx=4)

        db_frame = ttk.LabelFrame(left, text="Geometry database", padding=6)
        db_frame.pack(fill=tk.X, pady=4)
        row = ttk.Frame(db_frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Button(row, text="Load SQLite DB", command=self.load_database_dialog).pack(side=tk.LEFT)
        self.db_label = ttk.Label(row, text="(optional)", width=42)
        self.db_label.pack(side=tk.LEFT)
        ttk.Button(db_frame, text="Print DB schema", command=self.print_db_schema).pack(fill=tk.X, pady=2)
        ttk.Button(db_frame, text="Rebuild DB geometry", command=self.rebuild_db_geometry).pack(fill=tk.X, pady=2)

        ctrl = ttk.LabelFrame(left, text="Display", padding=6)
        ctrl.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="Dataset").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.display_mode,
            values=["gather_1", "gather_2", "overlay"],
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="ew")

        ttk.Label(ctrl, text="Mode").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.current_mode,
            values=["pick", "line"],
            state="readonly",
            width=22,
        ).grid(row=1, column=1, sticky="ew")

        ttk.Label(ctrl, text="Shade").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.shade_mode,
            values=["none", "positive", "negative", "both"],
            state="readonly",
            width=22,
        ).grid(row=2, column=1, sticky="ew")

        ttk.Label(ctrl, text="Component").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.component,
            values=["Z", "N", "E", "1", "2", "3", "All"],
            state="readonly",
            width=22,
        ).grid(row=3, column=1, sticky="ew")

        numeric = [
            ("tmin", self.tmin),
            ("tmax", self.tmax),
            ("gain", self.gain),
            ("trace scale", self.trace_scale),
            ("clip %", self.clip_percentile),
            ("gather 2 shift s", self.gather_2_shift),
            ("station code scale", self.station_scale),
        ]
        for r, (name, var) in enumerate(numeric, start=4):
            ttk.Label(ctrl, text=name).grid(row=r, column=0, sticky="w")
            ttk.Entry(ctrl, textvariable=var, width=10).grid(row=r, column=1, sticky="ew")

        proc = ttk.LabelFrame(left, text="Processing", padding=6)
        proc.pack(fill=tk.X, pady=4)

        ttk.Checkbutton(proc, text="Detrend", variable=self.detrend_on).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(proc, text="Filter", variable=self.filter_on).grid(row=0, column=1, sticky="w")

        proc_items = [
            ("taper pct", self.taper_pct),
            ("freqmin", self.freqmin),
            ("freqmax", self.freqmax),
            ("corners", self.corners),
        ]
        for r, (name, var) in enumerate(proc_items, start=1):
            ttk.Label(proc, text=name).grid(row=r, column=0, sticky="w")
            ttk.Entry(proc, textvariable=var, width=10).grid(row=r, column=1, sticky="ew")

        actions = ttk.LabelFrame(left, text="Actions", padding=6)
        actions.pack(fill=tk.X, pady=4)

        ttk.Button(actions, text="Redraw", command=self.redraw).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Clear picks", command=self.clear_picks).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Clear lines", command=self.clear_lines).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Save picks/lines", command=self.save_outputs).pack(fill=tk.X, pady=2)

        help_text = (
            "Pick mode: left-click adds pick; right-click deletes nearest pick.\n"
            "Line mode: two left-clicks add velocity line; right-click deletes last line/point.\n"
            "Overlay: black=gather 1, red=gather 2. Middle-click picks gather 2.\n"
            "Geometry order: SQLite DB if matched, waveform headers/metadata; numeric nodal station codes use station code scale=0.01 by default."
        )
        ttk.Label(left, text=help_text, wraplength=340, justify=tk.LEFT).pack(fill=tk.X, pady=4)

        ttk.Label(left, textvariable=self.status, wraplength=340, foreground="blue").pack(fill=tk.X, pady=4)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, main)
        toolbar.update()

        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)

    def _short_path(self, p: Optional[Path]) -> str:
        if p is None:
            return "(not loaded)"
        p = Path(p)
        parts = p.parts
        if len(parts) > 4:
            return ".../" + "/".join(parts[-3:])
        return str(p)

    def load_file_dialog(self, key: str):
        path = filedialog.askopenfilename(
            title=f"Load {key} MiniSEED, SEG-Y, SU, or SEG-2",
            filetypes=[
                ("Waveform files", "*.mseed *.miniseed *.ms *.seed *.sgy *.segy *.SEG *.SGY *.su *.SU *.dat *.DAT *.seg2 *.SEG2"),
                ("MiniSEED", "*.mseed *.miniseed *.ms *.seed"),
                ("SEG-Y", "*.sgy *.segy *.SEG *.SGY"),
                ("Seismic Unix", "*.su *.SU"),
                ("SEG-2", "*.dat *.DAT *.seg2 *.SEG2"),
                ("All files", "*.*"),
            ],
            initialdir=str(DEFAULT_BASE_DIR if DEFAULT_BASE_DIR.exists() else Path.cwd()),
        )
        if not path:
            return
        try:
            p = Path(path)
            st = read_waveform_file(p)
            if len(st) == 0:
                raise ValueError("ObsPy read succeeded, but the stream contains no traces.")
            self.paths[key] = p
            self.raw_streams[key] = st
            self.starttime_by_dataset[key] = min(tr.stats.starttime for tr in st) if len(st) else None
            self.file_labels[key].configure(text=self._short_path(p))
            if self.display_mode.get() not in (key, "overlay"):
                self.display_mode.set(key)
            self.status.set(f"Loaded {key}: {p.name} ({len(st)} traces)")
            print("\n" + "=" * 80)
            print(f"WAVEFORM DEBUG: {key} {p}")
            print("=" * 80)
            print(st)
            for i, tr in enumerate(st):
                print("\n" + "-" * 80)
                print(f"Trace {i}")
                print(tr)
                print(dict(tr.stats))
            self.rebuild_db_geometry_for_dataset(key)
            self.redraw()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Load error", f"Could not load {path}\n\n{e}")

    def load_database_dialog(self):
        path = filedialog.askopenfilename(
            title="Load project SQLite geometry database",
            filetypes=[
                ("SQLite databases", "*.sqlite *.sqlite3 *.db *.DB"),
                ("All files", "*.*"),
            ],
            initialdir=str(DEFAULT_BASE_DIR if DEFAULT_BASE_DIR.exists() else Path.cwd()),
        )
        if not path:
            return
        self.db_path = Path(path)
        self.db_label.configure(text=self._short_path(self.db_path))
        debug_database_schema(self.db_path)
        self.rebuild_db_geometry()

    def print_db_schema(self):
        if not self.db_path:
            self.status.set("No SQLite DB loaded.")
            return
        debug_database_schema(self.db_path)
        self.status.set(f"Printed DB schema: {self.db_path.name}")

    def rebuild_db_geometry_for_dataset(self, key: str):
        self.db_geometry_by_dataset[key] = {}
        self.db_geometry_note_by_dataset[key] = ""
        if not self.db_path or not self.paths.get(key) or self.raw_streams.get(key) is None:
            return
        geom, note = load_db_geometry_for_file(self.db_path, self.paths[key], len(self.raw_streams[key]))
        self.db_geometry_by_dataset[key] = geom
        self.db_geometry_note_by_dataset[key] = note
        print(f"DB geometry for {key}: {note}")
        if geom:
            xs = [v[0] for v in geom.values()]
            print(f"  receiver_x_m range: {min(xs):.3f} to {max(xs):.3f}; n={len(xs)}")
            for idx in sorted(geom)[:10]:
                rx, sx = geom[idx]
                print(f"  trace {idx}: receiver_x_m={rx}, source_x_m={sx}")

    def rebuild_db_geometry(self):
        for key in DATASET_KEYS:
            self.rebuild_db_geometry_for_dataset(key)
        notes = [f"{k}: {len(v)}" for k, v in self.db_geometry_by_dataset.items() if v]
        if notes:
            self.status.set("DB geometry loaded: " + ", ".join(notes))
        elif self.db_path:
            self.status.set("DB loaded, but no matching geometry found for loaded waveform files.")
        else:
            self.status.set("No SQLite DB loaded.")
        self.redraw()

    def _source_time_title_piece(self, key: str) -> str:
        pieces = []
        sx = self.source_x_by_dataset.get(key)
        if sx is not None and np.isfinite(sx):
            pieces.append(f"shot x={float(sx):.2f} m")
        st0 = self.starttime_by_dataset.get(key)
        if st0 is not None:
            pieces.append(f"UTC {st0}")
        note = self.db_geometry_note_by_dataset.get(key, "")
        if note:
            pieces.append(f"DB: {note}")
        return " | ".join(pieces)

    def _all_source_time_title_piece(self) -> str:
        visible = self._visible_datasets()
        return " ; ".join(
            f"{key}: {piece}" for key in visible
            for piece in [self._source_time_title_piece(key)]
            if piece
        )

    def processed_rows_for(self, key: str, *, display_shift_s: float = 0.0) -> List[TraceRow]:
        st = self.raw_streams.get(key)
        if st is None:
            return []

        try:
            geom = self.db_geometry_by_dataset.get(key, {})
            if geom:
                st = apply_geometry_map(st, geom)

            # MiniSEED nodal gathers often contain Z/N/E traces in one Stream.
            # Filter by channel suffix before plotting so components are not
            # drawn on top of each other at the same receiver position.
            st = filter_stream_by_component(st, self.component.get())
            if len(st) == 0:
                self.status.set(f"No {self.component.get()} component traces found for {key}.")
                return []

            stp = preprocess_stream(
                st,
                detrend=self.detrend_on.get(),
                taper_pct=float(self.taper_pct.get()),
                filter_on=self.filter_on.get(),
                freqmin=float(self.freqmin.get()),
                freqmax=float(self.freqmax.get()),
                corners=int(self.corners.get()),
            )
            rows, sx = stream_to_rows(
                stp,
                dataset=key,
                tmin=float(self.tmin.get()),
                tmax=float(self.tmax.get()),
                display_time_shift_s=display_shift_s,
                station_scale=float(self.station_scale.get()),
            )
            if sx is not None and np.isfinite(sx):
                self.source_x_by_dataset[key] = sx
            return rows
        except Exception as e:
            self.status.set(f"Processing failed for {key}: {e}")
            traceback.print_exc()
            return []

    def redraw(self):
        self.ax.clear()
        self.active_rows_by_dataset.clear()

        mode = self.display_mode.get()
        shade = self.shade_mode.get()

        if mode == "overlay":
            rows1 = self.processed_rows_for("gather_1", display_shift_s=0.0)
            rows2 = self.processed_rows_for("gather_2", display_shift_s=float(self.gather_2_shift.get()))
            self.active_rows_by_dataset["gather_1"] = rows1
            self.active_rows_by_dataset["gather_2"] = rows2
            self._draw_rows(rows1, color="black", alpha=0.9, shade=shade)
            self._draw_rows(rows2, color="red", alpha=0.55, shade=shade)
            title = f"Overlay: gather 1 black + gather 2 red | component {self.component.get()} | gather 2 shift {self.gather_2_shift.get():+.4f}s"
        else:
            shift = float(self.gather_2_shift.get()) if mode == "gather_2" else 0.0
            rows = self.processed_rows_for(mode, display_shift_s=shift)
            self.active_rows_by_dataset[mode] = rows
            color = "red" if mode == "gather_2" else "black"
            self._draw_rows(rows, color=color, alpha=0.9, shade=shade)
            title = f"{mode} | component {self.component.get()}"

        for ds, sx in self.source_x_by_dataset.items():
            if sx is not None and np.isfinite(sx) and ds in self._visible_datasets():
                color = "green" if ds == "gather_1" else "darkgreen"
                self.ax.axvline(sx, color=color, linestyle="--", linewidth=1.2, label=f"{ds} source x={sx:.1f} m")

        self._draw_picks_and_lines()

        self.ax.invert_yaxis()
        self.ax.set_xlabel("Receiver x (m; DB/header/metadata; nodal station codes scaled by station code scale)")
        self.ax.set_ylabel("Time since earliest trace start (s)")
        meta = self._all_source_time_title_piece()
        full_title = f"Shot Gather Picker | {title} | {self.current_mode.get()} mode"
        if meta:
            full_title += f"\n{meta}"
        self.ax.set_title(full_title)
        self.ax.grid(True, alpha=0.25)
        if any(ds in self._visible_datasets() and sx is not None for ds, sx in self.source_x_by_dataset.items()):
            self.ax.legend(loc="best")

        self.canvas.draw_idle()
        self.status.set(f"Redrawn. Picks={len(self.picks)}, lines={len(self.lines)}")

    def _trace_spacing(self, rows: List[TraceRow]) -> float:
        xs = np.array([r.x for r in rows], dtype=float)
        ux = np.sort(np.unique(xs[np.isfinite(xs)]))
        dx = np.nanmedian(np.diff(ux)) if len(ux) > 1 else 1.0
        if not np.isfinite(dx) or dx <= 0:
            return 1.0
        return float(dx)

    def _draw_rows(self, rows: List[TraceRow], *, color: str, alpha: float, shade: str):
        if not rows:
            return

        dx = self._trace_spacing(rows)
        gain = float(self.gain.get())
        trace_scale = float(self.trace_scale.get())
        clip = float(self.clip_percentile.get())

        for r in rows:
            y = normalize_trace(r.y, clip)
            wig = r.x + gain * trace_scale * dx * y
            self.ax.plot(wig, r.t, color=color, alpha=alpha, linewidth=0.6)

            if shade in ("positive", "both"):
                self.ax.fill_betweenx(r.t, r.x, wig, where=(wig >= r.x), color="red", alpha=0.32, interpolate=True)
            if shade in ("negative", "both"):
                self.ax.fill_betweenx(r.t, r.x, wig, where=(wig < r.x), color="blue", alpha=0.25, interpolate=True)

    def _visible_datasets(self) -> List[str]:
        mode = self.display_mode.get()
        if mode == "overlay":
            return ["gather_1", "gather_2"]
        return [mode]

    def _draw_picks_and_lines(self):
        visible = set(self._visible_datasets())

        for p in self.picks:
            if p.dataset not in visible:
                continue
            c = "orange" if p.dataset == "gather_1" else "purple"
            self.ax.plot(p.receiver_x_m, p.pick_time_s, "o", color=c, markersize=5)
            self.ax.text(p.receiver_x_m, p.pick_time_s, f"{p.pick_time_s:.3f}", color=c, fontsize=7)

        for ds, x, t, idx, trace_id in self.line_pending:
            if ds in visible:
                self.ax.plot(x, t, "s", color="dodgerblue", markersize=5)

        for ln in self.lines:
            if ln.dataset not in visible:
                continue
            self.ax.plot([ln.x1_m, ln.x2_m], [ln.t1_s, ln.t2_s], "-", color="dodgerblue", linewidth=1.8)
            xm = 0.5 * (ln.x1_m + ln.x2_m)
            tm = 0.5 * (ln.t1_s + ln.t2_s)
            self.ax.text(
                xm,
                tm,
                f"{ln.abs_velocity_mps:.0f} m/s",
                color="dodgerblue",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
            )

    def _dataset_for_click(self, event) -> str:
        mode = self.display_mode.get()
        if mode == "overlay":
            # Middle-click picks gather 2 in overlay; left-click picks gather 1.
            if event.button == 2:
                return "gather_2"
            return "gather_1"
        return mode

    def _nearest_row(self, dataset: str, x: float) -> Optional[TraceRow]:
        rows = self.active_rows_by_dataset.get(dataset, [])
        if not rows:
            return None
        xs = np.array([r.x for r in rows], dtype=float)
        i = int(np.nanargmin(np.abs(xs - x)))
        return rows[i]

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        ds = self._dataset_for_click(event)
        row = self._nearest_row(ds, float(event.xdata))
        if row is None:
            self.status.set(f"No active traces for {ds}")
            return

        x = float(row.x)
        t = float(event.ydata)

        if self.current_mode.get() == "pick":
            if event.button == 3:
                self.delete_nearest_pick(ds, x, t)
            else:
                self.picks.append(
                    Pick(
                        dataset=ds,
                        receiver_x_m=x,
                        pick_time_s=t,
                        trace_index=row.trace_index,
                        trace_id=row.trace_id,
                        source_x_m=float(row.source_x) if np.isfinite(row.source_x) else np.nan,
                        display_mode=self.display_mode.get(),
                    )
                )
                self.status.set(f"Added pick: {ds} x={x:.2f} t={t:.4f}")

        elif self.current_mode.get() == "line":
            if event.button == 3:
                if self.lines:
                    removed = self.lines.pop()
                    self.status.set(f"Removed line |v|={removed.abs_velocity_mps:.1f} m/s")
                elif self.line_pending:
                    self.line_pending.pop()
                    self.status.set("Removed pending line point")
            else:
                self.line_pending.append((ds, x, t, row.trace_index, row.trace_id))
                if len(self.line_pending) >= 2:
                    p1 = self.line_pending[-2]
                    p2 = self.line_pending[-1]
                    if p1[0] == p2[0]:
                        self.add_velocity_line(p1, p2)
                        self.line_pending.pop()
                        self.line_pending.pop()
                    else:
                        self.status.set("Line points must be from the same dataset.")

        self.redraw()

    def delete_nearest_pick(self, dataset: str, x: float, t: float):
        candidates = [(i, p) for i, p in enumerate(self.picks) if p.dataset == dataset]
        if not candidates:
            self.status.set("No picks to delete.")
            return
        i, p = min(candidates, key=lambda ip: abs(ip[1].receiver_x_m - x) + 5.0 * abs(ip[1].pick_time_s - t))
        self.picks.pop(i)
        self.status.set(f"Deleted pick: {p.dataset} x={p.receiver_x_m:.2f} t={p.pick_time_s:.4f}")

    def add_velocity_line(self, p1, p2):
        ds1, x1, t1, _, _ = p1
        _, x2, t2, _, _ = p2
        dx = x2 - x1
        dt = t2 - t1
        v = dx / dt if abs(dt) > 1e-9 else np.nan
        sx = self.source_x_by_dataset.get(ds1)
        ln = VelocityLine(
            dataset=ds1,
            x1_m=float(x1),
            t1_s=float(t1),
            x2_m=float(x2),
            t2_s=float(t2),
            dx_m=float(dx),
            dt_s=float(dt),
            velocity_mps=float(v),
            abs_velocity_mps=float(abs(v)) if np.isfinite(v) else np.nan,
            source_x_m=float(sx) if sx is not None and np.isfinite(sx) else np.nan,
            display_mode=self.display_mode.get(),
        )
        self.lines.append(ln)
        self.status.set(f"Added velocity line: v={v:.1f} m/s, |v|={abs(v):.1f} m/s")

    def clear_picks(self):
        self.picks.clear()
        self.redraw()

    def clear_lines(self):
        self.lines.clear()
        self.line_pending.clear()
        self.redraw()

    def save_outputs(self):
        outdir = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=str(DEFAULT_BASE_DIR if DEFAULT_BASE_DIR.exists() else Path.cwd()),
        )
        if not outdir:
            return
        outdir = Path(outdir)

        picks_csv = outdir / "shot_gather_interactive_picks.csv"
        lines_csv = outdir / "shot_gather_interactive_velocity_lines.csv"

        picks_to_dataframe(self.picks).to_csv(picks_csv, index=False)
        velocity_lines_to_dataframe(self.lines).to_csv(lines_csv, index=False)

        self.status.set(f"Saved {picks_csv.name} and {lines_csv.name}")
        messagebox.showinfo("Saved", f"Saved:\n{picks_csv}\n{lines_csv}")


def main():
    print(f"Starting Shot Gather Wiggle Picker {APP_VERSION}")
    print("Required controls in this version: LOAD DB / GEOMETRY INDEX, Component Z/N/E, station code scale=0.01, SU/SEG-2 filetypes")
    root = tk.Tk()
    root.geometry("1650x950")
    ShotGatherPickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()