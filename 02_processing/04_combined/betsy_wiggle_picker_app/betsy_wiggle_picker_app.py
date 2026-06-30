#!/usr/bin/env python3
"""
Betsy Wiggle Picker standalone app.

A lightweight Tk/matplotlib application for reviewing and picking active-source
shot gathers from SEG-Y files, especially the Betsy Gun nodal/Geode products
exported by notebook 99.

Features
--------
- Loads SEG-Y files directly.
- Reads source/receiver x positions from standard SEG-Y trace headers.
- Optional detrend, taper, and bandpass/highpass filtering.
- Editable filter corners and filter order.
- Wiggle gain, trace scale, clipping, time-window controls.
- Red/blue positive/negative lobe shading.
- Nodal Z/N/E, Geode, and nodal-Z + Geode overlay display modes.
- Pick mode: left click adds pick, right click deletes nearest pick.
- Line mode: two left clicks define a line; velocity is dx/dt.
- Saves picks and velocity lines to CSV.

Dependencies
------------
conda install obspy scipy numpy pandas matplotlib
Python's tkinter is also required. On some systems it is a separate package.
"""

from __future__ import annotations

import csv
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from obspy import read, Stream, UTCDateTime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


DEFAULT_BASE_DIR = Path("/Volumes/tachyon/LBSSP_DATA/betsy_gun_alignment_v1/segy_for_refrapy")


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


def _header_value(header, name: str, default=None):
    try:
        return getattr(header, name)
    except Exception:
        return default


def _coord_scale_from_header(header) -> float:
    scalar = _header_value(header, "scalar_to_be_applied_to_all_coordinates", 1)
    if scalar in (None, 0):
        return 1.0
    scalar = int(scalar)
    if scalar > 0:
        return float(scalar)
    return 1.0 / abs(float(scalar))


def _read_segy_geometry(tr, i: int) -> Tuple[float, float]:
    """
    Return receiver_x_m, source_x_m from SEG-Y trace header.

    Coordinates should have been written as:
      scalar_to_be_applied_to_all_coordinates = -100
      source_coordinate_x = source_x_m * 100
      group_coordinate_x = receiver_x_m * 100
    """
    h = getattr(getattr(tr.stats, "segy", None), "trace_header", None)
    if h is None:
        return float(i), np.nan

    scale = _coord_scale_from_header(h)
    gx = _header_value(h, "group_coordinate_x", None)
    sx = _header_value(h, "source_coordinate_x", None)

    rx = float(i) if gx is None else float(gx) * scale
    src = np.nan if sx is None else float(sx) * scale

    return rx, src


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
                tr.filter(
                    "bandpass",
                    freqmin=float(freqmin),
                    freqmax=float(freqmax),
                    corners=int(corners),
                    zerophase=True,
                )
            else:
                tr.filter(
                    "highpass",
                    freq=float(freqmin),
                    corners=int(corners),
                    zerophase=True,
                )
    return out


def stream_to_rows(
    st: Stream,
    *,
    dataset: str,
    tmin: float,
    tmax: float,
    display_time_shift_s: float = 0.0,
) -> Tuple[List[TraceRow], Optional[float]]:
    rows: List[TraceRow] = []
    sources = []

    for i, tr in enumerate(st):
        rx, sx = _read_segy_geometry(tr, i)
        if np.isfinite(sx):
            sources.append(float(sx))

        dt = float(tr.stats.delta)
        # Exported SEG-Y starts at relative time. Use trace start relative to its
        # own file's earliest start if needed.
        t0 = float(tr.stats.starttime - UTCDateTime(0))
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
    y = np.asarray(y, dtype=float)
    y = y - np.nanmedian(y)
    scale = np.nanpercentile(np.abs(y), clip_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(y))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(y / scale, -1.0, 1.0)


def find_default_files(base: Path = DEFAULT_BASE_DIR) -> Dict[str, Optional[Path]]:
    """
    Find likely files written by notebook 99.
    """
    out = {"nodal_Z": None, "nodal_N": None, "nodal_E": None, "geode": None, "combined": None}
    if not base.exists():
        return out

    patterns = {
        "nodal_Z": "*_nodal_Z_normalized.sgy",
        "nodal_N": "*_nodal_N_normalized.sgy",
        "nodal_E": "*_nodal_E_normalized.sgy",
        "geode": "*_geode_shifted_normalized.sgy",
        "combined": "*_combined_nodal_geode_shifted_normalized.sgy",
    }

    # Original Z export may have been named without _Z_.
    fallback_patterns = {
        "nodal_Z": "*_nodal_normalized.sgy",
    }

    for key, pat in patterns.items():
        matches = sorted(base.glob(pat))
        if matches:
            out[key] = matches[0]
        elif key in fallback_patterns:
            matches = sorted(base.glob(fallback_patterns[key]))
            if matches:
                out[key] = matches[0]

    return out


class BetsyPickerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Betsy Gun Wiggle Picker")

        self.paths: Dict[str, Optional[Path]] = find_default_files()
        self.raw_streams: Dict[str, Optional[Stream]] = {k: None for k in self.paths}
        self.rows_cache: Dict[str, List[TraceRow]] = {}
        self.source_x_m: Optional[float] = None

        self.picks: List[Pick] = []
        self.line_pending: List[Tuple[str, float, float, int, str]] = []
        self.lines: List[VelocityLine] = []

        self.active_rows_by_dataset: Dict[str, List[TraceRow]] = {}
        self.current_mode = tk.StringVar(value="pick")
        self.display_mode = tk.StringVar(value="nodal_Z")
        self.shade_mode = tk.StringVar(value="both")

        self.tmin = tk.DoubleVar(value=0.0)
        self.tmax = tk.DoubleVar(value=0.8)
        self.gain = tk.DoubleVar(value=1.0)
        self.trace_scale = tk.DoubleVar(value=0.75)
        self.clip_percentile = tk.DoubleVar(value=99.0)
        self.geode_shift = tk.DoubleVar(value=0.0)

        self.filter_on = tk.BooleanVar(value=True)
        self.detrend_on = tk.BooleanVar(value=True)
        self.taper_pct = tk.DoubleVar(value=0.02)
        self.freqmin = tk.DoubleVar(value=5.0)
        self.freqmax = tk.DoubleVar(value=150.0)
        self.corners = tk.IntVar(value=4)

        self.status = tk.StringVar(value="Ready")

        self._build_gui()
        self._load_default_files()
        self.redraw()

    def _build_gui(self):
        left = ttk.Frame(self.root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)

        main = ttk.Frame(self.root)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(left, text="SEG-Y files", padding=6)
        file_frame.pack(fill=tk.X, pady=4)

        self.file_labels = {}
        for key in ["nodal_Z", "nodal_N", "nodal_E", "geode", "combined"]:
            row = ttk.Frame(file_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Button(row, text=f"Load {key}", command=lambda k=key: self.load_file_dialog(k)).pack(side=tk.LEFT)
            lab = ttk.Label(row, text=self._short_path(self.paths.get(key)), width=42)
            lab.pack(side=tk.LEFT)
            self.file_labels[key] = lab

        ctrl = ttk.LabelFrame(left, text="Display", padding=6)
        ctrl.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="Dataset").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.display_mode,
            values=["nodal_Z", "nodal_N", "nodal_E", "geode", "nodal_Z_plus_geode", "combined"],
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

        numeric = [
            ("tmin", self.tmin),
            ("tmax", self.tmax),
            ("gain", self.gain),
            ("trace scale", self.trace_scale),
            ("clip %", self.clip_percentile),
            ("Geode shift s", self.geode_shift),
        ]
        for r, (name, var) in enumerate(numeric, start=3):
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
            "Pick mode: left-click adds pick; right-click deletes nearest pick.\\n"
            "Line mode: two left-clicks add velocity line; right-click deletes last line/point.\\n"
            "Overlay: black=nodal Z, red=Geode. Middle-click picks Geode in overlay."
        )
        ttk.Label(left, text=help_text, wraplength=330, justify=tk.LEFT).pack(fill=tk.X, pady=4)

        ttk.Label(left, textvariable=self.status, wraplength=330, foreground="blue").pack(fill=tk.X, pady=4)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, main)
        toolbar.update()

        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)

    def _short_path(self, p: Optional[Path]) -> str:
        if p is None:
            return "(not found)"
        p = Path(p)
        parts = p.parts
        if len(parts) > 4:
            return ".../" + "/".join(parts[-3:])
        return str(p)

    def _load_default_files(self):
        for key, path in self.paths.items():
            if path and path.exists():
                try:
                    self.raw_streams[key] = read(str(path), format="SEGY")
                except Exception as e:
                    print(f"Failed to load {key}: {path}: {e}")

    def load_file_dialog(self, key: str):
        path = filedialog.askopenfilename(
            title=f"Load {key} SEG-Y",
            filetypes=[("SEG-Y", "*.sgy *.segy *.SEG *.SGY"), ("All files", "*.*")],
            initialdir=str(DEFAULT_BASE_DIR if DEFAULT_BASE_DIR.exists() else Path.cwd()),
        )
        if not path:
            return
        try:
            self.paths[key] = Path(path)
            self.raw_streams[key] = read(str(path), format="SEGY")
            self.file_labels[key].configure(text=self._short_path(Path(path)))
            self.status.set(f"Loaded {key}: {path}")
            self.redraw()
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load {path}\\n\\n{e}")

    def processed_rows_for(self, key: str, *, display_shift_s: float = 0.0) -> List[TraceRow]:
        st = self.raw_streams.get(key)
        if st is None:
            return []

        try:
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
            )
            if sx is not None and np.isfinite(sx):
                self.source_x_m = sx
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

        if mode == "nodal_Z_plus_geode":
            nodal_rows = self.processed_rows_for("nodal_Z", display_shift_s=0.0)
            geode_rows = self.processed_rows_for("geode", display_shift_s=float(self.geode_shift.get()))
            self.active_rows_by_dataset["nodal_Z"] = nodal_rows
            self.active_rows_by_dataset["geode"] = geode_rows
            self._draw_rows(nodal_rows, color="black", alpha=0.9, shade=shade)
            self._draw_rows(geode_rows, color="red", alpha=0.55, shade=shade)
            title = f"Betsy overlay: nodal Z black + Geode red | shift {self.geode_shift.get():+.4f}s"
        else:
            rows = self.processed_rows_for(mode, display_shift_s=float(self.geode_shift.get()) if mode == "geode" else 0.0)
            self.active_rows_by_dataset[mode] = rows
            color = "red" if mode == "geode" else "black"
            self._draw_rows(rows, color=color, alpha=0.9, shade=shade)
            title = f"Betsy {mode}"

        sx = self.source_x_m
        if sx is not None and np.isfinite(sx):
            self.ax.axvline(sx, color="green", linestyle="--", linewidth=1.2, label=f"source x={sx:.1f} m")

        self._draw_picks_and_lines()

        self.ax.invert_yaxis()
        self.ax.set_xlabel("Receiver x (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title(f"{title} | {self.current_mode.get()} mode")
        self.ax.grid(True, alpha=0.25)
        if sx is not None and np.isfinite(sx):
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
        if mode == "nodal_Z_plus_geode":
            return ["nodal_Z", "geode"]
        return [mode]

    def _draw_picks_and_lines(self):
        visible = set(self._visible_datasets())

        for p in self.picks:
            if p.dataset not in visible:
                continue
            c = "orange" if p.dataset.startswith("nodal") else "purple"
            self.ax.plot(p.receiver_x_m, p.pick_time_s, "o", color=c, markersize=5)
            self.ax.text(p.receiver_x_m, p.pick_time_s, f"{p.pick_time_s:.3f}", color=c, fontsize=7)

        # Pending line points.
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
        if mode == "nodal_Z_plus_geode":
            # Middle-click picks Geode in overlay; left-click picks nodal_Z.
            if event.button == 2:
                return "geode"
            return "nodal_Z"
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
                        source_x_m=float(self.source_x_m) if self.source_x_m is not None else np.nan,
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
        candidates = [
            (i, p) for i, p in enumerate(self.picks)
            if p.dataset == dataset
        ]
        if not candidates:
            self.status.set("No picks to delete.")
            return
        i, p = min(candidates, key=lambda ip: abs(ip[1].receiver_x_m - x) + 5.0 * abs(ip[1].pick_time_s - t))
        self.picks.pop(i)
        self.status.set(f"Deleted pick: {p.dataset} x={p.receiver_x_m:.2f} t={p.pick_time_s:.4f}")

    def add_velocity_line(self, p1, p2):
        ds1, x1, t1, _, _ = p1
        ds2, x2, t2, _, _ = p2
        dx = x2 - x1
        dt = t2 - t1
        v = dx / dt if abs(dt) > 1e-9 else np.nan
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
            source_x_m=float(self.source_x_m) if self.source_x_m is not None else np.nan,
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

        picks_csv = outdir / "betsy_interactive_picks.csv"
        lines_csv = outdir / "betsy_interactive_velocity_lines.csv"

        pd.DataFrame([asdict(p) for p in self.picks]).to_csv(picks_csv, index=False)
        pd.DataFrame([asdict(ln) for ln in self.lines]).to_csv(lines_csv, index=False)

        self.status.set(f"Saved {picks_csv.name} and {lines_csv.name}")
        messagebox.showinfo("Saved", f"Saved:\\n{picks_csv}\\n{lines_csv}")


def main():
    root = tk.Tk()
    root.geometry("1500x900")
    app = BetsyPickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
