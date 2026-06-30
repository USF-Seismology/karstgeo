#!/usr/bin/env python3
"""
Geometry-aware SEG-Y wiggle picker.

Design goals
------------
* Built directly on lib/segy_tools.
* One unambiguous import path: <repo>/lib.
* SEG-Y headers are the primary geometry source.
* Optional SQLite DB is only used to patch streams whose files lack geometry.
* No legacy geometry-index button/state machine.

Expected repository layout
--------------------------
repo/
  apps/
    segy_wiggle_picker.py
  lib/
    segy_tools/
      __init__.py
      io.py
      gather.py
      db.py        # optional; only needed for DB geometry patching

Run from the repo root:
    python apps/segy_wiggle_picker.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import sys
import traceback
from typing import Optional

import numpy as np

# -----------------------------------------------------------------------------
# Single explicit import rule: app is in repo/apps, package is in repo/lib.
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = REPO_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    from obspy import Stream
    from segy_tools import io as st_io
    from segy_tools.gather import stream_to_gather_arrays
    try:
        from segy_tools import db as st_db
    except ImportError:
        st_db = None
except Exception as exc:  # fail clearly on startup
    raise RuntimeError(
        f"Could not import segy_tools from expected path: {LIB_DIR}\n"
        "Expected to run from repo root with lib/segy_tools installed."
    ) from exc

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


@dataclass
class GatherState:
    label: str
    path: Optional[Path] = None
    stream: Optional[Stream] = None
    db_note: str = ""
    geometry_source: str = "none"
    time_s: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    receiver_x_m: Optional[np.ndarray] = None
    source_x_m: Optional[float] = None
    geom: Optional[dict] = None


@dataclass
class Pick:
    gather_label: str
    receiver_x_m: float
    pick_time_s: float
    trace_index: int
    file_path: str


class SegyWigglePickerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SEG-Y Geometry-Aware Wiggle Picker")
        self.geometry("1350x850")

        self.db_path: Optional[Path] = None
        self.g1 = GatherState("gather_1")
        self.g2 = GatherState("overlay")
        self.picks: list[Pick] = []

        self._build_ui()
        self._connect_events()
        self._log("Ready. Load a SEG-Y/SU/MiniSEED file. DB is optional and only patches missing geometry.")
        self._log(f"Using segy_tools from: {LIB_DIR}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Button(top, text="Load SQLite DB", command=self.load_db).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Gather 1", command=lambda: self.load_gather(self.g1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Overlay", command=lambda: self.load_gather(self.g2)).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Clear Overlay", command=self.clear_overlay).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Redraw", command=self.redraw).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save Picks CSV", command=self.save_picks).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Clear Picks", command=self.clear_picks).pack(side=tk.LEFT, padx=3)

        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)

        self.plot_kind = tk.StringVar(value="wiggle")
        self.component = tk.StringVar(value="All")
        self.sort_by = tk.StringVar(value="receiver_x")
        self.normalize = tk.BooleanVar(value=True)
        self.scale = tk.DoubleVar(value=0.8)
        self.clip_pct = tk.DoubleVar(value=99.0)
        self.tmin = tk.StringVar(value="")
        self.tmax = tk.StringVar(value="")
        self.xmin = tk.StringVar(value="")
        self.xmax = tk.StringVar(value="")

        self._combo(controls, "Plot", self.plot_kind, ["wiggle", "image"])
        self._combo(controls, "Component", self.component, ["All", "Z", "N", "E"])
        self._combo(controls, "Sort", self.sort_by, ["receiver_x", "offset", "trace", "none"])
        ttk.Checkbutton(controls, text="Normalize", variable=self.normalize, command=self.redraw).pack(side=tk.LEFT, padx=8)
        self._entry(controls, "Scale", self.scale, width=6)
        self._entry(controls, "Clip %", self.clip_pct, width=6)
        self._entry(controls, "tmin", self.tmin, width=7)
        self._entry(controls, "tmax", self.tmax, width=7)
        self._entry(controls, "xmin", self.xmin, width=7)
        self._entry(controls, "xmax", self.xmax, width=7)

        status = ttk.Frame(self)
        status.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        self.status_var = tk.StringVar(value="No file loaded")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        main = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig_frame = ttk.Frame(main)
        main.add(fig_frame, weight=5)
        self.fig = Figure(figsize=(11, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        log_frame = ttk.Frame(main)
        main.add(log_frame, weight=1)
        self.log = tk.Text(log_frame, height=9, wrap=tk.WORD)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=yscroll.set)

    def _combo(self, parent, label, variable, values):
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(8, 2))
        cb = ttk.Combobox(parent, textvariable=variable, values=values, width=max(6, max(map(len, values))), state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda _evt: self.redraw())

    def _entry(self, parent, label, variable, width=8):
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(8, 2))
        ent = ttk.Entry(parent, textvariable=variable, width=width)
        ent.pack(side=tk.LEFT)
        ent.bind("<Return>", lambda _evt: self.redraw())

    def _connect_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_db(self):
        path = filedialog.askopenfilename(
            title="Load SQLite database",
            filetypes=[("SQLite DB", "*.sqlite *.sqlite3 *.db"), ("All files", "*")],
        )
        if not path:
            return
        self.db_path = Path(path)
        self._log(f"Loaded DB: {self.db_path}")
        if st_db is None:
            self._log("WARNING: segy_tools.db is not importable, so DB geometry patching is unavailable.")
        # Re-apply DB geometry to already-loaded gathers.
        for gs in (self.g1, self.g2):
            if gs.stream is not None and gs.path is not None:
                self._patch_geometry_from_db(gs)
                self._prepare_arrays(gs)
        self.redraw()

    def load_gather(self, gs: GatherState):
        path = filedialog.askopenfilename(
            title=f"Load {gs.label}",
            filetypes=[
                ("Waveform files", "*.sgy *.segy *.su *.mseed *.msd *.dat *.seg2"),
                ("SEG-Y", "*.sgy *.segy"),
                ("SU", "*.su"),
                ("MiniSEED", "*.mseed *.msd"),
                ("SEG-2 / DAT", "*.dat *.seg2"),
                ("All files", "*"),
            ],
        )
        if not path:
            return
        path = Path(path)
        try:
            st = st_io.read_waveform_file(path)
            if len(st) == 0:
                raise ValueError("File read returned an empty Stream.")
            gs.path = path
            gs.stream = st
            gs.db_note = ""
            gs.geometry_source = "headers/stats"
            self._log(f"Loaded {gs.label}: {path}")
            self._log(f"  traces={len(st)}, sr={st[0].stats.sampling_rate:g} Hz, npts={st[0].stats.npts}")

            if self.db_path is not None:
                self._patch_geometry_from_db(gs)
            self._prepare_arrays(gs)
            self._summarize_geometry(gs)
            self.redraw()
        except Exception as exc:
            self._log_exception(f"Failed to load {path}", exc)
            messagebox.showerror("Load failed", f"{exc}")

    def _patch_geometry_from_db(self, gs: GatherState):
        if self.db_path is None or gs.path is None or gs.stream is None:
            return
        if st_db is None:
            gs.db_note = "segy_tools.db not importable"
            self._log(f"DB geometry for {gs.label}: {gs.db_note}")
            return
        try:
            geom_map, note = st_db.load_db_geometry_for_file(self.db_path, gs.path, len(gs.stream))
            gs.db_note = note
            if geom_map:
                st_io.apply_geometry_map(gs.stream, geom_map, in_place=True)
                gs.geometry_source = "DB patch + headers/stats"
            self._log(f"DB geometry for {gs.label}: {note}")
        except Exception as exc:
            gs.db_note = f"DB geometry lookup failed: {exc}"
            self._log_exception(f"DB geometry lookup failed for {gs.label}", exc)

    def _prepare_arrays(self, gs: GatherState):
        if gs.stream is None:
            return
        component = self.component.get()
        comp_arg = None if component == "All" else component
        try:
            gs.time_s, gs.data, gs.receiver_x_m, gs.source_x_m, gs.geom = stream_to_gather_arrays(
                gs.stream,
                sort_by=self.sort_by.get(),
                component=comp_arg,
                fallback_receiver_spacing_m=1.0,
                fallback_first_receiver_x_m=0.0,
                fallback_source_x_m=np.nan,
            )
        except ValueError as exc:
            # Common for exported single-component SEG-Y with blank channel names.
            if comp_arg is not None and "Empty Stream" in str(exc):
                self._log(f"Component {comp_arg} produced no traces for {gs.label}; using all traces instead.")
                gs.time_s, gs.data, gs.receiver_x_m, gs.source_x_m, gs.geom = stream_to_gather_arrays(
                    gs.stream,
                    sort_by=self.sort_by.get(),
                    component=None,
                    fallback_receiver_spacing_m=1.0,
                    fallback_first_receiver_x_m=0.0,
                    fallback_source_x_m=np.nan,
                )
            else:
                raise

    def clear_overlay(self):
        self.g2 = GatherState("overlay")
        self.redraw()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def redraw(self):
        self.ax.clear()
        try:
            for gs, color, alpha in ((self.g1, "black", 1.0), (self.g2, "tab:blue", 0.7)):
                if gs.stream is None:
                    continue
                self._prepare_arrays(gs)
                if self.plot_kind.get() == "image" and gs is self.g1:
                    self._plot_image(gs)
                else:
                    self._plot_wiggles(gs, color=color, alpha=alpha)
            self._plot_picks()
            self._format_axes()
            self.canvas.draw_idle()
            self._update_status()
        except Exception as exc:
            self._log_exception("Redraw failed", exc)
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Redraw failed:\n{exc}", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()

    def _plot_wiggles(self, gs: GatherState, color="black", alpha=1.0):
        if gs.time_s is None or gs.data is None or gs.receiver_x_m is None:
            return
        t = gs.time_s
        data = np.asarray(gs.data, dtype=float)
        rx = np.asarray(gs.receiver_x_m, dtype=float)
        if data.size == 0 or len(rx) == 0:
            self._log(f"Nothing to plot for {gs.label}: empty data/geometry")
            return

        tmask = self._time_mask(t)
        tplot = t[tmask]
        dplot = data[:, tmask]

        if self.normalize.get():
            denom = np.nanmax(np.abs(dplot), axis=1)
            denom[~np.isfinite(denom) | (denom == 0)] = 1.0
            dplot = dplot / denom[:, None]
        else:
            clip = np.nanpercentile(np.abs(dplot), self._float_var(self.clip_pct, 99.0))
            if np.isfinite(clip) and clip > 0:
                dplot = dplot / clip

        scale = self._float_var(self.scale, 0.8)
        # Scale by median receiver spacing so wiggles occupy sensible x-width.
        if len(rx) > 1:
            diffs = np.diff(np.sort(rx))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            dx = float(np.median(diffs)) if len(diffs) else 1.0
        else:
            dx = 1.0

        for i, x in enumerate(rx):
            if not np.isfinite(x):
                x = float(i)
            self.ax.plot(x + dplot[i] * scale * dx, tplot, color=color, alpha=alpha, linewidth=0.7)

    def _plot_image(self, gs: GatherState):
        if gs.time_s is None or gs.data is None or gs.receiver_x_m is None:
            return
        t = gs.time_s
        rx = np.asarray(gs.receiver_x_m, dtype=float)
        data = np.asarray(gs.data, dtype=float)
        tmask = self._time_mask(t)
        tplot = t[tmask]
        dplot = data[:, tmask]
        clip = np.nanpercentile(np.abs(dplot), self._float_var(self.clip_pct, 98.0))
        if not np.isfinite(clip) or clip <= 0:
            clip = 1.0
        extent = [np.nanmin(rx), np.nanmax(rx), np.nanmax(tplot), np.nanmin(tplot)]
        self.ax.imshow(
            dplot.T,
            aspect="auto",
            extent=extent,
            cmap="seismic",
            vmin=-clip,
            vmax=clip,
            interpolation="nearest",
        )

    def _plot_picks(self):
        for pick in self.picks:
            self.ax.plot(pick.receiver_x_m, pick.pick_time_s, marker="o", markersize=6, color="orange", markeredgecolor="black")
            self.ax.vlines(pick.receiver_x_m, pick.pick_time_s - 0.01, pick.pick_time_s + 0.01, color="orange", linewidth=1.5)

    def _format_axes(self):
        self.ax.set_xlabel("Receiver position x [m]")
        self.ax.set_ylabel("Time [s]")
        self.ax.invert_yaxis()
        self.ax.grid(True, alpha=0.25)
        title_parts = []
        if self.g1.path:
            title_parts.append(self.g1.path.name)
        if self.g2.path:
            title_parts.append(f"overlay: {self.g2.path.name}")
        self.ax.set_title(" | ".join(title_parts) if title_parts else "No gather loaded")

        xmin = self._optional_float(self.xmin.get())
        xmax = self._optional_float(self.xmax.get())
        if xmin is not None or xmax is not None:
            cur = self.ax.get_xlim()
            self.ax.set_xlim(xmin if xmin is not None else cur[0], xmax if xmax is not None else cur[1])

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax or self.g1.receiver_x_m is None or self.g1.path is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        rx = np.asarray(self.g1.receiver_x_m, dtype=float)
        if len(rx) == 0:
            return
        idx = int(np.nanargmin(np.abs(rx - event.xdata)))
        if event.button == 1:
            pick = Pick(
                gather_label=self.g1.label,
                receiver_x_m=float(rx[idx]),
                pick_time_s=float(event.ydata),
                trace_index=idx,
                file_path=str(self.g1.path),
            )
            self.picks.append(pick)
            self._log(f"Pick added: x={pick.receiver_x_m:.3f} m, t={pick.pick_time_s:.5f} s, trace={idx}")
            self.redraw()
        elif event.button in (2, 3):
            self._delete_nearest_pick(float(event.xdata), float(event.ydata))
            self.redraw()

    def _delete_nearest_pick(self, x, t):
        if not self.picks:
            return
        # Normalize time distance so x/t are both relevant.
        xs = np.array([p.receiver_x_m for p in self.picks])
        ts = np.array([p.pick_time_s for p in self.picks])
        xscale = max(np.nanmax(xs) - np.nanmin(xs), 1.0)
        tscale = max(np.nanmax(ts) - np.nanmin(ts), 0.1)
        dist = ((xs - x) / xscale) ** 2 + ((ts - t) / tscale) ** 2
        i = int(np.argmin(dist))
        p = self.picks.pop(i)
        self._log(f"Pick deleted: x={p.receiver_x_m:.3f} m, t={p.pick_time_s:.5f} s")

    def save_picks(self):
        if not self.picks:
            messagebox.showinfo("No picks", "No picks to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save picks CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gather_label", "file_path", "trace_index", "receiver_x_m", "pick_time_s"])
            writer.writeheader()
            for p in self.picks:
                writer.writerow(p.__dict__)
        self._log(f"Saved {len(self.picks)} picks: {path}")

    def clear_picks(self):
        self.picks.clear()
        self.redraw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _time_mask(self, t):
        mask = np.ones_like(t, dtype=bool)
        tmin = self._optional_float(self.tmin.get())
        tmax = self._optional_float(self.tmax.get())
        if tmin is not None:
            mask &= t >= tmin
        if tmax is not None:
            mask &= t <= tmax
        return mask

    @staticmethod
    def _optional_float(s):
        s = str(s).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _float_var(var, default):
        try:
            return float(var.get())
        except Exception:
            return default

    def _summarize_geometry(self, gs: GatherState):
        if gs.receiver_x_m is None:
            self._log(f"{gs.label}: no receiver geometry available")
            return
        rx = np.asarray(gs.receiver_x_m, dtype=float)
        finite = rx[np.isfinite(rx)]
        if len(finite):
            self._log(
                f"{gs.label} geometry: x={finite.min():.3f}..{finite.max():.3f} m, "
                f"n={len(rx)}, source_x={gs.source_x_m} ({gs.geometry_source})"
            )
        else:
            self._log(f"{gs.label}: receiver geometry is all NaN")

    def _update_status(self):
        parts = []
        if self.db_path:
            parts.append(f"DB: {self.db_path.name}")
        if self.g1.path:
            n = len(self.g1.stream) if self.g1.stream else 0
            parts.append(f"G1: {self.g1.path.name} ({n} traces, {self.g1.geometry_source})")
        if self.g2.path:
            n = len(self.g2.stream) if self.g2.stream else 0
            parts.append(f"Overlay: {self.g2.path.name} ({n} traces)")
        parts.append(f"Picks: {len(self.picks)}")
        self.status_var.set("  |  ".join(parts))

    def _log(self, msg: str):
        self.log.insert(tk.END, str(msg).rstrip() + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _log_exception(self, prefix: str, exc: Exception):
        self._log(f"{prefix}: {exc}")
        self._log(traceback.format_exc())


def main():
    app = SegyWigglePickerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
