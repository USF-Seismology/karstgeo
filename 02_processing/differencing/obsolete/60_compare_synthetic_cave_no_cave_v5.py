#!/usr/bin/env python3
"""
60_compare_synthetic_cave_no_cave.py

Compare paired SPECFEM/SEG-Y synthetic shot gathers:

    1. model WITH cave/void
    2. model WITHOUT cave/void
    3. difference = WITH_CAVE - WITHOUT_CAVE

For each matched shot position this script writes:
    - 3-panel gather image: cave / no-cave / difference
    - optional individual wiggle plots
    - Charlie-style frequency-vs-receiver plots for cave / no-cave / difference
    - optional difference SEG-Y
    - CSV summary of matched shots

It is designed for the KarstGeo/SPECFEM2D workflow where shot and receiver
geometry live in the DATA folder:

    DATA/STATIONS
    DATA/SOURCES_LIST.txt

The STATIONS file is expected to be SPECFEM-style:

    station network x z burial elevation

Example:
    G0001    AA       1.0000000      50.0000000       0.0         0.0

Receiver x_m is read from column 3.
Receiver z_m is read from column 4.

The SOURCES_LIST.txt parser is deliberately tolerant because SPECFEM source
lists vary between workflows. It tries to find source x/z positions from either:
    - numeric columns; or
    - key=value style fields such as source_position_x = 87.0

Author: Glenn Thompson / ChatGPT
Date: 2026-06-23
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from obspy import Stream, Trace, read
    from obspy.io.segy.segy import SEGYTraceHeader
except Exception as exc:
    raise SystemExit(
        "This script requires ObsPy. Install with something like:\n"
        "    conda install -c conda-forge obspy\n"
        f"Original import error: {exc}"
    )


# -----------------------------------------------------------------------------
# Geometry containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Station:
    station: str
    network: str
    x_m: float
    z_m: float
    burial_m: float
    elevation_m: float


@dataclass(frozen=True)
class Source:
    source_id: str
    x_m: float
    z_m: Optional[float] = None
    raw_line: str = ""


@dataclass(frozen=True)
class GatherArrays:
    time_s: np.ndarray
    data: np.ndarray                 # shape = n_receivers x n_samples
    receiver_x_m: np.ndarray
    source_x_m: float
    dt_s: float
    source_label: str
    path: Path


@dataclass(frozen=True)
class PairResult:
    source_x_m: float
    source_label: str
    cave_path: Path
    nocave_path: Path
    n_receivers: int
    n_samples: int
    dt_s: float
    max_abs_cave: float
    max_abs_nocave: float
    max_abs_diff: float
    rms_cave: float
    rms_nocave: float
    rms_diff: float
    maxdiff_over_max_cave_pct: float
    maxdiff_over_max_nocave_pct: float
    rmsdiff_over_rms_cave_pct: float
    rmsdiff_over_rms_nocave_pct: float


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")


def _float_tokens(text: str) -> list[float]:
    vals = []
    for m in _FLOAT_RE.finditer(text.replace("D", "E").replace("d", "e")):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            pass
    return vals


def read_stations(path: str | Path) -> list[Station]:
    """Read SPECFEM STATIONS file."""
    path = Path(path)
    stations: list[Station] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: expected at least 4 columns, got: {s}")

            station = parts[0]
            network = parts[1]
            try:
                x_m = float(parts[2])
                z_m = float(parts[3])
                burial_m = float(parts[4]) if len(parts) > 4 else 0.0
                elevation_m = float(parts[5]) if len(parts) > 5 else 0.0
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: could not parse numeric fields: {s}") from exc

            stations.append(
                Station(
                    station=station,
                    network=network,
                    x_m=x_m,
                    z_m=z_m,
                    burial_m=burial_m,
                    elevation_m=elevation_m,
                )
            )

    if not stations:
        raise ValueError(f"No stations found in {path}")

    return stations


def read_sources_list(path: str | Path) -> list[Source]:
    """
    Read a tolerant SOURCES_LIST.txt.

    This handles several common patterns:
      1. one source per line with numeric x z columns
      2. source id followed by x z
      3. key=value style lines containing source_position_x / xs / x

    For pure numeric lines, the first numeric token is interpreted as source x
    unless the line appears to contain an integer source index followed by x.
    """
    path = Path(path)
    sources: list[Source] = []

    if not path.exists():
        return sources

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            # Prefer explicit key-value source_position_x-like fields.
            lower = s.lower()
            x_val = None
            z_val = None

            key_patterns_x = [
                r"(?:source_position_x|xs|x_source|source_x|x)\s*[:=]\s*(" + _FLOAT_RE.pattern + r")",
            ]
            key_patterns_z = [
                r"(?:source_position_z|zs|z_source|source_z|z)\s*[:=]\s*(" + _FLOAT_RE.pattern + r")",
            ]

            for pat in key_patterns_x:
                m = re.search(pat, lower)
                if m:
                    x_val = float(m.group(1).replace("d", "e"))
                    break

            for pat in key_patterns_z:
                m = re.search(pat, lower)
                if m:
                    z_val = float(m.group(1).replace("d", "e"))
                    break

            nums = _float_tokens(s)

            if x_val is None:
                if len(nums) == 0:
                    continue
                if len(nums) >= 2:
                    # If first token is a likely source index and second is plausible x,
                    # use second. Otherwise use first.
                    first = nums[0]
                    if abs(first - round(first)) < 1e-9 and 0 <= first <= 100000 and len(nums) >= 3:
                        x_val = nums[1]
                        z_val = nums[2]
                    else:
                        x_val = nums[0]
                        z_val = nums[1]
                else:
                    x_val = nums[0]

            source_id = f"S{len(sources) + 1:04d}"
            first_word = s.split()[0]
            if not _FLOAT_RE.fullmatch(first_word):
                source_id = first_word

            sources.append(Source(source_id=source_id, x_m=float(x_val), z_m=z_val, raw_line=raw))

    return sources


# -----------------------------------------------------------------------------
# SEG-Y / gather reading
# -----------------------------------------------------------------------------

def _read_stream(path: str | Path) -> Stream:
    """
    Read a SEG-Y/SU/MiniSEED-like file with ObsPy.

    ObsPy read() usually works when the extension is sensible. If it fails,
    try SEG-Y explicitly.
    """
    path = Path(path)
    try:
        return read(str(path))
    except Exception:
        try:
            return read(str(path), format="SEGY")
        except Exception:
            return read(str(path), format="SU")


def _header_attr(trace: Trace, name: str, default=None):
    try:
        return getattr(trace.stats.segy.trace_header, name)
    except Exception:
        return default


def _coord_scalar(trace: Trace) -> float:
    scalar = _header_attr(trace, "scalar_to_be_applied_to_all_coordinates", 1)
    try:
        scalar = int(scalar)
    except Exception:
        scalar = 1

    if scalar == 0:
        return 1.0
    if scalar > 0:
        return float(scalar)
    return 1.0 / abs(float(scalar))


def _maybe_header_coord_m(trace: Trace, names: Iterable[str]) -> Optional[float]:
    scalar = _coord_scalar(trace)
    for name in names:
        val = _header_attr(trace, name, None)
        if val is not None:
            try:
                return float(val) * scalar
            except Exception:
                pass
    return None


def stream_to_arrays(
    st: Stream,
    *,
    stations: list[Station],
    source_x_m: float,
    source_label: str,
    path: Path,
    component: Optional[str] = None,
    trust_segy_geometry: bool = False,
) -> GatherArrays:
    """
    Convert stream to arrays and attach receiver geometry.

    By default this uses DATA/STATIONS for receiver x positions, because
    synthetic SPECFEM outputs often lack useful SEG-Y trace geometry.

    If trust_segy_geometry is True, SEG-Y group_coordinate_x is used when present.
    """
    if component:
        st = Stream([tr for tr in st if str(getattr(tr.stats, "channel", "")).endswith(component)])

    if len(st) == 0:
        raise ValueError(f"No traces found in {path}")

    ntr = len(st)
    if len(stations) < ntr:
        raise ValueError(
            f"{path}: stream has {ntr} traces but STATIONS has only {len(stations)} receivers"
        )

    npts = min(int(tr.stats.npts) for tr in st)
    dt_s = float(st[0].stats.delta)
    time_s = np.arange(npts, dtype=float) * dt_s
    data = np.vstack([np.asarray(tr.data[:npts], dtype=np.float64) for tr in st])

    if trust_segy_geometry:
        rx = []
        for i, tr in enumerate(st):
            val = _maybe_header_coord_m(tr, ["group_coordinate_x"])
            rx.append(stations[i].x_m if val is None else val)
        receiver_x_m = np.asarray(rx, dtype=float)
    else:
        receiver_x_m = np.asarray([sta.x_m for sta in stations[:ntr]], dtype=float)

    order = np.argsort(receiver_x_m)
    return GatherArrays(
        time_s=time_s,
        data=data[order, :],
        receiver_x_m=receiver_x_m[order],
        source_x_m=float(source_x_m),
        dt_s=dt_s,
        source_label=source_label,
        path=Path(path),
    )


def read_gather(
    path: str | Path,
    *,
    stations: list[Station],
    source_x_m: float,
    source_label: str,
    component: Optional[str] = None,
    trust_segy_geometry: bool = False,
) -> GatherArrays:
    st = _read_stream(path)
    return stream_to_arrays(
        st,
        stations=stations,
        source_x_m=source_x_m,
        source_label=source_label,
        path=Path(path),
        component=component,
        trust_segy_geometry=trust_segy_geometry,
    )


# -----------------------------------------------------------------------------
# Matching files to shot positions
# -----------------------------------------------------------------------------

def find_gather_files(directory: str | Path, pattern: str) -> list[Path]:
    """
    Find gather files.

    Important for SPECFEM2D single-source runs:
    the files usually live below model_root/SURVEY_OUTPUT/<shot_folder>/,
    e.g.

        SURVEY_OUTPUT/source_000001/Uz_file_single_v.su
        SURVEY_OUTPUT/source_000001/Ux_file_single_v.su

    Therefore this uses recursive rglob() unless the pattern itself is
    explicitly absolute.
    """
    directory = Path(directory).expanduser()

    pat_path = Path(pattern)
    if pat_path.is_absolute():
        files = sorted(p for p in pat_path.parent.glob(pat_path.name) if p.is_file())
    else:
        files = sorted(p for p in directory.rglob(pattern) if p.is_file())

    if not files:
        raise FileNotFoundError(f"No files matched recursively under {directory} with pattern {pattern!r}")
    return files


# Backward-compatible name used by older code paths.
def find_segy_files(directory: str | Path, pattern: str) -> list[Path]:
    return find_gather_files(directory, pattern)


def infer_position_from_filename(path: str | Path) -> Optional[float]:
    """
    Infer shot x from file path.

    Looks across the filename and parent shot-folder names for tokens such as:
        x087.0m
        x087p0m
        shot_087.0m
        source087.0
        sx_87

    If your shot folders are only numbered sequentially, the script will fall
    back to sorted order and SOURCES_LIST order.
    """
    p = Path(path)
    candidates = [p.name] + [parent.name for parent in p.parents[:4]]
    text = " ".join(candidates).lower().replace("p", ".")
    patterns = [
        r"(?:source|shot|src|sx|x)[_\-]?(-?\d+(?:\.\d+)?)m?",
        r"(-?\d+(?:\.\d+)?)m",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _shot_sort_key(path: Path) -> tuple:
    """
    Stable sort key for SPECFEM shot folders/files.

    Handles names with embedded numbers, otherwise falls back to full path.
    """
    parent = path.parent.name
    nums = _float_tokens(parent)
    if not nums:
        nums = _float_tokens(path.name)
    if nums:
        return (0, nums[0], str(path))
    return (1, str(path))


def match_files_to_sources(
    files: list[Path],
    sources: list[Source],
    *,
    tolerance_m: float,
) -> dict[str, Path]:
    """
    Return source_id -> file by nearest inferred filename position.

    If no source list is available, caller should use pair_files_by_filename().
    """
    inferred: list[tuple[Path, float]] = []
    for f in files:
        x = infer_position_from_filename(f)
        if x is not None:
            inferred.append((f, x))

    if not inferred:
        raise ValueError(
            "Could not infer shot positions from filenames. "
            "Either use filenames containing x###m/source###/shot###, or adapt match_files_to_sources()."
        )

    matched: dict[str, Path] = {}
    used: set[Path] = set()

    for src in sources:
        best = None
        best_dx = np.inf
        for f, x in inferred:
            if f in used:
                continue
            dx = abs(x - src.x_m)
            if dx < best_dx:
                best = f
                best_dx = dx
        if best is not None and best_dx <= tolerance_m:
            matched[src.source_id] = best
            used.add(best)

    return matched


def pair_files_by_filename(cave_files: list[Path], nocave_files: list[Path]) -> list[tuple[str, float, Path, Path]]:
    """
    Fallback pairing when no usable SOURCES_LIST exists.

    Matches by inferred filename shot x. Returns:
        source_label, source_x_m, cave_path, nocave_path
    """
    cave = [(f, infer_position_from_filename(f)) for f in cave_files]
    nocave = [(f, infer_position_from_filename(f)) for f in nocave_files]
    cave = [(f, x) for f, x in cave if x is not None]
    nocave = [(f, x) for f, x in nocave if x is not None]

    if not cave or not nocave:
        # Absolute fallback: sorted order
        n = min(len(cave_files), len(nocave_files))
        return [(f"S{i+1:04d}", float(i + 1), cave_files[i], nocave_files[i]) for i in range(n)]

    pairs = []
    used: set[Path] = set()
    for cf, cx in cave:
        best = None
        best_dx = np.inf
        for nf, nx in nocave:
            if nf in used:
                continue
            dx = abs(cx - nx)
            if dx < best_dx:
                best = nf
                best_dx = dx
        if best is not None:
            used.add(best)
            pairs.append((f"x{cx:.3f}m", float(cx), cf, best))

    return pairs


def build_pairs(
    *,
    cave_dir: Path,
    nocave_dir: Path,
    cave_pattern: str,
    nocave_pattern: str,
    sources: list[Source],
    match_tolerance_m: float,
    pair_mode: str = "auto",
) -> list[tuple[str, float, Path, Path]]:
    """
    Build matched cave/no-cave file pairs.

    pair_mode:
      auto
        Try position matching from filenames/folders. If that fails or produces
        too few pairs, use sorted file order paired with SOURCES_LIST order.
      position
        Require filename/folder shot-position matching.
      order
        Pair sorted cave files and sorted no-cave files. If SOURCES_LIST exists,
        source positions are taken from the corresponding source-list row.

    This is the right default for SPECFEM2D runs where the actual files are:

        CAVE_MODEL/SURVEY_OUTPUT/<single-shot-folder>/Uz_file_single_v.su
        NO_CAVE_MODEL/SURVEY_OUTPUT/<single-shot-folder>/Uz_file_single_v.su

    and the source positions are defined by DATA/SOURCES_LIST.txt.
    """
    cave_files = sorted(find_gather_files(cave_dir, cave_pattern), key=_shot_sort_key)
    nocave_files = sorted(find_gather_files(nocave_dir, nocave_pattern), key=_shot_sort_key)

    print(f"Found {len(cave_files)} cave gather files using pattern {cave_pattern!r}")
    print(f"Found {len(nocave_files)} no-cave gather files using pattern {nocave_pattern!r}")

    if pair_mode not in {"auto", "position", "order"}:
        raise ValueError(f"Unsupported pair_mode={pair_mode!r}; use auto, position, or order")

    if pair_mode in {"auto", "position"} and sources:
        try:
            cave_map = match_files_to_sources(cave_files, sources, tolerance_m=match_tolerance_m)
            nocave_map = match_files_to_sources(nocave_files, sources, tolerance_m=match_tolerance_m)

            pairs = []
            for src in sources:
                cf = cave_map.get(src.source_id)
                nf = nocave_map.get(src.source_id)
                if cf is not None and nf is not None:
                    pairs.append((src.source_id, src.x_m, cf, nf))

            if pairs and (pair_mode == "position" or len(pairs) >= min(len(sources), len(cave_files), len(nocave_files)) // 2):
                return pairs
        except Exception as exc:
            if pair_mode == "position":
                raise
            print(f"Position-based matching was not usable; falling back to sorted order. Reason: {exc}")

    if pair_mode == "position":
        return pair_files_by_filename(cave_files, nocave_files)

    # Sorted-order pairing. This is usually correct for SPECFEM multiple
    # single-source folders when both model runs were generated from the same
    # SOURCES_LIST.txt.
    n = min(len(cave_files), len(nocave_files))
    if sources:
        n = min(n, len(sources))

    pairs: list[tuple[str, float, Path, Path]] = []
    for i in range(n):
        if sources:
            src = sources[i]
            source_label = src.source_id
            source_x_m = src.x_m
        else:
            inferred = infer_position_from_filename(cave_files[i])
            source_x_m = float(inferred if inferred is not None else i + 1)
            source_label = f"S{i+1:04d}"
        pairs.append((source_label, source_x_m, cave_files[i], nocave_files[i]))

    return pairs


# -----------------------------------------------------------------------------
# Alignment / differencing
# -----------------------------------------------------------------------------

def align_pair(
    a: GatherArrays,
    b: GatherArrays,
    *,
    receiver_tolerance_m: float = 0.05,
    time_tolerance_s: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two gathers by nearest receiver x.

    Returns:
        time_s, data_a_aligned, data_b_aligned, receiver_x_m
    """
    npts = min(a.data.shape[1], b.data.shape[1], len(a.time_s), len(b.time_s))
    if np.max(np.abs(a.time_s[:npts] - b.time_s[:npts])) > time_tolerance_s:
        raise ValueError(
            f"Time axes differ for {a.path.name} and {b.path.name}. "
            "Resampling is not yet enabled."
        )

    pairs = []
    used_b = set()
    for ia, rx in enumerate(a.receiver_x_m):
        ib = int(np.argmin(np.abs(b.receiver_x_m - rx)))
        if ib in used_b:
            continue
        if abs(b.receiver_x_m[ib] - rx) <= receiver_tolerance_m:
            pairs.append((ia, ib))
            used_b.add(ib)

    if not pairs:
        raise ValueError(f"No common receivers found for {a.path.name} and {b.path.name}")

    ia = np.asarray([p[0] for p in pairs], dtype=int)
    ib = np.asarray([p[1] for p in pairs], dtype=int)

    return a.time_s[:npts], a.data[ia, :npts], b.data[ib, :npts], a.receiver_x_m[ia]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _clip_value(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    vals = np.concatenate([np.ravel(np.abs(a[np.isfinite(a)])) for a in arrays if a.size])
    if vals.size == 0:
        return 1.0
    c = float(np.percentile(vals, percentile))
    return c if c > 0 else 1.0


def plot_three_panel_image(
    *,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    cave: np.ndarray,
    nocave: np.ndarray,
    diff: np.ndarray,
    source_x_m: float,
    cave_label: str,
    nocave_label: str,
    outfile: Path,
    tmin: Optional[float],
    tmax: Optional[float],
    clip_percentile: float,
    cave_x_m: Optional[tuple[float, float]] = None,
    difference_metrics: Optional[dict] = None,
    suptitle_extra: str = "",
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    datasets = [
        (cave, cave_label),
        (nocave, nocave_label),
        (diff, "Difference: cave - no cave"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    extent = [receiver_x_m.min(), receiver_x_m.max(), time_s.max(), time_s.min()]
    clim_main = _clip_value(cave, nocave, percentile=clip_percentile)
    clim_diff = _clip_value(diff, percentile=clip_percentile)

    for ax, (data, title) in zip(axes, datasets):
        clim = clim_diff if "Difference" in title else clim_main
        im = ax.imshow(
            data.T,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="seismic",
            vmin=-clim,
            vmax=clim,
        )
        ax.axvline(source_x_m, color="k", lw=1.0, ls="--", alpha=0.7, label="source")
        if cave_x_m is not None:
            ax.axvspan(cave_x_m[0], cave_x_m[1], color="0.5", alpha=0.15, label="cave")
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("amplitude")

    if difference_metrics is not None:
        txt = (
            f"max|diff| / max|with cave| = {_fmt_pct(difference_metrics.get('maxdiff_over_max_a_pct', np.nan))}\n"
            f"max|diff| / max|no cave|   = {_fmt_pct(difference_metrics.get('maxdiff_over_max_b_pct', np.nan))}\n"
            f"rms(diff) / rms(with cave) = {_fmt_pct(difference_metrics.get('rmsdiff_over_rms_a_pct', np.nan))}\n"
            f"rms(diff) / rms(no cave)   = {_fmt_pct(difference_metrics.get('rmsdiff_over_rms_b_pct', np.nan))}"
        )
        axes[2].text(
            0.02,
            0.98,
            txt,
            transform=axes[2].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.6"),
        )

    axes[0].set_ylabel("time (s)")
    if tmin is not None or tmax is not None:
        lo = time_s.min() if tmin is None else tmin
        hi = time_s.max() if tmax is None else tmax
        for ax in axes:
            ax.set_ylim(hi, lo)

    fig.suptitle(f"Synthetic shot comparison, source x = {source_x_m:.3f} m{suptitle_extra}", y=0.99)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_wiggle(
    *,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    data: np.ndarray,
    source_x_m: float,
    title: str,
    outfile: Path,
    tmin: Optional[float],
    tmax: Optional[float],
    normalize_traces: bool = True,
    scale: float = 0.45,
    clip_percentile: float = 99.0,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    d = np.asarray(data, dtype=float).copy()
    if normalize_traces:
        mx = np.nanmax(np.abs(d), axis=1)
        mx[mx == 0] = 1.0
        d = d / mx[:, None]
    else:
        c = _clip_value(d, percentile=clip_percentile)
        d = np.clip(d / c, -1, 1)

    dx = np.nanmedian(np.diff(np.sort(receiver_x_m))) if len(receiver_x_m) > 1 else 1.0
    amp_scale = scale * dx

    for tr, x in zip(d, receiver_x_m):
        y = tr * amp_scale + x
        ax.plot(y, time_s, lw=0.45, color="k")
        ax.fill_betweenx(time_s, x, y, where=(y >= x), color="k", alpha=0.25, linewidth=0)

    ax.axvline(source_x_m, color="r", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    ax.invert_yaxis()

    if tmin is not None or tmax is not None:
        lo = time_s.min() if tmin is None else tmin
        hi = time_s.max() if tmax is None else tmax
        ax.set_ylim(hi, lo)

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def frequency_offset_fft(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    max_freq_hz: float,
    normalize_per_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charlie-style frequency plot product:
        receiver x on horizontal axis
        frequency on vertical axis
        FFT amplitude as image value
    """
    dt = float(np.median(np.diff(time_s)))
    freqs = np.fft.rfftfreq(data.shape[1], d=dt)
    amps = np.abs(np.fft.rfft(data, axis=1))

    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    amps = amps[:, keep]

    if normalize_per_trace:
        mx = np.nanmax(amps, axis=1)
        mx[mx == 0] = 1.0
        amps = amps / mx[:, None]

    return receiver_x_m, freqs, amps


def plot_frequency_offset(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    source_x_m: float,
    title: str,
    outfile: Path,
    max_freq_hz: float,
    clip_percentile: float = 99.0,
    normalize_per_trace: bool = True,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    x, f, amp = frequency_offset_fft(
        data=data,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        max_freq_hz=max_freq_hz,
        normalize_per_trace=normalize_per_trace,
    )

    extent = [x.min(), x.max(), f.min(), f.max()]
    clim = _clip_value(amp, percentile=clip_percentile)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        amp.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=clim,
    )
    ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("normalized FFT amplitude" if normalize_per_trace else "FFT amplitude")
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Optional SEG-Y output
# -----------------------------------------------------------------------------


def _pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a = a[:n] - np.nanmean(a[:n])
    b = b[:n] - np.nanmean(b[:n])
    den = np.sqrt(np.nansum(a * a) * np.nansum(b * b))
    return float(np.nansum(a * b) / den) if den else np.nan


def _lsq_scale(reference_trace, comparison_trace):
    """
    Scale factor a such that:
        reference_trace ~= a * comparison_trace
    """
    r = np.asarray(reference_trace, dtype=float)
    c = np.asarray(comparison_trace, dtype=float)
    n = min(r.size, c.size)
    r = r[:n]
    c = c[:n]
    den = float(np.nansum(c * c))
    return float(np.nansum(r * c) / den) if den else np.nan


def _local_window_mask(t, center_s, halfwidth_s):
    keep = np.abs(t - center_s) <= halfwidth_s
    if not np.any(keep):
        i = int(np.argmin(np.abs(t - center_s)))
        keep = np.zeros_like(t, dtype=bool)
        keep[i] = True
    return keep


def compute_trace_peak_scaling(
    time_s,
    receiver_x_m,
    reference,
    comparison,
    *,
    reference_name,
    comparison_name,
    scale_tmin,
    scale_tmax,
    halfwidth_s,
):
    """
    Per-trace amplitude diagnostics for two synthetic gathers.

    reference is the target gather, e.g. WITH_CAVE.
    comparison is the gather being scaled, e.g. WITHOUT_CAVE.

    Report scale factors that multiply comparison to match reference:
      - positive peak ratio
      - negative peak ratio
      - mean/median of positive and negative ratios
      - LSQ scale in a small window around strongest reference peak
      - correlation in the full scale window and peak-centered window
    """
    rows = []
    keep = np.ones_like(time_s, dtype=bool)
    if scale_tmin is not None:
        keep &= time_s >= scale_tmin
    if scale_tmax is not None:
        keep &= time_s <= scale_tmax
    if not np.any(keep):
        keep = np.ones_like(time_s, dtype=bool)

    t_win = time_s[keep]

    for i, x in enumerate(receiver_x_m):
        ref = np.asarray(reference[i, keep], dtype=float)
        cmp = np.asarray(comparison[i, keep], dtype=float)

        ref_pos = float(np.nanmax(ref)) if ref.size else np.nan
        ref_neg = float(np.nanmin(ref)) if ref.size else np.nan
        cmp_pos = float(np.nanmax(cmp)) if cmp.size else np.nan
        cmp_neg = float(np.nanmin(cmp)) if cmp.size else np.nan

        scale_pos = ref_pos / cmp_pos if np.isfinite(ref_pos) and np.isfinite(cmp_pos) and cmp_pos != 0 else np.nan
        scale_neg = ref_neg / cmp_neg if np.isfinite(ref_neg) and np.isfinite(cmp_neg) and cmp_neg != 0 else np.nan

        vals = [v for v in (scale_pos, scale_neg) if np.isfinite(v)]
        scale_peak_mean = float(np.mean(vals)) if vals else np.nan
        scale_peak_median = float(np.median(vals)) if vals else np.nan

        if ref.size:
            j = int(np.nanargmax(np.abs(ref)))
            t_peak = float(t_win[j])
            pkeep = _local_window_mask(t_win, t_peak, halfwidth_s)
            scale_peak_lsq = _lsq_scale(ref[pkeep], cmp[pkeep])
            corr_peak_window = _pearson_corr(ref[pkeep], cmp[pkeep])
            corr_scale_window = _pearson_corr(ref, cmp)
        else:
            t_peak = np.nan
            scale_peak_lsq = np.nan
            corr_peak_window = np.nan
            corr_scale_window = np.nan

        ref_rms = rms(ref)
        cmp_rms = rms(cmp)

        rows.append({
            "trace_index_1based": i + 1,
            "receiver_x_m": float(x),
            "reference_name": reference_name,
            "comparison_name": comparison_name,
            "scale_window_tmin_s": scale_tmin,
            "scale_window_tmax_s": scale_tmax,
            "peak_halfwidth_s": float(halfwidth_s),
            "reference_peak_time_s": t_peak,
            "reference_pos_peak": ref_pos,
            "reference_neg_peak": ref_neg,
            "comparison_pos_peak": cmp_pos,
            "comparison_neg_peak": cmp_neg,
            "scale_pos_peak_reference_over_comparison": scale_pos,
            "scale_neg_peak_reference_over_comparison": scale_neg,
            "scale_peak_mean_reference_over_comparison": scale_peak_mean,
            "scale_peak_median_reference_over_comparison": scale_peak_median,
            "scale_peak_lsq_reference_over_comparison": scale_peak_lsq,
            "scale_rms_reference_over_comparison": ref_rms / cmp_rms if cmp_rms else np.nan,
            "corr_peak_window": corr_peak_window,
            "corr_scale_window": corr_scale_window,
            "reference_rms_scale_window": ref_rms,
            "comparison_rms_scale_window": cmp_rms,
        })

    return rows


def write_trace_peak_scaling_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, (np.floating, np.integer)):
                    v = v.item()
                if isinstance(v, float):
                    clean[k] = "" if not np.isfinite(v) else f"{v:.10g}"
                else:
                    clean[k] = v
            w.writerow(clean)


def plot_overlay_wiggles(
    *,
    time_s,
    receiver_x_m,
    reference,
    comparison,
    source_x_m,
    outfile,
    reference_label,
    comparison_label,
    reference_color="red",
    comparison_color="blue",
    normalize="pair",
    wiggle_scale=0.45,
    title_extra="",
):
    """
    Overlay two wiggle gathers.

    Default colors mirror the real-vs-synthetic plot convention:
      comparison = blue
      reference  = red

    Negative lobes are shaded transparently in each trace's color.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    ref = np.asarray(reference, dtype=float).copy()
    cmp = np.asarray(comparison, dtype=float).copy()

    if normalize == "pair":
        den = np.maximum(np.nanmax(np.abs(ref), axis=1), np.nanmax(np.abs(cmp), axis=1))
        den = np.asarray(den, dtype=float)
        den[~np.isfinite(den) | (den == 0)] = 1.0
        ref = ref / den[:, None]
        cmp = cmp / den[:, None]
    elif normalize == "trace":
        rden = np.nanmax(np.abs(ref), axis=1)
        cden = np.nanmax(np.abs(cmp), axis=1)
        rden[~np.isfinite(rden) | (rden == 0)] = 1.0
        cden[~np.isfinite(cden) | (cden == 0)] = 1.0
        ref = ref / rden[:, None]
        cmp = cmp / cden[:, None]
    elif normalize == "none":
        den = np.nanmax(np.abs(np.concatenate([ref.ravel(), cmp.ravel()])))
        den = den if np.isfinite(den) and den != 0 else 1.0
        ref = ref / den
        cmp = cmp / den
    else:
        raise ValueError(f"Unknown overlay normalization: {normalize}")

    dx = np.nanmedian(np.diff(np.sort(receiver_x_m))) if len(receiver_x_m) > 1 else 1.0
    amp = wiggle_scale * dx

    fig, ax = plt.subplots(figsize=(13, 7))

    # Draw comparison first, then reference on top.
    for tr, x in zip(cmp, receiver_x_m):
        y = x + tr * amp
        ax.plot(y, time_s, color=comparison_color, lw=0.55, alpha=0.75)
        ax.fill_betweenx(time_s, x, y, where=(tr < 0), color=comparison_color, alpha=0.18, linewidth=0)

    for tr, x in zip(ref, receiver_x_m):
        y = x + tr * amp
        ax.plot(y, time_s, color=reference_color, lw=0.55, alpha=0.75)
        ax.fill_betweenx(time_s, x, y, where=(tr < 0), color=reference_color, alpha=0.18, linewidth=0)

    ax.axvline(source_x_m, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Overlay wiggles: {comparison_label} blue, {reference_label} red{title_extra}")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=comparison_color, lw=1.2, label=comparison_label),
        Line2D([0], [0], color=reference_color, lw=1.2, label=reference_label),
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)




def trace_normalize(data: np.ndarray, method: str = "rms", eps: float = 1e-20) -> np.ndarray:
    """
    Normalize each trace independently.

    method:
      rms    -> divide each trace by RMS
      maxabs -> divide each trace by max(abs(trace))

    This is intended for morphology/timing/frequency comparison, not for
    physical-amplitude interpretation.
    """
    d = np.asarray(data, dtype=float).copy()
    if d.ndim != 2:
        return d

    if method == "rms":
        den = np.sqrt(np.nanmean(d * d, axis=1))
    elif method == "maxabs":
        den = np.nanmax(np.abs(d), axis=1)
    else:
        raise ValueError(f"Unknown trace normalization method: {method}")

    den = np.asarray(den, dtype=float)
    den[~np.isfinite(den) | (den < eps)] = 1.0
    return d / den[:, None]


def relative_difference_metrics(a: np.ndarray, b: np.ndarray, diff: np.ndarray) -> dict:
    """
    Return difference amplitude metrics relative to the two parent gathers.

    Here a and b are usually cave and no-cave.
    """
    max_a = float(np.nanmax(np.abs(a))) if a.size else np.nan
    max_b = float(np.nanmax(np.abs(b))) if b.size else np.nan
    max_diff = float(np.nanmax(np.abs(diff))) if diff.size else np.nan
    rms_a = rms(a)
    rms_b = rms(b)
    rms_diff = rms(diff)

    return {
        "max_abs_a": max_a,
        "max_abs_b": max_b,
        "max_abs_diff": max_diff,
        "rms_a": rms_a,
        "rms_b": rms_b,
        "rms_diff": rms_diff,
        "maxdiff_over_max_a_pct": 100.0 * max_diff / max_a if max_a else np.nan,
        "maxdiff_over_max_b_pct": 100.0 * max_diff / max_b if max_b else np.nan,
        "rmsdiff_over_rms_a_pct": 100.0 * rms_diff / rms_a if rms_a else np.nan,
        "rmsdiff_over_rms_b_pct": 100.0 * rms_diff / rms_b if rms_b else np.nan,
    }


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}%"

def write_difference_segy(
    *,
    outfile: Path,
    diff: np.ndarray,
    dt_s: float,
    receiver_x_m: np.ndarray,
    source_x_m: float,
) -> None:
    """
    Write difference gather as SEG-Y.

    This is intentionally simple. If your segy_tools package is importable,
    you may prefer to replace this with segy_tools.gather.gather_arrays_to_stream
    plus segy_tools.io.write_segy.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    st = Stream()
    for i, (x, trdata) in enumerate(zip(receiver_x_m, diff), start=1):
        tr = Trace(data=np.asarray(trdata, dtype=np.float32))
        tr.stats.delta = float(dt_s)
        tr.stats.network = "SY"
        tr.stats.station = f"D{i:04d}"
        tr.stats.channel = "Z"
        tr.stats.receiver_x_m = float(x)
        tr.stats.source_x_m = float(source_x_m)

        tr.stats.segy = {}
        tr.stats.segy.trace_header = SEGYTraceHeader()
        tr.stats.segy.trace_header.trace_sequence_number_within_line = i
        tr.stats.segy.trace_header.trace_number_within_the_original_field_record = i
        tr.stats.segy.trace_header.original_field_record_number = int(round(source_x_m * 100))
        tr.stats.segy.trace_header.energy_source_point_number = int(round(source_x_m * 100))
        tr.stats.segy.trace_header.scalar_to_be_applied_to_all_coordinates = -1000
        tr.stats.segy.trace_header.source_coordinate_x = int(round(source_x_m * 1000))
        tr.stats.segy.trace_header.group_coordinate_x = int(round(float(x) * 1000))
        tr.stats.segy.trace_header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = int(round((float(x) - source_x_m) * 1000))
        st.append(tr)

    st.write(str(outfile), format="SEGY", data_encoding=1)


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x * x)))


def safe_shot_dirname(source_x_m: float, source_label: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_label))
    return f"{label}_x{source_x_m:08.3f}m".replace(".", "p")


def process_pair(
    *,
    source_label: str,
    source_x_m: float,
    cave_path: Path,
    nocave_path: Path,
    stations: list[Station],
    output_dir: Path,
    component: Optional[str],
    trust_segy_geometry: bool,
    receiver_tolerance_m: float,
    tmin: Optional[float],
    tmax: Optional[float],
    max_freq_hz: float,
    write_diff_segy: bool,
    write_individual_wiggles: bool,
    normalize_frequency_per_trace: bool,
    cave_extent_x_m: Optional[tuple[float, float]],
    write_overlay_wiggles: bool,
    overlay_normalize: str,
    overlay_wiggle_scale: float,
    peak_scale_tmin: Optional[float],
    peak_scale_tmax: Optional[float],
    peak_scale_halfwidth_s: float,
    write_trace_normalized_figures: bool,
    trace_normalize_method: str,
) -> tuple[PairResult, list[dict]]:
    cave_g = read_gather(
        cave_path,
        stations=stations,
        source_x_m=source_x_m,
        source_label=source_label,
        component=component,
        trust_segy_geometry=trust_segy_geometry,
    )
    nocave_g = read_gather(
        nocave_path,
        stations=stations,
        source_x_m=source_x_m,
        source_label=source_label,
        component=component,
        trust_segy_geometry=trust_segy_geometry,
    )

    time_s, cave, nocave, receiver_x_m = align_pair(
        cave_g,
        nocave_g,
        receiver_tolerance_m=receiver_tolerance_m,
    )
    diff = cave - nocave
    diff_metrics = relative_difference_metrics(cave, nocave, diff)

    shot_dir = output_dir / safe_shot_dirname(source_x_m, source_label)
    shot_dir.mkdir(parents=True, exist_ok=True)

    plot_three_panel_image(
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        cave=cave,
        nocave=nocave,
        diff=diff,
        source_x_m=source_x_m,
        cave_label="Synthetic WITH cave/void",
        nocave_label="Synthetic WITHOUT cave/void",
        outfile=shot_dir / "comparison_image_cave_nocave_difference.png",
        tmin=tmin,
        tmax=tmax,
        clip_percentile=99.0,
        cave_x_m=cave_extent_x_m,
        difference_metrics=diff_metrics,
        suptitle_extra="; physical amplitudes",
    )

    if write_trace_normalized_figures:
        cave_norm = trace_normalize(cave, method=trace_normalize_method)
        nocave_norm = trace_normalize(nocave, method=trace_normalize_method)
        diff_norm = cave_norm - nocave_norm
        diff_norm_metrics = relative_difference_metrics(cave_norm, nocave_norm, diff_norm)
        plot_three_panel_image(
            time_s=time_s,
            receiver_x_m=receiver_x_m,
            cave=cave_norm,
            nocave=nocave_norm,
            diff=diff_norm,
            source_x_m=source_x_m,
            cave_label=f"Synthetic WITH cave/void, trace-normalized ({trace_normalize_method})",
            nocave_label=f"Synthetic WITHOUT cave/void, trace-normalized ({trace_normalize_method})",
            outfile=shot_dir / "comparison_image_cave_nocave_difference_trace_normalized.png",
            tmin=tmin,
            tmax=tmax,
            clip_percentile=99.0,
            cave_x_m=cave_extent_x_m,
            difference_metrics=diff_norm_metrics,
            suptitle_extra=f"; trace-normalized ({trace_normalize_method})",
        )

    if write_individual_wiggles:
        plot_wiggle(
            time_s=time_s,
            receiver_x_m=receiver_x_m,
            data=cave,
            source_x_m=source_x_m,
            title=f"WITH cave, source x = {source_x_m:.3f} m",
            outfile=shot_dir / "wiggle_with_cave.png",
            tmin=tmin,
            tmax=tmax,
        )
        plot_wiggle(
            time_s=time_s,
            receiver_x_m=receiver_x_m,
            data=nocave,
            source_x_m=source_x_m,
            title=f"WITHOUT cave, source x = {source_x_m:.3f} m",
            outfile=shot_dir / "wiggle_without_cave.png",
            tmin=tmin,
            tmax=tmax,
        )
        plot_wiggle(
            time_s=time_s,
            receiver_x_m=receiver_x_m,
            data=diff,
            source_x_m=source_x_m,
            title=f"Difference: cave - no cave, source x = {source_x_m:.3f} m",
            outfile=shot_dir / "wiggle_difference_cave_minus_nocave.png",
            tmin=tmin,
            tmax=tmax,
        )

    peak_rows = compute_trace_peak_scaling(
        time_s,
        receiver_x_m,
        cave,
        nocave,
        reference_name="with_cave",
        comparison_name="without_cave",
        scale_tmin=peak_scale_tmin,
        scale_tmax=peak_scale_tmax,
        halfwidth_s=peak_scale_halfwidth_s,
    )
    for row in peak_rows:
        row.update({
            "source_label": source_label,
            "source_x_m": source_x_m,
            "cave_file": str(cave_path),
            "nocave_file": str(nocave_path),
        })
    write_trace_peak_scaling_csv(shot_dir / "trace_peak_scaling_factors.csv", peak_rows)

    if write_overlay_wiggles:
        plot_overlay_wiggles(
            time_s=time_s,
            receiver_x_m=receiver_x_m,
            reference=cave,
            comparison=nocave,
            source_x_m=source_x_m,
            outfile=shot_dir / "wiggle_overlay_nocave_blue_cave_red.png",
            reference_label="WITH cave",
            comparison_label="WITHOUT cave",
            reference_color="red",
            comparison_color="blue",
            normalize=overlay_normalize,
            wiggle_scale=overlay_wiggle_scale,
            title_extra=f", source x={source_x_m:.3f} m",
        )

    plot_frequency_offset(
        data=cave,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        title=f"Frequency vs receiver: WITH cave, source x = {source_x_m:.3f} m",
        outfile=shot_dir / "frequency_receiver_with_cave.png",
        max_freq_hz=max_freq_hz,
        normalize_per_trace=normalize_frequency_per_trace,
    )
    plot_frequency_offset(
        data=nocave,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        title=f"Frequency vs receiver: WITHOUT cave, source x = {source_x_m:.3f} m",
        outfile=shot_dir / "frequency_receiver_without_cave.png",
        max_freq_hz=max_freq_hz,
        normalize_per_trace=normalize_frequency_per_trace,
    )
    plot_frequency_offset(
        data=diff,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        title=f"Frequency vs receiver: Difference, source x = {source_x_m:.3f} m",
        outfile=shot_dir / "frequency_receiver_difference.png",
        max_freq_hz=max_freq_hz,
        normalize_per_trace=normalize_frequency_per_trace,
    )

    if write_diff_segy:
        write_difference_segy(
            outfile=shot_dir / "difference_cave_minus_nocave.sgy",
            diff=diff,
            dt_s=float(np.median(np.diff(time_s))),
            receiver_x_m=receiver_x_m,
            source_x_m=source_x_m,
        )

    result = PairResult(
        source_x_m=source_x_m,
        source_label=source_label,
        cave_path=cave_path,
        nocave_path=nocave_path,
        n_receivers=int(diff.shape[0]),
        n_samples=int(diff.shape[1]),
        dt_s=float(np.median(np.diff(time_s))),
        max_abs_cave=float(np.nanmax(np.abs(cave))),
        max_abs_nocave=float(np.nanmax(np.abs(nocave))),
        max_abs_diff=float(np.nanmax(np.abs(diff))),
        rms_cave=rms(cave),
        rms_nocave=rms(nocave),
        rms_diff=rms(diff),
        maxdiff_over_max_cave_pct=diff_metrics["maxdiff_over_max_a_pct"],
        maxdiff_over_max_nocave_pct=diff_metrics["maxdiff_over_max_b_pct"],
        rmsdiff_over_rms_cave_pct=diff_metrics["rmsdiff_over_rms_a_pct"],
        rmsdiff_over_rms_nocave_pct=diff_metrics["rmsdiff_over_rms_b_pct"],
    )

    return result, peak_rows



def write_summary_csv(path: Path, results: list[PairResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_label",
        "source_x_m",
        "cave_path",
        "nocave_path",
        "n_receivers",
        "n_samples",
        "dt_s",
        "max_abs_cave",
        "max_abs_nocave",
        "max_abs_diff",
        "rms_cave",
        "rms_nocave",
        "rms_diff",
        "maxdiff_over_max_cave_pct",
        "maxdiff_over_max_nocave_pct",
        "rmsdiff_over_rms_cave_pct",
        "rmsdiff_over_rms_nocave_pct",
        "diff_over_cave_rms",
        "diff_over_nocave_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "source_label": r.source_label,
                    "source_x_m": f"{r.source_x_m:.6f}",
                    "cave_path": str(r.cave_path),
                    "nocave_path": str(r.nocave_path),
                    "n_receivers": r.n_receivers,
                    "n_samples": r.n_samples,
                    "dt_s": f"{r.dt_s:.9g}",
                    "max_abs_cave": f"{r.max_abs_cave:.9g}",
                    "max_abs_nocave": f"{r.max_abs_nocave:.9g}",
                    "max_abs_diff": f"{r.max_abs_diff:.9g}",
                    "rms_cave": f"{r.rms_cave:.9g}",
                    "rms_nocave": f"{r.rms_nocave:.9g}",
                    "rms_diff": f"{r.rms_diff:.9g}",
                    "maxdiff_over_max_cave_pct": f"{r.maxdiff_over_max_cave_pct:.9g}",
                    "maxdiff_over_max_nocave_pct": f"{r.maxdiff_over_max_nocave_pct:.9g}",
                    "rmsdiff_over_rms_cave_pct": f"{r.rmsdiff_over_rms_cave_pct:.9g}",
                    "rmsdiff_over_rms_nocave_pct": f"{r.rmsdiff_over_rms_nocave_pct:.9g}",
                    "diff_over_cave_rms": f"{r.rms_diff / r.rms_cave:.9g}" if r.rms_cave else "",
                    "diff_over_nocave_rms": f"{r.rms_diff / r.rms_nocave:.9g}" if r.rms_nocave else "",
                }
            )


def parse_cave_extent(text: Optional[str]) -> Optional[tuple[float, float]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--cave-extent-x-m must be like 120,135")
    a, b = float(parts[0]), float(parts[1])
    return (min(a, b), max(a, b))



def resolve_data_dir(
    *,
    requested_data_dir: Path,
    cave_dir: Path,
    nocave_dir: Path,
    stations_file: str,
    sources_file: str,
) -> Path:
    """
    Find the DATA directory containing STATIONS/SOURCES_LIST.txt.

    Search order:
      1. requested --data-dir
      2. cave_dir/DATA
      3. nocave_dir/DATA
      4. sibling model folders' DATA directories
      5. recursive search below the SOURCES_GROUNDED parent
    """
    requested_data_dir = Path(requested_data_dir)
    cave_dir = Path(cave_dir)
    nocave_dir = Path(nocave_dir)

    def has_stations(d: Path) -> bool:
        return (d / stations_file).exists()

    candidates = [
        requested_data_dir,
        cave_dir / "DATA",
        nocave_dir / "DATA",
    ]

    parent = cave_dir.parent
    if parent.exists():
        for child in sorted(parent.iterdir()):
            if child.is_dir():
                candidates.append(child / "DATA")

    seen = set()
    unique = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    for d in unique:
        if has_stations(d):
            if d != requested_data_dir:
                print(
                    "WARNING: requested --data-dir did not contain STATIONS. "
                    f"Using discovered DATA directory instead:\n  {d}"
                )
            return d

    if parent.exists():
        hits = sorted(parent.rglob(stations_file))
        for hit in hits:
            d = hit.parent
            if has_stations(d):
                print(
                    "WARNING: requested --data-dir did not contain STATIONS. "
                    f"Using recursively discovered DATA directory instead:\n  {d}"
                )
                return d

    checked = "\n".join(f"  {c}" for c in unique[:30])
    raise FileNotFoundError(
        f"Could not find {stations_file}.\n"
        f"Requested --data-dir was:\n  {requested_data_dir}\n\n"
        f"Checked:\n{checked}\n\n"
        "Set DATA_DIR in the wrapper to the folder that actually contains "
        "STATIONS and SOURCES_LIST.txt."
    )



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare synthetic SPECFEM/SEG-Y shot gathers with cave vs without cave."
    )

    p.add_argument("--data-dir", required=True, type=Path,
                   help="DATA directory containing STATIONS and optionally SOURCES_LIST.txt.")
    p.add_argument("--cave-dir", required=True, type=Path,
                   help="Directory containing synthetic SEG-Y files WITH cave/void.")
    p.add_argument("--nocave-dir", required=True, type=Path,
                   help="Directory containing synthetic SEG-Y files WITHOUT cave/void.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output directory for figures and CSV summary.")

    p.add_argument("--stations-file", default="STATIONS",
                   help="STATIONS filename inside --data-dir. Default: STATIONS")
    p.add_argument("--sources-file", default="SOURCES_LIST.txt",
                   help="SOURCES_LIST filename inside --data-dir. Default: SOURCES_LIST.txt")

    p.add_argument("--cave-pattern", default="SURVEY_OUTPUT/**/Uz_file_single_v.su",
                   help="Recursive glob for cave gather files. Default: SURVEY_OUTPUT/**/Uz_file_single_v.su")
    p.add_argument("--nocave-pattern", default="SURVEY_OUTPUT/**/Uz_file_single_v.su",
                   help="Recursive glob for no-cave gather files. Default: SURVEY_OUTPUT/**/Uz_file_single_v.su")

    p.add_argument("--pair-mode", choices=["auto", "position", "order"], default="auto",
                   help="How to pair shot folders: auto, position, or order. Default: auto")

    p.add_argument("--match-tolerance-m", type=float, default=0.51,
                   help="Tolerance for matching filename-inferred shot x to SOURCES_LIST. Default: 0.51")
    p.add_argument("--receiver-tolerance-m", type=float, default=0.05,
                   help="Tolerance for aligning receiver x between paired gathers. Default: 0.05")

    p.add_argument("--component", default=None,
                   help="Optional component suffix to select, e.g. Z. Usually not needed for synthetic SEG-Y.")
    p.add_argument("--trust-segy-geometry", action="store_true",
                   help="Use SEG-Y receiver coordinate headers when present instead of DATA/STATIONS.")

    p.add_argument("--tmin", type=float, default=None,
                   help="Minimum time to show in plots.")
    p.add_argument("--tmax", type=float, default=None,
                   help="Maximum time to show in plots.")
    p.add_argument("--max-freq-hz", type=float, default=150.0,
                   help="Maximum frequency for Charlie-style frequency-vs-receiver plots. Default: 150 Hz")

    p.add_argument("--cave-extent-x-m", default=None,
                   help="Optional cave x extent to shade on gather plots, e.g. 120,135")

    p.add_argument("--write-diff-segy", action="store_true",
                   help="Write difference_cave_minus_nocave.sgy for each shot.")
    p.add_argument("--write-individual-wiggles", action="store_true",
                   help="Also write separate wiggle plots for cave/no-cave/difference.")
    p.add_argument("--no-frequency-trace-normalization", action="store_true",
                   help="Disable per-trace normalization in frequency-vs-receiver plots.")
    p.add_argument("--write-trace-normalized-figures", action=argparse.BooleanOptionalAction, default=True,
                   help="Also write trace-normalized comparison/difference figures. Default: true.")
    p.add_argument("--trace-normalize-method", choices=["rms", "maxabs"], default="rms",
                   help="Trace normalization method for secondary figures. Default: rms.")
    p.add_argument("--write-overlay-wiggles", action="store_true",
                   help="Write overlaid no-cave-blue / cave-red wiggle plot for each shot.")
    p.add_argument("--overlay-normalize", choices=["pair", "trace", "none"], default="pair",
                   help="Normalization for overlay wiggles. Default: pair.")
    p.add_argument("--overlay-wiggle-scale", type=float, default=0.45,
                   help="Wiggle scale as fraction of receiver spacing. Default: 0.45.")
    p.add_argument("--peak-scale-tmin", type=float, default=None,
                   help="Start time for peak-scaling diagnostics. Default: plot/data window.")
    p.add_argument("--peak-scale-tmax", type=float, default=None,
                   help="End time for peak-scaling diagnostics. Default: plot/data window.")
    p.add_argument("--peak-scale-halfwidth-s", type=float, default=0.015,
                   help="Half-width around strongest reference peak for peak-window LSQ scaling. Default: 0.015 s.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional maximum number of shot pairs to process, useful for testing.")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_dir = resolve_data_dir(
        requested_data_dir=args.data_dir,
        cave_dir=args.cave_dir,
        nocave_dir=args.nocave_dir,
        stations_file=args.stations_file,
        sources_file=args.sources_file,
    )
    stations_path = data_dir / args.stations_file
    sources_path = data_dir / args.sources_file

    stations = read_stations(stations_path)
    sources = read_sources_list(sources_path)

    print(f"Read {len(stations)} receivers from {stations_path}")
    if sources:
        print(f"Read {len(sources)} sources from {sources_path}")
    else:
        print(f"No usable sources found in {sources_path}; will pair files by filename/sorted order")

    pairs = build_pairs(
        cave_dir=args.cave_dir,
        nocave_dir=args.nocave_dir,
        cave_pattern=args.cave_pattern,
        nocave_pattern=args.nocave_pattern,
        sources=sources,
        match_tolerance_m=args.match_tolerance_m,
        pair_mode=args.pair_mode,
    )

    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        raise SystemExit("No cave/no-cave file pairs found.")

    print(f"Matched {len(pairs)} cave/no-cave shot pairs")

    cave_extent = parse_cave_extent(args.cave_extent_x_m)

    results: list[PairResult] = []
    all_peak_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, (source_label, source_x_m, cave_path, nocave_path) in enumerate(pairs, start=1):
        print(
            f"[{i}/{len(pairs)}] source {source_label} x={source_x_m:.3f} m\n"
            f"    cave:   {cave_path.name}\n"
            f"    nocave: {nocave_path.name}"
        )
        try:
            res, peak_rows = process_pair(
                source_label=source_label,
                source_x_m=source_x_m,
                cave_path=cave_path,
                nocave_path=nocave_path,
                stations=stations,
                output_dir=args.output_dir,
                component=args.component,
                trust_segy_geometry=args.trust_segy_geometry,
                receiver_tolerance_m=args.receiver_tolerance_m,
                tmin=args.tmin,
                tmax=args.tmax,
                max_freq_hz=args.max_freq_hz,
                write_diff_segy=args.write_diff_segy,
                write_individual_wiggles=args.write_individual_wiggles,
                normalize_frequency_per_trace=not args.no_frequency_trace_normalization,
                cave_extent_x_m=cave_extent,
                write_overlay_wiggles=args.write_overlay_wiggles,
                overlay_normalize=args.overlay_normalize,
                overlay_wiggle_scale=args.overlay_wiggle_scale,
                peak_scale_tmin=args.peak_scale_tmin if args.peak_scale_tmin is not None else args.tmin,
                peak_scale_tmax=args.peak_scale_tmax if args.peak_scale_tmax is not None else args.tmax,
                peak_scale_halfwidth_s=args.peak_scale_halfwidth_s,
                write_trace_normalized_figures=args.write_trace_normalized_figures,
                trace_normalize_method=args.trace_normalize_method,
            )
            results.append(res)
            all_peak_rows.extend(peak_rows)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {msg}", file=sys.stderr)
            failures.append((source_label, msg))

    write_summary_csv(args.output_dir / "synthetic_cave_nocave_comparison_summary.csv", results)
    write_trace_peak_scaling_csv(args.output_dir / "synthetic_cave_nocave_trace_peak_scaling_factors.csv", all_peak_rows)

    if failures:
        failure_path = args.output_dir / "synthetic_cave_nocave_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_label", "error"])
            w.writerows(failures)
        print(f"Wrote failures: {failure_path}")

    print(f"Wrote summary: {args.output_dir / 'synthetic_cave_nocave_comparison_summary.csv'}")
    print(f"Completed {len(results)} / {len(pairs)} pairs")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
