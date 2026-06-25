#!/usr/bin/env python3
"""
61_compare_real_geode_vs_synthetic_novoid_v3.py

Compare real T1 Geode SEG-2 shot gathers against synthetic SPECFEM2D NO-VOID
single-shot Seismic Unix files.

Expected real files:
    /.../04_FieldData/051826/051826_Seismics_T1/3005.dat ... 3046.dat

Real shot-position correction:
    3005.dat = x 82.5 m
    nominal spacing = 2 m
    3015.dat = x 102.5 m
    3016.dat = x 102.5 m also
    3017.dat = x 104.5 m
    ...
    3046.dat = x 162.5 m

Expected synthetic files:
    NO_VOID_MODEL/SURVEY_OUTPUT/**/Uz_file_single_v.su
or:
    NO_VOID_MODEL/SURVEY_OUTPUT/**/Ux_file_single_v.su

Geometry:
    - synthetic receiver geometry comes from DATA/STATIONS
    - real receiver geometry is assigned as 72 geophones from 87 to 158 m
    - comparison/difference is only over the common receiver range 87..158 m

Difference:
    diff = real - scale_factor * synthetic

Default scale mode is fixed, using the current empirical best estimate from
lag-aligned direct-arrival-window metrics. The synthetic traces are also shifted
by the current best global timing offset before clipping to the real record
length.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import Stream, Trace, read
from obspy.io.segy.segy import SEGYTraceHeader


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")


@dataclass(frozen=True)
class Station:
    name: str
    net: str
    x_m: float
    z_m: float


@dataclass(frozen=True)
class Source:
    source_id: str
    x_m: float
    z_m: Optional[float] = None
    raw: str = ""


@dataclass(frozen=True)
class Gather:
    path: Path
    label: str
    source_x_m: float
    receiver_x_m: np.ndarray
    time_s: np.ndarray
    data: np.ndarray
    dt_s: float


def float_tokens(text: str) -> list[float]:
    out = []
    for m in _FLOAT_RE.finditer(text.replace("D", "E").replace("d", "e")):
        try:
            out.append(float(m.group(0)))
        except Exception:
            pass
    return out


def read_stations(path: Path) -> list[Station]:
    stations = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 4:
                raise ValueError(f"{path}:{line_no}: expected STATIONS columns: station net x z")
            stations.append(Station(p[0], p[1], float(p[2]), float(p[3])))
    if not stations:
        raise ValueError(f"No stations read from {path}")
    return stations


def read_sources(path: Path) -> list[Source]:
    """Tolerant SOURCES_LIST parser."""
    if not path.exists():
        return []

    sources = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            lower = s.lower()
            x = None
            z = None

            for key in ("source_position_x", "source_x", "x_source", "xs", "x"):
                m = re.search(rf"{key}\s*[:=]\s*({_FLOAT_RE.pattern})", lower)
                if m:
                    x = float(m.group(1).replace("d", "e"))
                    break

            for key in ("source_position_z", "source_z", "z_source", "zs", "z"):
                m = re.search(rf"{key}\s*[:=]\s*({_FLOAT_RE.pattern})", lower)
                if m:
                    z = float(m.group(1).replace("d", "e"))
                    break

            nums = float_tokens(s)
            if x is None and nums:
                # Common forms are either "index x z ..." or "x z ..."
                if len(nums) >= 3 and abs(nums[0] - round(nums[0])) < 1e-9:
                    x = nums[1]
                    z = nums[2]
                elif len(nums) >= 2:
                    x = nums[0]
                    z = nums[1]
                else:
                    x = nums[0]

            if x is None:
                continue

            first = s.split()[0]
            sid = first if not _FLOAT_RE.fullmatch(first) else f"S{len(sources)+1:04d}"
            sources.append(Source(sid, float(x), z, raw))

    return sources


def read_stream_any(path: Path) -> Stream:
    suffix = path.suffix.lower()
    formats = {
        ".dat": ["SEG2", None],
        ".su": ["SU", None],
        ".sgy": ["SEGY", None],
        ".segy": ["SEGY", None],
    }.get(suffix, [None, "SEG2", "SU", "SEGY"])

    last = None
    for fmt in formats:
        try:
            return read(str(path)) if fmt is None else read(str(path), format=fmt)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not read {path}; last error: {last}")


def preprocess_stream(
    st: Stream,
    *,
    demean: bool = True,
    detrend: bool = True,
    taper_fraction: float = 0.05,
    highpass_hz: Optional[float] = 10.0,
    lowpass_hz: Optional[float] = None,
    bandpass: Optional[tuple[float, float]] = None,
    filter_corners: int = 4,
    zerophase: bool = True,
) -> Stream:
    """
    Apply identical preprocessing to real and synthetic gathers.

    Current adopted default:
        demean + linear detrend + 5% cosine taper + 10 Hz high-pass.
    """
    st = st.copy()

    if demean:
        try:
            st.detrend("demean")
        except Exception:
            for tr in st:
                tr.data = np.asarray(tr.data, dtype=np.float64) - np.nanmean(tr.data)

    if detrend:
        try:
            st.detrend("linear")
        except Exception:
            pass

    if taper_fraction and taper_fraction > 0:
        try:
            st.taper(max_percentage=float(taper_fraction), type="cosine")
        except Exception:
            pass

    if bandpass is not None:
        fmin, fmax = bandpass
        st.filter("bandpass", freqmin=float(fmin), freqmax=float(fmax),
                  corners=int(filter_corners), zerophase=bool(zerophase))
    else:
        if highpass_hz is not None and highpass_hz > 0:
            st.filter("highpass", freq=float(highpass_hz),
                      corners=int(filter_corners), zerophase=bool(zerophase))
        if lowpass_hz is not None and lowpass_hz > 0:
            st.filter("lowpass", freq=float(lowpass_hz),
                      corners=int(filter_corners), zerophase=bool(zerophase))

    return st


def detrend_demean(st: Stream, *, demean: bool = True, detrend: bool = True) -> Stream:
    """Backward-compatible helper; new code should call preprocess_stream."""
    return preprocess_stream(
        st,
        demean=demean,
        detrend=detrend,
        taper_fraction=0.0,
        highpass_hz=None,
        lowpass_hz=None,
        bandpass=None,
    )


def stream_to_matrix(st: Stream) -> tuple[np.ndarray, float, np.ndarray]:
    if len(st) == 0:
        raise ValueError("empty stream")
    npts = min(int(tr.stats.npts) for tr in st)
    dt = float(st[0].stats.delta)
    data = np.vstack([np.asarray(tr.data[:npts], dtype=np.float64) for tr in st])
    time_s = np.arange(npts, dtype=float) * dt
    return data, dt, time_s


def read_real_gather(
    path: Path,
    *,
    source_x_m: float,
    label: str,
    real_first_trace_x_m: float,
    real_dx_m: float,
    receiver_x_min: float,
    receiver_x_max: float,
    reverse_real_traces: bool,
    preprocess_kwargs: dict,
) -> Gather:
    st = read_stream_any(path)
    if reverse_real_traces:
        st = Stream(list(reversed(st)))
    st = preprocess_stream(st, **preprocess_kwargs)

    data, dt, time_s = stream_to_matrix(st)
    rx = real_first_trace_x_m + np.arange(data.shape[0], dtype=float) * real_dx_m

    keep = (rx >= receiver_x_min - 1e-9) & (rx <= receiver_x_max + 1e-9)
    if not np.any(keep):
        raise ValueError(f"No real traces within x={receiver_x_min}..{receiver_x_max} m")

    return Gather(path, label, source_x_m, rx[keep], time_s, data[keep, :], dt)


def read_synthetic_gather(
    path: Path,
    *,
    stations: list[Station],
    source_x_m: float,
    label: str,
    receiver_x_min: float,
    receiver_x_max: float,
    preprocess_kwargs: dict,
) -> Gather:
    st = preprocess_stream(read_stream_any(path), **preprocess_kwargs)
    data, dt, time_s = stream_to_matrix(st)

    if data.shape[0] > len(stations):
        raise ValueError(f"{path}: {data.shape[0]} traces but only {len(stations)} STATIONS rows")

    rx = np.asarray([s.x_m for s in stations[:data.shape[0]]], dtype=float)
    keep = (rx >= receiver_x_min - 1e-9) & (rx <= receiver_x_max + 1e-9)
    if not np.any(keep):
        raise ValueError(f"No synthetic traces within x={receiver_x_min}..{receiver_x_max} m")

    data = data[keep, :]
    rx = rx[keep]
    order = np.argsort(rx)
    return Gather(path, label, source_x_m, rx[order], time_s, data[order, :], dt)


def real_files(real_dir: Path, first: int, last: int) -> list[Path]:
    files = []
    for n in range(first, last + 1):
        p = real_dir / f"{n}.dat"
        if not p.exists():
            raise FileNotFoundError(f"Missing real SEG-2 file: {p}")
        files.append(p)
    return files


def natural_key(path: Path):
    text = str(path.parent)
    parts = re.split(r"(\d+(?:\.\d+)?)", text)
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            out.append(p)
    return out


def synthetic_files(model_root: Path, component_file: str) -> list[Path]:
    root = model_root / "SURVEY_OUTPUT"
    if not root.exists():
        root = model_root
    files = sorted(root.rglob(component_file), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No {component_file} files found under {root}")
    return files

def real_shot_x_from_file_number(
    file_number: int,
    *,
    first_file: int = 3005,
    first_x_m: float = 82.5,
    dx_m: float = 2.0,
    duplicate_x_m: float = 102.5,
    duplicate_files: tuple[int, int] = (3015, 3016),
) -> float:
    """
    Return the real Geode shot x position for a .dat file number.

    Survey correction:
        3005.dat starts at x = 82.5 m.
        Shot positions then advance by 2 m.
        3015.dat and 3016.dat were both at x = 102.5 m.
        After 3016.dat, positions continue advancing by 2 m, so 3017.dat
        is x = 104.5 m and 3046.dat is x = 162.5 m.
    """
    if file_number in duplicate_files:
        return float(duplicate_x_m)
    if file_number < min(duplicate_files):
        return float(first_x_m + dx_m * (file_number - first_file))
    if file_number > max(duplicate_files):
        return float(duplicate_x_m + dx_m * (file_number - max(duplicate_files)))
    return float(first_x_m + dx_m * (file_number - first_file))


def real_file_number(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception as exc:
        raise ValueError(f"Cannot infer real file number from {path.name}") from exc


def map_synthetic_files_to_sources(
    synthetic_paths: list[Path],
    sources: list[Source],
) -> dict[float, tuple[Path, Source]]:
    """
    Map rounded source x position to synthetic file and source metadata.

    We assume synthetic files are sorted in the same order as SOURCES_LIST.
    This is appropriate for the SPECFEM single-source SURVEY_OUTPUT folders.
    """
    n = min(len(synthetic_paths), len(sources))
    mapping = {}
    for i in range(n):
        src = sources[i]
        mapping[round(float(src.x_m), 6)] = (synthetic_paths[i], src)
    return mapping


def nearest_synthetic_for_x(
    x_m: float,
    synthetic_by_x: dict[float, tuple[Path, Source]],
    *,
    tolerance_m: float,
) -> tuple[Path, Source]:
    if not synthetic_by_x:
        raise ValueError("No synthetic source/file mapping is available")
    xs = np.asarray(sorted(synthetic_by_x.keys()), dtype=float)
    j = int(np.argmin(np.abs(xs - x_m)))
    best_x = float(xs[j])
    if abs(best_x - x_m) > tolerance_m:
        raise ValueError(
            f"No synthetic shot within {tolerance_m:g} m of real shot x={x_m:.3f} m; "
            f"nearest is x={best_x:.3f} m"
        )
    return synthetic_by_x[round(best_x, 6)]


def interp_to_time(data: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], len(new_t)), dtype=np.float64)
    for i in range(data.shape[0]):
        out[i] = np.interp(new_t, old_t, data[i], left=0.0, right=0.0)
    return out


def shift_data_time(data: np.ndarray, time_s: np.ndarray, lag_s: float) -> np.ndarray:
    """
    Shift synthetic data on the real time axis.

    Sign convention inherited from 63_estimate_real_synthetic_lags.py:
        lag_s > 0 delays synthetic to align with real.
        shifted_syn(t) = syn(t - lag_s)

    The adopted current global lag is about -31.6 ms, so the synthetic data are
    advanced by 31.6 ms before clipping to the real 0.4 s record.
    """
    out = np.zeros_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        out[i] = np.interp(time_s - lag_s, time_s, data[i], left=0.0, right=0.0)
    return out


def align(real: Gather, syn: Gather, *, receiver_tol_m: float, tmin: float, tmax: float, synthetic_time_shift_ms: float = 0.0):
    pairs = []
    used = set()
    for ir, x in enumerate(real.receiver_x_m):
        js = int(np.argmin(np.abs(syn.receiver_x_m - x)))
        if js in used:
            continue
        if abs(syn.receiver_x_m[js] - x) <= receiver_tol_m:
            pairs.append((ir, js))
            used.add(js)

    if not pairs:
        raise ValueError("No common receiver positions")

    ir = np.asarray([p[0] for p in pairs])
    js = np.asarray([p[1] for p in pairs])

    r = real.data[ir]
    s = syn.data[js]
    rx = real.receiver_x_m[ir]

    # Use real time axis as authoritative, interpolate synthetic onto it,
    # then apply the adopted gather-wide timing correction before clipping.
    t = real.time_s.copy()
    s = interp_to_time(s, syn.time_s, t)
    if synthetic_time_shift_ms:
        s = shift_data_time(s, t, synthetic_time_shift_ms / 1000.0)

    keep = (t >= tmin) & (t <= tmax)
    if not np.any(keep):
        raise ValueError("No samples in requested time window")

    return t[keep], rx, r[:, keep], s[:, keep]


def scale_factor(real: np.ndarray, syn: np.ndarray, t: np.ndarray, mode: str, scale_tmin, scale_tmax, fixed_scale_factor: float = 1.0) -> float:
    mode = mode.lower()
    if mode == "none":
        return 1.0
    if mode == "fixed":
        return float(fixed_scale_factor)

    keep = np.ones_like(t, dtype=bool)
    if scale_tmin is not None:
        keep &= t >= scale_tmin
    if scale_tmax is not None:
        keep &= t <= scale_tmax

    r = real[:, keep].ravel()
    s = syn[:, keep].ravel()
    good = np.isfinite(r) & np.isfinite(s)
    r = r[good]
    s = s[good]
    if r.size == 0:
        return 1.0

    if mode == "maxabs":
        den = np.nanmax(np.abs(s))
        return float(np.nanmax(np.abs(r)) / den) if den else 1.0
    if mode == "rms":
        den = np.sqrt(np.nanmean(s * s))
        return float(np.sqrt(np.nanmean(r * r)) / den) if den else 1.0
    if mode == "lsq":
        den = float(np.dot(s, s))
        return float(np.dot(r, s) / den) if den else 1.0

    raise ValueError(f"Unknown scale mode: {mode}")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))


def clip_value(*arrays, pct=99.0):
    vals = np.concatenate([np.ravel(np.abs(a[np.isfinite(a)])) for a in arrays if np.size(a)])
    if vals.size == 0:
        return 1.0
    c = float(np.percentile(vals, pct))
    return c if c > 0 else 1.0


def plot_three(t, rx, real, syn, diff, source_x, out, scale_mode, sf, cave_extent):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    extent = [rx.min(), rx.max(), t.max(), t.min()]
    main_c = clip_value(real, syn)
    diff_c = clip_value(diff)

    for ax, data, title, c in [
        (axes[0], real, "Real Geode", main_c),
        (axes[1], syn, "Synthetic no-void", main_c),
        (axes[2], diff, "Difference: real - synthetic", diff_c),
    ]:
        im = ax.imshow(data.T, aspect="auto", interpolation="nearest",
                       extent=extent, cmap="seismic", vmin=-c, vmax=c)
        ax.axvline(source_x, color="k", lw=1, ls="--", alpha=0.75)
        if cave_extent:
            ax.axvspan(cave_extent[0], cave_extent[1], color="0.5", alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    axes[0].set_ylabel("time (s)")
    fig.suptitle(f"Real vs synthetic no-void, source x={source_x:.3f} m; scale={scale_mode}, factor={sf:.5g}")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_wiggle(t, rx, data, source_x, title, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(data, dtype=float).copy()
    mx = np.nanmax(np.abs(d), axis=1)
    mx[mx == 0] = 1.0
    d /= mx[:, None]
    dx = np.nanmedian(np.diff(np.sort(rx))) if len(rx) > 1 else 1.0
    amp = 0.45 * dx

    fig, ax = plt.subplots(figsize=(12, 7))
    for tr, x in zip(d, rx):
        y = x + tr * amp
        ax.plot(y, t, color="k", lw=0.45)
        ax.fill_betweenx(t, x, y, where=(y >= x), color="k", alpha=0.25, linewidth=0)
    ax.axvline(source_x, color="r", lw=1, ls="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _window_mask(t, tmin, tmax):
    keep = np.ones_like(t, dtype=bool)
    if tmin is not None:
        keep &= t >= tmin
    if tmax is not None:
        keep &= t <= tmax
    return keep


def _safe_ratio(num, den):
    try:
        num = float(num)
        den = float(den)
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return np.nan
        return num / den
    except Exception:
        return np.nan


def _corrcoef(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a = a[:n] - np.nanmean(a[:n])
    b = b[:n] - np.nanmean(b[:n])
    den = np.sqrt(np.nansum(a * a) * np.nansum(b * b))
    return float(np.nansum(a * b) / den) if den else np.nan


def _lsq_scale(real, syn):
    r = np.asarray(real, dtype=float).ravel()
    s = np.asarray(syn, dtype=float).ravel()
    good = np.isfinite(r) & np.isfinite(s)
    r = r[good]
    s = s[good]
    if r.size == 0:
        return np.nan
    den = float(np.dot(s, s))
    return float(np.dot(r, s) / den) if den else np.nan


def _local_peak_window_mask(t, real_trace, syn_trace, scale_tmin, scale_tmax, halfwidth_s):
    """
    Return a local window around the largest absolute real/synthetic peak in the
    scale window. This is useful when a full direct-arrival window contains
    multiple wavelet lobes and we want a compact peak-centered scale diagnostic.
    """
    base = _window_mask(t, scale_tmin, scale_tmax)
    if not np.any(base):
        return base
    idxs = np.where(base)[0]
    r = np.asarray(real_trace, dtype=float)
    s = np.asarray(syn_trace, dtype=float)
    score = np.maximum(np.abs(r[idxs]), np.abs(s[idxs]))
    if score.size == 0 or not np.any(np.isfinite(score)):
        return base
    center_idx = idxs[int(np.nanargmax(score))]
    center_t = float(t[center_idx])
    return _window_mask(t, center_t - halfwidth_s, center_t + halfwidth_s)


def trace_peak_scaling_rows(
    t,
    rx,
    real,
    syn_raw,
    syn_scaled,
    *,
    shot_index,
    source_label,
    source_x,
    real_file,
    synthetic_file,
    global_scale_factor,
    scale_tmin,
    scale_tmax,
    peak_halfwidth_s,
):
    """
    Compute per-trace synthetic-to-real scale diagnostics after all processing,
    receiver matching, time shifting, and clipping.

    syn_raw is the processed/time-shifted synthetic before the global scale.
    syn_scaled is the processed/time-shifted synthetic after the global scale.

    The scale factors reported here are multipliers to apply to syn_raw so that
    it matches real in the specified window. Values can be compared with the
    global scale factor used for plotting/differencing.
    """
    rows = []
    full_win = _window_mask(t, scale_tmin, scale_tmax)
    for i, x in enumerate(rx):
        r_full = real[i, full_win]
        sraw_full = syn_raw[i, full_win]
        sscaled_full = syn_scaled[i, full_win]

        local_win = _local_peak_window_mask(
            t,
            real[i],
            syn_scaled[i],
            scale_tmin,
            scale_tmax,
            peak_halfwidth_s,
        )
        r_loc = real[i, local_win]
        sraw_loc = syn_raw[i, local_win]
        sscaled_loc = syn_scaled[i, local_win]

        # Positive and negative peak matching in the full scale window.
        r_pos = float(np.nanmax(r_full)) if r_full.size else np.nan
        r_neg = float(np.nanmin(r_full)) if r_full.size else np.nan
        sraw_pos = float(np.nanmax(sraw_full)) if sraw_full.size else np.nan
        sraw_neg = float(np.nanmin(sraw_full)) if sraw_full.size else np.nan
        sscaled_pos = float(np.nanmax(sscaled_full)) if sscaled_full.size else np.nan
        sscaled_neg = float(np.nanmin(sscaled_full)) if sscaled_full.size else np.nan

        scale_pos = _safe_ratio(r_pos, sraw_pos)
        # Both are usually negative, so r_neg/sraw_neg is a positive scale if polarity matches.
        scale_neg = _safe_ratio(r_neg, sraw_neg)
        scale_abspeak = _safe_ratio(np.nanmax(np.abs(r_full)) if r_full.size else np.nan,
                                    np.nanmax(np.abs(sraw_full)) if sraw_full.size else np.nan)

        # Robust combined peak scale: median of valid positive and negative peak scales.
        peak_candidates = np.asarray([scale_pos, scale_neg], dtype=float)
        peak_candidates = peak_candidates[np.isfinite(peak_candidates) & (peak_candidates > 0)]
        scale_peak_mean_posneg = float(np.nanmean(peak_candidates)) if peak_candidates.size else np.nan
        scale_peak_median_posneg = float(np.nanmedian(peak_candidates)) if peak_candidates.size else np.nan

        # Least-squares/correlation scale in full and local peak windows.
        scale_lsq_full = _lsq_scale(r_full, sraw_full)
        scale_lsq_local = _lsq_scale(r_loc, sraw_loc)
        corr_full_after_global = _corrcoef(r_full, sscaled_full)
        corr_local_after_global = _corrcoef(r_loc, sscaled_loc)

        rows.append({
            "shot_index": shot_index,
            "source_label": source_label,
            "source_x_m": source_x,
            "real_file": str(real_file),
            "synthetic_file": str(synthetic_file),
            "trace_index_1based": i + 1,
            "receiver_x_m": float(x),
            "offset_m": float(x - source_x),
            "scale_window_tmin_s": scale_tmin,
            "scale_window_tmax_s": scale_tmax,
            "peak_local_halfwidth_s": peak_halfwidth_s,
            "global_scale_factor_used": global_scale_factor,
            "real_pos_peak": r_pos,
            "real_neg_peak": r_neg,
            "synthetic_raw_pos_peak": sraw_pos,
            "synthetic_raw_neg_peak": sraw_neg,
            "synthetic_scaled_pos_peak": sscaled_pos,
            "synthetic_scaled_neg_peak": sscaled_neg,
            "scale_pos_peak_real_over_synthetic_raw": scale_pos,
            "scale_neg_peak_real_over_synthetic_raw": scale_neg,
            "scale_abspeak_real_over_synthetic_raw": scale_abspeak,
            "scale_peak_mean_posneg_real_over_synthetic_raw": scale_peak_mean_posneg,
            "scale_peak_median_posneg_real_over_synthetic_raw": scale_peak_median_posneg,
            "scale_lsq_full_window_real_over_synthetic_raw": scale_lsq_full,
            "scale_lsq_local_peak_window_real_over_synthetic_raw": scale_lsq_local,
            "corr_full_window_after_global_scale": corr_full_after_global,
            "corr_local_peak_window_after_global_scale": corr_local_after_global,
            "real_rms_full_window": rms(r_full),
            "synthetic_raw_rms_full_window": rms(sraw_full),
            "synthetic_scaled_rms_full_window": rms(sscaled_full),
            "real_maxabs_full_window": float(np.nanmax(np.abs(r_full))) if r_full.size else np.nan,
            "synthetic_raw_maxabs_full_window": float(np.nanmax(np.abs(sraw_full))) if sraw_full.size else np.nan,
            "synthetic_scaled_maxabs_full_window": float(np.nanmax(np.abs(sscaled_full))) if sscaled_full.size else np.nan,
        })
    return rows


def _clean_csv_value(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return f"{v:.10g}"
    return v


def write_dict_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _clean_csv_value(v) for k, v in row.items()})


def plot_overlay_wiggles(
    t,
    rx,
    real,
    synthetic_scaled,
    source_x,
    title,
    out,
    *,
    cave_extent=None,
    normalize="pair",
    wiggle_scale=0.45,
):
    """
    Overlay real and synthetic wiggles on the same axes.

    Color convention requested by Glenn:
        synthetic = blue
        real      = red

    Negative lobes are transparently shaded in their own colors.

    normalize='pair' uses one normalization per receiver pair:
        norm_i = max(maxabs(real_i), maxabs(synthetic_i))
    This preserves the relative amplitude mismatch between real and synthetic
    for each receiver while keeping all traces readable.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    r = np.asarray(real, dtype=float).copy()
    s = np.asarray(synthetic_scaled, dtype=float).copy()

    if normalize == "pair":
        norm = np.maximum(np.nanmax(np.abs(r), axis=1), np.nanmax(np.abs(s), axis=1))
        norm[~np.isfinite(norm) | (norm == 0)] = 1.0
        rp = r / norm[:, None]
        sp = s / norm[:, None]
        norm_label = "pair-normalized display"
    elif normalize == "global":
        norm = np.nanmax(np.abs(np.concatenate([r.ravel(), s.ravel()])))
        if not np.isfinite(norm) or norm == 0:
            norm = 1.0
        rp = r / norm
        sp = s / norm
        norm_label = "global-normalized display"
    elif normalize == "none":
        # Still convert to a readable plotting amplitude using a robust global clip.
        norm = clip_value(r, s, pct=99.0)
        rp = np.clip(r / norm, -1, 1)
        sp = np.clip(s / norm, -1, 1)
        norm_label = "clipped physical-amplitude display"
    else:
        raise ValueError(f"Unknown overlay normalize mode: {normalize}")

    dx = np.nanmedian(np.diff(np.sort(rx))) if len(rx) > 1 else 1.0
    amp = float(wiggle_scale) * dx

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for rr, ss, x in zip(rp, sp, rx):
        yr = x + rr * amp
        ys = x + ss * amp
        ax.plot(ys, t, color="blue", lw=0.55, alpha=0.85)
        ax.fill_betweenx(t, x, ys, where=(ss < 0), color="blue", alpha=0.18, linewidth=0)
        ax.plot(yr, t, color="red", lw=0.55, alpha=0.85)
        ax.fill_betweenx(t, x, yr, where=(rr < 0), color="red", alpha=0.18, linewidth=0)

    ax.axvline(source_x, color="k", lw=1.0, ls="--", alpha=0.75, label="source")
    if cave_extent:
        ax.axvspan(cave_extent[0], cave_extent[1], color="0.5", alpha=0.12, label="cave extent")
    ax.set_title(f"{title}\nSynthetic blue, real red; negative lobes shaded; {norm_label}")
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()
    # Lightweight legend using dummy lines.
    ax.plot([], [], color="blue", lw=1.5, label="Synthetic no-void")
    ax.plot([], [], color="red", lw=1.5, label="Real Geode")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_freq(t, rx, data, source_x, title, out, max_freq, normalize=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(t)))
    freqs = np.fft.rfftfreq(data.shape[1], d=dt)
    amp = np.abs(np.fft.rfft(data, axis=1))
    keep = freqs <= max_freq
    freqs = freqs[keep]
    amp = amp[:, keep]

    if normalize:
        mx = np.nanmax(amp, axis=1)
        mx[mx == 0] = 1.0
        amp = amp / mx[:, None]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(amp.T, aspect="auto", origin="lower", interpolation="nearest",
                   extent=[rx.min(), rx.max(), freqs.min(), freqs.max()],
                   cmap="viridis", vmin=0, vmax=clip_value(amp))
    ax.axvline(source_x, color="w", lw=1, ls="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(alpha=0.15)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_diff_segy(path, diff, dt, rx, source_x):
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Stream()
    for i, (x, y) in enumerate(zip(rx, diff), 1):
        tr = Trace(data=np.asarray(y, dtype=np.float32))
        tr.stats.delta = dt
        tr.stats.network = "DF"
        tr.stats.station = f"D{i:04d}"
        tr.stats.channel = "Z"
        tr.stats.segy = {}
        tr.stats.segy.trace_header = SEGYTraceHeader()
        h = tr.stats.segy.trace_header
        h.trace_sequence_number_within_line = i
        h.trace_number_within_the_original_field_record = i
        h.original_field_record_number = int(round(source_x * 100))
        h.energy_source_point_number = int(round(source_x * 100))
        h.scalar_to_be_applied_to_all_coordinates = -1000
        h.source_coordinate_x = int(round(source_x * 1000))
        h.group_coordinate_x = int(round(float(x) * 1000))
        h.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = int(round((float(x) - source_x) * 1000))
        st.append(tr)
    st.write(str(path), format="SEGY", data_encoding=1)


def safe_dir(i, real_file, source_x):
    return f"shot{i:03d}_{real_file.stem}_x{source_x:08.3f}m".replace(".", "p")


def parse_extent(txt):
    if not txt:
        return None
    a, b = [float(x.strip()) for x in txt.split(",")]
    return (min(a, b), max(a, b))


def build_parser():
    p = argparse.ArgumentParser(description="Compare real SEG-2 Geode gathers vs synthetic no-void SU gathers.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--synthetic-novoid-dir", required=True, type=Path)
    p.add_argument("--real-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--stations-file", default="STATIONS")
    p.add_argument("--sources-file", default="SOURCES_LIST.txt")
    p.add_argument("--real-first-file", type=int, default=3005)
    p.add_argument("--real-last-file", type=int, default=3046)
    p.add_argument("--real-shot-first-x-m", type=float, default=82.5,
                   help="Shot x for --real-first-file. Default: 82.5 m for 3005.dat")
    p.add_argument("--real-shot-dx-m", type=float, default=2.0,
                   help="Nominal shot spacing for real files. Default: 2 m")
    p.add_argument("--real-shot-duplicate-x-m", type=float, default=102.5,
                   help="Corrected shot x for duplicate real shots 3015 and 3016. Default: 102.5 m")
    p.add_argument("--real-shot-duplicate-files", default="3015,3016",
                   help="Comma-separated real file numbers that share --real-shot-duplicate-x-m. Default: 3015,3016")
    p.add_argument("--shot-match-tolerance-m", type=float, default=0.05,
                   help="Tolerance for matching real shot x to synthetic source x. Default: 0.05 m")
    p.add_argument("--component-file", default="Uz_file_single_v.su")
    p.add_argument("--receiver-x-min", type=float, default=87.0)
    p.add_argument("--receiver-x-max", type=float, default=158.0)
    p.add_argument("--real-first-trace-x-m", type=float, default=87.0)
    p.add_argument("--real-dx-m", type=float, default=1.0)
    p.add_argument("--reverse-real-traces", action="store_true")
    p.add_argument("--receiver-tolerance-m", type=float, default=0.05)
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=0.4,
                   help="Comparison time-window end. Default: 0.4 s to match real SEG-2 records.")
    p.add_argument("--max-freq-hz", type=float, default=150.0)
    p.add_argument("--scale-mode", choices=["none", "fixed", "maxabs", "rms", "lsq"], default="fixed",
                   help="Default fixed uses --fixed-scale-factor from lag-aligned direct-arrival metrics.")
    p.add_argument("--fixed-scale-factor", type=float, default=2.96e7,
                   help="Empirical global synthetic multiplier. Default: 2.96e7.")
    p.add_argument("--scale-tmin", type=float, default=0.02)
    p.add_argument("--scale-tmax", type=float, default=0.12)
    p.add_argument("--synthetic-time-shift-ms", type=float, default=-31.6,
                   help="Gather-wide synthetic timing shift. Negative advances synthetic. Default: -31.6 ms.")
    p.add_argument("--demean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detrend", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taper-fraction", type=float, default=0.05)
    p.add_argument("--highpass-hz", type=float, default=10.0)
    p.add_argument("--lowpass-hz", type=float, default=None)
    p.add_argument("--filter-corners", type=int, default=4)
    p.add_argument("--zerophase", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cave-extent-x-m", default="122,130")
    p.add_argument("--write-individual-wiggles", action="store_true")
    p.add_argument("--write-overlay-wiggles", action=argparse.BooleanOptionalAction, default=True,
                   help="Write overlaid synthetic-blue / real-red wiggle plot for each shot. Default: true")
    p.add_argument("--overlay-normalize", choices=["pair", "global", "none"], default="pair",
                   help="Display normalization for overlaid wiggles. Default: pair")
    p.add_argument("--overlay-wiggle-scale", type=float, default=0.45,
                   help="Horizontal wiggle excursion as fraction of receiver spacing. Default: 0.45")
    p.add_argument("--peak-scale-halfwidth-s", type=float, default=0.015,
                   help="Half-width around local peak for local LSQ/correlation scale diagnostics. Default: 0.015 s")
    p.add_argument("--write-diff-segy", action="store_true")
    p.add_argument("--no-frequency-trace-normalization", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    stations = read_stations(args.data_dir / args.stations_file)
    sources = read_sources(args.data_dir / args.sources_file)
    rfiles = real_files(args.real_dir, args.real_first_file, args.real_last_file)
    sfiles = synthetic_files(args.synthetic_novoid_dir, args.component_file)

    if not sources:
        raise SystemExit(
            "SOURCES_LIST.txt is required for this corrected real-vs-synthetic pairing, "
            "because 3015.dat and 3016.dat both map to x=102.5 m."
        )

    duplicate_files = tuple(
        int(x.strip()) for x in str(args.real_shot_duplicate_files).split(",") if x.strip()
    )

    synthetic_by_x = map_synthetic_files_to_sources(sfiles, sources)

    pairs = []
    for rf in rfiles:
        fn = real_file_number(rf)
        real_x = real_shot_x_from_file_number(
            fn,
            first_file=args.real_first_file,
            first_x_m=args.real_shot_first_x_m,
            dx_m=args.real_shot_dx_m,
            duplicate_x_m=args.real_shot_duplicate_x_m,
            duplicate_files=duplicate_files,
        )
        sfp, src_from_model = nearest_synthetic_for_x(
            real_x, synthetic_by_x, tolerance_m=args.shot_match_tolerance_m
        )
        src = Source(
            source_id=f"real_{fn}_x{real_x:.3f}m__model_{src_from_model.source_id}",
            x_m=real_x,
            z_m=src_from_model.z_m,
            raw=getattr(src_from_model, "raw", ""),
        )
        pairs.append((rf, sfp, src))

    if args.limit:
        pairs = pairs[:args.limit]

    n = len(pairs)

    print(f"Read {len(stations)} synthetic receivers")
    print(f"Read {len(sources)} unique modeled source positions")
    print(f"Found {len(rfiles)} real SEG-2 files")
    print(f"Found {len(sfiles)} synthetic {args.component_file} files")
    print(f"Built {n} real/synthetic shot pairs using real file shot-position correction")
    print(f"Correction: 3015.dat and 3016.dat both use x={args.real_shot_duplicate_x_m:g} m")
    print(f"Processing over receiver x={args.receiver_x_min:g}..{args.receiver_x_max:g} m")

    cave_extent = parse_extent(args.cave_extent_x_m)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_kwargs = dict(
        demean=args.demean,
        detrend=args.detrend,
        taper_fraction=args.taper_fraction,
        highpass_hz=args.highpass_hz,
        lowpass_hz=args.lowpass_hz,
        bandpass=None,
        filter_corners=args.filter_corners,
        zerophase=args.zerophase,
    )
    print(
        "Preprocessing: "
        f"demean={args.demean}, detrend={args.detrend}, "
        f"taper_fraction={args.taper_fraction}, highpass_hz={args.highpass_hz}, "
        f"lowpass_hz={args.lowpass_hz}, corners={args.filter_corners}, zerophase={args.zerophase}"
    )
    print(
        f"Synthetic timing shift: {args.synthetic_time_shift_ms:g} ms; "
        f"scale_mode={args.scale_mode}; fixed_scale_factor={args.fixed_scale_factor:g}; "
        f"time clip={args.tmin:g}..{args.tmax:g} s"
    )

    results = []
    failures = []
    all_trace_scaling_rows = []

    for idx, (rf, sfp, src) in enumerate(pairs):
        shot_no = idx + 1
        print(f"[{shot_no}/{n}] {rf.name} vs {sfp.parent.name}/{sfp.name}; corrected source x={src.x_m:.3f} m")

        try:
            real = read_real_gather(
                rf,
                source_x_m=src.x_m,
                label=src.source_id,
                real_first_trace_x_m=args.real_first_trace_x_m,
                real_dx_m=args.real_dx_m,
                receiver_x_min=args.receiver_x_min,
                receiver_x_max=args.receiver_x_max,
                reverse_real_traces=args.reverse_real_traces,
                preprocess_kwargs=preprocess_kwargs,
            )
            syn = read_synthetic_gather(
                sfp,
                stations=stations,
                source_x_m=src.x_m,
                label=src.source_id,
                receiver_x_min=args.receiver_x_min,
                receiver_x_max=args.receiver_x_max,
                preprocess_kwargs=preprocess_kwargs,
            )

            t, rx, rd, sd_raw = align(
                real,
                syn,
                receiver_tol_m=args.receiver_tolerance_m,
                tmin=args.tmin,
                tmax=args.tmax,
                synthetic_time_shift_ms=args.synthetic_time_shift_ms,
            )

            sfactor = scale_factor(
                rd, sd_raw, t,
                mode=args.scale_mode,
                scale_tmin=args.scale_tmin,
                scale_tmax=args.scale_tmax,
                fixed_scale_factor=args.fixed_scale_factor,
            )
            sd = sfactor * sd_raw
            diff = rd - sd

            odir = args.output_dir / safe_dir(shot_no, rf, src.x_m)

            trace_scaling_rows = trace_peak_scaling_rows(
                t,
                rx,
                rd,
                sd_raw,
                sd,
                shot_index=shot_no,
                source_label=src.source_id,
                source_x=src.x_m,
                real_file=rf,
                synthetic_file=sfp,
                global_scale_factor=sfactor,
                scale_tmin=args.scale_tmin,
                scale_tmax=args.scale_tmax,
                peak_halfwidth_s=args.peak_scale_halfwidth_s,
            )
            all_trace_scaling_rows.extend(trace_scaling_rows)
            write_dict_csv(odir / "trace_peak_scaling_factors.csv", trace_scaling_rows)

            plot_three(t, rx, rd, sd, diff, src.x_m, odir / "comparison_image_real_synthetic_novoid_difference.png",
                       args.scale_mode, sfactor, cave_extent)

            if args.write_overlay_wiggles:
                plot_overlay_wiggles(
                    t,
                    rx,
                    rd,
                    sd,
                    src.x_m,
                    f"Real vs synthetic no-void overlay, {rf.name}, source x={src.x_m:.3f} m; scale={sfactor:.5g}",
                    odir / "wiggle_overlay_synthetic_blue_real_red.png",
                    cave_extent=cave_extent,
                    normalize=args.overlay_normalize,
                    wiggle_scale=args.overlay_wiggle_scale,
                )

            if args.write_individual_wiggles:
                plot_wiggle(t, rx, rd, src.x_m, f"Real {rf.name}", odir / "wiggle_real.png")
                plot_wiggle(t, rx, sd, src.x_m, "Synthetic no-void", odir / "wiggle_synthetic_novoid.png")
                plot_wiggle(t, rx, diff, src.x_m, "Difference real - synthetic", odir / "wiggle_difference_real_minus_synthetic.png")

            norm_freq = not args.no_frequency_trace_normalization
            plot_freq(t, rx, rd, src.x_m, f"Frequency vs receiver: real {rf.name}", odir / "frequency_receiver_real.png", args.max_freq_hz, norm_freq)
            plot_freq(t, rx, sd, src.x_m, "Frequency vs receiver: synthetic no-void", odir / "frequency_receiver_synthetic_novoid.png", args.max_freq_hz, norm_freq)
            plot_freq(t, rx, diff, src.x_m, "Frequency vs receiver: difference real - synthetic", odir / "frequency_receiver_difference.png", args.max_freq_hz, norm_freq)

            if args.write_diff_segy:
                write_diff_segy(odir / "difference_real_minus_synthetic_novoid.sgy", diff, float(np.median(np.diff(t))), rx, src.x_m)

            results.append({
                "shot_index": shot_no,
                "source_label": src.source_id,
                "source_x_m": src.x_m,
                "real_file": str(rf),
                "synthetic_file": str(sfp),
                "scale_mode": args.scale_mode,
                "scale_factor": sfactor,
                "synthetic_time_shift_ms": args.synthetic_time_shift_ms,
                "tmin_s": args.tmin,
                "tmax_s": args.tmax,
                "highpass_hz": args.highpass_hz,
                "taper_fraction": args.taper_fraction,
                "n_receivers": diff.shape[0],
                "n_samples": diff.shape[1],
                "dt_s": float(np.median(np.diff(t))),
                "rms_real": rms(rd),
                "rms_synthetic_raw": rms(sd_raw),
                "rms_synthetic_scaled": rms(sd),
                "rms_difference": rms(diff),
                "maxabs_real": float(np.nanmax(np.abs(rd))),
                "maxabs_synthetic_scaled": float(np.nanmax(np.abs(sd))),
                "maxabs_difference": float(np.nanmax(np.abs(diff))),
            })

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {msg}", file=sys.stderr)
            failures.append({"shot_index": shot_no, "real_file": rf.name, "error": msg})

    summary = args.output_dir / "real_vs_synthetic_novoid_summary.csv"
    fields = [
        "shot_index", "source_label", "source_x_m", "real_file", "synthetic_file",
        "scale_mode", "scale_factor", "synthetic_time_shift_ms", "tmin_s", "tmax_s",
        "highpass_hz", "taper_fraction", "n_receivers", "n_samples", "dt_s",
        "rms_real", "rms_synthetic_raw", "rms_synthetic_scaled", "rms_difference",
        "maxabs_real", "maxabs_synthetic_scaled", "maxabs_difference",
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow(row)

    trace_scale_summary = args.output_dir / "real_vs_synthetic_trace_peak_scaling_factors.csv"
    write_dict_csv(trace_scale_summary, all_trace_scaling_rows)

    if failures:
        fpath = args.output_dir / "real_vs_synthetic_novoid_failures.csv"
        with fpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["shot_index", "real_file", "error"])
            w.writeheader()
            w.writerows(failures)
        print(f"Wrote failures: {fpath}")

    print(f"Wrote summary: {summary}")
    print(f"Wrote trace peak scaling factors: {trace_scale_summary}")
    print(f"Completed {len(results)} / {n} pairs")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
