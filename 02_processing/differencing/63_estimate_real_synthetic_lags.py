#!/usr/bin/env python3
"""
63_estimate_real_synthetic_lags.py

Estimate timing offsets between real Geode SEG-2 shot gathers and synthetic
SPECFEM2D no-void SU shot gathers.

Goal
----
For each real/synthetic shot pair:

    1. Read real SEG-2 gather.
    2. Read matching synthetic no-void SU gather.
    3. Clip both to the common receiver range, typically 87..158 m.
    4. Apply identical preprocessing:
           demean, detrend, taper, optional highpass/bandpass.
    5. Resample synthetic onto the real time axis.
    6. For each matched receiver trace:
           correlate real trace with synthetic trace in a direct-arrival window.
    7. Estimate robust gather-wide lag:
           median lag
           amplitude-weighted median lag
           amplitude-weighted mean lag
    8. Apply the selected gather-wide lag to synthetic.
    9. Recompute correlation and scale candidates after alignment.

This is deliberately NOT trace-by-trace alignment for final differencing.
Trace-by-trace lags are diagnostics only. The recommended correction is one
lag per shot gather, so that travel-time residuals and cave-related anomalies
are preserved.

Geometry assumptions
--------------------
Real files:
    3005.dat to 3046.dat

Shot positions:
    3005.dat = 82.5 m
    then +2 m per file,
    except 3015.dat and 3016.dat are both 102.5 m.
    Therefore 3017.dat = 104.5 m and 3046.dat = 162.5 m.

Receivers:
    real Geode traces are 72 geophones from 87 m to 158 m at 1 m spacing.

Synthetic:
    no-void model root contains SURVEY_OUTPUT/**/Uz_file_single_v.su
    or Ux_file_single_v.su.
    DATA/STATIONS supplies synthetic receiver positions.

Outputs
-------
    trace_lag_metrics.csv
        One row per shot/receiver trace.

    shot_lag_metrics.csv
        One row per shot gather.

    aligned_window_scale_metrics.csv
        One row per shot, containing scale factors computed after applying
        the selected gather-wide lag.

    failures.csv
        Any shots that failed.

Recommended first run
---------------------
Use a fairly narrow direct-arrival/correlation window and a modest lag search:

    --corr-window 0.02,0.12
    --max-lag-ms 50
    --analysis-tmin 0.0
    --analysis-tmax 0.4
    --highpass-hz 10
    --taper-fraction 0.05

Author: Glenn Thompson / ChatGPT
Date: 2026-06-24
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from obspy import Stream, Trace, read
except Exception as exc:
    raise SystemExit(
        "This script requires ObsPy. Install with:\n"
        "    conda install -c conda-forge obspy\n"
        f"Original import error: {exc}"
    )


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Station:
    station: str
    network: str
    x_m: float
    z_m: float
    burial_m: float = 0.0
    elevation_m: float = 0.0


@dataclass(frozen=True)
class Source:
    source_id: str
    x_m: float
    z_m: Optional[float] = None
    raw: str = ""


@dataclass(frozen=True)
class Gather:
    time_s: np.ndarray
    data: np.ndarray
    receiver_x_m: np.ndarray
    source_x_m: float
    label: str
    path: Path
    dt_s: float
    sampling_rate_hz: float


@dataclass(frozen=True)
class Pair:
    real_file: Path
    synthetic_file: Path
    source: Source
    real_shot_x_m: float
    synthetic_shot_x_m: float
    shot_index: int


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def rms(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def maxabs(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def abs_percentile(x: np.ndarray, pct: float) -> float:
    x = finite_1d(x)
    return float(np.percentile(np.abs(x), pct)) if x.size else float("nan")


def mad(x: np.ndarray) -> float:
    x = finite_1d(x)
    if not x.size:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


def dot_lsq_scale(real: np.ndarray, syn: np.ndarray) -> float:
    r = finite_1d(real)
    s = finite_1d(syn)
    n = min(r.size, s.size)
    if n == 0:
        return float("nan")
    r = r[:n]
    s = s[:n]
    denom = float(np.dot(s, s))
    if denom == 0:
        return float("nan")
    return float(np.dot(r, s) / denom)


def correlation_coeff(a: np.ndarray, b: np.ndarray) -> float:
    x = finite_1d(a)
    y = finite_1d(b)
    n = min(x.size, y.size)
    if n < 3:
        return float("nan")
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(keep):
        return float("nan")
    return float(np.sum(v[keep] * w[keep]) / np.sum(w[keep]))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(keep):
        return float("nan")
    v = v[keep]
    w = w[keep]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cdf, 0.5)])


def clean_for_csv_value(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.10g}"
    return v


# -----------------------------------------------------------------------------
# Parsers and geometry
# -----------------------------------------------------------------------------

def _float_tokens(text: str) -> list[float]:
    vals = []
    for m in _FLOAT_RE.finditer(text.replace("D", "E").replace("d", "e")):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            pass
    return vals


def read_stations(path: Path) -> list[Station]:
    stations: list[Station] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: expected at least 4 columns")
            stations.append(
                Station(
                    station=parts[0],
                    network=parts[1],
                    x_m=float(parts[2]),
                    z_m=float(parts[3]),
                    burial_m=float(parts[4]) if len(parts) > 4 else 0.0,
                    elevation_m=float(parts[5]) if len(parts) > 5 else 0.0,
                )
            )
    if not stations:
        raise ValueError(f"No stations found in {path}")
    return stations


def read_sources_list(path: Path) -> list[Source]:
    sources: list[Source] = []
    if not path.exists():
        return sources

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            lower = s.lower()
            x_val = None
            z_val = None

            for key in ("source_position_x", "source_x", "x_source", "xs", "x"):
                m = re.search(rf"{key}\s*[:=]\s*({_FLOAT_RE.pattern})", lower)
                if m:
                    x_val = float(m.group(1).replace("d", "e"))
                    break

            for key in ("source_position_z", "source_z", "z_source", "zs", "z"):
                m = re.search(rf"{key}\s*[:=]\s*({_FLOAT_RE.pattern})", lower)
                if m:
                    z_val = float(m.group(1).replace("d", "e"))
                    break

            nums = _float_tokens(s)
            if x_val is None and nums:
                if len(nums) >= 3 and abs(nums[0] - round(nums[0])) < 1e-9:
                    x_val = nums[1]
                    z_val = nums[2]
                elif len(nums) >= 2:
                    x_val = nums[0]
                    z_val = nums[1]
                else:
                    x_val = nums[0]

            if x_val is None:
                continue

            first_word = s.split()[0]
            source_id = first_word if not _FLOAT_RE.fullmatch(first_word) else f"S{len(sources)+1:04d}"
            sources.append(Source(source_id=source_id, x_m=float(x_val), z_m=z_val, raw=raw))

    return sources


def real_dat_files(real_dir: Path, first_file: int, last_file: int) -> list[Path]:
    files = []
    for i in range(first_file, last_file + 1):
        p = real_dir / f"{i}.dat"
        if not p.exists():
            raise FileNotFoundError(f"Missing real SEG-2 file: {p}")
        files.append(p)
    return files


def find_synthetic_single_shot_files(model_root: Path, component_file: str) -> list[Path]:
    survey_output = model_root / "SURVEY_OUTPUT"
    search_root = survey_output if survey_output.exists() else model_root
    files = list(search_root.rglob(component_file))
    if not files:
        raise FileNotFoundError(f"No synthetic files named {component_file!r} found below {search_root}")

    def natural_key(p: Path):
        try:
            text = str(p.parent.relative_to(search_root))
        except Exception:
            text = str(p.parent)
        parts = re.split(r"(\d+(?:\.\d+)?)", text)
        key = []
        for part in parts:
            if not part:
                continue
            try:
                key.append(float(part))
            except Exception:
                key.append(part)
        return key

    return sorted(files, key=natural_key)


def parse_int_list(text: str) -> set[int]:
    out = set()
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def real_shot_x_for_file_number(
    file_number: int,
    *,
    first_file: int,
    first_x_m: float,
    dx_m: float,
    duplicate_x_m: Optional[float],
    duplicate_files: set[int],
) -> float:
    if duplicate_x_m is not None and file_number in duplicate_files:
        return float(duplicate_x_m)

    duplicate_files_sorted = sorted(duplicate_files)
    duplicate_shift_count = 0
    if duplicate_x_m is not None and len(duplicate_files_sorted) >= 2:
        for dup_file in duplicate_files_sorted[1:]:
            if file_number > dup_file:
                duplicate_shift_count += 1

    effective_index = (file_number - first_file) - duplicate_shift_count
    return float(first_x_m + effective_index * dx_m)


def nearest_source_for_x(sources: list[Source], x_m: float, tolerance_m: float) -> Source:
    if not sources:
        return Source(source_id=f"x{x_m:.3f}m", x_m=x_m)
    dx = np.asarray([abs(src.x_m - x_m) for src in sources], dtype=float)
    i = int(np.argmin(dx))
    if dx[i] > tolerance_m:
        raise ValueError(
            f"No modeled source within {tolerance_m:g} m of real shot x={x_m:.3f} m; "
            f"nearest is {sources[i].source_id} x={sources[i].x_m:.3f} m"
        )
    src = sources[i]
    return Source(source_id=src.source_id, x_m=src.x_m, z_m=src.z_m, raw=src.raw)


def synthetic_file_for_source(synthetic_files: list[Path], sources: list[Source], source: Source) -> Path:
    if sources:
        matches = [i for i, s in enumerate(sources) if abs(s.x_m - source.x_m) < 1e-9]
        if matches:
            i = matches[0]
            if i < len(synthetic_files):
                return synthetic_files[i]

    m = re.search(r"(\d+)", source.source_id)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(synthetic_files):
            return synthetic_files[i]

    raise ValueError(f"Could not map source {source.source_id} x={source.x_m} to synthetic file")


def build_pairs(
    *,
    real_files: list[Path],
    synthetic_files: list[Path],
    sources: list[Source],
    first_file: int,
    first_x_m: float,
    dx_m: float,
    duplicate_x_m: Optional[float],
    duplicate_files: set[int],
    shot_match_tolerance_m: float,
) -> list[Pair]:
    pairs: list[Pair] = []
    for shot_index, real_file in enumerate(real_files, start=1):
        file_number = int(real_file.stem)
        real_x = real_shot_x_for_file_number(
            file_number,
            first_file=first_file,
            first_x_m=first_x_m,
            dx_m=dx_m,
            duplicate_x_m=duplicate_x_m,
            duplicate_files=duplicate_files,
        )
        src = nearest_source_for_x(sources, real_x, shot_match_tolerance_m)
        syn_file = synthetic_file_for_source(synthetic_files, sources, src)
        pairs.append(
            Pair(
                real_file=real_file,
                synthetic_file=syn_file,
                source=src,
                real_shot_x_m=real_x,
                synthetic_shot_x_m=src.x_m,
                shot_index=shot_index,
            )
        )
    return pairs


# -----------------------------------------------------------------------------
# Data reading and preprocessing
# -----------------------------------------------------------------------------

def read_stream_any(path: Path) -> Stream:
    suffix = path.suffix.lower()
    if suffix == ".dat":
        formats = ["SEG2", None]
    elif suffix == ".su":
        formats = ["SU", None]
    elif suffix in {".sgy", ".segy"}:
        formats = ["SEGY", None]
    else:
        formats = [None, "SEG2", "SU", "SEGY"]

    tried = []
    last_exc = None
    for fmt in formats:
        try:
            if fmt is None:
                tried.append("auto")
                return read(str(path))
            tried.append(fmt)
            return read(str(path), format=fmt)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Could not read {path}; tried {tried}; last error: {last_exc}")


def channel_component(tr: Trace) -> str:
    ch = str(getattr(tr.stats, "channel", "") or "")
    return ch[-1].upper() if ch else ""


def select_component_if_possible(st: Stream, component: Optional[str]) -> Stream:
    if not component:
        return st
    comp = component.upper()
    selected = Stream([tr for tr in st if channel_component(tr) == comp])
    return selected if selected else st


def preprocess_stream(
    st: Stream,
    *,
    demean: bool,
    detrend: bool,
    taper_fraction: float,
    highpass_hz: Optional[float],
    lowpass_hz: Optional[float],
    bandpass: Optional[tuple[float, float]],
    filter_corners: int,
    zerophase: bool,
) -> Stream:
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
        st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=filter_corners, zerophase=zerophase)
    else:
        if highpass_hz is not None and highpass_hz > 0:
            st.filter("highpass", freq=float(highpass_hz), corners=filter_corners, zerophase=zerophase)
        if lowpass_hz is not None and lowpass_hz > 0:
            st.filter("lowpass", freq=float(lowpass_hz), corners=filter_corners, zerophase=zerophase)

    return st


def stream_to_data(st: Stream) -> tuple[np.ndarray, float, float]:
    if len(st) == 0:
        raise ValueError("Empty stream")
    npts = min(int(tr.stats.npts) for tr in st)
    dt = float(st[0].stats.delta)
    sr = float(st[0].stats.sampling_rate)
    data = np.vstack([np.asarray(tr.data[:npts], dtype=np.float64) for tr in st])
    return data, dt, sr


def read_real_gather(
    path: Path,
    *,
    source_x_m: float,
    source_label: str,
    receiver_x_min: float,
    receiver_x_max: float,
    real_first_trace_x_m: float,
    real_dx_m: float,
    component: Optional[str],
    reverse_real_traces: bool,
    preprocess_kwargs: dict,
) -> Gather:
    st = read_stream_any(path)
    st = select_component_if_possible(st, component)
    if reverse_real_traces:
        st = Stream(list(reversed(st)))
    st = preprocess_stream(st, **preprocess_kwargs)

    data, dt, sr = stream_to_data(st)
    rx = real_first_trace_x_m + np.arange(data.shape[0], dtype=float) * real_dx_m
    keep = (rx >= receiver_x_min - 1e-9) & (rx <= receiver_x_max + 1e-9)
    if not np.any(keep):
        raise ValueError(f"No real receivers in {receiver_x_min}..{receiver_x_max} m for {path}")

    data = data[keep, :]
    rx = rx[keep]
    t = np.arange(data.shape[1], dtype=float) * dt

    return Gather(t, data, rx, source_x_m, source_label, path, dt, sr)


def read_synthetic_gather(
    path: Path,
    *,
    stations: list[Station],
    source_x_m: float,
    source_label: str,
    receiver_x_min: float,
    receiver_x_max: float,
    component: Optional[str],
    preprocess_kwargs: dict,
) -> Gather:
    st = read_stream_any(path)
    st = select_component_if_possible(st, component)
    st = preprocess_stream(st, **preprocess_kwargs)

    data, dt, sr = stream_to_data(st)
    if data.shape[0] > len(stations):
        raise ValueError(f"{path}: {data.shape[0]} traces but only {len(stations)} stations")

    rx = np.asarray([sta.x_m for sta in stations[: data.shape[0]]], dtype=float)
    keep = (rx >= receiver_x_min - 1e-9) & (rx <= receiver_x_max + 1e-9)
    if not np.any(keep):
        raise ValueError(f"No synthetic receivers in {receiver_x_min}..{receiver_x_max} m for {path}")

    data = data[keep, :]
    rx = rx[keep]
    order = np.argsort(rx)
    data = data[order, :]
    rx = rx[order]
    t = np.arange(data.shape[1], dtype=float) * dt

    return Gather(t, data, rx, source_x_m, source_label, path, dt, sr)


def interp_time_axis(data: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    out = np.empty((data.shape[0], len(new_t)), dtype=np.float64)
    for i in range(data.shape[0]):
        out[i, :] = np.interp(new_t, old_t, data[i, :], left=0.0, right=0.0)
    return out


def shift_data_time(data: np.ndarray, time_s: np.ndarray, lag_s: float) -> np.ndarray:
    """
    Shift synthetic by lag_s onto the real time axis.

    Sign convention:
        lag_s > 0 means synthetic must be delayed to align with real.
        shifted_syn(t) = syn(t - lag_s)
    """
    out = np.empty_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        out[i, :] = np.interp(time_s - lag_s, time_s, data[i, :], left=0.0, right=0.0)
    return out


def align_real_and_synthetic(
    real: Gather,
    syn: Gather,
    *,
    receiver_tolerance_m: float,
    analysis_tmin: Optional[float],
    analysis_tmax: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = []
    used_syn = set()
    for ir, rx in enumerate(real.receiver_x_m):
        isyn = int(np.argmin(np.abs(syn.receiver_x_m - rx)))
        if isyn in used_syn:
            continue
        if abs(syn.receiver_x_m[isyn] - rx) <= receiver_tolerance_m:
            pairs.append((ir, isyn))
            used_syn.add(isyn)

    if not pairs:
        raise ValueError(f"No common receiver positions for {real.path.name} and {syn.path.name}")

    ir = np.asarray([p[0] for p in pairs], dtype=int)
    isyn = np.asarray([p[1] for p in pairs], dtype=int)

    real_data = real.data[ir, :]
    syn_data = syn.data[isyn, :]
    rx = real.receiver_x_m[ir]
    t = real.time_s.copy()

    syn_data = interp_time_axis(syn_data, syn.time_s, t)

    keep_t = np.ones_like(t, dtype=bool)
    if analysis_tmin is not None:
        keep_t &= t >= analysis_tmin
    if analysis_tmax is not None:
        keep_t &= t <= analysis_tmax

    if not np.any(keep_t):
        raise ValueError("Analysis time window removed all samples")

    return t[keep_t], real_data[:, keep_t], syn_data[:, keep_t], rx


# -----------------------------------------------------------------------------
# Correlation / lag estimation
# -----------------------------------------------------------------------------

def parse_float_pair(text: str, name: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be like 0.02,0.12")
    a, b = float(parts[0]), float(parts[1])
    return min(a, b), max(a, b)


def window_mask(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t >= t0) & (t <= t1)


def normalized_xcorr_lag(
    real: np.ndarray,
    syn: np.ndarray,
    dt_s: float,
    max_lag_s: float,
    demean: bool = True,
    use_abs_peak: bool = False,
) -> dict:
    """
    Estimate lag by normalized cross-correlation.

    Sign convention:
        lag_s > 0 means synthetic should be delayed to align with real.

    Implementation:
        For each integer lag L in samples:
            corr(L) = corrcoef(real[t], syn[t - L])
        Therefore positive L compares real with delayed synthetic.
    """
    r = np.asarray(real, dtype=np.float64)
    s = np.asarray(syn, dtype=np.float64)
    n = min(r.size, s.size)
    r = r[:n]
    s = s[:n]

    if n < 4 or dt_s <= 0:
        return {
            "lag_s": float("nan"),
            "lag_ms": float("nan"),
            "corr": float("nan"),
            "abs_corr": float("nan"),
            "n_samples": n,
        }

    if demean:
        r = r - np.mean(r)
        s = s - np.mean(s)

    max_lag_n = int(round(max_lag_s / dt_s))
    max_lag_n = max(0, min(max_lag_n, n - 3))

    best_lag = 0
    best_corr = float("nan")
    best_score = -np.inf

    for lag in range(-max_lag_n, max_lag_n + 1):
        if lag > 0:
            rr = r[lag:]
            ss = s[:-lag]
        elif lag < 0:
            rr = r[:lag]
            ss = s[-lag:]
        else:
            rr = r
            ss = s

        if rr.size < 4:
            continue

        denom = math.sqrt(float(np.dot(rr, rr) * np.dot(ss, ss)))
        if denom == 0:
            continue

        c = float(np.dot(rr, ss) / denom)
        score = abs(c) if use_abs_peak else c
        if score > best_score:
            best_score = score
            best_corr = c
            best_lag = lag

    lag_s = best_lag * dt_s
    return {
        "lag_s": float(lag_s),
        "lag_ms": float(1000.0 * lag_s),
        "corr": float(best_corr),
        "abs_corr": float(abs(best_corr)) if np.isfinite(best_corr) else float("nan"),
        "n_samples": n,
    }


def compute_weight(real_window: np.ndarray, syn_window: np.ndarray, mode: str) -> float:
    mode = mode.lower()
    rrms = rms(real_window)
    srms = rms(syn_window)
    rmax = maxabs(real_window)

    if mode == "none":
        return 1.0
    if mode == "real_rms":
        return rrms
    if mode == "real_rms2":
        return rrms * rrms
    if mode == "real_maxabs":
        return rmax
    if mode == "real_snr_like":
        # Placeholder SNR-like weight without a separate noise estimate:
        # strong real traces get more weight, but not quadratically.
        return math.sqrt(rrms) if np.isfinite(rrms) and rrms > 0 else 0.0
    if mode == "real_syn_rms_product":
        return rrms * srms

    raise ValueError(f"Unknown weight mode {mode!r}")


def lag_quality_filter(rows: list[dict], *, min_corr: float, min_abs_corr: float) -> list[dict]:
    good = []
    for row in rows:
        c = row.get("corr", float("nan"))
        ac = row.get("abs_corr", float("nan"))
        if np.isfinite(c) and np.isfinite(ac) and c >= min_corr and ac >= min_abs_corr:
            good.append(row)
    return good


def candidate_scales(real: np.ndarray, syn: np.ndarray) -> dict:
    return {
        "scale_rms_real_over_syn": safe_ratio(rms(real), rms(syn)),
        "scale_maxabs_real_over_syn": safe_ratio(maxabs(real), maxabs(syn)),
        "scale_abs_p99_real_over_syn": safe_ratio(abs_percentile(real, 99), abs_percentile(syn, 99)),
        "scale_abs_p999_real_over_syn": safe_ratio(abs_percentile(real, 99.9), abs_percentile(syn, 99.9)),
        "scale_lsq_syn_to_real": dot_lsq_scale(real, syn),
        "corr_real_syn": correlation_coeff(real, syn),
    }


def process_pair(
    *,
    pair: Pair,
    stations: list[Station],
    args,
    preprocess_kwargs: dict,
    corr_window: tuple[float, float],
    scale_window: tuple[float, float],
) -> tuple[list[dict], dict, dict]:
    real = read_real_gather(
        pair.real_file,
        source_x_m=pair.real_shot_x_m,
        source_label=pair.source.source_id,
        receiver_x_min=args.receiver_x_min,
        receiver_x_max=args.receiver_x_max,
        real_first_trace_x_m=args.real_first_trace_x_m,
        real_dx_m=args.real_dx_m,
        component=args.component,
        reverse_real_traces=args.reverse_real_traces,
        preprocess_kwargs=preprocess_kwargs,
    )

    syn = read_synthetic_gather(
        pair.synthetic_file,
        stations=stations,
        source_x_m=pair.synthetic_shot_x_m,
        source_label=pair.source.source_id,
        receiver_x_min=args.receiver_x_min,
        receiver_x_max=args.receiver_x_max,
        component=args.component,
        preprocess_kwargs=preprocess_kwargs,
    )

    t, real_data, syn_data, rx = align_real_and_synthetic(
        real,
        syn,
        receiver_tolerance_m=args.receiver_tolerance_m,
        analysis_tmin=args.analysis_tmin,
        analysis_tmax=args.analysis_tmax,
    )

    dt_s = float(np.median(np.diff(t))) if len(t) > 1 else real.dt_s
    corr_mask = window_mask(t, corr_window[0], corr_window[1])
    if not np.any(corr_mask):
        raise ValueError(f"Correlation window {corr_window} removed all samples")

    scale_mask = window_mask(t, scale_window[0], scale_window[1])
    if not np.any(scale_mask):
        raise ValueError(f"Scale window {scale_window} removed all samples")

    trace_rows: list[dict] = []
    for i, x in enumerate(rx):
        rwin = real_data[i, corr_mask]
        swin = syn_data[i, corr_mask]

        lag_info = normalized_xcorr_lag(
            rwin,
            swin,
            dt_s=dt_s,
            max_lag_s=args.max_lag_ms / 1000.0,
            demean=True,
            use_abs_peak=args.use_abs_corr_peak,
        )

        weight = compute_weight(rwin, swin, args.weight_mode)

        # Correlation at zero lag for reference.
        zero_corr = correlation_coeff(rwin, swin)

        row = {
            "shot_index": pair.shot_index,
            "real_file": pair.real_file.name,
            "synthetic_file": str(pair.synthetic_file),
            "synthetic_shot_folder": pair.synthetic_file.parent.name,
            "source_id": pair.source.source_id,
            "real_shot_x_m": pair.real_shot_x_m,
            "synthetic_source_x_m": pair.synthetic_shot_x_m,
            "receiver_x_m": float(x),
            "offset_real_m": float(x - pair.real_shot_x_m),
            "offset_synthetic_m": float(x - pair.synthetic_shot_x_m),
            "trace_index_1based": i + 1,
            "corr_window_tmin_s": corr_window[0],
            "corr_window_tmax_s": corr_window[1],
            "dt_s": dt_s,
            "max_lag_ms": args.max_lag_ms,
            "zero_lag_corr": zero_corr,
            "lag_s": lag_info["lag_s"],
            "lag_ms": lag_info["lag_ms"],
            "corr": lag_info["corr"],
            "abs_corr": lag_info["abs_corr"],
            "lag_n_samples": int(round(lag_info["lag_s"] / dt_s)) if np.isfinite(lag_info["lag_s"]) else "",
            "n_corr_samples": lag_info["n_samples"],
            "weight_mode": args.weight_mode,
            "weight": weight,
            "real_corrwin_rms": rms(rwin),
            "synthetic_corrwin_rms": rms(swin),
            "real_corrwin_maxabs": maxabs(rwin),
            "synthetic_corrwin_maxabs": maxabs(swin),
            "scale_rms_corrwin": safe_ratio(rms(rwin), rms(swin)),
            "scale_lsq_corrwin": dot_lsq_scale(rwin, swin),
        }
        trace_rows.append(row)

    good_rows = lag_quality_filter(
        trace_rows,
        min_corr=args.min_corr_for_shot_lag,
        min_abs_corr=args.min_abs_corr_for_shot_lag,
    )
    if len(good_rows) < args.min_good_traces_for_shot_lag:
        # Fallback to all finite rows if quality threshold is too strict.
        good_rows = [r for r in trace_rows if np.isfinite(r["lag_ms"])]

    lags_ms = np.asarray([r["lag_ms"] for r in good_rows], dtype=float)
    weights = np.asarray([r["weight"] for r in good_rows], dtype=float)
    corrs = np.asarray([r["corr"] for r in good_rows], dtype=float)
    abs_corrs = np.asarray([r["abs_corr"] for r in good_rows], dtype=float)

    median_lag_ms = float(np.median(lags_ms)) if lags_ms.size else float("nan")
    weighted_median_lag_ms = weighted_median(lags_ms, weights)
    weighted_mean_lag_ms = weighted_mean(lags_ms, weights)

    if args.selected_lag == "median":
        selected_lag_ms = median_lag_ms
    elif args.selected_lag == "weighted_median":
        selected_lag_ms = weighted_median_lag_ms
    elif args.selected_lag == "weighted_mean":
        selected_lag_ms = weighted_mean_lag_ms
    else:
        raise ValueError(f"Unknown selected lag method {args.selected_lag!r}")

    selected_lag_s = selected_lag_ms / 1000.0 if np.isfinite(selected_lag_ms) else 0.0
    syn_shifted = shift_data_time(syn_data, t, selected_lag_s)

    before_scale = candidate_scales(real_data[:, scale_mask], syn_data[:, scale_mask])
    after_scale = candidate_scales(real_data[:, scale_mask], syn_shifted[:, scale_mask])

    # Per-trace after-alignment correlations and scales in scale window.
    after_trace_corrs = []
    after_trace_rms_scales = []
    after_trace_lsq_scales = []
    for i in range(real_data.shape[0]):
        r = real_data[i, scale_mask]
        s = syn_shifted[i, scale_mask]
        after_trace_corrs.append(correlation_coeff(r, s))
        after_trace_rms_scales.append(safe_ratio(rms(r), rms(s)))
        after_trace_lsq_scales.append(dot_lsq_scale(r, s))

    shot_row = {
        "shot_index": pair.shot_index,
        "real_file": pair.real_file.name,
        "synthetic_file": str(pair.synthetic_file),
        "synthetic_shot_folder": pair.synthetic_file.parent.name,
        "source_id": pair.source.source_id,
        "real_shot_x_m": pair.real_shot_x_m,
        "synthetic_source_x_m": pair.synthetic_shot_x_m,
        "n_receivers": len(rx),
        "receiver_x_min_m": float(np.min(rx)),
        "receiver_x_max_m": float(np.max(rx)),
        "analysis_tmin_s": args.analysis_tmin,
        "analysis_tmax_s": args.analysis_tmax,
        "corr_window_tmin_s": corr_window[0],
        "corr_window_tmax_s": corr_window[1],
        "scale_window_tmin_s": scale_window[0],
        "scale_window_tmax_s": scale_window[1],
        "dt_s": dt_s,
        "max_lag_ms": args.max_lag_ms,
        "weight_mode": args.weight_mode,
        "selected_lag_method": args.selected_lag,
        "n_trace_lags_total": len(trace_rows),
        "n_trace_lags_used": len(good_rows),
        "median_lag_ms": median_lag_ms,
        "mad_lag_ms": mad(lags_ms),
        "weighted_median_lag_ms": weighted_median_lag_ms,
        "weighted_mean_lag_ms": weighted_mean_lag_ms,
        "selected_lag_ms": selected_lag_ms,
        "median_corr_used": float(np.median(corrs)) if corrs.size else float("nan"),
        "median_abs_corr_used": float(np.median(abs_corrs)) if abs_corrs.size else float("nan"),
        "mean_corr_used": float(np.mean(corrs)) if corrs.size else float("nan"),
        "mean_abs_corr_used": float(np.mean(abs_corrs)) if abs_corrs.size else float("nan"),
        "median_zero_lag_corr_all": float(np.median([r["zero_lag_corr"] for r in trace_rows if np.isfinite(r["zero_lag_corr"])])),
        "median_after_align_trace_corr": float(np.median(finite_1d(np.asarray(after_trace_corrs)))),
        "median_after_align_trace_rms_scale": float(np.median(finite_1d(np.asarray(after_trace_rms_scales)))),
        "median_after_align_trace_lsq_scale": float(np.median(finite_1d(np.asarray(after_trace_lsq_scales)))),
    }

    scale_row = {
        "shot_index": pair.shot_index,
        "real_file": pair.real_file.name,
        "source_id": pair.source.source_id,
        "real_shot_x_m": pair.real_shot_x_m,
        "synthetic_source_x_m": pair.synthetic_shot_x_m,
        "selected_lag_ms": selected_lag_ms,
        "scale_window_tmin_s": scale_window[0],
        "scale_window_tmax_s": scale_window[1],
        "before_corr_real_syn": before_scale["corr_real_syn"],
        "after_corr_real_syn": after_scale["corr_real_syn"],
        "before_scale_rms_real_over_syn": before_scale["scale_rms_real_over_syn"],
        "after_scale_rms_real_over_syn": after_scale["scale_rms_real_over_syn"],
        "before_scale_lsq_syn_to_real": before_scale["scale_lsq_syn_to_real"],
        "after_scale_lsq_syn_to_real": after_scale["scale_lsq_syn_to_real"],
        "before_scale_abs_p99_real_over_syn": before_scale["scale_abs_p99_real_over_syn"],
        "after_scale_abs_p99_real_over_syn": after_scale["scale_abs_p99_real_over_syn"],
        "before_scale_abs_p999_real_over_syn": before_scale["scale_abs_p999_real_over_syn"],
        "after_scale_abs_p999_real_over_syn": after_scale["scale_abs_p999_real_over_syn"],
    }

    return trace_rows, shot_row, scale_row


# -----------------------------------------------------------------------------
# CSV writing
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: clean_for_csv_value(v) for k, v in row.items()})


def parse_bandpass(text: Optional[str]) -> Optional[tuple[float, float]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--bandpass must be like 5,150")
    return float(parts[0]), float(parts[1])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate real-vs-synthetic gather-wide timing lags by trace cross-correlation."
    )

    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--synthetic-novoid-dir", required=True, type=Path)
    p.add_argument("--real-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--stations-file", default="STATIONS")
    p.add_argument("--sources-file", default="SOURCES_LIST.txt")
    p.add_argument("--component-file", default="Uz_file_single_v.su")

    p.add_argument("--real-first-file", type=int, default=3005)
    p.add_argument("--real-last-file", type=int, default=3046)
    p.add_argument("--real-shot-first-x-m", type=float, default=82.5)
    p.add_argument("--real-shot-dx-m", type=float, default=2.0)
    p.add_argument("--real-shot-duplicate-x-m", type=float, default=102.5)
    p.add_argument("--real-shot-duplicate-files", default="3015,3016")
    p.add_argument("--shot-match-tolerance-m", type=float, default=0.05)

    p.add_argument("--receiver-x-min", type=float, default=87.0)
    p.add_argument("--receiver-x-max", type=float, default=158.0)
    p.add_argument("--real-first-trace-x-m", type=float, default=87.0)
    p.add_argument("--real-dx-m", type=float, default=1.0)
    p.add_argument("--reverse-real-traces", action="store_true")
    p.add_argument("--receiver-tolerance-m", type=float, default=0.05)

    p.add_argument("--component", default=None)

    p.add_argument("--analysis-tmin", type=float, default=0.0)
    p.add_argument("--analysis-tmax", type=float, default=0.4)
    p.add_argument("--corr-window", default="0.02,0.12")
    p.add_argument("--scale-window", default="0.02,0.12")
    p.add_argument("--max-lag-ms", type=float, default=50.0)
    p.add_argument("--use-abs-corr-peak", action="store_true",
                   help="Use absolute correlation peak. Default uses positive peak only.")

    p.add_argument("--weight-mode",
                   choices=["none", "real_rms", "real_rms2", "real_maxabs", "real_snr_like", "real_syn_rms_product"],
                   default="real_rms")
    p.add_argument("--selected-lag",
                   choices=["median", "weighted_median", "weighted_mean"],
                   default="weighted_median")

    p.add_argument("--min-corr-for-shot-lag", type=float, default=-1.0,
                   help="Minimum signed correlation for a trace lag to contribute. Default -1 includes all.")
    p.add_argument("--min-abs-corr-for-shot-lag", type=float, default=0.0,
                   help="Minimum abs correlation for a trace lag to contribute. Default 0 includes all.")
    p.add_argument("--min-good-traces-for-shot-lag", type=int, default=8,
                   help="Fallback to all finite lags if fewer than this pass quality thresholds.")

    p.add_argument("--demean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detrend", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taper-fraction", type=float, default=0.05)
    p.add_argument("--highpass-hz", type=float, default=10.0)
    p.add_argument("--lowpass-hz", type=float, default=None)
    p.add_argument("--bandpass", type=parse_bandpass, default=None)
    p.add_argument("--filter-corners", type=int, default=4)
    p.add_argument("--zerophase", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--limit", type=int, default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stations = read_stations(args.data_dir / args.stations_file)
    sources = read_sources_list(args.data_dir / args.sources_file)

    print(f"Read {len(stations)} synthetic receivers from {args.data_dir / args.stations_file}")
    print(f"Read {len(sources)} modeled sources from {args.data_dir / args.sources_file}")

    real_files = real_dat_files(args.real_dir, args.real_first_file, args.real_last_file)
    synthetic_files = find_synthetic_single_shot_files(args.synthetic_novoid_dir, args.component_file)

    pairs = build_pairs(
        real_files=real_files,
        synthetic_files=synthetic_files,
        sources=sources,
        first_file=args.real_first_file,
        first_x_m=args.real_shot_first_x_m,
        dx_m=args.real_shot_dx_m,
        duplicate_x_m=args.real_shot_duplicate_x_m,
        duplicate_files=parse_int_list(args.real_shot_duplicate_files),
        shot_match_tolerance_m=args.shot_match_tolerance_m,
    )

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Built {len(pairs)} real/synthetic pairs")
    if pairs:
        print("Real shot x check:")
        preview = pairs[:3] + pairs[9:13] + pairs[-3:]
        seen = set()
        for pcheck in preview:
            key = pcheck.real_file.name
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  {pcheck.real_file.name}: real_x={pcheck.real_shot_x_m:.3f} m "
                f"-> synthetic_x={pcheck.synthetic_shot_x_m:.3f} m"
            )

    corr_window = parse_float_pair(args.corr_window, "--corr-window")
    scale_window = parse_float_pair(args.scale_window, "--scale-window")

    preprocess_kwargs = dict(
        demean=args.demean,
        detrend=args.detrend,
        taper_fraction=args.taper_fraction,
        highpass_hz=args.highpass_hz,
        lowpass_hz=args.lowpass_hz,
        bandpass=args.bandpass,
        filter_corners=args.filter_corners,
        zerophase=args.zerophase,
    )

    all_trace_rows: list[dict] = []
    shot_rows: list[dict] = []
    scale_rows: list[dict] = []
    failures: list[dict] = []

    for i, pair in enumerate(pairs, start=1):
        print(
            f"[{i}/{len(pairs)}] {pair.real_file.name} "
            f"real_x={pair.real_shot_x_m:.3f} m -> "
            f"{pair.synthetic_file.parent.name}/{pair.synthetic_file.name}"
        )
        try:
            trace_rows, shot_row, scale_row = process_pair(
                pair=pair,
                stations=stations,
                args=args,
                preprocess_kwargs=preprocess_kwargs,
                corr_window=corr_window,
                scale_window=scale_window,
            )
            all_trace_rows.extend(trace_rows)
            shot_rows.append(shot_row)
            scale_rows.append(scale_row)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {msg}", file=sys.stderr)
            failures.append(
                {
                    "shot_index": pair.shot_index,
                    "real_file": pair.real_file.name,
                    "synthetic_file": str(pair.synthetic_file),
                    "real_shot_x_m": pair.real_shot_x_m,
                    "synthetic_source_x_m": pair.synthetic_shot_x_m,
                    "error": msg,
                }
            )

    write_csv(args.output_dir / "trace_lag_metrics.csv", all_trace_rows)
    write_csv(args.output_dir / "shot_lag_metrics.csv", shot_rows)
    write_csv(args.output_dir / "aligned_window_scale_metrics.csv", scale_rows)
    write_csv(args.output_dir / "failures.csv", failures)

    config_path = args.output_dir / "lag_run_config.txt"
    with config_path.open("w", encoding="utf-8") as f:
        f.write("63_estimate_real_synthetic_lags.py\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"synthetic_novoid_dir: {args.synthetic_novoid_dir}\n")
        f.write(f"real_dir: {args.real_dir}\n")
        f.write(f"component_file: {args.component_file}\n")
        f.write(f"analysis_tmin/tmax: {args.analysis_tmin}, {args.analysis_tmax}\n")
        f.write(f"corr_window: {args.corr_window}\n")
        f.write(f"scale_window: {args.scale_window}\n")
        f.write(f"max_lag_ms: {args.max_lag_ms}\n")
        f.write(f"weight_mode: {args.weight_mode}\n")
        f.write(f"selected_lag: {args.selected_lag}\n")
        f.write(f"preprocess: {preprocess_kwargs}\n")

    print(f"Wrote {args.output_dir / 'trace_lag_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'shot_lag_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'aligned_window_scale_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'failures.csv'}")
    print(f"Wrote {config_path}")
    print(f"Completed {len(shot_rows)} / {len(pairs)} shot pairs")
    return 0 if shot_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
