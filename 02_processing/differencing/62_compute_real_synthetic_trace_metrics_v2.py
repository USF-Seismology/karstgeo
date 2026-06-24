#!/usr/bin/env python3
"""
62_compute_real_synthetic_trace_metrics.py

Compute diagnostic amplitude, noise, frequency, and scaling metrics for
paired real Geode SEG-2 shot gathers and synthetic SPECFEM2D no-void SU gathers.

Purpose
-------
Before deciding how to scale real-vs-synthetic data, compute objective metrics
for every matched trace in every matched shot gather.

This script writes:

    1. trace_metrics.csv
       One row per matched real/synthetic receiver trace.

    2. shot_metrics.csv
       One row per shot gather, including candidate scale factors:
           - rms scale
           - maxabs scale
           - robust percentile scale
           - least-squares scale
           - median of per-trace RMS ratios
           - median of per-trace LSQ scales

    3. window_metrics.csv
       One row per shot per analysis time window.

    4. failures.csv
       Any files/shots that could not be read or matched.

It does NOT change or plot the data.

Geometry assumptions
--------------------
Real files:
    3005.dat to 3046.dat

Shot positions:
    3005.dat = 82.5 m
    then +2 m per file,
    except 3015.dat and 3016.dat are both 102.5 m.

Receivers:
    real Geode traces are 72 geophones from 87 m to 158 m at 1 m spacing.

Synthetic:
    no-void model root contains SURVEY_OUTPUT/**/Uz_file_single_v.su
    or Ux_file_single_v.su.

The synthetic model DATA/STATIONS provides receiver geometry. Synthetic receivers
are clipped to the same 87..158 m interval.

Important
---------
Real Geode instrument sensitivities are unknown. These metrics are therefore
for empirical scaling/normalization decisions only; they are not an absolute
instrument-calibrated physical amplitude comparison.

Examples
--------
python 62_compute_real_synthetic_trace_metrics.py \
  --data-dir /path/to/NO_VOID_MODEL/DATA \
  --synthetic-novoid-dir /path/to/NO_VOID_MODEL \
  --real-dir /path/to/051826_Seismics_T1 \
  --output-dir /path/to/differencing/real_vs_synthetic_metrics \
  --component-file Uz_file_single_v.su

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
# Basic numerical helpers
# -----------------------------------------------------------------------------

def finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def safe_float(x) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return float("nan")
    except Exception:
        return float("nan")


def mean(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.mean(x)) if x.size else float("nan")


def median(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.median(x)) if x.size else float("nan")


def std(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.std(x)) if x.size else float("nan")


def rms(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def mad(x: np.ndarray) -> float:
    x = finite_1d(x)
    if not x.size:
        return float("nan")
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def robust_sigma_mad(x: np.ndarray) -> float:
    m = mad(x)
    return float(1.4826 * m) if np.isfinite(m) else float("nan")


def maxabs(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def peak_to_peak(x: np.ndarray) -> float:
    x = finite_1d(x)
    return float(np.max(x) - np.min(x)) if x.size else float("nan")


def abs_percentile(x: np.ndarray, pct: float) -> float:
    x = finite_1d(x)
    return float(np.percentile(np.abs(x), pct)) if x.size else float("nan")


def energy(x: np.ndarray, dt_s: float) -> float:
    x = finite_1d(x)
    return float(np.sum(x * x) * dt_s) if x.size else float("nan")


def zero_crossing_rate(x: np.ndarray, dt_s: float) -> float:
    x = finite_1d(x)
    if x.size < 2 or dt_s <= 0:
        return float("nan")
    s = np.signbit(x)
    crossings = np.count_nonzero(s[1:] != s[:-1])
    duration = (x.size - 1) * dt_s
    return float(crossings / duration) if duration > 0 else float("nan")


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


def correlation(real: np.ndarray, syn: np.ndarray) -> float:
    r = finite_1d(real)
    s = finite_1d(syn)
    n = min(r.size, s.size)
    if n < 3:
        return float("nan")
    r = r[:n] - np.mean(r[:n])
    s = s[:n] - np.mean(s[:n])
    denom = np.sqrt(np.dot(r, r) * np.dot(s, s))
    if denom == 0:
        return float("nan")
    return float(np.dot(r, s) / denom)


def safe_ratio(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return float("nan")
    if num is None or not np.isfinite(num):
        return float("nan")
    return float(num / den)


def dominant_frequency(x: np.ndarray, dt_s: float, fmin: float = 0.0, fmax: Optional[float] = None) -> float:
    x = finite_1d(x)
    if x.size < 4 or dt_s <= 0:
        return float("nan")
    y = x - np.mean(x)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dt_s)
    keep = freqs >= fmin
    if fmax is not None:
        keep &= freqs <= fmax
    keep &= freqs > 0
    if not np.any(keep):
        return float("nan")
    idx_local = np.argmax(spec[keep])
    return float(freqs[keep][idx_local])


def spectral_centroid(x: np.ndarray, dt_s: float, fmin: float = 0.0, fmax: Optional[float] = None) -> float:
    x = finite_1d(x)
    if x.size < 4 or dt_s <= 0:
        return float("nan")
    y = x - np.mean(x)
    amp = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dt_s)
    keep = freqs >= fmin
    if fmax is not None:
        keep &= freqs <= fmax
    keep &= freqs > 0
    amp = amp[keep]
    freqs = freqs[keep]
    denom = np.sum(amp)
    if denom == 0:
        return float("nan")
    return float(np.sum(freqs * amp) / denom)


def band_rms(x: np.ndarray, dt_s: float, fmin: float, fmax: float) -> float:
    """
    Frequency-domain band-limited RMS diagnostic.

    This is not intended as a filtered time series replacement. It is a compact
    metric for comparing relative spectral energy by band.
    """
    x = finite_1d(x)
    if x.size < 4 or dt_s <= 0:
        return float("nan")
    y = x - np.mean(x)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=dt_s)
    keep = (freqs >= fmin) & (freqs < fmax)
    if not np.any(keep):
        return float("nan")
    # Parseval-compatible relative metric.
    power = np.abs(spec[keep]) ** 2
    return float(np.sqrt(np.mean(power)))


def first_exceedance_time(x: np.ndarray, t: np.ndarray, noise_rms_value: float, threshold: float) -> float:
    x = finite_1d(x)
    if x.size == 0 or not np.isfinite(noise_rms_value) or noise_rms_value <= 0:
        return float("nan")
    n = min(x.size, t.size)
    y = np.abs(x[:n])
    idx = np.where(y >= threshold * noise_rms_value)[0]
    if idx.size == 0:
        return float("nan")
    return float(t[idx[0]])


# -----------------------------------------------------------------------------
# Parsers and file discovery
# -----------------------------------------------------------------------------

def _float_tokens(text: str) -> list[float]:
    vals: list[float] = []
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
    files: list[Path] = []
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
    out: set[int] = set()
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
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
    """
    Map real SEG-2 file number to shot x position.

    Normal case:
        3005 = 82.5 m
        3006 = 84.5 m
        ...

    Special case:
        3015 = 102.5 m
        3016 = 102.5 m

    After the duplicate, the sequence must continue from the duplicate location:
        3017 = 104.5 m
        3018 = 106.5 m
        ...
        3046 = 162.5 m

    Therefore, for files after a duplicated shot, subtract the number of extra
    duplicate entries that occurred before that file.
    """
    if duplicate_x_m is not None and file_number in duplicate_files:
        return float(duplicate_x_m)

    duplicate_files_sorted = sorted(duplicate_files)

    # Count duplicate files that occurred before this file, beyond the first
    # member of each duplicated location. For the usual case 3015,3016 this
    # subtracts 1 from every file after 3016.
    duplicate_shift_count = 0
    if duplicate_x_m is not None and len(duplicate_files_sorted) >= 2:
        first_duplicate_file = duplicate_files_sorted[0]
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


def synthetic_file_for_source(
    synthetic_files: list[Path],
    sources: list[Source],
    source: Source,
) -> Path:
    """
    Map source to synthetic file by index in SOURCES_LIST.

    This assumes SURVEY_OUTPUT single-shot folders sort in the same order as
    SOURCES_LIST. That matched the previous comparison workflow.
    """
    if sources:
        matches = [i for i, s in enumerate(sources) if abs(s.x_m - source.x_m) < 1e-9]
        if matches:
            i = matches[0]
            if i < len(synthetic_files):
                return synthetic_files[i]

    # fallback: source_id S0001-like
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
# Reading data
# -----------------------------------------------------------------------------

def read_stream_any(path: Path) -> Stream:
    suffix = path.suffix.lower()
    formats = []
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

    return Gather(
        time_s=t,
        data=data,
        receiver_x_m=rx,
        source_x_m=source_x_m,
        label=source_label,
        path=path,
        dt_s=dt,
        sampling_rate_hz=sr,
    )


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

    return Gather(
        time_s=t,
        data=data,
        receiver_x_m=rx,
        source_x_m=source_x_m,
        label=source_label,
        path=path,
        dt_s=dt,
        sampling_rate_hz=sr,
    )


def interp_time_axis(data: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    out = np.empty((data.shape[0], len(new_t)), dtype=np.float64)
    for i in range(data.shape[0]):
        out[i, :] = np.interp(new_t, old_t, data[i, :], left=0.0, right=0.0)
    return out


def align_real_and_synthetic(
    real: Gather,
    syn: Gather,
    *,
    receiver_tolerance_m: float,
    analysis_tmin: Optional[float],
    analysis_tmax: Optional[float],
    resample_synthetic_to_real: bool = True,
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

    if resample_synthetic_to_real:
        syn_data = interp_time_axis(syn_data, syn.time_s, t)
    else:
        n = min(real_data.shape[1], syn_data.shape[1], t.size)
        real_data = real_data[:, :n]
        syn_data = syn_data[:, :n]
        t = t[:n]

    keep_t = np.ones_like(t, dtype=bool)
    if analysis_tmin is not None:
        keep_t &= t >= analysis_tmin
    if analysis_tmax is not None:
        keep_t &= t <= analysis_tmax
    if not np.any(keep_t):
        raise ValueError("Analysis time window removed all samples")

    return t[keep_t], real_data[:, keep_t], syn_data[:, keep_t], rx


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def parse_windows(text: str) -> list[tuple[str, Optional[float], Optional[float]]]:
    """
    Parse windows like:
        full:0:0.6,early:0.02:0.12,pre:0:0.02,late:0.12:0.6
    """
    windows = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad window {item!r}; expected name:tmin:tmax")
        name = parts[0]
        t0 = None if parts[1] == "" else float(parts[1])
        t1 = None if parts[2] == "" else float(parts[2])
        windows.append((name, t0, t1))
    if not windows:
        windows.append(("full", None, None))
    return windows


def window_mask(t: np.ndarray, tmin: Optional[float], tmax: Optional[float]) -> np.ndarray:
    keep = np.ones_like(t, dtype=bool)
    if tmin is not None:
        keep &= t >= tmin
    if tmax is not None:
        keep &= t <= tmax
    return keep


def metrics_for_array(x: np.ndarray, t: np.ndarray, *, max_freq_hz: float, noise_rms_value: Optional[float] = None) -> dict:
    dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    r = rms(x)
    ma = maxabs(x)
    p95 = abs_percentile(x, 95)
    p99 = abs_percentile(x, 99)
    p999 = abs_percentile(x, 99.9)
    out = {
        "mean": mean(x),
        "median": median(x),
        "std": std(x),
        "rms": r,
        "mad": mad(x),
        "robust_sigma_mad": robust_sigma_mad(x),
        "maxabs": ma,
        "peak_to_peak": peak_to_peak(x),
        "abs_p95": p95,
        "abs_p99": p99,
        "abs_p999": p999,
        "energy": energy(x, dt),
        "zero_crossing_rate_hz": zero_crossing_rate(x, dt),
        "dominant_freq_hz": dominant_frequency(x, dt, fmin=0.5, fmax=max_freq_hz),
        "spectral_centroid_hz": spectral_centroid(x, dt, fmin=0.5, fmax=max_freq_hz),
        "band_rms_0_10": band_rms(x, dt, 0.0, 10.0),
        "band_rms_10_30": band_rms(x, dt, 10.0, 30.0),
        "band_rms_30_80": band_rms(x, dt, 30.0, 80.0),
        "band_rms_80_150": band_rms(x, dt, 80.0, 150.0),
    }
    if noise_rms_value is not None:
        out["snr_rms_over_noise"] = safe_ratio(r, noise_rms_value)
        out["snr_maxabs_over_noise"] = safe_ratio(ma, noise_rms_value)
        out["first_time_abs_gt_3x_noise_s"] = first_exceedance_time(x, t, noise_rms_value, 3.0)
        out["first_time_abs_gt_5x_noise_s"] = first_exceedance_time(x, t, noise_rms_value, 5.0)
    return out


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def candidate_scales(real: np.ndarray, syn: np.ndarray) -> dict:
    real_rms = rms(real)
    syn_rms = rms(syn)
    real_max = maxabs(real)
    syn_max = maxabs(syn)
    real_p99 = abs_percentile(real, 99)
    syn_p99 = abs_percentile(syn, 99)
    real_p999 = abs_percentile(real, 99.9)
    syn_p999 = abs_percentile(syn, 99.9)

    return {
        "scale_rms_real_over_syn": safe_ratio(real_rms, syn_rms),
        "scale_maxabs_real_over_syn": safe_ratio(real_max, syn_max),
        "scale_abs_p99_real_over_syn": safe_ratio(real_p99, syn_p99),
        "scale_abs_p999_real_over_syn": safe_ratio(real_p999, syn_p999),
        "scale_lsq_syn_to_real": dot_lsq_scale(real, syn),
        "corr_real_syn": correlation(real, syn),
    }


def trace_rows_for_pair(
    *,
    pair: Pair,
    t: np.ndarray,
    real_data: np.ndarray,
    syn_data: np.ndarray,
    rx: np.ndarray,
    windows: list[tuple[str, Optional[float], Optional[float]]],
    max_freq_hz: float,
    noise_window_name: str,
) -> list[dict]:
    rows: list[dict] = []

    # Compute optional per-trace real noise RMS from the named window.
    noise_masks = {name: window_mask(t, t0, t1) for name, t0, t1 in windows}
    noise_mask = noise_masks.get(noise_window_name)
    if noise_mask is None or not np.any(noise_mask):
        noise_rms_per_trace = np.full(real_data.shape[0], np.nan)
    else:
        noise_rms_per_trace = np.asarray([rms(real_data[i, noise_mask]) for i in range(real_data.shape[0])])

    for itrace, x in enumerate(rx, start=1):
        for window_name, t0, t1 in windows:
            keep = window_mask(t, t0, t1)
            if not np.any(keep):
                continue

            r = real_data[itrace - 1, keep]
            s = syn_data[itrace - 1, keep]
            tt = t[keep]
            n_rms = noise_rms_per_trace[itrace - 1]

            real_m = metrics_for_array(r, tt, max_freq_hz=max_freq_hz, noise_rms_value=n_rms)
            syn_m = metrics_for_array(s, tt, max_freq_hz=max_freq_hz)
            scale_m = candidate_scales(r, s)

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
                "trace_index_1based": itrace,
                "window": window_name,
                "window_tmin_s": t0,
                "window_tmax_s": t1,
                "n_samples": int(np.count_nonzero(keep)),
                "dt_s": float(np.median(np.diff(tt))) if tt.size > 1 else float("nan"),
                "noise_window": noise_window_name,
                "real_noise_rms_from_noise_window": n_rms,
            }
            row.update(prefix_dict("real", real_m))
            row.update(prefix_dict("synthetic", syn_m))
            row.update(scale_m)
            rows.append(row)

    return rows


def shot_window_rows_for_pair(
    *,
    pair: Pair,
    t: np.ndarray,
    real_data: np.ndarray,
    syn_data: np.ndarray,
    rx: np.ndarray,
    windows: list[tuple[str, Optional[float], Optional[float]]],
    max_freq_hz: float,
    noise_window_name: str,
) -> list[dict]:
    rows: list[dict] = []

    noise_mask = None
    for name, t0, t1 in windows:
        if name == noise_window_name:
            noise_mask = window_mask(t, t0, t1)
            break

    noise_rms = rms(real_data[:, noise_mask]) if noise_mask is not None and np.any(noise_mask) else float("nan")

    for window_name, t0, t1 in windows:
        keep = window_mask(t, t0, t1)
        if not np.any(keep):
            continue

        r = real_data[:, keep]
        s = syn_data[:, keep]
        tt = t[keep]

        real_m = metrics_for_array(r, tt, max_freq_hz=max_freq_hz, noise_rms_value=noise_rms)
        syn_m = metrics_for_array(s, tt, max_freq_hz=max_freq_hz)
        scale_m = candidate_scales(r, s)

        # Per-trace scaling distributions for this shot/window.
        per_trace_rms_scales = []
        per_trace_lsq_scales = []
        per_trace_corrs = []
        for i in range(r.shape[0]):
            per_trace_rms_scales.append(safe_ratio(rms(r[i, :]), rms(s[i, :])))
            per_trace_lsq_scales.append(dot_lsq_scale(r[i, :], s[i, :]))
            per_trace_corrs.append(correlation(r[i, :], s[i, :]))

        row = {
            "shot_index": pair.shot_index,
            "real_file": pair.real_file.name,
            "synthetic_file": str(pair.synthetic_file),
            "synthetic_shot_folder": pair.synthetic_file.parent.name,
            "source_id": pair.source.source_id,
            "real_shot_x_m": pair.real_shot_x_m,
            "synthetic_source_x_m": pair.synthetic_shot_x_m,
            "window": window_name,
            "window_tmin_s": t0,
            "window_tmax_s": t1,
            "n_receivers": int(r.shape[0]),
            "receiver_x_min_m": float(np.min(rx)),
            "receiver_x_max_m": float(np.max(rx)),
            "n_samples": int(np.count_nonzero(keep)),
            "dt_s": float(np.median(np.diff(tt))) if tt.size > 1 else float("nan"),
            "noise_window": noise_window_name,
            "real_noise_rms_from_noise_window": noise_rms,
            "median_per_trace_scale_rms": median(np.asarray(per_trace_rms_scales)),
            "mad_per_trace_scale_rms": mad(np.asarray(per_trace_rms_scales)),
            "median_per_trace_scale_lsq": median(np.asarray(per_trace_lsq_scales)),
            "mad_per_trace_scale_lsq": mad(np.asarray(per_trace_lsq_scales)),
            "median_per_trace_corr": median(np.asarray(per_trace_corrs)),
        }
        row.update(prefix_dict("real", real_m))
        row.update(prefix_dict("synthetic", syn_m))
        row.update(scale_m)
        rows.append(row)

    return rows


def shot_summary_row(pair: Pair, shot_window_rows: list[dict], preferred_window: str) -> dict:
    candidates = [r for r in shot_window_rows if r["window"] == preferred_window]
    if not candidates and shot_window_rows:
        candidates = [shot_window_rows[0]]
    r = candidates[0] if candidates else {}

    return {
        "shot_index": pair.shot_index,
        "real_file": pair.real_file.name,
        "synthetic_file": str(pair.synthetic_file),
        "synthetic_shot_folder": pair.synthetic_file.parent.name,
        "source_id": pair.source.source_id,
        "real_shot_x_m": pair.real_shot_x_m,
        "synthetic_source_x_m": pair.synthetic_shot_x_m,
        "preferred_window": r.get("window", preferred_window),
        "n_receivers": r.get("n_receivers", ""),
        "n_samples": r.get("n_samples", ""),
        "real_rms": r.get("real_rms", ""),
        "synthetic_rms": r.get("synthetic_rms", ""),
        "scale_rms_real_over_syn": r.get("scale_rms_real_over_syn", ""),
        "scale_maxabs_real_over_syn": r.get("scale_maxabs_real_over_syn", ""),
        "scale_abs_p99_real_over_syn": r.get("scale_abs_p99_real_over_syn", ""),
        "scale_abs_p999_real_over_syn": r.get("scale_abs_p999_real_over_syn", ""),
        "scale_lsq_syn_to_real": r.get("scale_lsq_syn_to_real", ""),
        "median_per_trace_scale_rms": r.get("median_per_trace_scale_rms", ""),
        "median_per_trace_scale_lsq": r.get("median_per_trace_scale_lsq", ""),
        "median_per_trace_corr": r.get("median_per_trace_corr", ""),
        "corr_real_syn": r.get("corr_real_syn", ""),
        "real_noise_rms_from_noise_window": r.get("real_noise_rms_from_noise_window", ""),
        "real_snr_rms_over_noise": r.get("real_snr_rms_over_noise", ""),
        "real_snr_maxabs_over_noise": r.get("real_snr_maxabs_over_noise", ""),
    }


# -----------------------------------------------------------------------------
# CSV writing
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Preserve first-row order, then append any later keys.
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, (np.floating, np.integer)):
                    v = v.item()
                if isinstance(v, float):
                    if math.isnan(v):
                        clean[k] = ""
                    else:
                        clean[k] = f"{v:.10g}"
                else:
                    clean[k] = v
            w.writerow(clean)


def parse_bandpass(text: Optional[str]) -> Optional[tuple[float, float]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--bandpass must be like 5,150")
    return (float(parts[0]), float(parts[1]))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute real/synthetic per-trace and per-shot metrics for scaling diagnostics."
    )

    p.add_argument("--data-dir", required=True, type=Path,
                   help="Synthetic no-void DATA directory containing STATIONS and SOURCES_LIST.txt.")
    p.add_argument("--synthetic-novoid-dir", required=True, type=Path,
                   help="Synthetic no-void model root containing SURVEY_OUTPUT.")
    p.add_argument("--real-dir", required=True, type=Path,
                   help="Directory containing real Geode SEG-2 .dat files.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output directory for CSV files.")

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

    p.add_argument("--component", default=None,
                   help="Optional ObsPy component selector. Usually not needed.")

    p.add_argument("--analysis-tmin", type=float, default=0.0)
    p.add_argument("--analysis-tmax", type=float, default=0.6)
    p.add_argument("--windows", default="full:0:0.6,pre:0:0.02,early:0.02:0.12,mid:0.12:0.30,late:0.30:0.60",
                   help="Comma-separated windows name:tmin:tmax.")
    p.add_argument("--preferred-scale-window", default="early",
                   help="Window used for shot_metrics headline scale factors. Default: early")
    p.add_argument("--noise-window-name", default="pre",
                   help="Window used as within-shot real noise estimate. Default: pre")

    p.add_argument("--max-freq-hz", type=float, default=150.0)

    p.add_argument("--demean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detrend", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taper-fraction", type=float, default=0.0,
                   help="Cosine taper fraction applied before filtering. Default: 0")
    p.add_argument("--highpass-hz", type=float, default=None,
                   help="Optional highpass filter frequency.")
    p.add_argument("--lowpass-hz", type=float, default=None,
                   help="Optional lowpass filter frequency.")
    p.add_argument("--bandpass", type=parse_bandpass, default=None,
                   help="Optional bandpass as fmin,fmax, e.g. 5,150")
    p.add_argument("--filter-corners", type=int, default=4)
    p.add_argument("--zerophase", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--limit", type=int, default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = read_stations(args.data_dir / args.stations_file)
    sources = read_sources_list(args.data_dir / args.sources_file)

    print(f"Read {len(stations)} synthetic receivers from {args.data_dir / args.stations_file}")
    print(f"Read {len(sources)} modeled sources from {args.data_dir / args.sources_file}")

    real_files = real_dat_files(args.real_dir, args.real_first_file, args.real_last_file)
    synthetic_files = find_synthetic_single_shot_files(args.synthetic_novoid_dir, args.component_file)

    duplicate_files = parse_int_list(args.real_shot_duplicate_files)

    pairs = build_pairs(
        real_files=real_files,
        synthetic_files=synthetic_files,
        sources=sources,
        first_file=args.real_first_file,
        first_x_m=args.real_shot_first_x_m,
        dx_m=args.real_shot_dx_m,
        duplicate_x_m=args.real_shot_duplicate_x_m,
        duplicate_files=duplicate_files,
        shot_match_tolerance_m=args.shot_match_tolerance_m,
    )

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Built {len(pairs)} real/synthetic pairs")
    if pairs:
        print("Real shot x check:")
        preview = pairs[:3] + pairs[9:13] + pairs[-3:]
        seen_preview = set()
        for pcheck in preview:
            key = (pcheck.real_file.name, pcheck.real_shot_x_m)
            if key in seen_preview:
                continue
            seen_preview.add(key)
            print(f"  {pcheck.real_file.name}: real_x={pcheck.real_shot_x_m:.3f} m -> synthetic_x={pcheck.synthetic_shot_x_m:.3f} m")
    print(f"Receiver range: {args.receiver_x_min:g}..{args.receiver_x_max:g} m")
    print(f"Analysis time range: {args.analysis_tmin:g}..{args.analysis_tmax:g} s")

    windows = parse_windows(args.windows)

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
    all_window_rows: list[dict] = []
    all_shot_rows: list[dict] = []
    failures: list[dict] = []

    for ipair, pair in enumerate(pairs, start=1):
        print(
            f"[{ipair}/{len(pairs)}] {pair.real_file.name} "
            f"real_x={pair.real_shot_x_m:.3f} m -> "
            f"{pair.synthetic_file.parent.name}/{pair.synthetic_file.name} "
            f"synthetic_x={pair.synthetic_shot_x_m:.3f} m"
        )

        try:
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
                resample_synthetic_to_real=True,
            )

            trace_rows = trace_rows_for_pair(
                pair=pair,
                t=t,
                real_data=real_data,
                syn_data=syn_data,
                rx=rx,
                windows=windows,
                max_freq_hz=args.max_freq_hz,
                noise_window_name=args.noise_window_name,
            )
            window_rows = shot_window_rows_for_pair(
                pair=pair,
                t=t,
                real_data=real_data,
                syn_data=syn_data,
                rx=rx,
                windows=windows,
                max_freq_hz=args.max_freq_hz,
                noise_window_name=args.noise_window_name,
            )
            shot_row = shot_summary_row(pair, window_rows, args.preferred_scale_window)

            all_trace_rows.extend(trace_rows)
            all_window_rows.extend(window_rows)
            all_shot_rows.append(shot_row)

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

    write_csv(output_dir / "trace_metrics.csv", all_trace_rows)
    write_csv(output_dir / "window_metrics.csv", all_window_rows)
    write_csv(output_dir / "shot_metrics.csv", all_shot_rows)
    write_csv(output_dir / "failures.csv", failures)

    # Also write a simple run configuration text file.
    config_path = output_dir / "metrics_run_config.txt"
    with config_path.open("w", encoding="utf-8") as f:
        f.write("62_compute_real_synthetic_trace_metrics.py\n")
        f.write("\nInputs:\n")
        f.write(f"  data_dir: {args.data_dir}\n")
        f.write(f"  synthetic_novoid_dir: {args.synthetic_novoid_dir}\n")
        f.write(f"  real_dir: {args.real_dir}\n")
        f.write(f"  component_file: {args.component_file}\n")
        f.write("\nGeometry:\n")
        f.write(f"  real files: {args.real_first_file}..{args.real_last_file}\n")
        f.write(f"  real shot x: first={args.real_shot_first_x_m}, dx={args.real_shot_dx_m}\n")
        f.write(f"  duplicate x: {args.real_shot_duplicate_x_m}, files={args.real_shot_duplicate_files}\n")
        f.write(f"  receiver x: {args.receiver_x_min}..{args.receiver_x_max}\n")
        f.write("\nProcessing:\n")
        f.write(f"  demean: {args.demean}\n")
        f.write(f"  detrend: {args.detrend}\n")
        f.write(f"  taper_fraction: {args.taper_fraction}\n")
        f.write(f"  highpass_hz: {args.highpass_hz}\n")
        f.write(f"  lowpass_hz: {args.lowpass_hz}\n")
        f.write(f"  bandpass: {args.bandpass}\n")
        f.write(f"  filter_corners: {args.filter_corners}\n")
        f.write(f"  zerophase: {args.zerophase}\n")
        f.write("\nWindows:\n")
        f.write(f"  analysis: {args.analysis_tmin}..{args.analysis_tmax}\n")
        f.write(f"  windows: {args.windows}\n")
        f.write(f"  preferred_scale_window: {args.preferred_scale_window}\n")
        f.write(f"  noise_window_name: {args.noise_window_name}\n")

    print(f"Wrote {output_dir / 'trace_metrics.csv'}")
    print(f"Wrote {output_dir / 'window_metrics.csv'}")
    print(f"Wrote {output_dir / 'shot_metrics.csv'}")
    print(f"Wrote {output_dir / 'failures.csv'}")
    print(f"Wrote {config_path}")
    print(f"Completed {len(all_shot_rows)} / {len(pairs)} shot pairs")
    return 0 if all_shot_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
