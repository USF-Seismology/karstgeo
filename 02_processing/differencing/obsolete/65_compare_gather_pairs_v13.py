#!/usr/bin/env python3
"""
65_compare_gather_pairs.py

Unified gather-pair comparison engine.

This replaces the duplicated 60_* and 61_* workflows with one engine that can run:

    1. synthetic_vs_synthetic
       reference  = synthetic WITH cave/void, usually SU
       comparison = synthetic WITHOUT cave/void, usually SU

    2. real_vs_synthetic
       reference  = real Geode SEG-2 files
       comparison = synthetic WITHOUT cave/void, usually SU

Common outputs
--------------
For every shot pair:
    - three-panel image: reference, comparison, difference
    - trace-normalized three-panel image
    - individual wiggles
    - overlaid wiggles
    - frequency-vs-receiver plots
    - sliding-window band-energy plots
    - trace peak/LSQ scaling CSV
    - optional difference SEG-Y

Global outputs:
    - comparison_summary.csv
    - trace_peak_scaling_factors.csv
    - failures.csv, if needed

Key idea
--------
Everything after reading/alignment is shared:

    reference gather - comparison gather

Only input adapters differ.

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
import matplotlib.pyplot as plt

try:
    from obspy import Stream, Trace, read
    from obspy.io.segy.segy import SEGYTraceHeader
except Exception as exc:
    raise SystemExit(
        "This script requires ObsPy. Install with:\n"
        "    conda install -c conda-forge obspy\n"
        f"Original import error: {exc}"
    )

try:
    from segy_tools import spectral as segy_spectral
except Exception:
    segy_spectral = None


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
    shot_index: int
    source: Source
    reference_path: Path
    comparison_path: Path
    reference_label: str
    comparison_label: str
    reference_kind: str
    comparison_kind: str


@dataclass(frozen=True)
class PairResult:
    shot_index: int
    source_x_m: float
    source_label: str
    reference_path: Path
    comparison_path: Path
    reference_label: str
    comparison_label: str
    n_receivers: int
    n_samples: int
    dt_s: float
    scale_mode: str
    scale_factor: float
    time_shift_ms: float
    max_abs_reference: float
    max_abs_comparison: float
    max_abs_difference: float
    rms_reference: float
    rms_comparison: float
    rms_difference: float
    maxdiff_over_max_reference_pct: float
    maxdiff_over_max_comparison_pct: float
    rmsdiff_over_rms_reference_pct: float
    rmsdiff_over_rms_comparison_pct: float
    physical_zero_lag_corr_mean: float = np.nan
    physical_zero_lag_corr_median: float = np.nan
    physical_nrmse_reference_rms_mean_pct: float = np.nan
    physical_nrmse_reference_rms_median_pct: float = np.nan
    trace_normalized_zero_lag_corr_mean: float = np.nan
    trace_normalized_zero_lag_corr_median: float = np.nan
    trace_normalized_nrmse_reference_rms_mean_pct: float = np.nan
    trace_normalized_nrmse_reference_rms_median_pct: float = np.nan


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")


# -----------------------------------------------------------------------------
# SPECFEM synthetic source-amplitude normalization
# -----------------------------------------------------------------------------

def _parse_fortran_float(text: str) -> float:
    """Parse SPECFEM/Fortran numeric strings such as 1.d10, 1.0e5."""
    return float(str(text).strip().replace("D", "e").replace("d", "e"))


def read_specfem_source_factor(source_file: Path) -> Optional[float]:
    """
    Read the `factor = ...` value from a SPECFEM DATA/SOURCE file.

    Returns None if the file or the factor line is not available. This is
    deliberately non-fatal so older workflows without DATA/SOURCE still run.
    """
    source_file = Path(source_file)
    if not source_file.exists():
        return None

    try:
        for line in source_file.read_text(errors="ignore").splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean or "=" not in clean:
                continue
            key, value = clean.split("=", 1)
            if key.strip().lower() != "factor":
                continue
            m = _FLOAT_RE.search(value)
            if not m:
                continue
            factor = _parse_fortran_float(m.group(0))
            if np.isfinite(factor) and factor != 0.0:
                return float(factor)
    except Exception:
        return None

    return None


def find_specfem_source_file(gather_path: Path) -> Optional[Path]:
    """
    Find the DATA/SOURCE file belonging to a synthetic gather.

    Handles both common SPECFEM layouts used here, for example:
        <run>/OUTPUT_FILES/Ux_file_single_v.su
        <run>/SURVEY_OUTPUT/shot_001_xs00134p5/Ux_file_single_v.su
    by walking up the parent chain and looking for <parent>/DATA/SOURCE.
    """
    gather_path = Path(gather_path)
    starts = [gather_path.parent, *gather_path.parents]
    seen = set()
    for parent in starts:
        if parent in seen:
            continue
        seen.add(parent)
        cand = parent / "DATA" / "SOURCE"
        if cand.exists():
            return cand
    return None


def synthetic_source_normalization_factor(gather_path: Path, target_factor: Optional[float]) -> tuple[float, Optional[float], Optional[Path]]:
    """
    Return multiplier needed to express a synthetic gather as if it used
    `target_factor` in DATA/SOURCE.

    If target_factor is None, <=0, or DATA/SOURCE:factor cannot be found,
    returns a neutral multiplier of 1.0.
    """
    if target_factor is None or not np.isfinite(target_factor) or float(target_factor) <= 0.0:
        return 1.0, None, None

    source_file = find_specfem_source_file(gather_path)
    if source_file is None:
        return 1.0, None, None

    source_factor = read_specfem_source_factor(source_file)
    if source_factor is None or source_factor == 0.0 or not np.isfinite(source_factor):
        return 1.0, None, source_file

    return float(target_factor) / float(source_factor), float(source_factor), source_file


# -----------------------------------------------------------------------------
# Basic numeric utilities
# -----------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(x * x)))


def clip_value(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    vals = []
    for a in arrays:
        aa = np.asarray(a, dtype=float)
        vals.append(np.abs(aa[np.isfinite(aa)]).ravel())
    vals = [v for v in vals if v.size]
    if not vals:
        return 1.0
    x = np.concatenate(vals)
    c = float(np.nanpercentile(x, percentile))
    if not np.isfinite(c) or c == 0:
        c = float(np.nanmax(x)) if x.size else 1.0
    return c if c and np.isfinite(c) else 1.0


def trace_normalize(data: np.ndarray, method: str = "rms", eps: float = 1e-20) -> np.ndarray:
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


def comparison_metrics(reference: np.ndarray, comparison: np.ndarray, diff: np.ndarray) -> dict:
    max_ref = float(np.nanmax(np.abs(reference))) if reference.size else np.nan
    max_cmp = float(np.nanmax(np.abs(comparison))) if comparison.size else np.nan
    max_diff = float(np.nanmax(np.abs(diff))) if diff.size else np.nan
    rms_ref = rms(reference)
    rms_cmp = rms(comparison)
    rms_diff = rms(diff)

    return {
        "max_abs_reference": max_ref,
        "max_abs_comparison": max_cmp,
        "max_abs_difference": max_diff,
        "rms_reference": rms_ref,
        "rms_comparison": rms_cmp,
        "rms_difference": rms_diff,
        "maxdiff_over_max_reference_pct": 100.0 * max_diff / max_ref if max_ref else np.nan,
        "maxdiff_over_max_comparison_pct": 100.0 * max_diff / max_cmp if max_cmp else np.nan,
        "rmsdiff_over_rms_reference_pct": 100.0 * rms_diff / rms_ref if rms_ref else np.nan,
        "rmsdiff_over_rms_comparison_pct": 100.0 * rms_diff / rms_cmp if rms_cmp else np.nan,
    }


def fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}%"


def pearson_corr(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a = a[:n] - np.nanmean(a[:n])
    b = b[:n] - np.nanmean(b[:n])
    den = np.sqrt(np.nansum(a * a) * np.nansum(b * b))
    return float(np.nansum(a * b) / den) if den else np.nan


def lsq_scale(reference_trace, comparison_trace) -> float:
    r = np.asarray(reference_trace, dtype=float).ravel()
    c = np.asarray(comparison_trace, dtype=float).ravel()
    n = min(r.size, c.size)
    if n == 0:
        return np.nan
    r = r[:n]
    c = c[:n]
    den = float(np.nansum(c * c))
    return float(np.nansum(r * c) / den) if den else np.nan


def zero_lag_corr(a, b) -> float:
    """Pearson correlation coefficient at zero lag."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a = a[:n]
    b = b[:n]
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return np.nan
    a = a[keep] - np.mean(a[keep])
    b = b[keep] - np.mean(b[keep])
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den else np.nan


def normalized_rmse(a, b, denom="reference_rms") -> float:
    """NRMSE for two traces."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 1:
        return np.nan
    a = a[:n]
    b = b[:n]
    diff = a - b
    e = rms(diff)
    if denom == "reference_rms":
        d = rms(a)
    elif denom == "comparison_rms":
        d = rms(b)
    elif denom == "mean_rms":
        d = 0.5 * (rms(a) + rms(b))
    elif denom == "reference_peak":
        d = float(np.nanmax(np.abs(a)))
    else:
        raise ValueError(f"Unknown NRMSE denominator: {denom}")
    return float(e / d) if d else np.nan


def best_lag_ms(a, b, dt_s, max_lag_s=None) -> tuple[float, float]:
    """
    Disabled fast placeholder.

    Best-lag cross-correlation is expensive for every trace in every shot and
    is not needed for the current diagnostic workflow. Kept only for backward
    compatibility with older CSV schemas.
    """
    return np.nan, np.nan


def compute_trace_similarity_metrics(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    max_lag_s: Optional[float] = None,
    label: str = "physical",
) -> list[dict]:
    """
    Fast per-trace similarity metrics.

    Computes only zero-lag Pearson correlation and NRMSE. Best-lag
    cross-correlation was intentionally removed for speed.
    """
    rows = []
    for i, x in enumerate(rx):
        ref = reference[i, :]
        cmp = comparison[i, :]
        zcorr = zero_lag_corr(ref, cmp)
        rows.append({
            "metric_set": label,
            "trace_index_1based": i + 1,
            "receiver_x_m": float(x),
            "zero_lag_corr": zcorr,
            "best_lag_ms": np.nan,
            "best_lag_corr": np.nan,
            "nrmse_reference_rms": normalized_rmse(ref, cmp, denom="reference_rms"),
            "nrmse_mean_rms": normalized_rmse(ref, cmp, denom="mean_rms"),
            "relative_rms_difference_pct": 100.0 * normalized_rmse(ref, cmp, denom="reference_rms"),
            "reference_rms": rms(ref),
            "comparison_rms": rms(cmp),
            "difference_rms": rms(ref - cmp),
            "reference_peak_abs": float(np.nanmax(np.abs(ref))) if len(ref) else np.nan,
            "comparison_peak_abs": float(np.nanmax(np.abs(cmp))) if len(cmp) else np.nan,
            "difference_peak_abs": float(np.nanmax(np.abs(ref - cmp))) if len(ref) else np.nan,
        })
    return rows


def summarize_similarity_rows(rows: list[dict], prefix: str = "") -> dict:
    out = {}
    if not rows:
        return out
    for key in ["zero_lag_corr", "best_lag_corr", "best_lag_ms", "nrmse_reference_rms", "nrmse_mean_rms", "relative_rms_difference_pct"]:
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"{prefix}{key}_mean"] = float(np.mean(vals))
            out[f"{prefix}{key}_median"] = float(np.median(vals))
            out[f"{prefix}{key}_min"] = float(np.min(vals))
            out[f"{prefix}{key}_max"] = float(np.max(vals))
        else:
            out[f"{prefix}{key}_mean"] = np.nan
            out[f"{prefix}{key}_median"] = np.nan
            out[f"{prefix}{key}_min"] = np.nan
            out[f"{prefix}{key}_max"] = np.nan
    return out



# -----------------------------------------------------------------------------
# SPECFEM / geometry parsing
# -----------------------------------------------------------------------------

def float_tokens(text: str) -> list[float]:
    vals = []
    for m in _FLOAT_RE.finditer(text.replace("D", "E").replace("d", "e")):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            pass
    return vals


def read_stations(path: Path) -> list[Station]:
    stations: list[Station] = []
    if not path.exists():
        raise FileNotFoundError(f"STATIONS file not found: {path}")

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


def read_sources(path: Path) -> list[Source]:
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

            nums = float_tokens(s)
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


def par_float(value: str) -> float:
    return float(str(value).strip().replace("d", "e").replace("D", "E"))


def parse_void_extent_from_par_file(
    par_file: Path,
    *,
    void_material_id: Optional[int] = None,
) -> Optional[tuple[float, float]]:
    par_file = Path(par_file)
    if not par_file.exists():
        return None

    lines = par_file.read_text(encoding="utf-8", errors="replace").splitlines()

    xmin = xmax = None
    nx = None
    for line in lines:
        clean = line.split("#", 1)[0].strip()
        if "=" not in clean:
            continue
        key, value = [p.strip() for p in clean.split("=", 1)]
        key_l = key.lower()
        if key_l == "xmin":
            xmin = par_float(value)
        elif key_l == "xmax":
            xmax = par_float(value)
        elif key_l == "nx":
            nx = int(float(value.strip()))

    if xmin is None or xmax is None or nx is None:
        return None

    zero_vs_materials = set()
    nbmodels_idx = None
    nbmodels = 0
    for i, line in enumerate(lines):
        clean = line.split("#", 1)[0].strip()
        if clean.lower().startswith("nbmodels") and "=" in clean:
            nbmodels_idx = i
            nbmodels = int(float(clean.split("=", 1)[1].strip()))
            break

    if nbmodels_idx is not None:
        count = 0
        for line in lines[nbmodels_idx + 1:]:
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            parts = clean.split()
            if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
                try:
                    mid = int(parts[0])
                    vs = par_float(parts[4])
                    if abs(vs) < 1e-12:
                        zero_vs_materials.add(mid)
                    count += 1
                    if count >= nbmodels:
                        break
                except Exception:
                    pass

    target_materials = {int(void_material_id)} if void_material_id is not None else zero_vs_materials
    if not target_materials:
        return None

    nbregions_idx = None
    nbregions = 0
    for i, line in enumerate(lines):
        clean = line.split("#", 1)[0].strip()
        if clean.lower().startswith("nbregions") and "=" in clean:
            nbregions_idx = i
            nbregions = int(float(clean.split("=", 1)[1].strip()))
            break

    if nbregions_idx is None:
        return None

    dx = (xmax - xmin) / nx
    extents = []
    count = 0
    for line in lines[nbregions_idx + 1:]:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 5:
            try:
                ixmin = int(parts[0])
                ixmax = int(parts[1])
                material_id = int(parts[4])
                count += 1
                if material_id in target_materials:
                    x_left = xmin + (ixmin - 1) * dx
                    x_right = xmin + ixmax * dx
                    extents.append((min(x_left, x_right), max(x_left, x_right)))
                if count >= nbregions:
                    break
            except Exception:
                pass

    if not extents:
        return None

    return min(e[0] for e in extents), max(e[1] for e in extents)


def find_par_file(*roots: Path) -> Optional[Path]:
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        for candidate in [
            root / "Par_file",
            root / "DATA" / "Par_file",
            root / "DATA" / "Par_file_single_source",
            root / "DATA" / "Par_file_multiple_source",
        ]:
            if candidate.exists():
                return candidate
    return None


def parse_extent(text: Optional[str]) -> Optional[tuple[float, float]]:
    if text is None or str(text).strip() == "":
        return None
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("--cave-extent-x-m must be like 140.5,160.0")
    a, b = float(parts[0]), float(parts[1])
    return min(a, b), max(a, b)


def resolve_cave_extent(args) -> Optional[tuple[float, float]]:
    roots = []
    for attr in ("reference_dir", "comparison_dir", "synthetic_novoid_dir", "cave_dir", "nocave_dir", "data_dir"):
        if hasattr(args, attr) and getattr(args, attr) is not None:
            roots.append(getattr(args, attr))

    par_file = Path(args.par_file) if args.par_file else find_par_file(*roots)

    if par_file is not None and par_file.exists():
        extent = parse_void_extent_from_par_file(par_file, void_material_id=args.void_material_id)
        if extent is not None:
            print(
                f"Using cave/void extent from Par_file: {par_file}\n"
                f"  x = {extent[0]:.3f} .. {extent[1]:.3f} m"
            )
            return extent
        print(f"WARNING: Could not parse void extent from Par_file: {par_file}")

    extent = parse_extent(args.cave_extent_x_m)
    if extent is not None:
        print(
            "Using manually supplied cave/void extent from --cave-extent-x-m: "
            f"x = {extent[0]:.3f} .. {extent[1]:.3f} m"
        )
        return extent

    print("No cave/void extent supplied or parsed; plots will not shade cave location.")
    return None


def resolve_data_dir(args) -> Path:
    requested = Path(args.data_dir)
    candidates = [requested]

    for attr in ("comparison_dir", "reference_dir", "synthetic_novoid_dir", "cave_dir", "nocave_dir"):
        if hasattr(args, attr):
            root = getattr(args, attr)
            if root is not None:
                candidates.append(Path(root) / "DATA")

    # Sibling search.
    for attr in ("comparison_dir", "reference_dir", "synthetic_novoid_dir", "cave_dir", "nocave_dir"):
        if hasattr(args, attr):
            root = getattr(args, attr)
            if root is not None:
                parent = Path(root).parent
                if parent.exists():
                    for child in sorted(parent.iterdir()):
                        if child.is_dir():
                            candidates.append(child / "DATA")
                break

    seen = set()
    for d in candidates:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        if (d / args.stations_file).exists():
            if d != requested:
                print(f"WARNING: using discovered DATA directory instead of requested --data-dir:\n  {d}")
            return d

    raise FileNotFoundError(f"Could not find {args.stations_file}; requested --data-dir was {requested}")


# -----------------------------------------------------------------------------
# File discovery and pairing
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

    last_exc = None
    tried = []
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


def natural_key(p: Path):
    text = str(p)
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


def find_files(root: Path, pattern: str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    # pathlib glob handles ** patterns.
    files = sorted(root.glob(pattern), key=natural_key)
    if not files:
        # fallback recursive basename search
        basename = Path(pattern).name
        files = sorted(root.rglob(basename), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No files matched recursively under {root} with pattern {pattern!r}")

    return files


def real_files(real_dir: Path, first_file: int, last_file: int) -> list[Path]:
    files = []
    for i in range(first_file, last_file + 1):
        p = Path(real_dir) / f"{i}.dat"
        if not p.exists():
            raise FileNotFoundError(f"Missing real SEG-2 file: {p}")
        files.append(p)
    return files


def real_file_number(path: Path) -> int:
    m = re.search(r"(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"Cannot infer real file number from {path}")
    return int(m.group(1))


def real_shot_x_from_file_number(
    file_number: int,
    *,
    first_file: int,
    first_x_m: float,
    dx_m: float,
    duplicate_x_m: Optional[float],
    duplicate_files: tuple[int, ...],
) -> float:
    duplicate_set = set(duplicate_files)
    if duplicate_x_m is not None and file_number in duplicate_set:
        return float(duplicate_x_m)

    duplicate_files_sorted = sorted(duplicate_set)
    duplicate_shift_count = 0
    if duplicate_x_m is not None and len(duplicate_files_sorted) >= 2:
        for dup_file in duplicate_files_sorted[1:]:
            if file_number > dup_file:
                duplicate_shift_count += 1

    effective_index = (file_number - first_file) - duplicate_shift_count
    return float(first_x_m + effective_index * dx_m)


def map_synthetic_files_to_sources(files: list[Path], sources: list[Source]) -> dict[float, tuple[Path, Source]]:
    out = {}
    for i, src in enumerate(sources):
        if i >= len(files):
            break
        out[round(float(src.x_m), 6)] = (files[i], src)
    return out


def nearest_synthetic_for_x(
    x_m: float,
    synthetic_by_x: dict[float, tuple[Path, Source]],
    tolerance_m: float,
) -> tuple[Path, Source]:
    xs = np.asarray(list(synthetic_by_x.keys()), dtype=float)
    if xs.size == 0:
        raise ValueError("No synthetic sources available for matching")
    i = int(np.argmin(np.abs(xs - x_m)))
    nearest_x = float(xs[i])
    if abs(nearest_x - x_m) > tolerance_m:
        raise ValueError(f"No synthetic source within {tolerance_m:g} m of x={x_m:.3f}; nearest={nearest_x:.3f}")
    return synthetic_by_x[round(nearest_x, 6)]


def build_pairs_synthetic_vs_synthetic(args, sources: list[Source]) -> list[Pair]:
    ref_files = find_files(args.reference_dir, args.reference_pattern)
    cmp_files = find_files(args.comparison_dir, args.comparison_pattern)

    n = min(len(ref_files), len(cmp_files), len(sources) if sources else min(len(ref_files), len(cmp_files)))
    if n == 0:
        raise ValueError("No synthetic pairs could be built")

    pairs = []
    for i in range(n):
        src = sources[i] if sources else Source(f"S{i+1:04d}", float(i + 1))
        pairs.append(
            Pair(
                shot_index=i + 1,
                source=src,
                reference_path=ref_files[i],
                comparison_path=cmp_files[i],
                reference_label=args.reference_label,
                comparison_label=args.comparison_label,
                reference_kind="synthetic",
                comparison_kind="synthetic",
            )
        )
    return pairs


def build_pairs_real_vs_synthetic(args, sources: list[Source]) -> list[Pair]:
    rfiles = real_files(args.real_dir, args.real_first_file, args.real_last_file)
    sfiles = find_files(args.comparison_dir, args.comparison_pattern)

    synthetic_by_x = map_synthetic_files_to_sources(sfiles, sources)
    duplicate_files = tuple(int(x.strip()) for x in str(args.real_shot_duplicate_files).split(",") if x.strip())

    pairs = []
    for idx, rf in enumerate(rfiles, start=1):
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
            real_x,
            synthetic_by_x,
            tolerance_m=args.shot_match_tolerance_m,
        )
        src = Source(
            source_id=f"real_{fn}_x{real_x:.3f}m__model_{src_from_model.source_id}",
            x_m=real_x,
            z_m=src_from_model.z_m,
            raw=src_from_model.raw,
        )
        pairs.append(
            Pair(
                shot_index=idx,
                source=src,
                reference_path=rf,
                comparison_path=sfp,
                reference_label=args.reference_label,
                comparison_label=args.comparison_label,
                reference_kind="real",
                comparison_kind="synthetic",
            )
        )
    return pairs


# -----------------------------------------------------------------------------
# Gather readers, preprocessing, alignment, scaling
# -----------------------------------------------------------------------------

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
                tr.data = np.asarray(tr.data, dtype=float) - np.nanmean(tr.data)

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


def stream_to_matrix(st: Stream) -> tuple[np.ndarray, float, float]:
    if len(st) == 0:
        raise ValueError("Empty stream")
    npts = min(int(tr.stats.npts) for tr in st)
    dt = float(st[0].stats.delta)
    sr = float(st[0].stats.sampling_rate)
    data = np.vstack([np.asarray(tr.data[:npts], dtype=float) for tr in st])
    return data, dt, sr


def read_synthetic_gather(
    path: Path,
    *,
    stations: list[Station],
    source: Source,
    receiver_x_min: Optional[float],
    receiver_x_max: Optional[float],
    component: Optional[str],
    preprocess_kwargs: dict,
) -> Gather:
    st = read_stream_any(path)
    st = select_component_if_possible(st, component)
    st = preprocess_stream(st, **preprocess_kwargs)

    data, dt, sr = stream_to_matrix(st)
    if data.shape[0] > len(stations):
        raise ValueError(f"{path}: {data.shape[0]} traces but only {len(stations)} stations")

    rx = np.asarray([sta.x_m for sta in stations[: data.shape[0]]], dtype=float)
    keep = np.ones_like(rx, dtype=bool)
    if receiver_x_min is not None:
        keep &= rx >= receiver_x_min - 1e-9
    if receiver_x_max is not None:
        keep &= rx <= receiver_x_max + 1e-9

    if not np.any(keep):
        raise ValueError(f"No synthetic receivers retained for {path}")

    data = data[keep, :]
    rx = rx[keep]
    order = np.argsort(rx)
    data = data[order, :]
    rx = rx[order]
    t = np.arange(data.shape[1], dtype=float) * dt

    return Gather(t, data, rx, source.x_m, source.source_id, path, dt, sr)


def read_real_gather(
    path: Path,
    *,
    source: Source,
    receiver_x_min: Optional[float],
    receiver_x_max: Optional[float],
    real_first_trace_x_m: float,
    real_dx_m: float,
    reverse_real_traces: bool,
    component: Optional[str],
    preprocess_kwargs: dict,
) -> Gather:
    st = read_stream_any(path)
    st = select_component_if_possible(st, component)
    if reverse_real_traces:
        st = Stream(list(reversed(st)))
    st = preprocess_stream(st, **preprocess_kwargs)

    data, dt, sr = stream_to_matrix(st)
    rx = real_first_trace_x_m + np.arange(data.shape[0], dtype=float) * real_dx_m

    keep = np.ones_like(rx, dtype=bool)
    if receiver_x_min is not None:
        keep &= rx >= receiver_x_min - 1e-9
    if receiver_x_max is not None:
        keep &= rx <= receiver_x_max + 1e-9

    if not np.any(keep):
        raise ValueError(f"No real receivers retained for {path}")

    data = data[keep, :]
    rx = rx[keep]
    t = np.arange(data.shape[1], dtype=float) * dt

    return Gather(t, data, rx, source.x_m, source.source_id, path, dt, sr)


def read_gather_for_pair(
    path: Path,
    kind: str,
    *,
    stations: list[Station],
    source: Source,
    args,
    preprocess_kwargs: dict,
) -> Gather:
    if kind == "real":
        return read_real_gather(
            path,
            source=source,
            receiver_x_min=args.receiver_x_min,
            receiver_x_max=args.receiver_x_max,
            real_first_trace_x_m=args.real_first_trace_x_m,
            real_dx_m=args.real_dx_m,
            reverse_real_traces=args.reverse_real_traces,
            component=args.component,
            preprocess_kwargs=preprocess_kwargs,
        )
    if kind == "synthetic":
        g = read_synthetic_gather(
            path,
            stations=stations,
            source=source,
            receiver_x_min=args.receiver_x_min,
            receiver_x_max=args.receiver_x_max,
            component=args.component,
            preprocess_kwargs=preprocess_kwargs,
        )

        if getattr(args, "normalize_synthetic_source_factor", True):
            mult, source_factor, source_file = synthetic_source_normalization_factor(
                path,
                getattr(args, "synthetic_source_target_factor", None),
            )
            if mult != 1.0:
                g = Gather(
                    g.time_s,
                    g.data * mult,
                    g.receiver_x_m,
                    g.source_x_m,
                    g.label,
                    g.path,
                    g.dt_s,
                    g.sampling_rate_hz,
                )
            if getattr(args, "print_synthetic_source_scaling", False):
                if source_factor is None:
                    print(f"    source-factor normalization: {path}: factor not found; multiplier=1")
                else:
                    print(
                        f"    source-factor normalization: {path}: "
                        f"DATA/SOURCE factor={source_factor:.10g}; "
                        f"target={args.synthetic_source_target_factor:.10g}; multiplier={mult:.10g}"
                    )
        return g
    raise ValueError(f"Unknown gather kind: {kind}")


def interp_time_axis(data: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    out = np.empty((data.shape[0], len(new_t)), dtype=float)
    for i in range(data.shape[0]):
        out[i, :] = np.interp(new_t, old_t, data[i, :], left=0.0, right=0.0)
    return out


def shift_data_time(data: np.ndarray, time_s: np.ndarray, shift_ms: float) -> np.ndarray:
    """
    Shift comparison gather by shift_ms onto reference time axis.

    shift_ms > 0 delays comparison:
        shifted(t) = original(t - shift)
    """
    lag_s = float(shift_ms) / 1000.0
    out = np.empty_like(data, dtype=float)
    for i in range(data.shape[0]):
        out[i, :] = np.interp(time_s - lag_s, time_s, data[i, :], left=0.0, right=0.0)
    return out


def align_gathers(
    reference: Gather,
    comparison: Gather,
    *,
    receiver_tolerance_m: float,
    tmin: Optional[float],
    tmax: Optional[float],
    comparison_time_shift_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = []
    used = set()
    for ir, rx in enumerate(reference.receiver_x_m):
        ic = int(np.argmin(np.abs(comparison.receiver_x_m - rx)))
        if ic in used:
            continue
        if abs(comparison.receiver_x_m[ic] - rx) <= receiver_tolerance_m:
            pairs.append((ir, ic))
            used.add(ic)

    if not pairs:
        raise ValueError(f"No common receiver positions between {reference.path.name} and {comparison.path.name}")

    ir = np.asarray([p[0] for p in pairs], dtype=int)
    ic = np.asarray([p[1] for p in pairs], dtype=int)

    t = reference.time_s.copy()
    ref = reference.data[ir, :]
    cmp = comparison.data[ic, :]

    cmp = interp_time_axis(cmp, comparison.time_s, t)

    if comparison_time_shift_ms:
        cmp = shift_data_time(cmp, t, comparison_time_shift_ms)

    keep_t = np.ones_like(t, dtype=bool)
    if tmin is not None:
        keep_t &= t >= tmin
    if tmax is not None:
        keep_t &= t <= tmax

    if not np.any(keep_t):
        raise ValueError("Time window removed all samples")

    return t[keep_t], ref[:, keep_t], cmp[:, keep_t], reference.receiver_x_m[ir]


def compute_scale_factor(
    reference: np.ndarray,
    comparison: np.ndarray,
    t: np.ndarray,
    *,
    mode: str,
    fixed_scale: float,
    scale_tmin: Optional[float],
    scale_tmax: Optional[float],
) -> float:
    if mode == "none":
        return 1.0
    if mode == "fixed":
        return float(fixed_scale)

    keep = np.ones_like(t, dtype=bool)
    if scale_tmin is not None:
        keep &= t >= scale_tmin
    if scale_tmax is not None:
        keep &= t <= scale_tmax
    if not np.any(keep):
        keep = np.ones_like(t, dtype=bool)

    r = reference[:, keep]
    c = comparison[:, keep]

    if mode == "rms":
        return rms(r) / rms(c) if rms(c) else 1.0
    if mode == "lsq":
        return lsq_scale(r, c)
    if mode == "maxabs":
        den = float(np.nanmax(np.abs(c)))
        return float(np.nanmax(np.abs(r)) / den) if den else 1.0

    raise ValueError(f"Unknown scale mode: {mode}")



# -----------------------------------------------------------------------------
# Optional post-alignment diagnostic filters
# -----------------------------------------------------------------------------

def matrix_to_stream(
    data: np.ndarray,
    dt_s: float,
    *,
    receiver_x_m: Optional[np.ndarray] = None,
    source_x_m: Optional[float] = None,
) -> Stream:
    """
    Convert a gather matrix (n_traces x n_samples) to an ObsPy Stream.

    This is used for optional diagnostic filters applied after reference and
    comparison gathers have already been aligned to common receivers and times.
    """
    st = Stream()
    for i, trdata in enumerate(np.asarray(data, dtype=float)):
        tr = Trace(data=np.asarray(trdata, dtype=np.float32))
        tr.stats.delta = float(dt_s)
        tr.stats.sampling_rate = 1.0 / float(dt_s)
        tr.stats.network = "XX"
        tr.stats.station = f"{i+1:05d}"
        tr.stats.channel = "Z"
        if receiver_x_m is not None:
            tr.stats.receiver_x_m = float(receiver_x_m[i])
        if source_x_m is not None:
            tr.stats.source_x_m = float(source_x_m)
        st.append(tr)
    return st


def stream_to_matrix_exact(st: Stream, n_samples: Optional[int] = None) -> np.ndarray:
    if len(st) == 0:
        raise ValueError("Cannot convert empty Stream to matrix")
    if n_samples is None:
        n_samples = min(int(tr.stats.npts) for tr in st)
    return np.vstack([np.asarray(tr.data[:n_samples], dtype=float) for tr in st])


def apply_obspy_bandpass_to_matrix(
    data: np.ndarray,
    t: np.ndarray,
    *,
    freqmin: float,
    freqmax: float,
    corners: int,
    zerophase: bool,
) -> np.ndarray:
    """
    Apply an ObsPy bandpass to each trace in a matrix.

    Used for optional 25-400 Hz style diagnostic products. This is deliberately
    applied to both parent gathers before differencing.
    """
    dt = float(np.nanmedian(np.diff(t)))
    nyq = 0.5 / dt
    fmax = min(float(freqmax), 0.999 * nyq)
    fmin = float(freqmin)

    if fmin <= 0 or fmax <= fmin:
        raise ValueError(
            f"Invalid diagnostic bandpass {freqmin}-{freqmax} Hz for dt={dt:g} s "
            f"(Nyquist={nyq:g} Hz)"
        )

    st = matrix_to_stream(data, dt)
    st.filter(
        "bandpass",
        freqmin=fmin,
        freqmax=fmax,
        corners=int(corners),
        zerophase=bool(zerophase),
    )
    return stream_to_matrix_exact(st, n_samples=data.shape[1])


def _cosine_taper_1d(n: int, fraction: float) -> np.ndarray:
    """Symmetric raised-cosine taper."""
    w = np.ones(int(n), dtype=float)
    m = int(round(float(fraction) * n))
    if m <= 0:
        return w
    m = min(m, n // 2)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, m)))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def _fk_velocity_filter_matrix_core(
    data: np.ndarray,
    *,
    dt: float,
    dx: float,
    min_velocity_mps: float,
    taper_width_mps: float,
    use_taper: bool,
    spatial_taper_fraction: float = 0.0,
    pad_factor: int = 1,
) -> np.ndarray:
    """
    Core 2-D f-k velocity fan filter.

    This intentionally follows the original working implementation:
      fk = fft2(data)
      apparent_velocity = abs(F / K)
      mute apparent_velocity < min_velocity

    Additions:
      - optional spatial taper before FFT
      - optional zero padding along receiver axis
    """
    d = np.asarray(data, dtype=float)
    ntr, npts = d.shape
    if ntr < 2 or npts < 2:
        return d.copy()

    # Remove trace means to avoid a strong DC stripe. This is usually harmless
    # for the f-k diagnostic product and reduces wrap/ringing.
    means = np.nanmean(d, axis=1, keepdims=True)
    d0 = d - means

    taper = _cosine_taper_1d(ntr, spatial_taper_fraction)[:, None]
    d0 = d0 * taper

    pad_factor = max(1, int(pad_factor))
    ntr_fft = int(ntr * pad_factor)
    if ntr_fft > ntr:
        d_fft = np.zeros((ntr_fft, npts), dtype=float)
        d_fft[:ntr, :] = d0
    else:
        d_fft = d0

    fk = np.fft.fft2(d_fft)
    freqs = np.fft.fftfreq(npts, d=dt)
    wavenumbers = np.fft.fftfreq(ntr_fft, d=dx)
    k_grid, f_grid = np.meshgrid(wavenumbers, freqs, indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        apparent_velocity = np.abs(f_grid / k_grid)

    mask = np.ones_like(fk, dtype=float)
    v0 = float(min_velocity_mps)

    if use_taper:
        v1 = v0 + float(taper_width_mps)
        v1 = max(v1, v0 + 1e-9)
        mask[apparent_velocity <= v0] = 0.0
        transition = (apparent_velocity > v0) & (apparent_velocity < v1)
        x = (apparent_velocity[transition] - v0) / (v1 - v0)
        mask[transition] = 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        mask[apparent_velocity < v0] = 0.0

    # Preserve k=0 vertical plane, matching the original implementation.
    mask[k_grid == 0] = 1.0

    filtered = np.real(np.fft.ifft2(fk * mask))[:ntr, :]

    # Do not try to divide by taper near edges; that would amplify edge noise.
    # Add the removed trace means back only in the k=0 sense would be ambiguous
    # after filtering, so leave the diagnostic f-k product mean-free.
    return filtered


def apply_fk_filter_to_matrix(
    data: np.ndarray,
    t: np.ndarray,
    rx: np.ndarray,
    *,
    source_x_m: float,
    min_velocity_mps: float,
    taper_width_mps: float,
    use_taper: bool,
    split_at_source: bool = True,
    spatial_taper_fraction: float = 0.05,
    pad_factor: int = 2,
) -> np.ndarray:
    """
    Apply a 2-D f-k velocity fan filter to a gather matrix.

    The old implementation and segy_tools.spectral implementation apply one
    global f-k transform to the whole shot gather. That is fine for one-sided
    gathers, but it can produce wraparound / apparent reverse-dip artifacts
    for split-spread gathers where energy propagates away from a source in the
    middle of the receiver line.

    This version defaults to split-at-source filtering:
      - receivers left of the source are filtered as one gather
      - receivers right of the source are filtered as another gather
      - the two pieces are reassembled

    This avoids forcing two opposite-propagating wavefields into one periodic
    spatial FFT. Optional spatial tapering and zero padding further reduce
    f-k wraparound/ringing.
    """
    dt = float(np.nanmedian(np.diff(t)))
    rx = np.asarray(rx, dtype=float)
    d = np.asarray(data, dtype=float)
    out = np.zeros_like(d, dtype=float)

    def _filter_indices(indices):
        if len(indices) < 4:
            out[indices, :] = d[indices, :]
            return
        # Preserve physical receiver order in x.
        indices = np.asarray(indices, dtype=int)
        order = indices[np.argsort(rx[indices])]
        dx_local = float(np.nanmedian(np.diff(np.sort(rx[order])))) if len(order) > 1 else 1.0
        filtered = _fk_velocity_filter_matrix_core(
            d[order, :],
            dt=dt,
            dx=dx_local,
            min_velocity_mps=min_velocity_mps,
            taper_width_mps=taper_width_mps,
            use_taper=use_taper,
            spatial_taper_fraction=spatial_taper_fraction,
            pad_factor=pad_factor,
        )
        out[order, :] = filtered

    if split_at_source:
        left = np.where(rx < source_x_m)[0]
        right = np.where(rx >= source_x_m)[0]
        if len(left) >= 4 and len(right) >= 4:
            _filter_indices(left)
            _filter_indices(right)
            return out

    # Fallback to one global gather, matching the original implementation but
    # with optional taper/padding.
    order = np.arange(d.shape[0])
    dx = float(np.nanmedian(np.diff(np.sort(rx)))) if len(rx) > 1 else 1.0
    return _fk_velocity_filter_matrix_core(
        d[order, :],
        dt=dt,
        dx=dx,
        min_velocity_mps=min_velocity_mps,
        taper_width_mps=taper_width_mps,
        use_taper=use_taper,
        spatial_taper_fraction=spatial_taper_fraction,
        pad_factor=pad_factor,
    )


def write_diagnostic_comparison_product(
    *,
    name: str,
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    pair: Pair,
    shot_dir: Path,
    args,
    cave_extent: Optional[tuple[float, float]],
    bands: list[tuple[float, float]],
    title_extra: str,
) -> dict:
    """
    Write a complete secondary comparison product from already-filtered parents.

    This is used for optional products such as:
      - diagnostic bandpass, e.g. 25-400 Hz
      - f-k filtered body-wave emphasized gathers

    The difference is always computed after filtering both parent gathers:
        filtered(reference) - filtered(comparison)
    """
    outdir = shot_dir / name
    outdir.mkdir(parents=True, exist_ok=True)

    diff = reference - comparison
    metrics = comparison_metrics(reference, comparison, diff)

    (outdir / "difference_amplitude_metrics.txt").write_text(
        "\n".join([
            f"diagnostic_product = {name}",
            f"source_x_m = {pair.source.x_m:.6g}",
            f"reference_label = {pair.reference_label}",
            f"comparison_label = {pair.comparison_label}",
            f"max_abs_reference_left_panel = {metrics['max_abs_reference']:.10g}",
            f"max_abs_comparison_center_panel = {metrics['max_abs_comparison']:.10g}",
            f"max_abs_difference_right_panel = {metrics['max_abs_difference']:.10g}",
            f"rms_reference_left_panel = {metrics['rms_reference']:.10g}",
            f"rms_comparison_center_panel = {metrics['rms_comparison']:.10g}",
            f"rms_difference_right_panel = {metrics['rms_difference']:.10g}",
            f"peak_right_over_peak_left_pct = {metrics['maxdiff_over_max_reference_pct']:.10g}",
            f"peak_right_over_peak_center_pct = {metrics['maxdiff_over_max_comparison_pct']:.10g}",
            f"rms_right_over_rms_left_pct = {metrics['rmsdiff_over_rms_reference_pct']:.10g}",
            f"rms_right_over_rms_center_pct = {metrics['rmsdiff_over_rms_comparison_pct']:.10g}",
            "",
        ]),
        encoding="utf-8",
    )

    plot_three_panel(
        t,
        rx,
        reference,
        comparison,
        diff,
        source_x_m=pair.source.x_m,
        reference_label=f"{pair.reference_label} [{name}]",
        comparison_label=f"{pair.comparison_label} [{name}]",
        outfile=outdir / "combined_image_reference_comparison_difference.png",
        cave_extent=cave_extent,
        metrics=metrics,
        title_extra=title_extra,
    )

    if args.write_combined_three_panel_products:
        plot_wiggle_three_panel(
            t,
            rx,
            reference,
            comparison,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=f"{pair.reference_label} [{name}]",
            comparison_label=f"{pair.comparison_label} [{name}]",
            outfile=outdir / "combined_wiggles_reference_comparison_difference.png",
            cave_extent=cave_extent,
            normalize_traces=args.combined_wiggle_trace_normalize,
            scale=args.overlay_wiggle_scale,
        )
        plot_frequency_three_panel(
            t,
            rx,
            reference,
            comparison,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=f"{pair.reference_label} [{name}]",
            comparison_label=f"{pair.comparison_label} [{name}]",
            outfile=outdir / "combined_frequency_receiver_reference_comparison_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
        )
        if args.write_spectral_contours:
            plot_spectral_contours_three_panel(
                t,
                rx,
                reference,
                comparison,
                diff,
                source_x_m=pair.source.x_m,
                reference_label=f"{pair.reference_label} [{name}]",
                comparison_label=f"{pair.comparison_label} [{name}]",
                outfile=outdir / "combined_spectral_contours_reference_comparison_difference.png",
                max_freq_hz=args.max_freq_hz,
                cave_extent=cave_extent,
                normalize_per_trace=args.frequency_trace_normalization,
                log10=args.spectral_contour_log10,
                levels=args.spectral_contour_levels,
                smooth_bins=args.spectral_contour_smooth_bins,
            )
        if args.write_band_energy:
            plot_band_energy_three_panel(
                reference,
                comparison,
                diff,
                t,
                rx,
                source_x_m=pair.source.x_m,
                reference_label=f"{pair.reference_label} [{name}]",
                comparison_label=f"{pair.comparison_label} [{name}]",
                outfile=outdir / "combined_band_energy_reference_comparison_difference.png",
                bands=bands,
                window_s=args.band_energy_window_s,
                step_s=args.band_energy_step_s,
                cave_extent=cave_extent,
                normalize_per_trace=args.band_energy_normalize_per_trace,
                log10=args.band_energy_log10,
            )

    if args.write_trace_normalized_figures:
        ref_norm = trace_normalize(reference, method=args.trace_normalize_method)
        cmp_norm = trace_normalize(comparison, method=args.trace_normalize_method)
        diff_norm = ref_norm - cmp_norm
        norm_metrics = comparison_metrics(ref_norm, cmp_norm, diff_norm)

        if args.write_trace_similarity:
            trace_normalized_similarity_rows = compute_trace_similarity_metrics(
                t,
                rx,
                ref_norm,
                cmp_norm,
                max_lag_s=args.similarity_max_lag_s,
                label=f"trace_normalized_{args.trace_normalize_method}",
            )
            write_csv(shot_dir / "trace_similarity_metrics_trace_normalized.csv", trace_normalized_similarity_rows)
            plot_trace_similarity_summary(
                trace_normalized_similarity_rows,
                source_x_m=pair.source.x_m,
                title=(
                    f"Trace similarity, trace-normalized ({args.trace_normalize_method}): "
                    f"{pair.reference_label} vs {pair.comparison_label}"
                ),
                outfile=shot_dir / "trace_similarity_metrics_trace_normalized.png",
                cave_extent=cave_extent,
            )

        plot_three_panel(
            t,
            rx,
            ref_norm,
            cmp_norm,
            diff_norm,
            source_x_m=pair.source.x_m,
            reference_label=f"{pair.reference_label} [{name}, trace-normalized]",
            comparison_label=f"{pair.comparison_label} [{name}, trace-normalized]",
            outfile=outdir / "combined_image_reference_comparison_difference_trace_normalized.png",
            cave_extent=cave_extent,
            metrics=norm_metrics,
            title_extra=f"{title_extra}; trace-normalized ({args.trace_normalize_method})",
        )

    if args.write_combined_three_panel_products:
        plot_wiggle_three_panel(
            t,
            rx,
            ref,
            cmp_scaled,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_wiggles_reference_comparison_difference.png",
            cave_extent=cave_extent,
            normalize_traces=args.combined_wiggle_trace_normalize,
            scale=args.overlay_wiggle_scale,
        )

    if args.write_individual_wiggles:
        plot_wiggle(
            t,
            rx,
            reference,
            source_x_m=pair.source.x_m,
            title=f"Reference [{name}]: {pair.reference_label}",
            outfile=outdir / "wiggle_reference.png",
            cave_extent=cave_extent,
        )
        plot_wiggle(
            t,
            rx,
            comparison,
            source_x_m=pair.source.x_m,
            title=f"Comparison [{name}]: {pair.comparison_label}",
            outfile=outdir / "wiggle_comparison.png",
            cave_extent=cave_extent,
        )
        plot_wiggle(
            t,
            rx,
            diff,
            source_x_m=pair.source.x_m,
            title=f"Difference [{name}]: reference - comparison",
            outfile=outdir / "wiggle_difference.png",
            cave_extent=cave_extent,
        )

    if args.write_overlay_wiggles:
        plot_overlay_wiggles(
            t,
            rx,
            reference,
            comparison,
            source_x_m=pair.source.x_m,
            reference_label=f"{pair.reference_label} [{name}]",
            comparison_label=f"{pair.comparison_label} [{name}]",
            outfile=outdir / "wiggle_overlay_comparison_blue_reference_red.png",
            cave_extent=cave_extent,
            normalize=args.overlay_normalize,
            wiggle_scale=args.overlay_wiggle_scale,
        )

    if args.write_standalone_frequency:
        plot_frequency(
            t,
            rx,
            reference,
            source_x_m=pair.source.x_m,
            title=f"Frequency vs receiver [{name}]: reference",
            outfile=outdir / "frequency_receiver_reference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
        )
        plot_frequency(
            t,
            rx,
            comparison,
            source_x_m=pair.source.x_m,
            title=f"Frequency vs receiver [{name}]: comparison",
            outfile=outdir / "frequency_receiver_comparison.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
        )
        plot_frequency(
            t,
            rx,
            diff,
            source_x_m=pair.source.x_m,
            title=f"Frequency vs receiver [{name}]: difference",
            outfile=outdir / "frequency_receiver_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
        )

    if args.write_spectral_contours and args.write_combined_three_panel_products:
        plot_spectral_contours_three_panel(
            t,
            rx,
            ref,
            cmp_scaled,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_spectral_contours_reference_comparison_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )

    if args.write_spectral_contours and args.write_standalone_spectral_contours:
        plot_spectral_contours(
            t,
            rx,
            reference,
            source_x_m=pair.source.x_m,
            title=f"Spectral contours [{name}]: reference",
            outfile=outdir / "spectral_contours_reference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )
        plot_spectral_contours(
            t,
            rx,
            comparison,
            source_x_m=pair.source.x_m,
            title=f"Spectral contours [{name}]: comparison",
            outfile=outdir / "spectral_contours_comparison.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )
        plot_spectral_contours(
            t,
            rx,
            diff,
            source_x_m=pair.source.x_m,
            title=f"Spectral contours [{name}]: difference",
            outfile=outdir / "spectral_contours_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )

    if args.write_band_energy and args.write_standalone_band_energy:
        for label, d in [
            ("reference", reference),
            ("comparison", comparison),
            ("difference", diff),
        ]:
            plot_band_energy(
                d,
                t,
                rx,
                source_x_m=pair.source.x_m,
                title=f"Band energy [{name}]: {label}",
                outfile=outdir / f"band_energy_{label}.png",
                bands=bands,
                window_s=args.band_energy_window_s,
                step_s=args.band_energy_step_s,
                cave_extent=cave_extent,
                normalize_per_trace=args.band_energy_normalize_per_trace,
                log10=args.band_energy_log10,
            )

    return metrics


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def safe_dir(shot_index: int, source: Source) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.source_id))
    return f"{shot_index:03d}_{label}_x{source.x_m:08.3f}m".replace(".", "p")


def _cave_extent_for_panel(cave_extent, panel_index: int):
    """Return cave extent for a panel index 0=reference, 1=comparison, 2=difference.

    For backward compatibility, cave_extent may be a simple (xmin, xmax) tuple,
    which shades every panel. Newer workflows may pass a dict with keys:
        extent: (xmin, xmax)
        panels: set/list of panel names or indices, e.g. {"reference", "difference"}
    """
    if cave_extent is None:
        return None
    if isinstance(cave_extent, dict):
        extent = cave_extent.get("extent")
        panels = cave_extent.get("panels", {"reference", "comparison", "difference"})
        names = {0: "reference", 1: "comparison", 2: "difference"}
        if panel_index in panels or names.get(panel_index) in panels or "all" in panels:
            return extent
        return None
    return cave_extent


def _cave_extent_any(cave_extent):
    """Return the numeric (xmin, xmax) cave extent from either old tuple or new dict form.

    The dict form is used only to control which panels are shaded in three-panel
    figures. Standalone plots should either use this helper, or pass None when
    they intentionally should not show cave shading.
    """
    if cave_extent is None:
        return None
    if isinstance(cave_extent, dict):
        return cave_extent.get("extent")
    return cave_extent


def annotate_cave_and_source(ax, source_x_m: float, cave_extent: Optional[tuple[float, float]], color_source="k", color_cave="0.5"):
    ce = _cave_extent_any(cave_extent)
    if ce is not None:
        ax.axvspan(ce[0], ce[1], color=color_cave, alpha=0.15)
    ax.axvline(source_x_m, color=color_source, lw=1.0, ls="--", alpha=0.75)


def plot_three_panel(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    diff: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    cave_extent: Optional[tuple[float, float]],
    metrics: dict,
    title_extra: str,
    clip_percentile: float = 99.0,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    extent = [rx.min(), rx.max(), t.max(), t.min()]
    main_c = clip_value(reference, comparison, percentile=clip_percentile)
    diff_c = clip_value(diff, percentile=clip_percentile)

    peak_ref_pct = metrics.get("maxdiff_over_max_reference_pct", np.nan)
    peak_cmp_pct = metrics.get("maxdiff_over_max_comparison_pct", np.nan)
    rms_ref_pct = metrics.get("rmsdiff_over_rms_reference_pct", np.nan)
    rms_cmp_pct = metrics.get("rmsdiff_over_rms_comparison_pct", np.nan)

    diff_title = f"Difference: {reference_label} - {comparison_label}"

    panels = [
        (reference, reference_label, main_c),
        (comparison, comparison_label, main_c),
        (diff, diff_title, diff_c),
    ]

    for ipanel, (ax, (data, title, c)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            data.T,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="seismic",
            vmin=-c,
            vmax=c,
        )
        annotate_cave_and_source(ax, source_x_m, _cave_extent_for_panel(cave_extent, ipanel))
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("amplitude")

    txt = (
        "Difference strength relative to signal\n"
        f"peak(right) / peak(left)   = {fmt_pct(peak_ref_pct)}\n"
        f"peak(right) / peak(center) = {fmt_pct(peak_cmp_pct)}\n"
        f"RMS(right) / RMS(left)     = {fmt_pct(rms_ref_pct)}\n"
        f"RMS(right) / RMS(center)   = {fmt_pct(rms_cmp_pct)}"
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
    fig.suptitle(f"Gather comparison, source x={source_x_m:.3f} m{title_extra}", y=0.99)
    fig.tight_layout(rect=[0, 0.00, 1, 0.97])
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_wiggle(
    t: np.ndarray,
    rx: np.ndarray,
    data: np.ndarray,
    *,
    source_x_m: float,
    title: str,
    outfile: Path,
    cave_extent: Optional[tuple[float, float]],
    normalize_traces: bool = True,
    scale: float = 0.45,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    d = np.asarray(data, dtype=float).copy()
    if normalize_traces:
        mx = np.nanmax(np.abs(d), axis=1)
        mx[~np.isfinite(mx) | (mx == 0)] = 1.0
        d = d / mx[:, None]
    else:
        c = clip_value(d)
        d = np.clip(d / c, -1, 1)

    dx = np.nanmedian(np.diff(np.sort(rx))) if len(rx) > 1 else 1.0
    amp = scale * dx

    fig, ax = plt.subplots(figsize=(12, 7))
    ce = _cave_extent_any(cave_extent)
    if ce is not None:
        ax.axvspan(ce[0], ce[1], color="0.5", alpha=0.15)

    for tr, x in zip(d, rx):
        y = x + tr * amp
        ax.plot(y, t, color="k", lw=0.45)
        ax.fill_betweenx(t, x, y, where=(tr >= 0), color="k", alpha=0.25, linewidth=0)

    ax.axvline(source_x_m, color="r", lw=1.0, ls="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_overlay_wiggles(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    cave_extent: Optional[tuple[float, float]],
    normalize: str = "pair",
    wiggle_scale: float = 0.45,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    ref = np.asarray(reference, dtype=float).copy()
    cmp = np.asarray(comparison, dtype=float).copy()

    if normalize == "pair":
        den = np.maximum(np.nanmax(np.abs(ref), axis=1), np.nanmax(np.abs(cmp), axis=1))
        den[~np.isfinite(den) | (den == 0)] = 1.0
        ref /= den[:, None]
        cmp /= den[:, None]
    elif normalize == "trace":
        rden = np.nanmax(np.abs(ref), axis=1)
        cden = np.nanmax(np.abs(cmp), axis=1)
        rden[~np.isfinite(rden) | (rden == 0)] = 1.0
        cden[~np.isfinite(cden) | (cden == 0)] = 1.0
        ref /= rden[:, None]
        cmp /= cden[:, None]
    elif normalize == "none":
        den = np.nanmax(np.abs(np.concatenate([ref.ravel(), cmp.ravel()])))
        den = den if np.isfinite(den) and den else 1.0
        ref /= den
        cmp /= den
    else:
        raise ValueError(f"Unknown overlay normalization: {normalize}")

    dx = np.nanmedian(np.diff(np.sort(rx))) if len(rx) > 1 else 1.0
    amp = wiggle_scale * dx

    fig, ax = plt.subplots(figsize=(13, 7))
    ce = _cave_extent_any(cave_extent)
    if ce is not None:
        ax.axvspan(ce[0], ce[1], color="0.5", alpha=0.15)

    # comparison blue first, reference red second
    for tr, x in zip(cmp, rx):
        y = x + tr * amp
        ax.plot(y, t, color="blue", lw=0.55, alpha=0.75)
        ax.fill_betweenx(t, x, y, where=(tr < 0), color="blue", alpha=0.18, linewidth=0)

    for tr, x in zip(ref, rx):
        y = x + tr * amp
        ax.plot(y, t, color="red", lw=0.55, alpha=0.75)
        ax.fill_betweenx(t, x, y, where=(tr < 0), color="red", alpha=0.18, linewidth=0)

    ax.axvline(source_x_m, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(
        f"Overlay wiggles: comparison blue, reference red; "
        f"source x={source_x_m:.3f} m; display normalization={normalize}"
    )
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("time (s)")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="red", lw=1.2, label=f"Reference: {reference_label}"),
        Line2D([0], [0], color="blue", lw=1.2, label=f"Comparison: {comparison_label}"),
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_frequency(
    t: np.ndarray,
    rx: np.ndarray,
    data: np.ndarray,
    *,
    source_x_m: float,
    title: str,
    outfile: Path,
    max_freq_hz: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = True,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    dt = float(np.nanmedian(np.diff(t)))
    freqs = np.fft.rfftfreq(data.shape[1], d=dt)
    amp = np.abs(np.fft.rfft(data, axis=1))
    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    amp = amp[:, keep]

    if normalize_per_trace:
        mx = np.nanmax(amp, axis=1)
        mx[~np.isfinite(mx) | (mx == 0)] = 1.0
        amp = amp / mx[:, None]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        amp.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[rx.min(), rx.max(), freqs.min(), freqs.max()],
        cmap="viridis",
        vmin=0,
        vmax=clip_value(amp),
    )
    ce = _cave_extent_any(cave_extent)
    if ce is not None:
        ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
    ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(alpha=0.15)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_spectral_contours(
    t: np.ndarray,
    rx: np.ndarray,
    data: np.ndarray,
    *,
    source_x_m: float,
    title: str,
    outfile: Path,
    max_freq_hz: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = True,
    log10: bool = True,
    levels: int = 24,
    smooth_bins: int = 1,
) -> None:
    """
    Charlie-style frequency contour plot.

    This is similar to the frequency-vs-receiver image, but uses filled
    contours instead of imshow. It is useful for emphasizing coherent spectral
    ridges across receiver position.

    x-axis: receiver position
    y-axis: frequency
    color: FFT amplitude or log10 FFT amplitude
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    dt = float(np.nanmedian(np.diff(t)))
    freqs = np.fft.rfftfreq(data.shape[1], d=dt)
    amp = np.abs(np.fft.rfft(data, axis=1))
    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    amp = amp[:, keep]

    if normalize_per_trace:
        mx = np.nanmax(amp, axis=1)
        mx[~np.isfinite(mx) | (mx == 0)] = 1.0
        amp = amp / mx[:, None]

    # Optional tiny moving-average smoothing in frequency bins only, for contours.
    if smooth_bins and smooth_bins > 1:
        k = int(smooth_bins)
        kernel = np.ones(k, dtype=float) / k
        amp = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, amp)

    z = np.log10(np.maximum(amp, 1e-20)) if log10 else amp

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return

    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    X, Y = np.meshgrid(rx, freqs)

    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.contourf(
        X,
        Y,
        z.T,
        levels=np.linspace(vmin, vmax, int(levels)),
        cmap="viridis",
        extend="both",
    )

    # Add sparse contour lines for readability.
    try:
        ax.contour(
            X,
            Y,
            z.T,
            levels=np.linspace(vmin, vmax, max(5, int(levels) // 3)),
            colors="k",
            linewidths=0.25,
            alpha=0.35,
        )
    except Exception:
        pass

    ce = _cave_extent_any(cave_extent)
    if ce is not None:
        ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
    ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(alpha=0.15)
    cb = fig.colorbar(cf, ax=ax)
    label = "log10 normalized FFT amplitude" if (log10 and normalize_per_trace) else (
        "log10 FFT amplitude" if log10 else ("normalized FFT amplitude" if normalize_per_trace else "FFT amplitude")
    )
    cb.set_label(label)

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)



def plot_trace_similarity_summary(
    rows: list[dict],
    *,
    source_x_m: float,
    title: str,
    outfile: Path,
    cave_extent: Optional[tuple[float, float]],
) -> None:
    """Plot fast per-trace zero-lag similarity metrics versus receiver position."""
    if not rows:
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rx = np.asarray([r["receiver_x_m"] for r in rows], dtype=float)
    zcorr = np.asarray([r["zero_lag_corr"] for r in rows], dtype=float)
    nrmse_pct = 100.0 * np.asarray([r["nrmse_reference_rms"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    panels = [
        (zcorr, "zero-lag correlation", (-1.05, 1.05)),
        (nrmse_pct, "NRMSE vs reference RMS (%)", None),
    ]

    for ax, (y, ylabel, ylim) in zip(axes, panels):
        ax.plot(rx, y, marker="o", ms=3.0, lw=1.0)
        ce = _cave_extent_any(cave_extent)
        if ce is not None:
            ax.axvspan(ce[0], ce[1], color="0.5", alpha=0.15)
        ax.axvline(source_x_m, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    axes[0].set_title(
        f"{title}\n"
        f"mean zero-lag r={np.nanmean(zcorr):.4f}, "
        f"median r={np.nanmedian(zcorr):.4f}; "
        f"mean NRMSE={np.nanmean(nrmse_pct):.2f}%"
    )
    axes[-1].set_xlabel("receiver x (m)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def _shared_spectral_arrays(
    t: np.ndarray,
    datasets: list[np.ndarray],
    *,
    max_freq_hz: float,
    normalize_per_trace: bool,
    log10: bool,
    smooth_bins: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    dt = float(np.nanmedian(np.diff(t)))
    freqs = np.fft.rfftfreq(datasets[0].shape[1], d=dt)
    keep = freqs <= max_freq_hz
    freqs = freqs[keep]

    out = []
    for data in datasets:
        amp = np.abs(np.fft.rfft(data, axis=1))[:, keep]
        if normalize_per_trace:
            mx = np.nanmax(amp, axis=1)
            mx[~np.isfinite(mx) | (mx == 0)] = 1.0
            amp = amp / mx[:, None]
        if smooth_bins and smooth_bins > 1:
            k = int(smooth_bins)
            kernel = np.ones(k, dtype=float) / k
            amp = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, amp)
        z = np.log10(np.maximum(amp, 1e-20)) if log10 else amp
        out.append(z)

    return freqs, out


def plot_spectral_contours_three_panel(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    diff: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    max_freq_hz: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = False,
    log10: bool = True,
    levels: int = 24,
    smooth_bins: int = 1,
) -> None:
    """Combined Charlie-style spectral contour figure with shared scale/colorbar."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    freqs, arrays = _shared_spectral_arrays(
        t, [reference, comparison, diff],
        max_freq_hz=max_freq_hz,
        normalize_per_trace=normalize_per_trace,
        log10=log10,
        smooth_bins=smooth_bins,
    )
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.any(np.isfinite(a))])
    if finite.size == 0:
        return
    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    X, Y = np.meshgrid(rx, freqs)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.12, top=0.86, wspace=0.10)
    cax = fig.add_axes([0.925, 0.18, 0.018, 0.62])
    titles = [f"Reference: {reference_label}", f"Comparison: {comparison_label}", "Difference: reference - comparison"]

    cf = None
    for ipanel, (ax, z, title) in enumerate(zip(axes, arrays, titles)):
        cf = ax.contourf(X, Y, z.T, levels=np.linspace(vmin, vmax, int(levels)),
                         cmap="viridis", extend="both")
        try:
            ax.contour(X, Y, z.T, levels=np.linspace(vmin, vmax, max(5, int(levels)//3)),
                       colors="k", linewidths=0.20, alpha=0.25)
        except Exception:
            pass
        ce = _cave_extent_for_panel(cave_extent, ipanel)
        if ce is not None:
            ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
        ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("frequency (Hz)")
    label = "log10 FFT amplitude"
    if normalize_per_trace:
        label = "log10 trace-normalized FFT amplitude" if log10 else "trace-normalized FFT amplitude"
    elif not log10:
        label = "FFT amplitude"
    if cf is not None:
        cb = fig.colorbar(cf, cax=cax)
        cb.set_label(label)

    diff_metrics = comparison_metrics(reference, comparison, diff)
    fig.suptitle(
        f"Spectral contours, source x={source_x_m:.3f} m; shared color scale; "
        f"RMS diff/ref={fmt_pct(diff_metrics['rmsdiff_over_rms_reference_pct'])}, "
        f"peak diff/ref={fmt_pct(diff_metrics['maxdiff_over_max_reference_pct'])}",
        y=0.965,
    )
    fig.savefig(outfile, dpi=180)
    plt.close(fig)

def plot_frequency_three_panel(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    diff: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    max_freq_hz: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = False,
) -> None:
    """Combined frequency-vs-receiver figure with one shared colorbar."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    dt = float(np.nanmedian(np.diff(t)))
    freqs = np.fft.rfftfreq(reference.shape[1], d=dt)
    keep = freqs <= max_freq_hz
    freqs = freqs[keep]

    arrays = []
    for data in [reference, comparison, diff]:
        amp = np.abs(np.fft.rfft(data, axis=1))[:, keep]
        if normalize_per_trace:
            mx = np.nanmax(amp, axis=1)
            mx[~np.isfinite(mx) | (mx == 0)] = 1.0
            amp = amp / mx[:, None]
        arrays.append(amp)

    c = clip_value(*arrays, percentile=98.0)
    if not np.isfinite(c) or c <= 0:
        c = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.12, top=0.86, wspace=0.10)
    cax = fig.add_axes([0.925, 0.18, 0.018, 0.62])

    extent = [rx.min(), rx.max(), freqs.min(), freqs.max()]
    titles = [f"Reference: {reference_label}", f"Comparison: {comparison_label}", "Difference: reference - comparison"]
    im = None
    for ipanel, (ax, amp, title) in enumerate(zip(axes, arrays, titles)):
        im = ax.imshow(amp.T, aspect="auto", origin="lower", interpolation="nearest",
                       extent=extent, cmap="viridis", vmin=0, vmax=c)
        ce = _cave_extent_for_panel(cave_extent, ipanel)
        if ce is not None:
            ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
        ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("frequency (Hz)")
    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("trace-normalized FFT amplitude" if normalize_per_trace else "FFT amplitude")

    diff_metrics = comparison_metrics(reference, comparison, diff)
    fig.suptitle(
        f"Frequency receiver plots, source x={source_x_m:.3f} m; shared color scale; "
        f"RMS diff/ref={fmt_pct(diff_metrics['rmsdiff_over_rms_reference_pct'])}",
        y=0.965,
    )
    fig.savefig(outfile, dpi=180)
    plt.close(fig)

def plot_wiggle_three_panel(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    diff: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    cave_extent: Optional[tuple[float, float]],
    normalize_traces: bool = False,
    scale: float = 0.45,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    datasets = [np.asarray(reference, dtype=float).copy(),
                np.asarray(comparison, dtype=float).copy(),
                np.asarray(diff, dtype=float).copy()]

    if normalize_traces:
        for k in range(len(datasets)):
            mx = np.nanmax(np.abs(datasets[k]), axis=1)
            mx[~np.isfinite(mx) | (mx == 0)] = 1.0
            datasets[k] = datasets[k] / mx[:, None]
    else:
        c = clip_value(*datasets, percentile=99.0)
        c = c if np.isfinite(c) and c > 0 else 1.0
        datasets = [np.clip(d / c, -1, 1) for d in datasets]

    dx = np.nanmedian(np.diff(np.sort(rx))) if len(rx) > 1 else 1.0
    amp = scale * dx

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    titles = [
        f"Reference: {reference_label}",
        f"Comparison: {comparison_label}",
        "Difference: reference - comparison",
    ]

    for ipanel, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
        ce = _cave_extent_for_panel(cave_extent, ipanel)
        if ce is not None:
            ax.axvspan(ce[0], ce[1], color="0.5", alpha=0.15)
        for tr, x in zip(data, rx):
            y = x + tr * amp
            ax.plot(y, t, color="k", lw=0.40)
            ax.fill_betweenx(t, x, y, where=(tr >= 0), color="k", alpha=0.20, linewidth=0)
        ax.axvline(source_x_m, color="r", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)
        ax.invert_yaxis()

    axes[0].set_ylabel("time (s)")
    mode = "trace-normalized display" if normalize_traces else "shared physical display scale"
    fig.suptitle(f"Wiggles, source x={source_x_m:.3f} m; {mode}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_band_energy_three_panel(
    reference: np.ndarray,
    comparison: np.ndarray,
    diff: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    *,
    source_x_m: float,
    reference_label: str,
    comparison_label: str,
    outfile: Path,
    bands: list[tuple[float, float]],
    window_s: float,
    step_s: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = False,
    log10: bool = False,
) -> None:
    """Combined band-energy figure with one shared colorbar across all panels."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Reference", "Comparison", "Difference"]
    datasets = [reference, comparison, diff]
    image_sets = []
    centers = None
    for data in datasets:
        centers_i, images = sliding_band_energy(
            data, time_s, bands, window_s=window_s, step_s=step_s,
            normalize_per_trace=normalize_per_trace, log10=log10)
        centers = centers_i
        image_sets.append(images)

    band_labels = list(image_sets[0].keys())
    nrows = len(band_labels)
    all_arrays = [images[band_label] for images in image_sets for band_label in band_labels]

    if log10:
        finite = np.concatenate([a[np.isfinite(a)].ravel() for a in all_arrays if np.any(np.isfinite(a))])
        vmin = float(np.nanpercentile(finite, 2)) if finite.size else -1.0
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    else:
        vmin = 0.0
        vmax = clip_value(*all_arrays, percentile=98.0)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    fig_h = max(2.0*nrows + 1.4, 7.0)
    fig, axes = plt.subplots(nrows, 3, figsize=(18.5, fig_h), sharex=True, sharey=True, constrained_layout=False)
    if nrows == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.060, right=0.90, bottom=0.08, top=0.90, wspace=0.08, hspace=0.16)
    cax = fig.add_axes([0.925, 0.18, 0.018, 0.66])

    extent = [receiver_x_m.min(), receiver_x_m.max(), centers.max(), centers.min()]
    im = None
    for r, band_label in enumerate(band_labels):
        band_arrays = [image_sets[c][band_label] for c in range(3)]
        for cidx, (label, img) in enumerate(zip(labels, band_arrays)):
            ax = axes[r, cidx]
            im = ax.imshow(img.T, aspect="auto", interpolation="nearest", extent=extent,
                           cmap="magma", vmin=vmin, vmax=vmax)
            ce = _cave_extent_for_panel(cave_extent, cidx)
            if ce is not None:
                ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
            ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
            ax.grid(alpha=0.15)
            if r == 0:
                ax.set_title(f"{label}: " + ([reference_label, comparison_label, "reference - comparison"][cidx]))
            if cidx == 0:
                ax.set_ylabel(f"{band_label}\ntime (s)")
            if r == nrows-1:
                ax.set_xlabel("receiver x (m)")

    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("log10 band energy" if log10 else ("trace-normalized band energy" if normalize_per_trace else "band energy"))

    diff_metrics = comparison_metrics(reference, comparison, diff)
    fig.suptitle(
        f"Band energy, source x={source_x_m:.3f} m; shared color scale across all bands/panels; "
        f"RMS diff/ref={fmt_pct(diff_metrics['rmsdiff_over_rms_reference_pct'])}",
        y=0.965,
    )
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Band energy
# -----------------------------------------------------------------------------

def parse_frequency_bands(text: str) -> list[tuple[float, float]]:
    bands = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
        elif ":" in item:
            a, b = item.split(":", 1)
        else:
            raise ValueError(f"Bad frequency band {item!r}; expected fmin-fmax")
        fmin = float(a)
        fmax = float(b)
        if fmax <= fmin:
            raise ValueError(f"Bad frequency band {item!r}; fmax must exceed fmin")
        bands.append((fmin, fmax))
    if not bands:
        raise ValueError("No valid frequency bands parsed")
    return bands


def sliding_band_energy(
    data: np.ndarray,
    time_s: np.ndarray,
    bands: list[tuple[float, float]],
    *,
    window_s: float,
    step_s: float,
    normalize_per_trace: bool = True,
    log10: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d = np.asarray(data, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if d.ndim != 2:
        raise ValueError("data must be 2-D")
    if t.size != d.shape[1]:
        raise ValueError("time_s length must match data sample count")
    if t.size < 4:
        raise ValueError("Too few samples")

    dt = float(np.nanmedian(np.diff(t)))
    nwin = max(8, int(round(window_s / dt)))
    nstep = max(1, int(round(step_s / dt)))
    if nwin > d.shape[1]:
        nwin = d.shape[1]

    starts = np.arange(0, d.shape[1] - nwin + 1, nstep, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
        nwin = d.shape[1]

    centers = t[starts + nwin // 2]
    freqs = np.fft.rfftfreq(nwin, d=dt)
    win = np.hanning(nwin)
    if not np.any(win):
        win = np.ones(nwin)

    images = {}
    for fmin, fmax in bands:
        label = f"{fmin:g}-{fmax:g} Hz"
        keep_f = (freqs >= fmin) & (freqs < fmax)
        img = np.zeros((d.shape[0], starts.size), dtype=float)

        if not np.any(keep_f):
            img[:] = np.nan
            images[label] = img
            continue

        for j, start in enumerate(starts):
            seg = d[:, start:start + nwin].astype(float, copy=True)
            seg = seg - np.nanmean(seg, axis=1, keepdims=True)
            seg *= win[None, :]
            spec = np.fft.rfft(seg, axis=1)
            power = np.abs(spec[:, keep_f]) ** 2
            img[:, j] = np.sqrt(np.nanmean(power, axis=1))

        if normalize_per_trace:
            mx = np.nanmax(img, axis=1)
            mx[~np.isfinite(mx) | (mx == 0)] = 1.0
            img = img / mx[:, None]

        if log10:
            img = np.log10(np.maximum(img, 1e-20))

        images[label] = img

    return centers, images


def plot_band_energy(
    data: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    *,
    source_x_m: float,
    title: str,
    outfile: Path,
    bands: list[tuple[float, float]],
    window_s: float,
    step_s: float,
    cave_extent: Optional[tuple[float, float]],
    normalize_per_trace: bool = True,
    log10: bool = False,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    centers, images = sliding_band_energy(
        data,
        time_s,
        bands,
        window_s=window_s,
        step_s=step_s,
        normalize_per_trace=normalize_per_trace,
        log10=log10,
    )

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(max(5.0 * n, 8.0), 6), sharey=True)
    if n == 1:
        axes = [axes]

    extent = [receiver_x_m.min(), receiver_x_m.max(), centers.max(), centers.min()]

    for ax, (label, img) in zip(axes, images.items()):
        c = clip_value(img)
        im = ax.imshow(
            img.T,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="magma",
            vmin=np.nanmin(img) if log10 else 0,
            vmax=c,
        )
        ce = _cave_extent_any(cave_extent)
        if ce is not None:
            ax.axvspan(ce[0], ce[1], color="w", alpha=0.18)
        ax.axvline(source_x_m, color="w", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("receiver x (m)")
        ax.grid(alpha=0.15)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("log10 band energy" if log10 else ("normalized band energy" if normalize_per_trace else "band energy"))

    axes[0].set_ylabel("time (s)")
    mode = "trace-normalized" if normalize_per_trace else "absolute"
    fig.suptitle(f"{title}; sliding-window band energy ({mode}; standalone color scale)", y=0.99)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scaling diagnostics and output
# -----------------------------------------------------------------------------

def local_window_mask(t, center_s, halfwidth_s):
    keep = np.abs(t - center_s) <= halfwidth_s
    if not np.any(keep):
        i = int(np.argmin(np.abs(t - center_s)))
        keep = np.zeros_like(t, dtype=bool)
        keep[i] = True
    return keep


def compute_trace_peak_scaling(
    t: np.ndarray,
    rx: np.ndarray,
    reference: np.ndarray,
    comparison_raw: np.ndarray,
    comparison_scaled: np.ndarray,
    *,
    reference_label: str,
    comparison_label: str,
    scale_tmin: Optional[float],
    scale_tmax: Optional[float],
    halfwidth_s: float,
) -> list[dict]:
    keep = np.ones_like(t, dtype=bool)
    if scale_tmin is not None:
        keep &= t >= scale_tmin
    if scale_tmax is not None:
        keep &= t <= scale_tmax
    if not np.any(keep):
        keep = np.ones_like(t, dtype=bool)

    t_win = t[keep]
    rows = []

    for i, x in enumerate(rx):
        ref = np.asarray(reference[i, keep], dtype=float)
        cmp_raw = np.asarray(comparison_raw[i, keep], dtype=float)
        cmp_scaled = np.asarray(comparison_scaled[i, keep], dtype=float)

        ref_pos = float(np.nanmax(ref)) if ref.size else np.nan
        ref_neg = float(np.nanmin(ref)) if ref.size else np.nan
        cmp_pos = float(np.nanmax(cmp_raw)) if cmp_raw.size else np.nan
        cmp_neg = float(np.nanmin(cmp_raw)) if cmp_raw.size else np.nan

        scale_pos = ref_pos / cmp_pos if np.isfinite(ref_pos) and np.isfinite(cmp_pos) and cmp_pos != 0 else np.nan
        scale_neg = ref_neg / cmp_neg if np.isfinite(ref_neg) and np.isfinite(cmp_neg) and cmp_neg != 0 else np.nan
        valid = [v for v in (scale_pos, scale_neg) if np.isfinite(v)]

        if ref.size:
            j = int(np.nanargmax(np.abs(ref)))
            t_peak = float(t_win[j])
            pkeep = local_window_mask(t_win, t_peak, halfwidth_s)
            scale_peak_lsq = lsq_scale(ref[pkeep], cmp_raw[pkeep])
            corr_peak = pearson_corr(ref[pkeep], cmp_raw[pkeep])
            corr_window = pearson_corr(ref, cmp_raw)
        else:
            t_peak = np.nan
            scale_peak_lsq = np.nan
            corr_peak = np.nan
            corr_window = np.nan

        rows.append(
            {
                "trace_index_1based": i + 1,
                "receiver_x_m": float(x),
                "reference_label": reference_label,
                "comparison_label": comparison_label,
                "scale_window_tmin_s": scale_tmin,
                "scale_window_tmax_s": scale_tmax,
                "peak_halfwidth_s": halfwidth_s,
                "reference_peak_time_s": t_peak,
                "reference_pos_peak": ref_pos,
                "reference_neg_peak": ref_neg,
                "comparison_raw_pos_peak": cmp_pos,
                "comparison_raw_neg_peak": cmp_neg,
                "scale_pos_peak_reference_over_comparison_raw": scale_pos,
                "scale_neg_peak_reference_over_comparison_raw": scale_neg,
                "scale_peak_mean_reference_over_comparison_raw": float(np.mean(valid)) if valid else np.nan,
                "scale_peak_median_reference_over_comparison_raw": float(np.median(valid)) if valid else np.nan,
                "scale_peak_lsq_reference_over_comparison_raw": scale_peak_lsq,
                "scale_rms_reference_over_comparison_raw": rms(ref) / rms(cmp_raw) if rms(cmp_raw) else np.nan,
                "corr_peak_window_raw": corr_peak,
                "corr_scale_window_raw": corr_window,
                "reference_rms_scale_window": rms(ref),
                "comparison_raw_rms_scale_window": rms(cmp_raw),
                "comparison_scaled_rms_scale_window": rms(cmp_scaled),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
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
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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


def write_diff_segy(
    outfile: Path,
    diff: np.ndarray,
    dt_s: float,
    rx: np.ndarray,
    source_x_m: float,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    st = Stream()
    for i, (x, trdata) in enumerate(zip(rx, diff), start=1):
        tr = Trace(data=np.asarray(trdata, dtype=np.float32))
        tr.stats.delta = float(dt_s)
        tr.stats.network = "DF"
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
# Main pair processing
# -----------------------------------------------------------------------------

def process_pair(pair: Pair, *, stations: list[Station], args, preprocess_kwargs: dict, cave_extent, bands) -> tuple[PairResult, list[dict]]:
    ref_g = read_gather_for_pair(
        pair.reference_path,
        pair.reference_kind,
        stations=stations,
        source=pair.source,
        args=args,
        preprocess_kwargs=preprocess_kwargs,
    )
    cmp_g = read_gather_for_pair(
        pair.comparison_path,
        pair.comparison_kind,
        stations=stations,
        source=pair.source,
        args=args,
        preprocess_kwargs=preprocess_kwargs,
    )

    t, ref, cmp_raw, rx = align_gathers(
        ref_g,
        cmp_g,
        receiver_tolerance_m=args.receiver_tolerance_m,
        tmin=args.tmin,
        tmax=args.tmax,
        comparison_time_shift_ms=args.comparison_time_shift_ms,
    )

    sfactor = compute_scale_factor(
        ref,
        cmp_raw,
        t,
        mode=args.scale_mode,
        fixed_scale=args.fixed_scale_factor,
        scale_tmin=args.scale_tmin,
        scale_tmax=args.scale_tmax,
    )

    cmp_scaled = cmp_raw * sfactor
    diff = ref - cmp_scaled
    metrics = comparison_metrics(ref, cmp_scaled, diff)

    shot_dir = args.output_dir / safe_dir(pair.shot_index, pair.source)
    shot_dir.mkdir(parents=True, exist_ok=True)

    physical_similarity_rows = []
    trace_normalized_similarity_rows = []

    if args.write_trace_similarity:
        physical_similarity_rows = compute_trace_similarity_metrics(
            t,
            rx,
            ref,
            cmp_scaled,
            max_lag_s=args.similarity_max_lag_s,
            label="physical",
        )
        write_csv(shot_dir / "trace_similarity_metrics_physical.csv", physical_similarity_rows)
        plot_trace_similarity_summary(
            physical_similarity_rows,
            source_x_m=pair.source.x_m,
            title=f"Trace similarity, physical amplitudes: {pair.reference_label} vs {pair.comparison_label}",
            outfile=shot_dir / "trace_similarity_metrics_physical.png",
            cave_extent=cave_extent,
        )

    physical_similarity_summary = summarize_similarity_rows(physical_similarity_rows, prefix="physical_")

    (shot_dir / "difference_amplitude_metrics.txt").write_text(
        "\n".join([
            f"source_x_m = {pair.source.x_m:.6g}",
            f"reference_label = {pair.reference_label}",
            f"comparison_label = {pair.comparison_label}",
            "",
            "PHYSICAL AMPLITUDE DIFFERENCE METRICS",
            f"max_abs_reference_left_panel = {metrics['max_abs_reference']:.10g}",
            f"max_abs_comparison_center_panel = {metrics['max_abs_comparison']:.10g}",
            f"max_abs_difference_right_panel = {metrics['max_abs_difference']:.10g}",
            f"rms_reference_left_panel = {metrics['rms_reference']:.10g}",
            f"rms_comparison_center_panel = {metrics['rms_comparison']:.10g}",
            f"rms_difference_right_panel = {metrics['rms_difference']:.10g}",
            f"peak_right_over_peak_left_pct = {metrics['maxdiff_over_max_reference_pct']:.10g}",
            f"peak_right_over_peak_center_pct = {metrics['maxdiff_over_max_comparison_pct']:.10g}",
            f"rms_right_over_rms_left_pct = {metrics['rmsdiff_over_rms_reference_pct']:.10g}",
            f"rms_right_over_rms_center_pct = {metrics['rmsdiff_over_rms_comparison_pct']:.10g}",
            "",
            "PHYSICAL TRACE SIMILARITY",
            f"mean_zero_lag_corr = {physical_similarity_summary.get('physical_zero_lag_corr_mean', np.nan):.10g}",
            f"median_zero_lag_corr = {physical_similarity_summary.get('physical_zero_lag_corr_median', np.nan):.10g}",
            f"mean_nrmse_reference_rms_pct = {100.0 * physical_similarity_summary.get('physical_nrmse_reference_rms_mean', np.nan):.10g}",
            f"median_nrmse_reference_rms_pct = {100.0 * physical_similarity_summary.get('physical_nrmse_reference_rms_median', np.nan):.10g}",
            "",
        ]),
        encoding="utf-8",
    )

    plot_three_panel(
        t,
        rx,
        ref,
        cmp_scaled,
        diff,
        source_x_m=pair.source.x_m,
        reference_label=pair.reference_label,
        comparison_label=pair.comparison_label,
        outfile=shot_dir / "combined_image_reference_comparison_difference.png",
        cave_extent=cave_extent,
        metrics=metrics,
        title_extra=(
            f"; scale={args.scale_mode}, factor={sfactor:.5g}; "
            f"synthetic source target={args.synthetic_source_target_factor:.5g}; "
            "physical/scaled amplitudes"
        ),
    )

    if args.write_trace_normalized_figures:
        ref_norm = trace_normalize(ref, method=args.trace_normalize_method)
        cmp_norm = trace_normalize(cmp_scaled, method=args.trace_normalize_method)
        diff_norm = ref_norm - cmp_norm
        norm_metrics = comparison_metrics(ref_norm, cmp_norm, diff_norm)
        plot_three_panel(
            t,
            rx,
            ref_norm,
            cmp_norm,
            diff_norm,
            source_x_m=pair.source.x_m,
            reference_label=f"{pair.reference_label}, trace-normalized",
            comparison_label=f"{pair.comparison_label}, trace-normalized",
            outfile=shot_dir / "combined_image_reference_comparison_difference_trace_normalized.png",
            cave_extent=cave_extent,
            metrics=norm_metrics,
            title_extra=(
                f"; trace-normalized ({args.trace_normalize_method}); "
                "metrics describe normalized waveform mismatch, not physical amplitude change"
            ),
        )
    else:
        diff_norm = None

    if args.write_combined_three_panel_products:
        plot_wiggle_three_panel(
            t,
            rx,
            ref,
            cmp_scaled,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_wiggles_reference_comparison_difference.png",
            cave_extent=cave_extent,
            normalize_traces=args.combined_wiggle_trace_normalize,
            scale=args.overlay_wiggle_scale,
        )

    if args.write_individual_wiggles:
        plot_wiggle(
            t,
            rx,
            ref,
            source_x_m=pair.source.x_m,
            title=f"Reference: {pair.reference_label}, source x={pair.source.x_m:.3f} m",
            outfile=shot_dir / "wiggle_reference.png",
            cave_extent=cave_extent,
        )
        plot_wiggle(
            t,
            rx,
            cmp_scaled,
            source_x_m=pair.source.x_m,
            title=f"Comparison: {pair.comparison_label}, source x={pair.source.x_m:.3f} m",
            outfile=shot_dir / "wiggle_comparison.png",
            cave_extent=cave_extent,
        )
        plot_wiggle(
            t,
            rx,
            diff,
            source_x_m=pair.source.x_m,
            title=f"Difference: reference - comparison, source x={pair.source.x_m:.3f} m",
            outfile=shot_dir / "wiggle_difference.png",
            cave_extent=cave_extent,
        )

    if args.write_overlay_wiggles:
        plot_overlay_wiggles(
            t,
            rx,
            ref,
            cmp_scaled,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "wiggle_overlay_comparison_blue_reference_red.png",
            cave_extent=cave_extent,
            normalize=args.overlay_normalize,
            wiggle_scale=args.overlay_wiggle_scale,
        )

    norm_freq = args.frequency_trace_normalization
    if args.write_combined_three_panel_products:
        plot_frequency_three_panel(
            t,
            rx,
            ref,
            cmp_scaled,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_frequency_receiver_reference_comparison_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
        )

    plot_frequency(
        t,
        rx,
        ref,
        source_x_m=pair.source.x_m,
        title=f"Frequency vs receiver: reference {pair.reference_label}",
        outfile=shot_dir / "frequency_receiver_reference.png",
        max_freq_hz=args.max_freq_hz,
        cave_extent=cave_extent,
        normalize_per_trace=norm_freq,
    )
    plot_frequency(
        t,
        rx,
        cmp_scaled,
        source_x_m=pair.source.x_m,
        title=f"Frequency vs receiver: comparison {pair.comparison_label}",
        outfile=shot_dir / "frequency_receiver_comparison.png",
        max_freq_hz=args.max_freq_hz,
        cave_extent=cave_extent,
        normalize_per_trace=norm_freq,
    )
    plot_frequency(
        t,
        rx,
        diff,
        source_x_m=pair.source.x_m,
        title="Frequency vs receiver: difference reference - comparison",
        outfile=shot_dir / "frequency_receiver_difference.png",
        max_freq_hz=args.max_freq_hz,
        cave_extent=cave_extent,
        normalize_per_trace=norm_freq,
    )

    if args.write_spectral_contours and args.write_combined_three_panel_products:
        plot_spectral_contours_three_panel(
            t,
            rx,
            ref,
            cmp_scaled,
            diff,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_spectral_contours_reference_comparison_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )

    if args.write_spectral_contours and args.write_standalone_spectral_contours:
        plot_spectral_contours(
            t,
            rx,
            ref,
            source_x_m=pair.source.x_m,
            title=f"Spectral contours: reference {pair.reference_label}",
            outfile=shot_dir / "spectral_contours_reference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )
        plot_spectral_contours(
            t,
            rx,
            cmp_scaled,
            source_x_m=pair.source.x_m,
            title=f"Spectral contours: comparison {pair.comparison_label}",
            outfile=shot_dir / "spectral_contours_comparison.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )
        plot_spectral_contours(
            t,
            rx,
            diff,
            source_x_m=pair.source.x_m,
            title="Spectral contours: difference reference - comparison",
            outfile=shot_dir / "spectral_contours_difference.png",
            max_freq_hz=args.max_freq_hz,
            cave_extent=cave_extent,
            normalize_per_trace=args.frequency_trace_normalization,
            log10=args.spectral_contour_log10,
            levels=args.spectral_contour_levels,
            smooth_bins=args.spectral_contour_smooth_bins,
        )

    if args.write_band_energy and args.write_combined_three_panel_products:
        plot_band_energy_three_panel(
            ref,
            cmp_scaled,
            diff,
            t,
            rx,
            source_x_m=pair.source.x_m,
            reference_label=pair.reference_label,
            comparison_label=pair.comparison_label,
            outfile=shot_dir / "combined_band_energy_reference_comparison_difference.png",
            bands=bands,
            window_s=args.band_energy_window_s,
            step_s=args.band_energy_step_s,
            cave_extent=cave_extent,
            normalize_per_trace=args.band_energy_normalize_per_trace,
            log10=args.band_energy_log10,
        )

    if args.write_band_energy and args.write_standalone_band_energy:
        for name, data in [
            ("reference", ref),
            ("comparison", cmp_scaled),
            ("difference", diff),
        ]:
            plot_band_energy(
                data,
                t,
                rx,
                source_x_m=pair.source.x_m,
                title=f"Band energy: {name}",
                outfile=shot_dir / f"band_energy_{name}.png",
                bands=bands,
                window_s=args.band_energy_window_s,
                step_s=args.band_energy_step_s,
                cave_extent=cave_extent,
                normalize_per_trace=args.band_energy_normalize_per_trace,
                log10=args.band_energy_log10,
            )

        if args.write_trace_normalized_figures and diff_norm is not None:
            plot_band_energy(
                diff_norm,
                t,
                rx,
                source_x_m=pair.source.x_m,
                title="Band energy: trace-normalized difference",
                outfile=shot_dir / "band_energy_difference_trace_normalized.png",
                bands=bands,
                window_s=args.band_energy_window_s,
                step_s=args.band_energy_step_s,
                cave_extent=cave_extent,
                normalize_per_trace=args.band_energy_normalize_per_trace,
                log10=args.band_energy_log10,
            )

    if args.write_diagnostic_bandpass:
        bp_ref = apply_obspy_bandpass_to_matrix(
            ref,
            t,
            freqmin=args.diagnostic_bandpass_fmin,
            freqmax=args.diagnostic_bandpass_fmax,
            corners=args.diagnostic_bandpass_corners,
            zerophase=args.diagnostic_bandpass_zerophase,
        )
        bp_cmp = apply_obspy_bandpass_to_matrix(
            cmp_scaled,
            t,
            freqmin=args.diagnostic_bandpass_fmin,
            freqmax=args.diagnostic_bandpass_fmax,
            corners=args.diagnostic_bandpass_corners,
            zerophase=args.diagnostic_bandpass_zerophase,
        )
        write_diagnostic_comparison_product(
            name=f"bandpass_{args.diagnostic_bandpass_fmin:g}_{args.diagnostic_bandpass_fmax:g}Hz",
            t=t,
            rx=rx,
            reference=bp_ref,
            comparison=bp_cmp,
            pair=pair,
            shot_dir=shot_dir,
            args=args,
            cave_extent=cave_extent,
            bands=bands,
            title_extra=(
                f"; diagnostic bandpass "
                f"{args.diagnostic_bandpass_fmin:g}-{args.diagnostic_bandpass_fmax:g} Hz"
            ),
        )

    if args.write_fk_filtered:
        fk_ref = apply_fk_filter_to_matrix(
            ref,
            t,
            rx,
            source_x_m=pair.source.x_m,
            min_velocity_mps=args.fk_min_velocity_mps,
            taper_width_mps=args.fk_taper_width_mps,
            use_taper=args.fk_use_taper,
            split_at_source=args.fk_split_at_source,
            spatial_taper_fraction=args.fk_spatial_taper_fraction,
            pad_factor=args.fk_pad_factor,
        )
        fk_cmp = apply_fk_filter_to_matrix(
            cmp_scaled,
            t,
            rx,
            source_x_m=pair.source.x_m,
            min_velocity_mps=args.fk_min_velocity_mps,
            taper_width_mps=args.fk_taper_width_mps,
            use_taper=args.fk_use_taper,
            split_at_source=args.fk_split_at_source,
            spatial_taper_fraction=args.fk_spatial_taper_fraction,
            pad_factor=args.fk_pad_factor,
        )
        write_diagnostic_comparison_product(
            name=f"fk_vmin_{args.fk_min_velocity_mps:g}mps",
            t=t,
            rx=rx,
            reference=fk_ref,
            comparison=fk_cmp,
            pair=pair,
            shot_dir=shot_dir,
            args=args,
            cave_extent=cave_extent,
            bands=bands,
            title_extra=(
                f"; f-k filtered, vmin={args.fk_min_velocity_mps:g} m/s"
            ),
        )

    peak_rows = compute_trace_peak_scaling(
        t,
        rx,
        ref,
        cmp_raw,
        cmp_scaled,
        reference_label=pair.reference_label,
        comparison_label=pair.comparison_label,
        scale_tmin=args.scale_tmin,
        scale_tmax=args.scale_tmax,
        halfwidth_s=args.peak_scale_halfwidth_s,
    )
    for row in peak_rows:
        row.update(
            {
                "shot_index": pair.shot_index,
                "source_id": pair.source.source_id,
                "source_x_m": pair.source.x_m,
                "reference_file": str(pair.reference_path),
                "comparison_file": str(pair.comparison_path),
                "global_scale_factor_applied": sfactor,
                "comparison_time_shift_ms": args.comparison_time_shift_ms,
            }
        )
    write_csv(shot_dir / "trace_peak_scaling_factors.csv", peak_rows)

    if args.write_diff_segy:
        write_diff_segy(
            shot_dir / "difference_reference_minus_comparison.sgy",
            diff,
            dt_s=float(np.nanmedian(np.diff(t))),
            rx=rx,
            source_x_m=pair.source.x_m,
        )

    result = PairResult(
        shot_index=pair.shot_index,
        source_x_m=pair.source.x_m,
        source_label=pair.source.source_id,
        reference_path=pair.reference_path,
        comparison_path=pair.comparison_path,
        reference_label=pair.reference_label,
        comparison_label=pair.comparison_label,
        n_receivers=ref.shape[0],
        n_samples=ref.shape[1],
        dt_s=float(np.nanmedian(np.diff(t))),
        scale_mode=args.scale_mode,
        scale_factor=sfactor,
        time_shift_ms=args.comparison_time_shift_ms,
        max_abs_reference=metrics["max_abs_reference"],
        max_abs_comparison=metrics["max_abs_comparison"],
        max_abs_difference=metrics["max_abs_difference"],
        rms_reference=metrics["rms_reference"],
        rms_comparison=metrics["rms_comparison"],
        rms_difference=metrics["rms_difference"],
        maxdiff_over_max_reference_pct=metrics["maxdiff_over_max_reference_pct"],
        maxdiff_over_max_comparison_pct=metrics["maxdiff_over_max_comparison_pct"],
        rmsdiff_over_rms_reference_pct=metrics["rmsdiff_over_rms_reference_pct"],
        rmsdiff_over_rms_comparison_pct=metrics["rmsdiff_over_rms_comparison_pct"],
        physical_zero_lag_corr_mean=physical_similarity_summary.get("physical_zero_lag_corr_mean", np.nan),
        physical_zero_lag_corr_median=physical_similarity_summary.get("physical_zero_lag_corr_median", np.nan),
        physical_nrmse_reference_rms_mean_pct=100.0 * physical_similarity_summary.get("physical_nrmse_reference_rms_mean", np.nan),
        physical_nrmse_reference_rms_median_pct=100.0 * physical_similarity_summary.get("physical_nrmse_reference_rms_median", np.nan),
        trace_normalized_zero_lag_corr_mean=summarize_similarity_rows(trace_normalized_similarity_rows, prefix="tn_").get("tn_zero_lag_corr_mean", np.nan),
        trace_normalized_zero_lag_corr_median=summarize_similarity_rows(trace_normalized_similarity_rows, prefix="tn_").get("tn_zero_lag_corr_median", np.nan),
        trace_normalized_nrmse_reference_rms_mean_pct=100.0 * summarize_similarity_rows(trace_normalized_similarity_rows, prefix="tn_").get("tn_nrmse_reference_rms_mean", np.nan),
        trace_normalized_nrmse_reference_rms_median_pct=100.0 * summarize_similarity_rows(trace_normalized_similarity_rows, prefix="tn_").get("tn_nrmse_reference_rms_median", np.nan),
    )

    return result, peak_rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_bandpass(text: Optional[str]) -> Optional[tuple[float, float]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--bandpass must be like 5,150")
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified gather-pair comparison engine.")

    p.add_argument("--mode", required=True, choices=["synthetic_vs_synthetic", "real_vs_synthetic"])

    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--stations-file", default="STATIONS")
    p.add_argument("--sources-file", default="SOURCES_LIST.txt")

    p.add_argument("--reference-dir", type=Path, default=None)
    p.add_argument("--comparison-dir", type=Path, required=True)
    p.add_argument("--reference-pattern", default="SURVEY_OUTPUT/**/Uz_file_single_v.su")
    p.add_argument("--comparison-pattern", default="SURVEY_OUTPUT/**/Uz_file_single_v.su")
    p.add_argument("--reference-label", default="reference")
    p.add_argument("--comparison-label", default="comparison")

    p.add_argument("--real-dir", type=Path, default=None)
    p.add_argument("--real-first-file", type=int, default=3005)
    p.add_argument("--real-last-file", type=int, default=3046)
    p.add_argument("--real-shot-first-x-m", type=float, default=82.5)
    p.add_argument("--real-shot-dx-m", type=float, default=2.0)
    p.add_argument("--real-shot-duplicate-x-m", type=float, default=102.5)
    p.add_argument("--real-shot-duplicate-files", default="3015,3016")
    p.add_argument("--real-first-trace-x-m", type=float, default=87.0)
    p.add_argument("--real-dx-m", type=float, default=1.0)
    p.add_argument("--reverse-real-traces", action="store_true")
    p.add_argument("--shot-match-tolerance-m", type=float, default=0.05)

    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--receiver-x-min", type=float, default=None)
    p.add_argument("--receiver-x-max", type=float, default=None)
    p.add_argument("--receiver-tolerance-m", type=float, default=0.05)
    p.add_argument("--component", default=None)

    p.add_argument("--tmin", type=float, default=None)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--comparison-time-shift-ms", type=float, default=0.0)

    p.add_argument("--scale-mode", choices=["none", "fixed", "rms", "lsq", "maxabs"], default="none")
    p.add_argument("--fixed-scale-factor", type=float, default=1.0)
    p.add_argument("--scale-tmin", type=float, default=None)
    p.add_argument("--scale-tmax", type=float, default=None)

    p.add_argument(
        "--normalize-synthetic-source-factor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For synthetic gathers, read each run's DATA/SOURCE factor and multiply "
            "the gather by synthetic_source_target_factor / source_factor before "
            "alignment, plotting, and differencing. This corrects otherwise identical "
            "SPECFEM runs that used different source amplitudes. Use "
            "--no-normalize-synthetic-source-factor to disable."
        ),
    )
    p.add_argument(
        "--synthetic-source-target-factor",
        type=float,
        default=1.0e10,
        help=(
            "Target SPECFEM DATA/SOURCE factor used for synthetic source-amplitude "
            "normalization. Default: 1e10."
        ),
    )
    p.add_argument(
        "--print-synthetic-source-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print source-factor normalization applied to each synthetic gather.",
    )

    p.add_argument("--demean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detrend", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taper-fraction", type=float, default=0.0)
    p.add_argument("--highpass-hz", type=float, default=None)
    p.add_argument("--lowpass-hz", type=float, default=None)
    p.add_argument("--bandpass", type=parse_bandpass, default=None)
    p.add_argument("--filter-corners", type=int, default=4)
    p.add_argument("--zerophase", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--write-diagnostic-bandpass", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write a secondary comparison product after applying an ObsPy bandpass to both parent gathers before differencing.")
    p.add_argument("--diagnostic-bandpass-fmin", type=float, default=25.0,
                   help="Low corner for optional diagnostic bandpass. Default: 25 Hz.")
    p.add_argument("--diagnostic-bandpass-fmax", type=float, default=400.0,
                   help="High corner for optional diagnostic bandpass. Default: 400 Hz; clipped below Nyquist if necessary.")
    p.add_argument("--diagnostic-bandpass-corners", type=int, default=4,
                   help="ObsPy bandpass corners for optional diagnostic bandpass. Default: 4.")
    p.add_argument("--diagnostic-bandpass-zerophase", action=argparse.BooleanOptionalAction, default=True,
                   help="Use zero-phase ObsPy bandpass for optional diagnostic product. Default: true.")

    p.add_argument("--write-fk-filtered", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write a secondary comparison product after f-k velocity filtering both parent gathers before differencing.")
    p.add_argument("--fk-min-velocity-mps", type=float, default=400.0,
                   help="Reject apparent velocities below this value in optional f-k filter. Default: 400 m/s.")
    p.add_argument("--fk-taper-width-mps", type=float, default=100.0,
                   help="Raised-cosine transition width for optional f-k filter. Default: 100 m/s.")
    p.add_argument("--fk-use-taper", action=argparse.BooleanOptionalAction, default=True,
                   help="Use tapered f-k velocity mute rather than hard mute. Default: true.")
    p.add_argument("--fk-split-at-source", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply f-k filter separately to receivers left/right of source. Recommended for split-spread gathers. Default: true.")
    p.add_argument("--fk-spatial-taper-fraction", type=float, default=0.05,
                   help="Fraction of receiver aperture tapered before f-k FFT. Default: 0.05.")
    p.add_argument("--fk-pad-factor", type=int, default=2,
                   help="Zero-padding factor along receiver axis before f-k FFT. Default: 2.")

    p.add_argument("--cave-extent-x-m", default=None)
    p.add_argument("--cave-shade-panels", default="all",
                   help=("Comma-separated panels that receive cave shading in three-panel figures: "
                         "all, reference, comparison, difference. Example: reference,difference"))
    p.add_argument("--par-file", type=Path, default=None)
    p.add_argument("--void-material-id", type=int, default=None)

    p.add_argument("--max-freq-hz", type=float, default=150.0)
    p.add_argument("--frequency-trace-normalization", action=argparse.BooleanOptionalAction, default=False,
                   help="Normalize each trace spectrum for frequency/spectral plots. Default false; use true mainly for real-vs-synthetic diagnostic plots.")

    p.add_argument("--write-spectral-contours", action=argparse.BooleanOptionalAction, default=True,
                   help="Write Charlie-style frequency contour plots. Default: true.")
    p.add_argument("--spectral-contour-log10", action=argparse.BooleanOptionalAction, default=True,
                   help="Plot log10 FFT amplitude in spectral contour plots. Default: true.")
    p.add_argument("--spectral-contour-levels", type=int, default=24,
                   help="Number of filled contour levels. Default: 24.")
    p.add_argument("--spectral-contour-smooth-bins", type=int, default=1,
                   help="Optional smoothing in frequency bins before contouring. Default: 1/no smoothing.")

    p.add_argument("--write-combined-three-panel-products", action=argparse.BooleanOptionalAction, default=True,
                   help="Write combined reference/comparison/difference figures with shared scales. Default: true.")
    p.add_argument("--combined-wiggle-trace-normalize", action=argparse.BooleanOptionalAction, default=False,
                   help="Trace-normalize combined wiggle display. Default false, preserving shared display scale.")

    p.add_argument("--write-diff-segy", action="store_true")
    p.add_argument("--write-individual-wiggles", action="store_true",
                   help="Write separate wiggle_reference/comparison/difference PNGs. Default off; combined products are preferred.")
    p.add_argument("--write-overlay-wiggles", action="store_true",
                   help="Write separate red/blue overlay wiggle PNGs. Default off; combined products are preferred.")
    p.add_argument("--write-standalone-frequency", action=argparse.BooleanOptionalAction, default=False,
                   help="Write separate frequency_receiver_reference/comparison/difference PNGs. Default false.")
    p.add_argument("--write-standalone-spectral-contours", action=argparse.BooleanOptionalAction, default=False,
                   help="Write separate spectral_contours_reference/comparison/difference PNGs. Default false.")
    p.add_argument("--write-standalone-band-energy", action=argparse.BooleanOptionalAction, default=False,
                   help="Write separate band_energy_reference/comparison/difference PNGs. Default false.")
    p.add_argument("--overlay-normalize", choices=["pair", "trace", "none"], default="pair",
                   help="Overlay display normalization. pair=each receiver pair shares one scale; trace=each trace independent; none=global scale. Default: pair.")
    p.add_argument("--overlay-wiggle-scale", type=float, default=0.45)

    p.add_argument("--write-trace-similarity", action=argparse.BooleanOptionalAction, default=True,
                   help="Write per-trace zero-lag correlation, NRMSE, and best-lag metrics/plots. Default: true.")
    p.add_argument("--similarity-max-lag-s", type=float, default=0.0,
                   help="Deprecated/ignored. Similarity metrics are zero-lag only for speed.")

    p.add_argument("--write-trace-normalized-figures", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trace-normalize-method", choices=["rms", "maxabs"], default="rms")

    p.add_argument("--peak-scale-halfwidth-s", type=float, default=0.015)

    p.add_argument("--write-band-energy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--band-energy-bands", default="10-30,30-80,80-150")
    p.add_argument("--band-energy-window-s", type=float, default=0.05)
    p.add_argument("--band-energy-step-s", type=float, default=0.01)
    p.add_argument("--band-energy-normalize-per-trace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--band-energy-log10", action="store_true")

    p.add_argument("--limit", type=int, default=None)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    data_dir = resolve_data_dir(args)
    stations = read_stations(data_dir / args.stations_file)
    sources = read_sources(data_dir / args.sources_file)

    print(f"Read {len(stations)} receivers from {data_dir / args.stations_file}")
    print(f"Read {len(sources)} sources from {data_dir / args.sources_file}")

    if args.mode == "synthetic_vs_synthetic":
        if args.reference_dir is None:
            raise SystemExit("--reference-dir is required for synthetic_vs_synthetic")
        pairs = build_pairs_synthetic_vs_synthetic(args, sources)
    elif args.mode == "real_vs_synthetic":
        if args.real_dir is None:
            raise SystemExit("--real-dir is required for real_vs_synthetic")
        pairs = build_pairs_real_vs_synthetic(args, sources)
    else:
        raise ValueError(args.mode)

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Built {len(pairs)} gather pairs for mode={args.mode}")

    cave_extent = resolve_cave_extent(args)
    if cave_extent is not None:
        panels_raw = str(args.cave_shade_panels or "all").replace(" ", "")
        panels = {p for p in panels_raw.split(",") if p}
        valid = {"all", "reference", "comparison", "difference", "0", "1", "2"}
        bad = panels - valid
        if bad:
            raise SystemExit(f"Invalid --cave-shade-panels value(s): {sorted(bad)}")
        panels = {int(p) if p in {"0", "1", "2"} else p for p in panels}
        cave_extent = {"extent": cave_extent, "panels": panels}
        print(f"Cave shading panels: {sorted(str(p) for p in panels)}")
    bands = parse_frequency_bands(args.band_energy_bands)

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    results = []
    all_peak_rows = []
    failures = []

    for i, pair in enumerate(pairs, start=1):
        print(
            f"[{i}/{len(pairs)}] x={pair.source.x_m:.3f} "
            f"{pair.reference_path.name} - {pair.comparison_path.name}"
        )
        try:
            result, peak_rows = process_pair(
                pair,
                stations=stations,
                args=args,
                preprocess_kwargs=preprocess_kwargs,
                cave_extent=cave_extent,
                bands=bands,
            )
            results.append(result)
            all_peak_rows.extend(peak_rows)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {msg}", file=sys.stderr)
            failures.append(
                {
                    "shot_index": pair.shot_index,
                    "source_x_m": pair.source.x_m,
                    "reference_file": str(pair.reference_path),
                    "comparison_file": str(pair.comparison_path),
                    "error": msg,
                }
            )

    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "shot_index": r.shot_index,
                "source_x_m": r.source_x_m,
                "source_label": r.source_label,
                "reference_file": str(r.reference_path),
                "comparison_file": str(r.comparison_path),
                "reference_label": r.reference_label,
                "comparison_label": r.comparison_label,
                "n_receivers": r.n_receivers,
                "n_samples": r.n_samples,
                "dt_s": r.dt_s,
                "scale_mode": r.scale_mode,
                "scale_factor": r.scale_factor,
                "time_shift_ms": r.time_shift_ms,
                "max_abs_reference": r.max_abs_reference,
                "max_abs_comparison": r.max_abs_comparison,
                "max_abs_difference": r.max_abs_difference,
                "rms_reference": r.rms_reference,
                "rms_comparison": r.rms_comparison,
                "rms_difference": r.rms_difference,
                "maxdiff_over_max_reference_pct": r.maxdiff_over_max_reference_pct,
                "maxdiff_over_max_comparison_pct": r.maxdiff_over_max_comparison_pct,
                "rmsdiff_over_rms_reference_pct": r.rmsdiff_over_rms_reference_pct,
                "rmsdiff_over_rms_comparison_pct": r.rmsdiff_over_rms_comparison_pct,
                "physical_zero_lag_corr_mean": r.physical_zero_lag_corr_mean,
                "physical_zero_lag_corr_median": r.physical_zero_lag_corr_median,
                "physical_nrmse_reference_rms_mean_pct": r.physical_nrmse_reference_rms_mean_pct,
                "physical_nrmse_reference_rms_median_pct": r.physical_nrmse_reference_rms_median_pct,
                "trace_normalized_zero_lag_corr_mean": r.trace_normalized_zero_lag_corr_mean,
                "trace_normalized_zero_lag_corr_median": r.trace_normalized_zero_lag_corr_median,
                "trace_normalized_nrmse_reference_rms_mean_pct": r.trace_normalized_nrmse_reference_rms_mean_pct,
                "trace_normalized_nrmse_reference_rms_median_pct": r.trace_normalized_nrmse_reference_rms_median_pct,
            }
        )

    write_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    write_csv(args.output_dir / "trace_peak_scaling_factors.csv", all_peak_rows)
    write_csv(args.output_dir / "failures.csv", failures)

    print(f"Wrote {args.output_dir / 'comparison_summary.csv'}")
    print(f"Wrote {args.output_dir / 'trace_peak_scaling_factors.csv'}")
    if failures:
        print(f"Wrote failures: {args.output_dir / 'failures.csv'}")
    print(f"Completed {len(results)} / {len(pairs)} pairs")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
