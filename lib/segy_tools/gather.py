from __future__ import annotations

from pathlib import Path
import warnings
from typing import Optional, Sequence

import numpy as np
from obspy import Stream

from .io import (
    read_segy_as_stream,
    write_segy,
    extract_geometry_from_stream,
    extract_geometry_from_segy_stream,
    gather_arrays_to_stream,
    apply_scalar as _io_apply_scalar,
    header_value as _io_header_value,
)


def _apply_scalar(value: float, scalar: int | None) -> float:
    """Deprecated shim. Use ``segy_tools.io.apply_scalar`` instead."""
    warnings.warn(
        "segy_tools.gather._apply_scalar is deprecated; use segy_tools.io.apply_scalar instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _io_apply_scalar(value, scalar)


def _header_value(header, names: Sequence[str], default=None):
    """Deprecated shim. Use ``segy_tools.io.header_value`` instead."""
    warnings.warn(
        "segy_tools.gather._header_value is deprecated; use segy_tools.io.header_value instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _io_header_value(header, names, default=default)


def _select_component(st: Stream, component: str | None = None) -> Stream:
    """Return a copied stream restricted to one component suffix, if requested."""
    if component in (None, "", "all", "ALL", "*"):
        return st.copy()
    component = str(component)
    return Stream([tr.copy() for tr in st if str(tr.stats.channel).endswith(component)])


def stream_to_gather_arrays(
    st: Stream,
    *,
    sort_by="receiver_x",
    component: str | None = None,
    fallback_receiver_spacing_m: Optional[float] = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: Optional[float] = None,
):
    """Convert an ObsPy Stream to gather arrays.

    This remains in ``gather.py`` because it is the bridge from generic ObsPy
    streams to gather-analysis arrays. Geometry extraction itself lives in
    ``segy_tools.io`` and is imported above.
    """
    st_work = _select_component(st, component=component)

    if len(st_work) == 0:
        raise ValueError("Empty Stream after component selection.")

    npts = min(tr.stats.npts for tr in st_work)
    dt = float(st_work[0].stats.delta)
    time = np.arange(npts, dtype=float) * dt
    data = np.vstack([tr.data[:npts].astype(float) for tr in st_work])

    geom = extract_geometry_from_stream(
        st_work,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )
    receiver_x_m = np.asarray(geom["receiver_x_m"], dtype=float)
    source_x_m = geom["source_x_m"]

    if sort_by == "receiver_x":
        order = np.argsort(receiver_x_m)
    elif sort_by == "offset":
        order = np.argsort(np.asarray(geom["offsets_m"], dtype=float))
    elif sort_by == "trace":
        order = np.argsort(np.asarray(geom["receiver_numbers"], dtype=float))
    elif sort_by in ("none", None):
        order = np.arange(len(st_work))
    else:
        raise ValueError("sort_by must be one of 'receiver_x', 'offset', 'trace', or 'none'.")

    data = data[order]
    receiver_x_m = receiver_x_m[order]
    geom = {k: v[order] if isinstance(v, np.ndarray) and len(v) == len(order) else v for k, v in geom.items()}

    return time, data, receiver_x_m, source_x_m, geom


# -----------------------------------------------------------------------------
# Deprecated plotting shims.  Plotting now lives in segy_tools.plotting.
# -----------------------------------------------------------------------------

def _plotting_shim(name: str, *args, **kwargs):
    warnings.warn(
        f"segy_tools.gather.{name} is deprecated; use segy_tools.plotting.{name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import plotting
    return getattr(plotting, name)(*args, **kwargs)


def plot_wiggle_gather_from_stream(*args, **kwargs):
    return _plotting_shim("plot_wiggle_gather_from_stream", *args, **kwargs)


def plot_image_gather_from_stream(*args, **kwargs):
    return _plotting_shim("plot_image_gather_from_stream", *args, **kwargs)


def plot_shot_gather_from_stream(*args, **kwargs):
    return _plotting_shim("plot_shot_gather_from_stream", *args, **kwargs)


def plot_wiggle_gather_from_segy(*args, **kwargs):
    return _plotting_shim("plot_wiggle_gather_from_segy", *args, **kwargs)


def plot_image_gather_from_segy(*args, **kwargs):
    return _plotting_shim("plot_image_gather_from_segy", *args, **kwargs)


def plot_shot_gather_from_segy(*args, **kwargs):
    return _plotting_shim("plot_shot_gather_from_segy", *args, **kwargs)


# -----------------------------------------------------------------------------
# Gather alignment/differencing utilities.
# -----------------------------------------------------------------------------

def _nearest_geometry_indices(rx_ref: np.ndarray, rx_other: np.ndarray, tolerance_m: float):
    """Return matching index pairs for nearest receiver positions."""
    pairs = []
    used = set()
    for i, x in enumerate(rx_ref):
        j = int(np.argmin(np.abs(rx_other - x)))
        if j in used:
            continue
        if abs(rx_other[j] - x) <= tolerance_m:
            pairs.append((i, j))
            used.add(j)
    return pairs


def align_gather_arrays_by_receiver_x(
    time_a,
    data_a,
    rx_a,
    time_b,
    data_b,
    rx_b,
    *,
    geometry_tolerance_m: float = 0.25,
    time_tolerance_s: float = 1e-6,
):
    """Align two gathers on common receiver positions."""
    time_a = np.asarray(time_a, dtype=float)
    time_b = np.asarray(time_b, dtype=float)
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    rx_a = np.asarray(rx_a, dtype=float)
    rx_b = np.asarray(rx_b, dtype=float)

    pairs = _nearest_geometry_indices(rx_a, rx_b, tolerance_m=geometry_tolerance_m)
    if not pairs:
        raise ValueError("No common receiver positions found within geometry_tolerance_m.")

    ia = np.asarray([p[0] for p in pairs], dtype=int)
    ib = np.asarray([p[1] for p in pairs], dtype=int)

    npts = min(data_a.shape[1], data_b.shape[1])
    if len(time_a) < npts or len(time_b) < npts:
        npts = min(len(time_a), len(time_b), npts)

    if np.nanmax(np.abs(time_a[:npts] - time_b[:npts])) > time_tolerance_s:
        raise ValueError("Time axes differ. Resample one gather before differencing.")

    return time_a[:npts], data_a[ia, :npts], data_b[ib, :npts], rx_a[ia]


def difference_segy_gathers(
    segy_a: str | Path,
    segy_b: str | Path,
    *,
    fallback_receiver_spacing_m: float = 1.0,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: float = 0.0,
    sort_by: str = "receiver_x",
    component: str | None = None,
    geometry_tolerance_m: float = 0.25,
    output_segy_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read two SEG-Y gathers and compute A - B after geometry alignment."""
    st_a = read_segy_as_stream(segy_a)
    st_b = read_segy_as_stream(segy_b)

    time_a, data_a, rx_a, sx_a, _ = stream_to_gather_arrays(
        st_a,
        sort_by=sort_by,
        component=component,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )
    time_b, data_b, rx_b, sx_b, _ = stream_to_gather_arrays(
        st_b,
        sort_by=sort_by,
        component=component,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )

    time_s, data_a2, data_b2, receiver_x_m = align_gather_arrays_by_receiver_x(
        time_a,
        data_a,
        rx_a,
        time_b,
        data_b,
        rx_b,
        geometry_tolerance_m=geometry_tolerance_m,
    )
    diff = data_a2 - data_b2

    if output_segy_path is not None:
        source_x = sx_a if sx_a is not None and np.isfinite(sx_a) else fallback_source_x_m
        diff_stream = gather_arrays_to_stream(
            diff,
            dt_s=float(st_a[0].stats.delta),
            starttime=st_a[0].stats.starttime,
            receiver_x_m=receiver_x_m,
            source_x_m=float(source_x),
            shot_number=1,
            station_prefix="D",
        )
        write_segy(diff_stream, output_segy_path, data_encoding=1)

    return time_s, data_a2, data_b2, diff, receiver_x_m
