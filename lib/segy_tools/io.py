"""Generic SEG-Y, SU, ObsPy Stream, and survey-geometry utilities.

This module is deliberately independent of SPECFEM, Deepwave, Geode, nodal
acquisition systems, or any other producer.  Producer-specific packages should
create ObsPy ``Stream`` objects and then call the geometry/header functions here
before writing SEG-Y.

Coordinate convention
---------------------
All public geometry arguments are in metres.  SEG-Y stores many coordinate and
height fields as integers, so this module writes integer centimetres by default
and sets the SEG-Y coordinate/elevation scalars to ``-100``.  A stored value of
``12345`` with scalar ``-100`` therefore means ``123.45 m``.

For a 2-D profile:

* source x-position -> ``source_coordinate_x``;
* receiver x-position -> ``group_coordinate_x``;
* source elevation/topography -> ``surface_elevation_at_source``;
* receiver elevation/topography -> ``receiver_group_elevation``;
* source depth below surface -> ``source_depth_below_surface``.

Many refraction pickers only use the x-coordinate fields.  The elevation fields
are still written using standard SEG-Y trace-header locations so downstream
software can use topography when supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader


@dataclass
class SeismicArrayData:
    """Container for common-shot gather data.

    The package convention is ``data.shape == (n_traces, n_samples)``.
    """

    data: np.ndarray
    fs: float
    dt: float
    time: np.ndarray
    offsets: np.ndarray
    source_file: Optional[str] = None


@dataclass(frozen=True)
class Geometry:
    """Simple regular 2-D survey geometry helper.

    This is useful as a fallback for synthetic gathers.  Irregular geometry is
    better represented by explicit receiver/source arrays or ``TraceGeometry``.
    """

    first_receiver_x_m: float = 0.0
    receiver_spacing_m: float = 1.0
    receiver_z_m: float = 0.0
    first_shot_x_m: float = 0.0
    shot_spacing_m: float = 1.0
    source_z_m: float = 0.0


@dataclass(frozen=True)
class TraceGeometry:
    """Geometry for one seismic trace, in metres."""

    source_x_m: float
    receiver_x_m: float
    source_z_m: float = 0.0
    receiver_z_m: float = 0.0
    shot_id: int = 1
    receiver_id: int = 1
    trace_id: int = 1
    source_y_m: float = 0.0
    receiver_y_m: float = 0.0
    source_depth_m: float = 0.0



def component_to_channel(component: str) -> str:
    """Map simple component names to project-standard synthetic channels."""
    c = str(component).upper()
    if c in ("X", "BXX"):
        return "BXX"
    if c in ("Z", "BXZ"):
        return "BXZ"
    if c in ("Y", "BXY"):
        return "BXY"
    return c

def receiver_x_from_station(station_idx: int, geom: Geometry) -> float:
    """Return receiver x position for a regular station-index geometry."""
    return geom.first_receiver_x_m + (int(station_idx) - 1) * geom.receiver_spacing_m


def source_x_from_shot(shot_number: int, geom: Geometry) -> float:
    """Return source x position for a regular shot-index geometry."""
    return geom.first_shot_x_m + (int(shot_number) - 1) * geom.shot_spacing_m


def values_for_traces(
    values,
    *,
    station_indices: Sequence[int] | None,
    ntraces: int,
    fallback,
    name: str,
) -> np.ndarray:
    """Resolve scalar/sequence/mapping geometry values to one value per trace.

    ``values`` may be ``None``, a scalar, a sequence in trace order, or a
    mapping keyed by integer station index or station names such as ``S0001``.
    ``fallback`` may be a scalar or a sequence with length ``ntraces``.
    """
    if values is None:
        values = fallback

    if isinstance(values, Mapping):
        if station_indices is None:
            raise ValueError(f"{name} was supplied as a mapping, but station_indices are unavailable.")
        out = []
        for sta in station_indices:
            keys = (int(sta), str(sta), f"S{int(sta):04d}", f"S{int(sta)}")
            for key in keys:
                if key in values:
                    out.append(float(values[key]))
                    break
            else:
                raise KeyError(f"No {name} value found for station {sta!r}.")
        return np.asarray(out, dtype=float)

    if np.isscalar(values):
        return np.full(ntraces, float(values), dtype=float)

    arr = np.asarray(values, dtype=float)
    if arr.size != ntraces:
        raise ValueError(f"{name} must be scalar, mapping, or length {ntraces}; got length {arr.size}.")
    return arr.astype(float, copy=False)


def regular_receiver_x(ntraces: int, first_receiver_x_m: float, receiver_spacing_m: float) -> np.ndarray:
    """Return regular receiver x positions in trace order."""
    return float(first_receiver_x_m) + np.arange(ntraces, dtype=float) * float(receiver_spacing_m)


def regular_receiver_x_from_station_indices(
    station_indices: Sequence[int],
    first_receiver_x_m: float,
    receiver_spacing_m: float,
) -> np.ndarray:
    """Return regular receiver x positions from arbitrary station indices."""
    stations = np.asarray(station_indices, dtype=int)
    station0 = int(stations.min())
    return float(first_receiver_x_m) + (stations - station0).astype(float) * float(receiver_spacing_m)


def _scaled_int(value_m: float, scalar: int) -> int:
    """Convert metres to the SEG-Y integer representation for ``scalar``."""
    value_m = 0.0 if value_m is None else float(value_m)
    if scalar == 0:
        scalar = 1
    if scalar < 0:
        stored = value_m * abs(scalar)
    else:
        stored = value_m / scalar
    return int(round(stored))



def scaled_int(value_m: float, scalar: int) -> int:
    """Public compatibility alias for converting metres to SEG-Y integer units."""
    return _scaled_int(value_m, scalar)

def _apply_scalar(stored_value: float, scalar: int) -> float:
    """Convert a SEG-Y integer coordinate/elevation back to metres."""
    if scalar == 0:
        scalar = 1
    if scalar < 0:
        return float(stored_value) / abs(scalar)
    return float(stored_value) * scalar


def apply_scalar(stored_value: float, scalar: int | None) -> float:
    """Public wrapper for applying a SEG-Y coordinate/elevation scalar.

    SEG-Y stores coordinates and elevations as integers plus a scalar.  Positive
    scalars multiply the stored value; negative scalars divide by their absolute
    value.  A missing or zero scalar is treated as 1.
    """
    if scalar in (None, 0):
        scalar = 1
    return _apply_scalar(stored_value, int(scalar))


def header_value(header, names: Sequence[str], default=None):
    """Return the first available attribute from a SEG-Y trace header.

    ``names`` should contain ObsPy SEG-Y trace-header attribute names, ordered
    from preferred to fallback.  ``default`` is returned when no name exists.
    """
    if header is None:
        return default
    for name in names:
        if hasattr(header, name):
            return getattr(header, name)
    return default


def get_trace_header(tr: Trace):
    """Return ``tr.stats.segy.trace_header`` if present, otherwise ``None``."""
    try:
        return tr.stats.segy.trace_header
    except Exception:
        return None


def stats_float(tr: Trace, names: Sequence[str], default=None):
    """Return the first finite float value found on ``tr.stats``.

    This allows MiniSEED or synthetic streams to carry geometry through custom
    ObsPy stats fields, e.g. ``tr.stats.receiver_x_m`` or
    ``tr.stats.source_x_m``, even before SEG-Y headers are attached.
    """
    for name in names:
        value = None
        try:
            value = getattr(tr.stats, name)
        except Exception:
            pass
        if value is None:
            try:
                value = tr.stats[name]
            except Exception:
                pass
        if value is not None:
            try:
                value = float(value)
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return default


def make_trace_header(
    *,
    station_idx: int,
    receiver_x_m: float,
    source_x_m: float,
    shot_number: int,
    dt_s: float | None = None,
    npts: int | None = None,
    receiver_z_m: float = 0.0,
    source_z_m: float = 0.0,
    trace_number: int | None = None,
    receiver_y_m: float = 0.0,
    source_y_m: float = 0.0,
    source_depth_m: float = 0.0,
    coord_scalar: int = -100,
    elev_scalar: int = -100,
    coordinate_units: int = 1,
) -> SEGYTraceHeader:
    """Create a standard ObsPy SEG-Y trace header.

    Parameters are in metres.  By default coordinates/elevations are stored as
    integer centimetres and the corresponding SEG-Y scalar fields are ``-100``.

    Notes for topography
    --------------------
    ``receiver_z_m`` is written to ``receiver_group_elevation`` and
    ``source_z_m`` is written to ``surface_elevation_at_source``.  Positive
    values should mean positive elevation in your project coordinate system.
    """
    trace_number = int(trace_number if trace_number is not None else station_idx)
    sx = _scaled_int(source_x_m, coord_scalar)
    sy = _scaled_int(source_y_m, coord_scalar)
    gx = _scaled_int(receiver_x_m, coord_scalar)
    gy = _scaled_int(receiver_y_m, coord_scalar)
    selev = _scaled_int(source_z_m, elev_scalar)
    gelev = _scaled_int(receiver_z_m, elev_scalar)
    sdepth = _scaled_int(source_depth_m, elev_scalar)
    offset_m = float(receiver_x_m) - float(source_x_m)

    th = SEGYTraceHeader()
    th.trace_sequence_number_within_line = trace_number
    th.trace_sequence_number_within_segy_file = trace_number
    th.original_field_record_number = int(shot_number)
    th.trace_number_within_the_original_field_record = int(station_idx)
    th.energy_source_point_number = int(shot_number)
    th.ensemble_number = int(shot_number)
    th.trace_number_within_the_ensemble = int(station_idx)
    th.source_coordinate_x = sx
    th.source_coordinate_y = sy
    th.group_coordinate_x = gx
    th.group_coordinate_y = gy
    th.coordinate_units = int(coordinate_units)
    th.scalar_to_be_applied_to_all_coordinates = int(coord_scalar)
    th.receiver_group_elevation = gelev
    th.surface_elevation_at_source = selev
    th.source_depth_below_surface = sdepth
    th.scalar_to_be_applied_to_all_elevations_and_depths = int(elev_scalar)
    th.source_to_receiver_offset_in_m = int(round(offset_m))
    # A few readers prefer this older alias/name if present; ObsPy ignores
    # unknown fields during write, but keeps them available on the Python side.
    th.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = int(round(offset_m))
    if dt_s is not None:
        sample_interval_us = int(round(float(dt_s) * 1_000_000.0))
        th.sample_interval_in_ms_for_this_trace = sample_interval_us
    if npts is not None:
        th.number_of_samples_in_this_trace = int(npts)
    th.trace_identification_code = 1
    th.number_of_vertically_summed_traces_yielding_this_trace = 1
    th.number_of_horizontally_stacked_traces_yielding_this_trace = 1
    return th


def attach_trace_header(
    tr: Trace,
    *,
    station_idx: int,
    receiver_x_m: float,
    source_x_m: float,
    shot_number: int,
    receiver_z_m: float = 0.0,
    source_z_m: float = 0.0,
    trace_number: int | None = None,
    receiver_y_m: float = 0.0,
    source_y_m: float = 0.0,
    source_depth_m: float = 0.0,
    coord_scalar: int = -100,
    elev_scalar: int = -100,
) -> Trace:
    """Attach a project-standard SEG-Y trace header to one ObsPy Trace."""
    tr.stats.segy = getattr(tr.stats, "segy", AttribDict())
    tr.stats.segy.trace_header = make_trace_header(
        station_idx=int(station_idx),
        receiver_x_m=float(receiver_x_m),
        source_x_m=float(source_x_m),
        shot_number=int(shot_number),
        dt_s=float(tr.stats.delta),
        npts=int(tr.stats.npts),
        receiver_z_m=float(receiver_z_m),
        source_z_m=float(source_z_m),
        trace_number=trace_number,
        receiver_y_m=float(receiver_y_m),
        source_y_m=float(source_y_m),
        source_depth_m=float(source_depth_m),
        coord_scalar=int(coord_scalar),
        elev_scalar=int(elev_scalar),
    )
    return tr


def apply_geometry_to_stream(
    stream: Stream,
    *,
    receiver_x_m=None,
    source_x_m: float = 0.0,
    shot_number: int = 1,
    station_indices: Sequence[int] | None = None,
    receiver_z_m=None,
    source_z_m: float = 0.0,
    receiver_y_m=None,
    source_y_m: float = 0.0,
    source_depth_m: float = 0.0,
    first_receiver_x_m: float = 0.0,
    receiver_spacing_m: float = 1.0,
    network: str | None = None,
    channel: str | None = None,
    location: str | None = None,
    coord_scalar: int = -100,
    elev_scalar: int = -100,
) -> Stream:
    """Attach SEG-Y geometry headers to every trace in a stream.

    ``receiver_x_m``/``receiver_z_m``/``receiver_y_m`` may be scalars,
    trace-order sequences, or mappings keyed by station index/name.  When
    ``receiver_x_m`` is omitted, a regular array generated from
    ``first_receiver_x_m`` and ``receiver_spacing_m`` is used.
    """
    st = stream.copy()
    ntr = len(st)
    if station_indices is None:
        station_indices = np.arange(1, ntr + 1, dtype=int)
    else:
        station_indices = np.asarray(station_indices, dtype=int)
        if len(station_indices) != ntr:
            raise ValueError("station_indices length must match stream length")

    rx_fallback = regular_receiver_x(ntr, first_receiver_x_m, receiver_spacing_m)
    rx = values_for_traces(receiver_x_m, station_indices=station_indices, ntraces=ntr, fallback=rx_fallback, name="receiver_x_m")
    rz = values_for_traces(receiver_z_m, station_indices=station_indices, ntraces=ntr, fallback=0.0, name="receiver_z_m")
    ry = values_for_traces(receiver_y_m, station_indices=station_indices, ntraces=ntr, fallback=0.0, name="receiver_y_m")

    for i, (tr, sta) in enumerate(zip(st, station_indices), start=1):
        if network is not None:
            tr.stats.network = network
        if not getattr(tr.stats, "station", None):
            tr.stats.station = f"S{int(sta):04d}"
        if location is not None:
            tr.stats.location = location
        if channel is not None:
            tr.stats.channel = channel
        attach_trace_header(
            tr,
            station_idx=int(sta),
            receiver_x_m=float(rx[i - 1]),
            source_x_m=float(source_x_m),
            shot_number=int(shot_number),
            receiver_z_m=float(rz[i - 1]),
            source_z_m=float(source_z_m),
            trace_number=i,
            receiver_y_m=float(ry[i - 1]),
            source_y_m=float(source_y_m),
            source_depth_m=float(source_depth_m),
            coord_scalar=coord_scalar,
            elev_scalar=elev_scalar,
        )
    return st


def apply_trace_geometries(stream: Stream, geometries: Sequence[TraceGeometry], *, network: str | None = None, channel: str | None = None) -> Stream:
    """Attach one explicit ``TraceGeometry`` object per trace."""
    if len(stream) != len(geometries):
        raise ValueError("stream and geometries must have the same length")
    st = stream.copy()
    for tr, g in zip(st, geometries):
        if network is not None:
            tr.stats.network = network
        if channel is not None:
            tr.stats.channel = channel
        if not getattr(tr.stats, "station", None):
            tr.stats.station = f"S{int(g.receiver_id):04d}"
        attach_trace_header(
            tr,
            station_idx=int(g.receiver_id),
            receiver_x_m=float(g.receiver_x_m),
            source_x_m=float(g.source_x_m),
            shot_number=int(g.shot_id),
            receiver_z_m=float(g.receiver_z_m),
            source_z_m=float(g.source_z_m),
            trace_number=int(g.trace_id),
            receiver_y_m=float(g.receiver_y_m),
            source_y_m=float(g.source_y_m),
            source_depth_m=float(g.source_depth_m),
        )
    return st



def force_trace_timing_and_headers(
    stream: Stream,
    receiver_x_m,
    source_x_m: float,
    shot_number: int,
    dt_s: float,
    t0_s: float = 0.0,
    component: str = "Z",
    receiver_z_m=0.0,
    source_z_m: float = 0.0,
    network: str = "SY",
    coord_scalar: int = -100,
    elev_scalar: int = -100,
) -> Stream:
    """Compatibility helper: set timing/channel metadata and attach SEG-Y headers.

    This replaces the old ``segy_tools.headers.force_trace_timing_and_headers``.
    ``receiver_x_m`` and ``receiver_z_m`` may be scalar, sequence, or mapping.
    Elevations use the project convention of positive-up metres.
    """
    st = stream.copy()
    for tr in st:
        tr.stats.delta = float(dt_s)
        tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(t0_s)
        tr.stats.network = network
        tr.stats.location = f"{int(shot_number) % 100:02d}"
        tr.stats.channel = component_to_channel(component)
    return apply_geometry_to_stream(
        st,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        shot_number=shot_number,
        receiver_z_m=receiver_z_m,
        source_z_m=source_z_m,
        network=network,
        channel=component_to_channel(component),
        coord_scalar=coord_scalar,
        elev_scalar=elev_scalar,
    )

def geometry_from_trace_header(tr: Trace) -> TraceGeometry:
    """Extract ``TraceGeometry`` from a trace header when possible."""
    th = tr.stats.segy.trace_header
    coord_scalar = int(getattr(th, "scalar_to_be_applied_to_all_coordinates", 1) or 1)
    elev_scalar = int(getattr(th, "scalar_to_be_applied_to_all_elevations_and_depths", 1) or 1)
    shot_id = int(getattr(th, "original_field_record_number", getattr(th, "energy_source_point_number", 1)))
    receiver_id = int(getattr(th, "trace_number_within_the_original_field_record", 1))
    trace_id = int(getattr(th, "trace_sequence_number_within_line", receiver_id))
    return TraceGeometry(
        source_x_m=_apply_scalar(getattr(th, "source_coordinate_x", 0), coord_scalar),
        receiver_x_m=_apply_scalar(getattr(th, "group_coordinate_x", 0), coord_scalar),
        source_y_m=_apply_scalar(getattr(th, "source_coordinate_y", 0), coord_scalar),
        receiver_y_m=_apply_scalar(getattr(th, "group_coordinate_y", 0), coord_scalar),
        source_z_m=_apply_scalar(getattr(th, "surface_elevation_at_source", 0), elev_scalar),
        receiver_z_m=_apply_scalar(getattr(th, "receiver_group_elevation", 0), elev_scalar),
        source_depth_m=_apply_scalar(getattr(th, "source_depth_below_surface", 0), elev_scalar),
        shot_id=shot_id,
        receiver_id=receiver_id,
        trace_id=trace_id,
    )


def geometry_from_stream(stream: Stream) -> list[TraceGeometry]:
    """Extract trace geometry from all traces in a stream."""
    return [geometry_from_trace_header(tr) for tr in stream]


def extract_geometry_from_stream(
    st: Stream,
    fallback_receiver_spacing_m: Optional[float] = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: Optional[float] = None,
):
    """Extract source/receiver geometry arrays from an ObsPy ``Stream``.

    Priority order for receiver x-coordinate:

    1. custom stats fields such as ``receiver_x_m`` or ``x_m``;
    2. SEG-Y ``group_coordinate_x`` plus coordinate scalar;
    3. fallback regular spacing/index.

    Priority order for source x-coordinate:

    1. custom stats fields such as ``source_x_m`` or ``shot_x_m``;
    2. SEG-Y ``source_coordinate_x`` plus coordinate scalar;
    3. ``fallback_source_x_m``.

    Elevations use the project convention of positive-up metres.  Receiver
    elevation is read from ``receiver_group_elevation`` and source elevation
    from ``surface_elevation_at_source`` when SEG-Y headers are available.
    """
    receiver_x, receiver_z = [], []
    source_x, source_z = [], []
    offsets, shot_numbers, receiver_numbers = [], [], []

    for i, tr in enumerate(st):
        h = get_trace_header(tr)
        coord_scalar = header_value(h, ["scalar_to_be_applied_to_all_coordinates"], default=1)
        elev_scalar = header_value(h, ["scalar_to_be_applied_to_all_elevations_and_depths"], default=1)

        rx = stats_float(
            tr,
            ["receiver_x_m", "receiver_x", "x_m", "x", "distance_m", "offset_receiver_x_m"],
            default=None,
        )
        sx = stats_float(
            tr,
            ["source_x_m", "source_x", "shot_x_m", "shot_x"],
            default=None,
        )

        if rx is None:
            rx_raw = header_value(h, ["group_coordinate_x"], default=None)
            if rx_raw is None:
                rx = (
                    fallback_first_receiver_x_m + i * fallback_receiver_spacing_m
                    if fallback_receiver_spacing_m is not None
                    else float(i)
                )
            else:
                rx = apply_scalar(rx_raw, coord_scalar)

        if sx is None:
            sx_raw = header_value(h, ["source_coordinate_x"], default=None)
            sx = fallback_source_x_m if sx_raw is None else apply_scalar(sx_raw, coord_scalar)

        rz = stats_float(
            tr,
            ["receiver_z_m", "receiver_elevation_m", "elevation_m", "z_m"],
            default=None,
        )
        if rz is None:
            rz_raw = header_value(h, ["receiver_group_elevation"], default=None)
            rz = np.nan if rz_raw is None else apply_scalar(rz_raw, elev_scalar)

        sz = stats_float(
            tr,
            ["source_z_m", "source_elevation_m", "shot_elevation_m", "shot_z_m"],
            default=None,
        )
        if sz is None:
            sz_raw = header_value(h, ["surface_elevation_at_source"], default=None)
            sz = np.nan if sz_raw is None else apply_scalar(sz_raw, elev_scalar)

        off = stats_float(tr, ["offset_m", "source_receiver_offset_m"], default=None)
        if off is None:
            off_raw = header_value(
                h,
                [
                    "distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group",
                    "source_to_receiver_offset_in_m",
                ],
                default=None,
            )
            if off_raw is None:
                off = np.nan if sx is None else float(rx) - float(sx)
            else:
                off = apply_scalar(off_raw, coord_scalar)

        shot = header_value(
            h,
            ["original_field_record_number", "energy_source_point_number"],
            default=np.nan,
        )
        recno = header_value(
            h,
            ["trace_number_within_the_original_field_record", "trace_sequence_number_within_line"],
            default=i + 1,
        )

        receiver_x.append(float(rx))
        receiver_z.append(np.nan if rz is None else float(rz))
        source_x.append(np.nan if sx is None else float(sx))
        source_z.append(np.nan if sz is None else float(sz))
        offsets.append(off)
        shot_numbers.append(shot)
        receiver_numbers.append(recno)

    receiver_x = np.asarray(receiver_x, dtype=float)
    source_x_arr = np.asarray(source_x, dtype=float)
    finite_sx = source_x_arr[np.isfinite(source_x_arr)]
    source_x_m = float(np.median(finite_sx)) if len(finite_sx) else fallback_source_x_m

    source_z_arr = np.asarray(source_z, dtype=float)
    finite_sz = source_z_arr[np.isfinite(source_z_arr)]
    source_z_m = float(np.median(finite_sz)) if len(finite_sz) else None

    return {
        "receiver_x_m": receiver_x,
        "receiver_z_m": np.asarray(receiver_z, dtype=float),
        "source_x_m": source_x_m,
        "source_z_m": source_z_m,
        "source_x_m_per_trace": source_x_arr,
        "source_z_m_per_trace": source_z_arr,
        "offsets_m": np.asarray(offsets, dtype=float),
        "shot_numbers": np.asarray(shot_numbers),
        "receiver_numbers": np.asarray(receiver_numbers),
    }


# Backward-compatible alias used by older notebooks/scripts.
extract_geometry_from_segy_stream = extract_geometry_from_stream


def gather_arrays_to_stream(
    data: np.ndarray,
    dt_s: float,
    starttime=None,
    receiver_x_m: Optional[Sequence[float]] = None,
    source_x_m: float = 0.0,
    receiver_z_m: Optional[Sequence[float] | float] = None,
    source_z_m: float = 0.0,
    shot_number: int = 1,
    station_prefix: str = "R",
    network: str = "SY",
    component: str = "Z",
) -> Stream:
    """Convert a gather array to an ObsPy ``Stream`` with SEG-Y headers.

    Parameters
    ----------
    data
        Array shaped ``(n_traces, n_samples)``.
    dt_s
        Sample interval in seconds.
    receiver_x_m
        Receiver x positions in metres.  May be irregularly spaced.  If omitted,
        trace indices ``0..n_traces-1`` are used.
    receiver_z_m, source_z_m
        Positive-up elevations/topography in metres.

    The geometry is stored both as custom ObsPy ``stats`` fields and as standard
    SEG-Y trace headers via ``force_trace_timing_and_headers``.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be shaped (n_traces, n_samples).")

    n_traces, _ = data.shape
    if receiver_x_m is None:
        receiver_x_m = np.arange(n_traces, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)
    if receiver_x_m.size != n_traces:
        raise ValueError("receiver_x_m length must match the number of traces.")

    if receiver_z_m is None:
        receiver_z_values = np.zeros(n_traces, dtype=float)
    elif np.isscalar(receiver_z_m):
        receiver_z_values = np.full(n_traces, float(receiver_z_m), dtype=float)
    else:
        receiver_z_values = np.asarray(receiver_z_m, dtype=float)
        if receiver_z_values.size != n_traces:
            raise ValueError("receiver_z_m length must match the number of traces.")

    st = Stream()
    for i in range(n_traces):
        tr = Trace(data=np.asarray(data[i, :], dtype=np.float32))
        tr.stats.delta = float(dt_s)
        if starttime is not None:
            tr.stats.starttime = starttime
        tr.stats.network = network
        tr.stats.station = f"{station_prefix}{i + 1:04d}"
        tr.stats.channel = component
        tr.stats.receiver_x_m = float(receiver_x_m[i])
        tr.stats.receiver_z_m = float(receiver_z_values[i])
        tr.stats.source_x_m = float(source_x_m)
        tr.stats.source_z_m = float(source_z_m)
        st.append(tr)

    st = force_trace_timing_and_headers(
        stream=st,
        receiver_x_m=receiver_x_m,
        source_x_m=float(source_x_m),
        receiver_z_m=receiver_z_values,
        source_z_m=float(source_z_m),
        shot_number=int(shot_number),
        dt_s=float(dt_s),
        t0_s=0.0,
        component=component,
        network=network,
    )
    if starttime is not None:
        for tr in st:
            tr.stats.starttime = starttime
    return st


def read_segy_as_stream(path, *, unpack_trace_headers=True) -> Stream:
    """Read a SEG-Y file as an ObsPy Stream."""
    return read(str(path), format="SEGY", unpack_trace_headers=unpack_trace_headers)


def read_su_file(path, dt_s=None, t0_s=0.0, byteorder="<") -> Stream:
    """Read a Seismic Unix file as an ObsPy Stream."""
    st = read(str(path), format="SU", byteorder=byteorder)
    if dt_s is not None:
        for tr in st:
            tr.stats.delta = float(dt_s)
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(t0_s)
    return st


def read_segy_obspy(
    filename: str | Path,
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
) -> SeismicArrayData:
    """Read SEG-Y/SU data using ObsPy and return array data.

    This preserves the public interface from older helper code but standardizes
    the returned data to ``(n_traces, n_samples)``.
    """
    filename = Path(filename)
    st = read(str(filename), format=format) if format is not None else read(str(filename))
    if len(st) == 0:
        raise ValueError(f"No traces found in {filename}")
    npts = min(tr.stats.npts for tr in st)
    data = np.vstack([tr.data[:npts].astype(float) for tr in st])
    dt = float(st[0].stats.delta)
    fs = 1.0 / dt
    time = np.arange(npts) * dt
    if offsets is None:
        try:
            offsets = np.asarray([geometry_from_trace_header(tr).receiver_x_m - geometry_from_trace_header(tr).source_x_m for tr in st], dtype=float)
        except Exception:
            offsets = np.arange(data.shape[0]) * dx
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.size != data.shape[0]:
            raise ValueError("offsets length must match number of traces")
    return SeismicArrayData(data=data, fs=fs, dt=dt, time=time, offsets=np.asarray(offsets, dtype=float), source_file=str(filename))


def write_segy(stream: Stream, path, data_encoding=5, byteorder=">") -> Path:
    """Write an ObsPy Stream as SEG-Y."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="SEGY", data_encoding=data_encoding, byteorder=byteorder)
    return path


def write_stream_as_segy(
    stream: Stream,
    path,
    *,
    receiver_x_m=None,
    source_x_m: float = 0.0,
    shot_number: int = 1,
    station_indices: Sequence[int] | None = None,
    receiver_z_m=None,
    source_z_m: float = 0.0,
    first_receiver_x_m: float = 0.0,
    receiver_spacing_m: float = 1.0,
    network: str | None = None,
    channel: str | None = None,
    data_encoding=5,
    byteorder=">",
) -> Path:
    """Apply geometry headers, then write a stream as SEG-Y."""
    st = apply_geometry_to_stream(
        stream,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        shot_number=shot_number,
        station_indices=station_indices,
        receiver_z_m=receiver_z_m,
        source_z_m=source_z_m,
        first_receiver_x_m=first_receiver_x_m,
        receiver_spacing_m=receiver_spacing_m,
        network=network,
        channel=channel,
    )
    return write_segy(st, path, data_encoding=data_encoding, byteorder=byteorder)


def write_mseed(stream: Stream, path) -> Path:
    """Write an ObsPy Stream as MiniSEED."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="MSEED")
    return path



# -----------------------------------------------------------------------------
# Generic waveform/picker-app helpers
# -----------------------------------------------------------------------------
# These helpers were factored out of the standalone wiggle picker so all tools
# can read receiver-position-aware waveform files consistently.  They are kept
# deliberately generic: they do not depend on the GUI.

import re
from typing import Dict, Tuple
import pandas as pd


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

def guess_obspy_format(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
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
    """Read MiniSEED, SEG-Y, or SEG-2, with format guess plus fallback."""
    fmt = guess_obspy_format(path)
    if fmt is not None:
        try:
            return read(str(path), format=fmt)
        except Exception:
            # Fall through to ObsPy autodetection below. This helps with files
            # that have a misleading suffix.
            pass
    return read(str(path))



def apply_geometry_map(st: Stream, geom: Dict[int, Tuple[float, float]]) -> Stream:
    """Attach database geometry to trace stats before plotting."""
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
    """Return the most useful one-character component code for a trace."""
    chan = str(getattr(tr.stats, "channel", "") or "").strip()
    if chan:
        return chan[-1].upper()
    # Some ad-hoc exports may put component in location or station.
    loc = str(getattr(tr.stats, "location", "") or "").strip()
    if loc and loc[-1].upper() in {"Z", "N", "E", "1", "2", "3"}:
        return loc[-1].upper()
    return ""

def filter_stream_by_component(st: Stream, component: str) -> Stream:
    """Return only traces matching the selected component/channel suffix."""
    comp = (component or "All").strip().upper()
    if comp == "ALL":
        return st.copy()
    out = Stream()
    for tr in st:
        if _component_code(tr) == comp:
            out += tr.copy()
    return out

def stream_starttime_string(st: Stream) -> str:
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