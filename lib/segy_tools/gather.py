from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from obspy import Stream

from .io import read_segy_as_stream
from .plotting import plot_wiggle_gather, plot_image_gather


def _get_trace_header(tr):
    try:
        return tr.stats.segy.trace_header
    except Exception:
        return None


def _apply_scalar(value: float, scalar: int | None) -> float:
    if scalar in (None, 0):
        return float(value)
    scalar = int(scalar)
    if scalar > 0:
        return float(value) * scalar
    return float(value) / abs(scalar)


def _header_value(header, names: Sequence[str], default=None):
    if header is None:
        return default
    for name in names:
        if hasattr(header, name):
            return getattr(header, name)
    return default


def _stats_float(tr, names: Sequence[str], default=None):
    """Return the first finite float value found on ``tr.stats``.

    This deliberately checks custom ObsPy stats fields before SEG-Y headers, so
    MiniSEED streams can carry geometry attached by catalog/Excel metadata, e.g.

        tr.stats.receiver_x_m = 96.05
        tr.stats.source_x_m = 94.50

    ObsPy ``Stats`` is dict-like, so both attribute and key access are tried.
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


def _select_component(st: Stream, component: str | None = None) -> Stream:
    """Return a copied stream restricted to one component suffix, if requested."""
    if component in (None, "", "all", "ALL", "*"):
        return st.copy()
    component = str(component)
    return Stream([tr.copy() for tr in st if str(tr.stats.channel).endswith(component)])


def extract_geometry_from_segy_stream(
    st: Stream,
    fallback_receiver_spacing_m: Optional[float] = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: Optional[float] = None,
):
    """Extract source/receiver geometry from a Stream.

    Priority order for receiver x:
    1. ``tr.stats.receiver_x_m`` / related custom stats fields
    2. SEG-Y trace header ``group_coordinate_x``
    3. fallback spacing/index

    Priority order for source x:
    1. ``tr.stats.source_x_m`` / related custom stats fields
    2. SEG-Y trace header ``source_coordinate_x``
    3. ``fallback_source_x_m``

    This makes MiniSEED catalog workflows work correctly after attaching
    geometry from SQLite/Excel metadata.
    """
    receiver_x, source_x, offsets, shot_numbers, receiver_numbers = [], [], [], [], []

    for i, tr in enumerate(st):
        h = _get_trace_header(tr)
        scalar = _header_value(h, ["scalar_to_be_applied_to_all_coordinates"], default=1)

        rx = _stats_float(
            tr,
            ["receiver_x_m", "receiver_x", "x_m", "x", "distance_m", "offset_receiver_x_m"],
            default=None,
        )
        sx = _stats_float(
            tr,
            ["source_x_m", "source_x", "shot_x_m", "shot_x"],
            default=None,
        )

        if rx is None:
            rx_raw = _header_value(h, ["group_coordinate_x"], default=None)
            if rx_raw is None:
                rx = (
                    fallback_first_receiver_x_m + i * fallback_receiver_spacing_m
                    if fallback_receiver_spacing_m is not None
                    else float(i)
                )
            else:
                rx = _apply_scalar(rx_raw, scalar)

        if sx is None:
            sx_raw = _header_value(h, ["source_coordinate_x"], default=None)
            sx = fallback_source_x_m if sx_raw is None else _apply_scalar(sx_raw, scalar)

        off = _stats_float(tr, ["offset_m", "source_receiver_offset_m"], default=None)
        if off is None:
            off_raw = _header_value(
                h,
                ["distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group"],
                default=None,
            )
            if off_raw is None:
                off = np.nan if sx is None else float(rx) - float(sx)
            else:
                off = _apply_scalar(off_raw, scalar)

        shot = _header_value(
            h,
            ["original_field_record_number", "energy_source_point_number"],
            default=np.nan,
        )
        recno = _header_value(
            h,
            ["trace_number_within_the_original_field_record", "trace_sequence_number_within_line"],
            default=i + 1,
        )

        receiver_x.append(float(rx))
        source_x.append(np.nan if sx is None else float(sx))
        offsets.append(off)
        shot_numbers.append(shot)
        receiver_numbers.append(recno)

    receiver_x = np.asarray(receiver_x, dtype=float)
    source_x_arr = np.asarray(source_x, dtype=float)
    finite_sx = source_x_arr[np.isfinite(source_x_arr)]
    source_x_m = float(np.median(finite_sx)) if len(finite_sx) else fallback_source_x_m

    return {
        "receiver_x_m": receiver_x,
        "source_x_m": source_x_m,
        "offsets_m": np.asarray(offsets, dtype=float),
        "shot_numbers": np.asarray(shot_numbers),
        "receiver_numbers": np.asarray(receiver_numbers),
    }


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

    Parameters
    ----------
    component
        Optional component suffix, e.g. ``"Z"``. Use this for 3C MiniSEED
        gathers so E/N/Z components are not plotted as separate receivers.
    """
    st_work = _select_component(st, component=component)

    if len(st_work) == 0:
        raise ValueError("Empty Stream after component selection.")

    npts = min(tr.stats.npts for tr in st_work)
    dt = float(st_work[0].stats.delta)
    time = np.arange(npts, dtype=float) * dt
    data = np.vstack([tr.data[:npts].astype(float) for tr in st_work])

    geom = extract_geometry_from_segy_stream(
        st_work,
        fallback_receiver_spacing_m,
        fallback_first_receiver_x_m,
        fallback_source_x_m,
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


def plot_wiggle_gather_from_stream(
    st: Stream,
    *,
    sort_by="receiver_x",
    component: str | None = None,
    fallback_receiver_spacing_m=None,
    fallback_first_receiver_x_m=0.0,
    fallback_source_x_m=None,
    title=None,
    tmin=None,
    tmax=None,
    omin=None,
    omax=None,
    scale=0.8,
    clip_percentile=99,
    normalize=True,
    cave=None,
    outfile=None,
    dpi=160,
    **style_kwargs,
):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by=sort_by,
        component=component,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )
    if title is None:
        comp_txt = "" if component in (None, "", "all", "ALL", "*") else f" ({component})"
        title = f"Shot gather{comp_txt}"
    fig = plot_wiggle_gather(
        time,
        data,
        receiver_x_m,
        source_x_m=source_x_m,
        title=title,
        tmin=tmin,
        tmax=tmax,
        omin=omin,
        omax=omax,
        scale=scale,
        clip_percentile=clip_percentile,
        normalize=normalize,
        cave=cave,
        outfile=outfile,
        dpi=dpi,
        **style_kwargs,
    )
    return fig


def plot_image_gather_from_stream(
    st: Stream,
    *,
    sort_by="receiver_x",
    component: str | None = None,
    fallback_receiver_spacing_m=None,
    fallback_first_receiver_x_m=0.0,
    fallback_source_x_m=None,
    title=None,
    tmin=None,
    tmax=None,
    omin=None,
    omax=None,
    clip_percentile=98,
    cave=None,
    outfile=None,
    dpi=160,
):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by=sort_by,
        component=component,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )
    if title is None:
        comp_txt = "" if component in (None, "", "all", "ALL", "*") else f" ({component})"
        title = f"Image gather{comp_txt}"
    return plot_image_gather(
        time,
        data,
        receiver_x_m,
        source_x_m=source_x_m,
        title=title,
        tmin=tmin,
        tmax=tmax,
        omin=omin,
        omax=omax,
        clip_percentile=clip_percentile,
        cave=cave,
        outfile=outfile,
        dpi=dpi,
    )


def plot_shot_gather_from_stream(st: Stream, *, kind="both", outfile_prefix=None, save_numpy=False, **kwargs):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by=kwargs.get("sort_by", "receiver_x"),
        component=kwargs.get("component"),
        fallback_receiver_spacing_m=kwargs.get("fallback_receiver_spacing_m"),
        fallback_first_receiver_x_m=kwargs.get("fallback_first_receiver_x_m", 0.0),
        fallback_source_x_m=kwargs.get("fallback_source_x_m"),
    )
    result = {"time": time, "data": data, "receiver_x_m": receiver_x_m, "source_x_m": source_x_m, "geometry": geom, "figures": {}}
    prefix = Path(outfile_prefix) if outfile_prefix is not None else None
    common = dict(tmin=kwargs.get("tmin"), tmax=kwargs.get("tmax"), omin=kwargs.get("omin"), omax=kwargs.get("omax"), cave=kwargs.get("cave"), dpi=kwargs.get("dpi", 160))
    if kind in ("wiggle", "both"):
        result["figures"]["wiggle"] = plot_wiggle_gather(time, data, receiver_x_m, source_x_m=source_x_m, title=kwargs.get("title", "Shot gather wiggle"), scale=kwargs.get("scale", 0.8), clip_percentile=kwargs.get("clip_percentile", 99), normalize=kwargs.get("normalize", True), outfile=None if prefix is None else f"{prefix}_wiggle.png", **common)
    if kind in ("image", "both"):
        result["figures"]["image"] = plot_image_gather(time, data, receiver_x_m, source_x_m=source_x_m, title=kwargs.get("title", "Shot gather image"), clip_percentile=kwargs.get("clip_percentile", 98), outfile=None if prefix is None else f"{prefix}_image.png", **common)
    if save_numpy and prefix is not None:
        np.save(f"{prefix}.npy", data); result["numpy_path"] = Path(f"{prefix}.npy")
    return result


def plot_wiggle_gather_from_segy(segy_path, **kwargs):
    st = read_segy_as_stream(segy_path)
    return plot_wiggle_gather_from_stream(st, **kwargs)


def plot_image_gather_from_segy(segy_path, **kwargs):
    st = read_segy_as_stream(segy_path)
    return plot_image_gather_from_stream(st, **kwargs)


def plot_shot_gather_from_segy(segy_path, **kwargs):
    st = read_segy_as_stream(segy_path)
    return plot_shot_gather_from_stream(st, **kwargs)


# -----------------------------------------------------------------------------
# Generic gather array/stream utilities migrated from seismic_gather_utils.
# -----------------------------------------------------------------------------

from obspy import Trace
from .headers import force_trace_timing_and_headers
from .io import write_segy


def gather_arrays_to_stream(
    data: np.ndarray,
    dt_s: float,
    starttime=None,
    receiver_x_m: Optional[Sequence[float]] = None,
    source_x_m: float = 0.0,
    shot_number: int = 1,
    station_prefix: str = "R",
    network: str = "SY",
    component: str = "Z",
) -> Stream:
    """Convert a gather array into an ObsPy ``Stream`` with SEG-Y headers.

    ``receiver_x_m`` may be irregularly spaced. It is written both as custom
    ObsPy stats fields and SEG-Y coordinate headers by ``force_trace_timing...``.
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
        tr.stats.source_x_m = float(source_x_m)
        st.append(tr)

    st = force_trace_timing_and_headers(
        stream=st,
        receiver_x_m=receiver_x_m,
        source_x_m=float(source_x_m),
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
    """Align two gathers on common receiver positions.

    This is intended for model-vs-real comparisons when receiver spacing may be
    irregular or when the two gathers do not have identical trace order.
    """
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

    # For now require compatible time samples; resampling can be added later if needed.
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
    """Read two SEG-Y gathers and compute A - B after geometry alignment.

    Unlike the older implementation, this does not assume the same trace order
    or a fixed receiver spacing. It aligns traces by nearest receiver x.
    """
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
