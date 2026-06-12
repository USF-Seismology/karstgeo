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


def extract_geometry_from_segy_stream(st: Stream, fallback_receiver_spacing_m: Optional[float] = None, fallback_first_receiver_x_m: float = 0.0, fallback_source_x_m: Optional[float] = None):
    receiver_x, source_x, offsets, shot_numbers, receiver_numbers = [], [], [], [], []
    for i, tr in enumerate(st):
        h = _get_trace_header(tr)
        scalar = _header_value(h, ["scalar_to_be_applied_to_all_coordinates"], default=1)
        rx_raw = _header_value(h, ["group_coordinate_x"], default=None)
        sx_raw = _header_value(h, ["source_coordinate_x"], default=None)
        if rx_raw is None:
            rx = fallback_first_receiver_x_m + i * fallback_receiver_spacing_m if fallback_receiver_spacing_m is not None else float(i)
        else:
            rx = _apply_scalar(rx_raw, scalar)
        sx = fallback_source_x_m if sx_raw is None else _apply_scalar(sx_raw, scalar)
        off_raw = _header_value(h, ["distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group"], default=None)
        off = (np.nan if sx is None else rx - sx) if off_raw is None else _apply_scalar(off_raw, scalar)
        shot = _header_value(h, ["original_field_record_number", "energy_source_point_number"], default=np.nan)
        recno = _header_value(h, ["trace_number_within_the_original_field_record", "trace_sequence_number_within_line"], default=i + 1)
        receiver_x.append(rx); source_x.append(np.nan if sx is None else sx); offsets.append(off); shot_numbers.append(shot); receiver_numbers.append(recno)
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


def stream_to_gather_arrays(st: Stream, *, sort_by="receiver_x", fallback_receiver_spacing_m: Optional[float] = None, fallback_first_receiver_x_m: float = 0.0, fallback_source_x_m: Optional[float] = None):
    if len(st) == 0:
        raise ValueError("Empty Stream.")
    npts = min(tr.stats.npts for tr in st)
    dt = float(st[0].stats.delta)
    time = np.arange(npts, dtype=float) * dt
    data = np.vstack([tr.data[:npts].astype(float) for tr in st])
    geom = extract_geometry_from_segy_stream(st, fallback_receiver_spacing_m, fallback_first_receiver_x_m, fallback_source_x_m)
    receiver_x_m = np.asarray(geom["receiver_x_m"], dtype=float)
    source_x_m = geom["source_x_m"]
    if sort_by == "receiver_x":
        order = np.argsort(receiver_x_m)
    elif sort_by == "offset":
        order = np.argsort(np.asarray(geom["offsets_m"], dtype=float))
    elif sort_by == "trace":
        order = np.argsort(np.asarray(geom["receiver_numbers"], dtype=float))
    elif sort_by in ("none", None):
        order = np.arange(len(st))
    else:
        raise ValueError("sort_by must be one of 'receiver_x', 'offset', 'trace', or 'none'.")
    data = data[order]; receiver_x_m = receiver_x_m[order]
    geom = {k: v[order] if isinstance(v, np.ndarray) and len(v) == len(order) else v for k, v in geom.items()}
    return time, data, receiver_x_m, source_x_m, geom


def plot_wiggle_gather_from_stream(st: Stream, *, sort_by="receiver_x", fallback_receiver_spacing_m=None, fallback_first_receiver_x_m=0.0, fallback_source_x_m=None, title=None, tmin=None, tmax=None, omin=None, omax=None, scale=0.8, clip_percentile=99, normalize=True, cave=None, outfile=None, dpi=160, **style_kwargs):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(st, sort_by=sort_by, fallback_receiver_spacing_m=fallback_receiver_spacing_m, fallback_first_receiver_x_m=fallback_first_receiver_x_m, fallback_source_x_m=fallback_source_x_m)
    if title is None:
        title = "SEG-Y shot gather"
    fig = plot_wiggle_gather(time, data, receiver_x_m, source_x_m=source_x_m, title=title, tmin=tmin, tmax=tmax, omin=omin, omax=omax, scale=scale, clip_percentile=clip_percentile, normalize=normalize, cave=cave, outfile=outfile, dpi=dpi, **style_kwargs)
    return fig


def plot_image_gather_from_stream(st: Stream, *, sort_by="receiver_x", fallback_receiver_spacing_m=None, fallback_first_receiver_x_m=0.0, fallback_source_x_m=None, title=None, tmin=None, tmax=None, omin=None, omax=None, clip_percentile=98, cave=None, outfile=None, dpi=160):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(st, sort_by=sort_by, fallback_receiver_spacing_m=fallback_receiver_spacing_m, fallback_first_receiver_x_m=fallback_first_receiver_x_m, fallback_source_x_m=fallback_source_x_m)
    if title is None:
        title = "SEG-Y image gather"
    return plot_image_gather(time, data, receiver_x_m, source_x_m=source_x_m, title=title, tmin=tmin, tmax=tmax, omin=omin, omax=omax, clip_percentile=clip_percentile, cave=cave, outfile=outfile, dpi=dpi)


def plot_shot_gather_from_stream(st: Stream, *, kind="both", outfile_prefix=None, save_numpy=False, **kwargs):
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(st, sort_by=kwargs.get("sort_by", "receiver_x"), fallback_receiver_spacing_m=kwargs.get("fallback_receiver_spacing_m"), fallback_first_receiver_x_m=kwargs.get("fallback_first_receiver_x_m", 0.0), fallback_source_x_m=kwargs.get("fallback_source_x_m"))
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

    Parameters
    ----------
    data
        Gather array using the package convention ``(n_traces, n_samples)``.
    dt_s
        Sample interval in seconds.
    starttime
        Optional ObsPy ``UTCDateTime``. If omitted, a deterministic epoch time
        is assigned by the SEG-Y header helper.
    receiver_x_m
        Receiver coordinates in metres. If omitted, receivers are numbered from
        0 to ``n_traces - 1`` in metres.
    source_x_m
        Source coordinate in metres, written to SEG-Y trace headers.
    shot_number
        Field record / shot number written to SEG-Y headers.
    station_prefix
        Prefix for generated ObsPy station codes.
    network
        ObsPy network code.
    component
        Component code, usually ``"Z"`` for vertical-component active-source
        gathers.

    Returns
    -------
    obspy.Stream
        Stream containing one trace per row of ``data``.
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


def difference_segy_gathers(
    segy_a: str | Path,
    segy_b: str | Path,
    *,
    fallback_receiver_spacing_m: float = 1.0,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: float = 0.0,
    sort_by: str = "receiver_x",
    output_segy_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read two SEG-Y gathers and compute a trace-by-trace difference.

    The two gathers are trimmed to their common number of traces and samples
    before differencing. The returned difference is ``A - B``.

    Parameters
    ----------
    segy_a, segy_b
        Input SEG-Y gather paths.
    fallback_receiver_spacing_m, fallback_first_receiver_x_m, fallback_source_x_m
        Fallback geometry used when SEG-Y trace headers do not contain usable
        source/receiver coordinates.
    sort_by
        Sorting mode passed to :func:`stream_to_gather_arrays`.
    output_segy_path
        Optional path for writing the difference gather as SEG-Y.

    Returns
    -------
    time_s, data_a, data_b, diff, receiver_x_m
        ``data_a``, ``data_b`` and ``diff`` are shaped
        ``(n_traces, n_samples)``.
    """
    st_a = read_segy_as_stream(segy_a)
    st_b = read_segy_as_stream(segy_b)

    time_a, data_a, rx_a, sx_a, _ = stream_to_gather_arrays(
        st_a,
        sort_by=sort_by,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )
    _, data_b, _, _, _ = stream_to_gather_arrays(
        st_b,
        sort_by=sort_by,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )

    ntr = min(data_a.shape[0], data_b.shape[0])
    npts = min(data_a.shape[1], data_b.shape[1])
    time_s = time_a[:npts]
    receiver_x_m = rx_a[:ntr]
    data_a = data_a[:ntr, :npts]
    data_b = data_b[:ntr, :npts]
    diff = data_a - data_b

    if output_segy_path is not None:
        diff_stream = gather_arrays_to_stream(
            diff,
            dt_s=float(st_a[0].stats.delta),
            starttime=st_a[0].stats.starttime,
            receiver_x_m=receiver_x_m,
            source_x_m=float(sx_a) if sx_a is not None and np.isfinite(sx_a) else fallback_source_x_m,
            shot_number=1,
            station_prefix="D",
        )
        write_segy(diff_stream, output_segy_path, data_encoding=1)

    return time_s, data_a, data_b, diff, receiver_x_m
