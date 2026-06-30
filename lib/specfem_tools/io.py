"""Utilities for discovering, reading, and exporting SPECFEM2D output gathers.

This module is intentionally focused on SPECFEM2D-specific output handling:

* discovering ``OUTPUT_FILES`` directories;
* loading SPECFEM2D SU files or SEM ASCII files;
* converting SPECFEM gather results to standardized ObsPy streams;
* exporting intermediate SEG-Y products for downstream QC and comparison.

General SEG-Y/SU/gather-processing utilities are kept out of this module where
possible. In this project, those more general functions should eventually live in
``segy_tools`` or a similar shared package.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict

try:
    from segy_tools.io import read_su_file, read_segy_as_stream, write_segy
    from lib.segy_tools.headers_shims import make_trace_header
    from segy_tools.gather import difference_segy_gathers, plot_wiggle_gather_from_stream
    from segy_tools.plotting import plot_difference_gathers
except Exception:  # pragma: no cover - allows import before local package is on path
    read_su_file = None
    read_segy_as_stream = None
    write_segy = None
    make_trace_header = None
    difference_segy_gathers = None
    plot_wiggle_gather_from_stream = None
    plot_difference_gathers = None

import re

@dataclass
class Timing:
    dt_s: Optional[float] = None
    t0_s: Optional[float] = None
    starttime_iso: str = "1970-01-01T00:00:00"


@dataclass(frozen=True)
class Geometry:
    """Simple 2-D survey geometry helper.

    This is retained as a convenience for regularly spaced legacy synthetic
    gathers, but the preferred path for RefraPick/field-style exports is to
    pass explicit ``receiver_x_m`` and, optionally, ``receiver_z_m`` arrays or
    station-index dictionaries.
    """

    first_receiver_x_m: float = 0.0
    receiver_spacing_m: float = 1.0
    receiver_z_m: float = 0.0
    first_shot_x_m: float = 0.0
    shot_spacing_m: float = 1.0
    source_z_m: float = 0.0


def receiver_x_from_station(station_idx: int, geom: Geometry) -> float:
    """Return receiver x position for a regular station-index geometry."""
    return geom.first_receiver_x_m + (int(station_idx) - 1) * geom.receiver_spacing_m


def source_x_from_shot(shot_number: int, geom: Geometry) -> float:
    """Return source x position for a regular shot-index geometry."""
    return geom.first_shot_x_m + (int(shot_number) - 1) * geom.shot_spacing_m


def _values_for_traces(
    values,
    *,
    station_indices: Sequence[int] | None,
    ntraces: int,
    fallback,
    name: str,
) -> np.ndarray:
    """Resolve scalar/array/mapping geometry values to one value per trace.

    ``values`` may be one of:

    * ``None``: use ``fallback``;
    * scalar: use the same value for every trace;
    * sequence with length ``ntraces``: assumed already in trace order;
    * mapping keyed by integer station index or station name like ``S0001``.

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


def _regular_receiver_x(ntraces: int, first_receiver_x_m: float, receiver_spacing_m: float) -> np.ndarray:
    return float(first_receiver_x_m) + np.arange(ntraces, dtype=float) * float(receiver_spacing_m)


def _regular_receiver_x_from_station_indices(
    station_indices: Sequence[int],
    first_receiver_x_m: float,
    receiver_spacing_m: float,
) -> np.ndarray:
    stations = np.asarray(station_indices, dtype=int)
    station0 = int(stations.min())
    return float(first_receiver_x_m) + (stations - station0).astype(float) * float(receiver_spacing_m)


def _attach_standard_trace_header(
    tr: Trace,
    *,
    station_idx: int,
    receiver_x_m: float,
    source_x_m: float,
    shot_number: int,
    receiver_z_m: float = 0.0,
    source_z_m: float = 0.0,
) -> None:
    """Attach a project-standard SEG-Y trace header to one ObsPy Trace."""
    if make_trace_header is None:
        raise ImportError("segy_tools.headers.make_trace_header is not available.")
    tr.stats.segy = AttribDict()
    tr.stats.segy.trace_header = make_trace_header(
        station_idx=int(station_idx),
        receiver_x_m=float(receiver_x_m),
        source_x_m=float(source_x_m),
        shot_number=int(shot_number),
        dt_s=float(tr.stats.delta),
        npts=int(tr.stats.npts),
        receiver_z_m=float(receiver_z_m),
        source_z_m=float(source_z_m),
    )


def component_to_channel(component: str) -> str:
    c = component.upper()
    if c in ("X", "BXX"):
        return "BXX"
    if c in ("Z", "BXZ"):
        return "BXZ"
    if c in ("Y", "BXY"):
        return "BXY"
    return c


def parse_sem_filename(path: Path) -> Optional[dict]:
    parts = Path(path).name.split(".")
    if len(parts) < 3:
        return None
    network, station, channel = parts[:3]
    extension = parts[3] if len(parts) > 3 else ""
    m = re.match(r"S(?P<num>\d+)$", station)
    if not m:
        return None
    return {
        "network": network,
        "station": station,
        "station_index": int(m.group("num")),
        "channel": channel,
        "extension": extension,
    }


def discover_sem_files(input_dir: Path, component: str = "Z", extension: str = "semv") -> list[Path]:
    input_dir = Path(input_dir).expanduser()
    channel = component_to_channel(component)
    patterns = []
    if extension:
        patterns.append(f"*.{channel}.{extension}")
    else:
        patterns.append(f"*.{channel}")
    patterns.extend([f"*.{channel}.semv", f"*.{channel}.semd", f"*.{channel}.sema", f"*.{channel}"])

    files, seen = [], set()
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                info = parse_sem_filename(path)
                if info and info["channel"] == channel:
                    files.append(path)
                    seen.add(path)

    def station_key(path):
        info = parse_sem_filename(path)
        return info["station_index"] if info else 10**12

    return sorted(files, key=station_key)


def read_sem_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return np.arange(len(arr), dtype=float), arr.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Cannot parse {path}; expected one or two columns.")
    return arr[:, 0].astype(float), arr[:, 1].astype(np.float32)


def read_sem_gather(input_dir: Path, component: str = "Z", extension: str = "semv", timing: Timing | None = None, verbose: bool = True):
    timing = timing or Timing()
    files = discover_sem_files(input_dir, component=component, extension=extension)
    if not files:
        raise FileNotFoundError(f"No SPECFEM ASCII files found in {input_dir} for component={component}, extension={extension}")
    if verbose:
        print(f"Found {len(files)} files in {input_dir}")

    records, station_indices, time = [], [], None
    for i, path in enumerate(files, start=1):
        info = parse_sem_filename(path)
        if info is None:
            continue
        t_file, y = read_sem_ascii(path)
        if time is None:
            if timing.dt_s is not None:
                t0 = timing.t0_s if timing.t0_s is not None else float(t_file[0])
                time = float(t0) + np.arange(len(y), dtype=float) * float(timing.dt_s)
            else:
                time = t_file
        elif len(y) != len(time):
            raise ValueError(f"Trace length mismatch in {path}: {len(y)} samples vs expected {len(time)}")
        records.append(y)
        station_indices.append(info["station_index"])
        if verbose and (i == 1 or i % 50 == 0 or i == len(files)):
            print(f"  read {i:5d}/{len(files):5d}: {path.name}")

    order = np.argsort(station_indices)
    return np.asarray(time), np.asarray(records, dtype=np.float32)[order], np.asarray(station_indices, dtype=int)[order]


def sem_gather_to_stream(time, data, station_indices, component="Z", network="SY") -> Stream:
    """Make a simple Stream from SEM gather arrays. Headers/geometries are applied downstream."""
    st = Stream()
    if len(time) < 2:
        dt = 1.0
    else:
        dt = float(np.median(np.diff(time)))
    t0 = float(time[0]) if len(time) else 0.0
    for sta, y in zip(station_indices, data):
        tr = Trace(data=np.asarray(y, dtype=np.float32))
        tr.stats.network = network
        tr.stats.station = f"S{int(sta):04d}"
        tr.stats.channel = component_to_channel(component)
        tr.stats.delta = dt
        st.append(tr)
    return st

@dataclass(frozen=True)
class SpecfemExportConfig:
    """Configuration for exporting SPECFEM2D model products.

    Parameters
    ----------
    segy_out_dir
        Directory where converted SEG-Y files are written.
    fig_dir
        Directory where wiggle plots are written.
    diff_fig_dir
        Directory where model-difference figures are written.
    receiver_spacing_m
        Fallback receiver spacing used for synthetic geometry.
    first_receiver_x_m
        Fallback coordinate of the first receiver.
    source_x_m
        Source coordinate written to headers and used for plotting.
    network
        Network code assigned to exported synthetic streams.
    """

    segy_out_dir: Path
    fig_dir: Path
    diff_fig_dir: Path
    receiver_spacing_m: float = 1.0
    first_receiver_x_m: float = 0.0
    source_x_m: float = 0.0
    receiver_x_m: object = None
    receiver_z_m: object = None
    source_z_m: float = 0.0
    network: str = "SY"

    def ensure_directories(self) -> None:
        """Create output directories if they do not already exist."""
        for path in (self.segy_out_dir, self.fig_dir, self.diff_fig_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


def find_specfem_model_outputs(root: str | Path, pattern: str = "[A-Z]/OUTPUT_FILES") -> list[Path]:
    """Find SPECFEM2D model ``OUTPUT_FILES`` directories.

    Parameters
    ----------
    root
        Directory containing model subdirectories, e.g. ``A/OUTPUT_FILES`` or
        ``Mod12/OUTPUT_FILES`` depending on ``pattern``.
    pattern
        Glob pattern relative to ``root``. The default matches single-letter
        model names used in the karst forward-model experiments.

    Returns
    -------
    list[pathlib.Path]
        Sorted matching ``OUTPUT_FILES`` directories.
    """
    root = Path(root).expanduser()
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def model_number_from_name(model_name: str) -> Optional[int]:
    """Return a stable integer code for a model name.

    Single-letter models map to their ASCII code (``A`` -> 65), preserving the
    ordering used in earlier notebooks. Numeric model suffixes such as ``Mod12``
    map to their integer value where possible.
    """
    text = str(model_name).strip()
    if not text:
        return None

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return ord(text[0].upper())


def model_name_from_output_dir(output_dir: str | Path) -> str:
    """Return the model directory name from a SPECFEM ``OUTPUT_FILES`` path."""
    return Path(output_dir).expanduser().parent.name


def discover_su_files(output_dir: str | Path, pattern: str = "*.su") -> list[Path]:
    """Find candidate Seismic Unix files in a SPECFEM ``OUTPUT_FILES`` directory."""
    return sorted(Path(output_dir).expanduser().glob(pattern))


def component_from_su_filename(path: str | Path, default: str = "BXZ") -> str:
    """Infer a SPECFEM-style component code from a common SPECFEM SU filename.

    Examples
    --------
    ``Ux_file_single_v.su`` -> ``BXX``
    ``Uz_file_single_v.su`` -> ``BXZ``
    """
    name = Path(path).name.lower()
    if name.startswith("ux"):
        return "BXX"
    if name.startswith("uz"):
        return "BXZ"
    return default.upper()


def read_su_try_both_byteorders(path: str | Path, dt_s: float | None = None, t0_s: float = 0.0) -> tuple[Stream, str]:
    """Read an SU file, trying little-endian and then big-endian byte order.

    Parameters
    ----------
    path
        SU file path.
    dt_s
        Optional sample interval override passed to ``segy_tools.io.read_su_file``.
    t0_s
        Optional start-time offset passed to ``read_su_file``.

    Returns
    -------
    stream, byteorder
        ObsPy stream and the byte order that succeeded.
    """
    if read_su_file is None:
        raise ImportError("segy_tools.io.read_su_file is not available.")

    last_error = None
    for byteorder in ("<", ">"):
        try:
            return read_su_file(path, dt_s=dt_s, t0_s=t0_s, byteorder=byteorder), byteorder
        except Exception as exc:  # keep trying
            last_error = exc
    raise last_error


def load_specfem_gather(
    output_dir: str | Path,
    component: str = "BXZ",
    extension: str = "semv",
    timing=None,
    prefer_su: bool = True,
    verbose: bool = True,
) -> dict:
    """Load one SPECFEM2D gather from an ``OUTPUT_FILES`` directory.

    The loader first tries binary SU files when ``prefer_su=True`` and falls
    back to SPECFEM ASCII files read by ``specfem_tools.io.read_sem_gather``.

    Parameters
    ----------
    output_dir
        SPECFEM ``OUTPUT_FILES`` directory.
    component
        Component to load when falling back to ASCII, e.g. ``BXZ`` or ``BXX``.
    extension
        SPECFEM ASCII extension, usually ``semv`` or ``semd``.
    timing
        Optional timing metadata passed through to ``read_sem_gather``.
    prefer_su
        If True, try ``*.su`` files before ASCII.
    verbose
        Print loading diagnostics.

    Returns
    -------
    dict
        Result dictionary. ``mode`` is either ``su`` or ``sem``.
    """
    output_dir = Path(output_dir).expanduser()
    component = component.upper()

    if prefer_su:
        for sufile in discover_su_files(output_dir):
            inferred_component = component_from_su_filename(sufile, default=component)
            try:
                st, byteorder = read_su_try_both_byteorders(sufile)
                if verbose:
                    print(f"Loaded SU: {sufile.name} byteorder={byteorder}")
                return {
                    "mode": "su",
                    "stream": st,
                    "path": sufile,
                    "component": inferred_component,
                    "byteorder": byteorder,
                }
            except Exception as exc:
                if verbose:
                    print(f"  SU failed: {sufile.name}: {type(exc).__name__}: {exc}")

    if read_sem_gather is None:
        raise ImportError("specfem_tools.io.read_sem_gather is not available for ASCII fallback.")

    if verbose:
        print(f"Falling back to SPECFEM ASCII: component={component}, extension={extension}")

    time_s, data, station_indices = read_sem_gather(
        output_dir,
        component=component,
        extension=extension,
        timing=timing,
        verbose=verbose,
    )
    return {
        "mode": "sem",
        "time": time_s,
        "data": data,
        "station_indices": station_indices,
        "component": component,
        "extension": extension,
    }


def specfem_gather_result_to_stream(
    result: dict,
    component: str = "BXZ",
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    source_x_m: float = 0.0,
    shot_number: int = 1,
    network: str = "SY",
    receiver_x_m=None,
    receiver_z_m=None,
    source_z_m: float = 0.0,
) -> Stream:
    """Convert a loaded SPECFEM gather result to a standardized ObsPy stream.

    This function supports both the older regular synthetic geometry and the
    newer irregular-geometry workflow needed for RefraPick/field-style SEG-Y.

    Parameters
    ----------
    result
        Dictionary returned by ``load_specfem_gather``.
    component
        Channel/component code to assign to traces.
    receiver_spacing_m, first_receiver_x_m, source_x_m
        Legacy regular synthetic geometry used when ``receiver_x_m`` is not
        supplied.
    shot_number
        Field-record number assigned in SEG-Y trace headers.
    network
        ObsPy network code.
    receiver_x_m
        Optional explicit receiver x positions. May be a scalar, a sequence in
        trace order, or a mapping keyed by station index/name. Supplying this is
        the preferred route for irregular receiver spacing.
    receiver_z_m
        Optional receiver elevations/depths in metres. May be scalar, sequence,
        or mapping. These are written through to the SEG-Y trace header using
        ``make_trace_header(..., receiver_z_m=...)``.
    source_z_m
        Optional source elevation/depth in metres.

    Returns
    -------
    obspy.Stream
        Stream with basic station/channel metadata and SEG-Y trace headers.
    """
    if make_trace_header is None:
        raise ImportError("segy_tools.headers.make_trace_header is not available.")

    component = component.upper()
    mode = result.get("mode")

    if mode == "su":
        st = result["stream"].copy()
        ntr = len(st)
        rx_fallback = _regular_receiver_x(ntr, first_receiver_x_m, receiver_spacing_m)
        rx = _values_for_traces(
            receiver_x_m,
            station_indices=None,
            ntraces=ntr,
            fallback=rx_fallback,
            name="receiver_x_m",
        )
        rz = _values_for_traces(
            receiver_z_m,
            station_indices=None,
            ntraces=ntr,
            fallback=0.0,
            name="receiver_z_m",
        )
        for i, tr in enumerate(st, start=1):
            tr.stats.network = network
            tr.stats.station = f"S{i:04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = result.get("component", component).upper()
            _attach_standard_trace_header(
                tr,
                station_idx=i,
                receiver_x_m=float(rx[i - 1]),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                receiver_z_m=float(rz[i - 1]),
                source_z_m=float(source_z_m),
            )
        return st

    if mode == "sem":
        time_s = np.asarray(result["time"], dtype=float)
        data = np.asarray(result["data"], dtype=np.float32)
        station_indices = np.asarray(result["station_indices"], dtype=int)
        if time_s.size < 2:
            raise ValueError("SEM gather time vector must contain at least two samples.")
        dt = float(np.median(np.diff(time_s)))
        ntr = len(station_indices)
        rx_fallback = _regular_receiver_x_from_station_indices(
            station_indices, first_receiver_x_m, receiver_spacing_m
        )
        rx = _values_for_traces(
            receiver_x_m,
            station_indices=station_indices,
            ntraces=ntr,
            fallback=rx_fallback,
            name="receiver_x_m",
        )
        rz = _values_for_traces(
            receiver_z_m,
            station_indices=station_indices,
            ntraces=ntr,
            fallback=0.0,
            name="receiver_z_m",
        )

        st = Stream()
        for row_index, station_index in enumerate(station_indices):
            tr = Trace(data=data[row_index].astype(np.float32))
            tr.stats.network = network
            tr.stats.station = f"S{int(station_index):04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = component
            tr.stats.delta = dt
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(time_s[0])
            _attach_standard_trace_header(
                tr,
                station_idx=int(station_index),
                receiver_x_m=float(rx[row_index]),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                receiver_z_m=float(rz[row_index]),
                source_z_m=float(source_z_m),
            )
            st.append(tr)
        return st

    raise ValueError(f"Unknown SPECFEM gather result mode: {mode!r}")


def load_model_as_stream(
    output_dir: str | Path,
    component: str = "BXZ",
    extension: str = "semv",
    timing=None,
    prefer_su: bool = True,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    source_x_m: float = 0.0,
    shot_number: Optional[int] = None,
    network: str = "SY",
    receiver_x_m=None,
    receiver_z_m=None,
    source_z_m: float = 0.0,
    verbose: bool = True,
) -> tuple[Stream, dict]:
    """Load a SPECFEM2D model directory and return a standardized stream.

    Returns both the stream and the raw loader result dictionary for provenance.
    """
    output_dir = Path(output_dir).expanduser()
    model_name = model_name_from_output_dir(output_dir)
    if shot_number is None:
        shot_number = model_number_from_name(model_name) or 1

    result = load_specfem_gather(
        output_dir=output_dir,
        component=component,
        extension=extension,
        timing=timing,
        prefer_su=prefer_su,
        verbose=verbose,
    )
    st = specfem_gather_result_to_stream(
        result=result,
        component=component,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
        source_x_m=source_x_m,
        shot_number=int(shot_number),
        network=network,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m,
        source_z_m=source_z_m,
    )
    return st, result


def write_model_products(
    output_dir: str | Path,
    config: SpecfemExportConfig,
    component: str = "BXZ",
    extension: str = "semv",
    timing=None,
    prefer_su: bool = True,
    write_segy_file: bool = True,
    make_plot: bool = True,
    tmin: float = 0.0,
    tmax: float = 0.3,
    scale: float = 0.02,
    normalize: bool = False,
    verbose: bool = True,
) -> tuple[Stream, dict, Path, Path]:
    """Load one model, then optionally write SEG-Y and a wiggle plot.

    Parameters
    ----------
    output_dir
        SPECFEM ``OUTPUT_FILES`` directory.
    config
        Export configuration containing output directories and fallback geometry.
    component, extension, timing, prefer_su
        Passed to ``load_model_as_stream``.
    write_segy_file
        Write ``<model>_<component>.segy`` under ``config.segy_out_dir``.
    make_plot
        Write ``<model>_<component>.png`` under ``config.fig_dir``.
    tmin, tmax, scale, normalize
        Plotting parameters passed to ``plot_wiggle_gather_from_stream``.

    Returns
    -------
    stream, result, segy_file, fig_file
        Processed stream, raw loader result, and output paths.
    """
    if write_segy_file and write_segy is None:
        raise ImportError("segy_tools.io.write_segy is not available.")
    if make_plot and plot_wiggle_gather_from_stream is None:
        raise ImportError("segy_tools.gather.plot_wiggle_gather_from_stream is not available.")

    config.ensure_directories()
    output_dir = Path(output_dir).expanduser()
    model_name = model_name_from_output_dir(output_dir)
    shot_number = model_number_from_name(model_name) or 1

    st, result = load_model_as_stream(
        output_dir=output_dir,
        component=component,
        extension=extension,
        timing=timing,
        prefer_su=prefer_su,
        receiver_spacing_m=config.receiver_spacing_m,
        first_receiver_x_m=config.first_receiver_x_m,
        source_x_m=config.source_x_m,
        shot_number=shot_number,
        network=config.network,
        receiver_x_m=config.receiver_x_m,
        receiver_z_m=config.receiver_z_m,
        source_z_m=config.source_z_m,
        verbose=verbose,
    )

    segy_file = Path(config.segy_out_dir) / f"{model_name}_{component}.segy"
    fig_file = Path(config.fig_dir) / f"{model_name}_{component}.png"

    if write_segy_file:
        segy_file.parent.mkdir(parents=True, exist_ok=True)
        write_segy(st, segy_file)
        if verbose:
            print(f"  wrote SEG-Y: {segy_file}")

    if make_plot:
        fig_file.parent.mkdir(parents=True, exist_ok=True)
        plot_wiggle_gather_from_stream(
            st,
            fallback_receiver_spacing_m=config.receiver_spacing_m,
            fallback_first_receiver_x_m=config.first_receiver_x_m,
            fallback_source_x_m=config.source_x_m,
            normalize=normalize,
            scale=scale,
            tmin=tmin,
            tmax=tmax,
            title=f"{model_name} {component}",
            outfile=fig_file,
        )
        if verbose:
            print(f"  wrote figure: {fig_file}")

    return st, result, segy_file, fig_file


def batch_write_model_products(
    model_output_dirs: Sequence[str | Path],
    config: SpecfemExportConfig,
    components: Sequence[str] = ("BXZ", "BXX"),
    extension: str = "semv",
    prefer_su: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Process multiple SPECFEM model directories into SEG-Y and figures.

    Parameters
    ----------
    model_output_dirs
        Iterable of ``OUTPUT_FILES`` directories.
    config
        Export configuration.
    components
        Components to attempt for each model.
    extension, prefer_su
        Loading options.
    **kwargs
        Additional keyword arguments passed to ``write_model_products``.

    Returns
    -------
    pandas.DataFrame
        Processing summary with one row per attempted model/component.
    """
    rows = []
    for output_dir in model_output_dirs:
        model_name = model_name_from_output_dir(output_dir)
        for component in components:
            try:
                st, result, segy_file, fig_file = write_model_products(
                    output_dir=output_dir,
                    config=config,
                    component=component,
                    extension=extension,
                    prefer_su=prefer_su,
                    **kwargs,
                )
                rows.append({
                    "model": model_name,
                    "component": component,
                    "mode": result.get("mode"),
                    "n_traces": len(st),
                    "segy_file": segy_file,
                    "figure_file": fig_file,
                    "error": None,
                })
            except Exception as exc:
                rows.append({
                    "model": model_name,
                    "component": component,
                    "mode": "failed",
                    "n_traces": 0,
                    "segy_file": None,
                    "figure_file": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return pd.DataFrame(rows)



def convert_sem_output_to_segy(
    input_dir: str | Path,
    output_dir: str | Path,
    component: str = "Z",
    extension: str = "semv",
    shot_number: int = 1,
    source_x_m: float | None = None,
    geom: Geometry | None = None,
    timing: Timing | None = None,
    network: str = "SY",
    receiver_x_m=None,
    receiver_z_m=None,
    source_z_m: float | None = None,
    verbose: bool = True,
) -> Path:
    """Convert one SPECFEM SEM ASCII shot gather directly to SEG-Y.

    This replaces the useful SEM conversion functionality that previously lived
    in ``converters.py``.  For irregular receiver spacing, pass ``receiver_x_m``
    as either a trace-order sequence or a mapping keyed by SPECFEM station index
    / station name.  For topography, pass matching ``receiver_z_m`` values.
    """
    if write_segy is None:
        raise ImportError("segy_tools.io.write_segy is not available.")

    geom = geom or Geometry()
    timing = timing or Timing()
    if source_x_m is None:
        source_x_m = source_x_from_shot(shot_number, geom)
    if source_z_m is None:
        source_z_m = geom.source_z_m

    time_s, data, stations = read_sem_gather(
        input_dir,
        component=component,
        extension=extension,
        timing=timing,
        verbose=verbose,
    )
    result = {
        "mode": "sem",
        "time": time_s,
        "data": data,
        "station_indices": stations,
        "component": component_to_channel(component),
        "extension": extension,
    }
    st = specfem_gather_result_to_stream(
        result,
        component=component_to_channel(component),
        receiver_spacing_m=geom.receiver_spacing_m,
        first_receiver_x_m=geom.first_receiver_x_m,
        source_x_m=float(source_x_m),
        shot_number=int(shot_number),
        network=network,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m if receiver_z_m is not None else geom.receiver_z_m,
        source_z_m=float(source_z_m),
    )

    channel = component_to_channel(component)
    outpath = Path(output_dir).expanduser() / channel / f"shot_{int(shot_number):03d}_{channel}_{extension or 'sem'}.segy"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    write_segy(st, outpath)
    if verbose:
        print(f"Wrote {outpath}")
    return outpath


def convert_su_shot_to_segy(
    su_path: str | Path,
    output_path: str | Path,
    receiver_x_m,
    source_x_m: float,
    shot_number: int,
    dt_s: float | None = None,
    t0_s: float = 0.0,
    component: str = "Z",
    receiver_z_m=None,
    source_z_m: float = 0.0,
    network: str = "SY",
) -> Path:
    """Convert one SPECFEM SU shot gather to SEG-Y with explicit geometry.

    ``receiver_x_m`` should normally be the true receiver positions in trace
    order.  This is the direct path to produce RefraPick-friendly SEG-Y for an
    irregular STATIONS file.  ``receiver_z_m`` may be supplied to preserve
    topography/elevation in the SEG-Y trace headers.
    """
    if read_su_file is None:
        raise ImportError("segy_tools.io.read_su_file is not available.")
    if write_segy is None:
        raise ImportError("segy_tools.io.write_segy is not available.")

    st_in = read_su_file(su_path, dt_s=dt_s, t0_s=t0_s)
    result = {
        "mode": "su",
        "stream": st_in,
        "component": component_to_channel(component),
    }
    # If dt_s was not supplied, preserve the sample interval returned by read_su_file.
    st = specfem_gather_result_to_stream(
        result,
        component=component_to_channel(component),
        source_x_m=float(source_x_m),
        shot_number=int(shot_number),
        network=network,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m,
        source_z_m=float(source_z_m),
    )
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_segy(st, output_path)
    return output_path

def plot_segy_file(
    segy_file: str | Path,
    outfile: str | Path | None = None,
    config: Optional[SpecfemExportConfig] = None,
    tmin: float = 0.0,
    tmax: float = 0.3,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    source_x_m: float = 0.0,
    scale: float = 0.02,
    normalize: bool = False,
) -> Stream:
    """Read and plot a converted SEG-Y gather for quick QC."""
    if plot_wiggle_gather_from_stream is None:
        raise ImportError("segy_tools.gather.plot_wiggle_gather_from_stream is not available.")

    segy_file = Path(segy_file).expanduser()
    if read_segy_as_stream is None:
        raise ImportError("segy_tools.io.read_segy_as_stream is not available.")
    st = read_segy_as_stream(segy_file)

    if config is not None:
        receiver_spacing_m = config.receiver_spacing_m
        first_receiver_x_m = config.first_receiver_x_m
        source_x_m = config.source_x_m
        if outfile is None:
            outfile = Path(config.fig_dir) / f"{segy_file.stem}_wiggle.png"

    plot_wiggle_gather_from_stream(
        st,
        fallback_receiver_spacing_m=receiver_spacing_m,
        fallback_first_receiver_x_m=first_receiver_x_m,
        fallback_source_x_m=source_x_m,
        tmin=tmin,
        tmax=tmax,
        scale=scale,
        normalize=normalize,
        title=f"{segy_file.name}: wiggle gather",
        outfile=outfile,
    )
    return st


def plot_su_directory(
    su_dir: str | Path,
    pattern: str = "*.su",
    outfile_dir: str | Path | None = None,
    tmin: float = -0.05,
    tmax: float = 0.25,
    receiver_spacing_m: float = 2.0,
    first_receiver_x_m: float = 0.0,
    source_x_m: float = 0.0,
    scale: float = 0.02,
    normalize: bool = False,
) -> pd.DataFrame:
    """Read all matching SU files in a directory and write QC wiggle plots."""
    if plot_wiggle_gather_from_stream is None:
        raise ImportError("segy_tools.gather.plot_wiggle_gather_from_stream is not available.")

    su_dir = Path(su_dir).expanduser()
    if outfile_dir is None:
        outfile_dir = su_dir / "su_wiggles"
    outfile_dir = Path(outfile_dir)
    outfile_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for su_file in sorted(su_dir.glob(pattern)):
        try:
            st, byteorder = read_su_try_both_byteorders(su_file)
            outfile = outfile_dir / f"{su_file.stem}_wiggle.png"
            plot_wiggle_gather_from_stream(
                st,
                fallback_receiver_spacing_m=receiver_spacing_m,
                fallback_first_receiver_x_m=first_receiver_x_m,
                fallback_source_x_m=source_x_m,
                tmin=tmin,
                tmax=tmax,
                scale=scale,
                normalize=normalize,
                title=f"{su_file.stem}: SU wiggle gather",
                outfile=outfile,
            )
            rows.append({"file": su_file, "byteorder": byteorder, "n_traces": len(st), "figure_file": outfile, "error": None})
        except Exception as exc:
            rows.append({"file": su_file, "byteorder": None, "n_traces": 0, "figure_file": None, "error": str(exc)})
    return pd.DataFrame(rows)


def plot_model_difference_from_segy(
    model_a: str,
    model_b: str,
    component: str = "BXZ",
    segy_dir: str | Path | None = None,
    diff_segy_dir: str | Path | None = None,
    config: Optional[SpecfemExportConfig] = None,
    source_x_m: float = 0.0,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    tmin: float = 0.0,
    tmax: float = 0.3,
    omin: float | None = None,
    omax: float | None = None,
    clip_percentile: float = 98.0,
    outfile: str | Path | None = None,
) -> tuple[object, np.ndarray]:
    """Compute and plot the difference between two converted model SEG-Y gathers."""
    if plot_difference_gathers is None:
        raise ImportError("segy_tools.plotting.plot_difference_gathers is not available.")

    if config is not None:
        segy_dir = config.segy_out_dir if segy_dir is None else segy_dir
        source_x_m = config.source_x_m
        receiver_spacing_m = config.receiver_spacing_m
        first_receiver_x_m = config.first_receiver_x_m
        if outfile is None:
            outfile = Path(config.diff_fig_dir) / f"{model_a}_minus_{model_b}_{component}.png"

    if segy_dir is None:
        raise ValueError("Either segy_dir or config must be supplied.")

    segy_dir = Path(segy_dir).expanduser()
    segy_a = segy_dir / f"{model_a}_{component}.segy"
    segy_b = segy_dir / f"{model_b}_{component}.segy"
    if not segy_a.exists():
        raise FileNotFoundError(segy_a)
    if not segy_b.exists():
        raise FileNotFoundError(segy_b)

    output_diff_segy = None
    if diff_segy_dir is not None:
        diff_segy_dir = Path(diff_segy_dir).expanduser()
        output_diff_segy = diff_segy_dir / f"{model_a}_minus_{model_b}_{component}.segy"

    if difference_segy_gathers is None:
        raise ImportError("segy_tools.gather.difference_segy_gathers is not available.")

    time_s, data_a, data_b, diff, receiver_x_m = difference_segy_gathers(
        segy_a,
        segy_b,
        fallback_receiver_spacing_m=receiver_spacing_m,
        fallback_first_receiver_x_m=first_receiver_x_m,
        fallback_source_x_m=source_x_m,
        output_segy_path=output_diff_segy,
    )

    fig = plot_difference_gathers(
        time=time_s,
        data_a=data_a,
        data_b=data_b,
        receiver_x_m=receiver_x_m,
        source_x_m=source_x_m,
        label_a=model_a,
        label_b=model_b,
        title=f"{model_a} - {model_b}, {component}",
        tmin=tmin,
        tmax=tmax,
        omin=omin,
        omax=omax,
        clip_percentile=clip_percentile,
        outfile=outfile,
    )
    return fig, diff
