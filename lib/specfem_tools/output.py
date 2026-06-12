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
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict

try:
    from segy_tools.io import read_su_file, read_segy_as_stream, write_segy
    from segy_tools.headers import make_trace_header
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

from .io import read_sem_gather


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
) -> Stream:
    """Convert a loaded SPECFEM gather result to a standardized ObsPy stream.

    Parameters
    ----------
    result
        Dictionary returned by ``load_specfem_gather``.
    component
        Channel/component code to assign to traces.
    receiver_spacing_m, first_receiver_x_m, source_x_m
        Synthetic geometry used when source/receiver coordinates are not stored
        in the native output.
    shot_number
        Field-record number assigned in SEG-Y trace headers.
    network
        ObsPy network code.

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
        for i, tr in enumerate(st, start=1):
            receiver_x_m = first_receiver_x_m + (i - 1) * receiver_spacing_m
            tr.stats.network = network
            tr.stats.station = f"S{i:04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = result.get("component", component).upper()
            tr.stats.segy = AttribDict()
            tr.stats.segy.trace_header = make_trace_header(
                station_idx=i,
                receiver_x_m=float(receiver_x_m),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                dt_s=float(tr.stats.delta),
                npts=int(tr.stats.npts),
                receiver_z_m=0.0,
                source_z_m=0.0,
            )
        return st

    if mode == "sem":
        time_s = np.asarray(result["time"], dtype=float)
        data = np.asarray(result["data"], dtype=np.float32)
        station_indices = np.asarray(result["station_indices"], dtype=int)
        if time_s.size < 2:
            raise ValueError("SEM gather time vector must contain at least two samples.")
        dt = float(np.median(np.diff(time_s)))
        st = Stream()
        station0 = int(station_indices.min())

        for row_index, station_index in enumerate(station_indices):
            receiver_x_m = first_receiver_x_m + (int(station_index) - station0) * receiver_spacing_m
            tr = Trace(data=data[row_index].astype(np.float32))
            tr.stats.network = network
            tr.stats.station = f"S{int(station_index):04d}"
            tr.stats.location = f"{int(shot_number) % 100:02d}"
            tr.stats.channel = component
            tr.stats.delta = dt
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(time_s[0])
            tr.stats.segy = AttribDict()
            tr.stats.segy.trace_header = make_trace_header(
                station_idx=int(station_index),
                receiver_x_m=float(receiver_x_m),
                source_x_m=float(source_x_m),
                shot_number=int(shot_number),
                dt_s=dt,
                npts=int(tr.stats.npts),
                receiver_z_m=0.0,
                source_z_m=0.0,
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
