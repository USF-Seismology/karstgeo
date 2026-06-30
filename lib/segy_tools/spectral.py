from __future__ import annotations

"""Spectral and f-k utilities for SEG-Y/ObsPy gathers.

This module contains numerical spectral-analysis routines.  Plotting-only
routines generally belong in :mod:`segy_tools.plotting`; a small number of
plotting helpers remain here where they are tightly coupled to spectral products
or retained as compatibility wrappers.

Geometry convention
-------------------
All gather-level routines use ``stream_to_gather_arrays()`` so receiver
coordinates come from SEG-Y headers / ObsPy stats when available.  Regular
receiver spacing is used only as a fallback.

Important f-k limitation
------------------------
A conventional 2-D FFT f-k transform requires uniformly spaced receivers.
For irregular receiver spacing, this module can either raise a clear error or
interpolate the gather onto a regular receiver grid before applying the f-k
operation.
"""

from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple
import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream

from .gather import stream_to_gather_arrays
from .io import read_segy_as_stream, gather_arrays_to_stream
from .processing import demean_traces, normalize_traces_by_range


# -----------------------------------------------------------------------------
# Basic spectra
# -----------------------------------------------------------------------------


def trace_spectrum(trace: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided amplitude spectrum for a single trace."""
    trace = np.asarray(trace, dtype=float).ravel()
    freqs = np.fft.rfftfreq(trace.size, d=1.0 / float(fs))
    amps = np.abs(np.fft.rfft(trace))
    return freqs, amps


def plot_trace_spectrum(
    data: np.ndarray,
    fs: float,
    trace_index: int = 0,
    max_freq: float = 250.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Frequency spectrum",
) -> plt.Axes:
    """Plot a single-trace amplitude spectrum from a trace or gather array."""
    data = np.asarray(data, dtype=float)
    trace = data if data.ndim == 1 else data[int(trace_index)]
    freqs, amps = trace_spectrum(trace, fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, amps)
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return ax


# -----------------------------------------------------------------------------
# Charlie Breithaupt MST-like transform retained for reproducibility
# -----------------------------------------------------------------------------


def g_window(length: int, freq: float, factor: float = 1.0) -> np.ndarray:
    """Gaussian window used in Charlie Breithaupt's custom MST transform."""
    v1 = np.arange(length, dtype=float)
    v2 = np.arange(-length, 0, dtype=float)
    vector = np.vstack([v1, v2]) ** 2
    vector *= -float(factor) * np.pi**2 / float(freq) ** 2
    return np.sum(np.exp(vector), axis=0)


def mst(
    trace: np.ndarray,
    time: np.ndarray,
    factor: float = 1.0,
    F: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charlie Breithaupt's custom MST-like transform.

    This is a close translation of the thesis MATLAB routine. It is preserved
    for reproducibility, but standard FFT/STFT products are usually easier to
    interpret.
    """
    h = np.asarray(trace, dtype=float).ravel()
    t = np.asarray(time, dtype=float).ravel()
    M = h.size
    H = np.concatenate([np.fft.fft(h), np.fft.fft(h)])
    ncols_half = int(np.ceil(M / 2.0))
    STR = np.zeros((M, ncols_half), dtype=complex)
    STR[:, 0] = np.mean(h)

    for fbin in range(1, int(np.floor((M - 1) / 2.0)) + 1):
        if fbin >= ncols_half:
            break
        T = g_window(M, F, factor)
        STR[:, fbin] = np.fft.ifft(H[fbin : fbin + M] * T)

    ST = np.fliplr(np.conj(STR[:, 1:]))
    STR = np.column_stack([STR, ST])

    if len(t) > 1:
        positive_time_indices = np.where(t > 0)[0]
        if positive_time_indices.size >= 2:
            aa1 = positive_time_indices[0]
            dt = t[aa1 + 1] - t[aa1]
        else:
            dt = t[1] - t[0]
    else:
        dt = 1.0
    fnyq = 1.0 / (2.0 * dt)
    m, n = STR.shape
    fout = np.linspace(0.0, fnyq, n)
    tout = np.linspace(0.0, float(np.max(t)) if len(t) else 0.0, m)
    return STR, tout, fout


# -----------------------------------------------------------------------------
# Frequency-vs-offset products
# -----------------------------------------------------------------------------


def frequency_offset_fft_arrays(
    data: np.ndarray,
    time: np.ndarray,
    receiver_x_m: Sequence[float],
    max_freq: float = 100.0,
    normalize: bool = True,
) -> dict:
    """Compute a frequency-vs-receiver/offset amplitude image using FFT.

    Parameters use the package convention ``data.shape == (n_traces, n_samples)``.
    ``receiver_x_m`` may be irregularly spaced; no spatial FFT is performed here.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be shaped (n_traces, n_samples).")
    if receiver_x_m.size != data.shape[0]:
        raise ValueError("receiver_x_m length must match number of traces.")

    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    work = normalize_traces_by_range(data) if normalize else demean_traces(data)
    spec = np.abs(np.fft.rfft(work, axis=1))
    freqs = np.fft.rfftfreq(work.shape[1], d=dt)
    keep = freqs <= float(max_freq)
    spec = spec[:, keep].T
    freqs = freqs[keep]
    if normalize:
        spec = spec - np.nanmin(spec)
        denom = np.nanmax(spec)
        if denom > 0:
            spec = spec / denom
    return {"spectrum": spec, "frequencies": freqs, "receiver_x_m": receiver_x_m}


def plot_frequency_offset(
    frequencies: np.ndarray,
    receiver_x_m: np.ndarray,
    amplitude: np.ndarray,
    title: str = "Frequency vs offset",
    max_freq: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    outfile: Optional[str | Path] = None,
    dpi: int = 160,
) -> plt.Axes:
    """Plot frequency-vs-position amplitudes as contours.

    This works with irregular receiver spacing because Matplotlib contours accept
    explicit x coordinates.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contour(receiver_x_m, frequencies, amplitude)
    ax.invert_yaxis()
    if max_freq is not None:
        ax.set_ylim(float(max_freq), 0)
    ax.set_title(title)
    ax.set_xlabel("Receiver x / offset (m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.figure.colorbar(cs, ax=ax, label="Scaled amplitude")
    ax.grid(True, alpha=0.2)
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(outfile, dpi=dpi)
        plt.close(ax.figure)
    return ax


def frequency_offset_fft(
    filename: str | Path,
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
    max_freq: float = 100.0,
    make_plot: bool = True,
) -> dict:
    """Read a SEG-Y/SU file and compute FFT frequency-vs-position amplitudes.

    Geometry is read from SEG-Y headers when present. ``dx`` is only a fallback.
    """
    # ``format`` is retained for API compatibility; read_segy_as_stream handles
    # SEG-Y here.  SU-specific workflows should use io.read_su_file first.
    st = read_segy_as_stream(filename)
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by="receiver_x",
        fallback_receiver_spacing_m=dx,
        fallback_first_receiver_x_m=0.0,
        fallback_source_x_m=0.0,
    )
    if offsets is not None:
        receiver_x_m = np.asarray(offsets, dtype=float)
    result = frequency_offset_fft_arrays(data, time, receiver_x_m, max_freq=max_freq)
    result.update({"stream": st, "time": time, "data": data, "source_x_m": source_x_m, "geometry": geom, "figures": []})
    if make_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_frequency_offset(
            result["frequencies"],
            result["receiver_x_m"],
            result["spectrum"],
            max_freq=max_freq,
            ax=ax,
            title="Frequency vs offset, FFT",
        )
        result["figures"].append(fig)
    return result


def frequency_offset_mst_arrays(
    data: np.ndarray,
    time: np.ndarray,
    receiver_x_m: Sequence[float],
    factor: float = 1.0,
    F_values: Sequence[float] = (30.0,),
) -> dict:
    """Compute Charlie MST frequency-vs-position integrated amplitudes."""
    data = np.asarray(data, dtype=float)
    outputs = []
    frequencies = []
    for F in F_values:
        integrals = []
        fout_ref = None
        for trace in data:
            STR, tout, fout = mst(trace, time, factor=factor, F=F)
            integrals.append(np.trapz(np.abs(STR), axis=0))
            fout_ref = fout
        mat = np.column_stack(integrals)  # freq x trace
        mat = mat - np.nanmin(mat)
        denom = np.nanmax(np.ptp(mat, axis=0))
        if denom > 0:
            mat = mat / denom
        outputs.append(mat)
        frequencies.append(fout_ref)
    return {
        "normalized_integrals": outputs,
        "frequencies": frequencies,
        "receiver_x_m": np.asarray(receiver_x_m),
        "F_values": list(F_values),
    }


# -----------------------------------------------------------------------------
# Regular-grid helpers for f-k operations
# -----------------------------------------------------------------------------


def receiver_spacing_summary(receiver_x_m: Sequence[float]) -> dict:
    """Return summary statistics for receiver spacing."""
    x = np.asarray(receiver_x_m, dtype=float)
    if x.size < 2:
        return {"n": int(x.size), "dx_median": np.nan, "dx_min": np.nan, "dx_max": np.nan, "dx_std": np.nan, "is_increasing": True}
    dx = np.diff(x)
    return {
        "n": int(x.size),
        "dx_median": float(np.nanmedian(dx)),
        "dx_min": float(np.nanmin(dx)),
        "dx_max": float(np.nanmax(dx)),
        "dx_std": float(np.nanstd(dx)),
        "is_increasing": bool(np.all(dx > 0)),
    }


def is_regular_receiver_spacing(receiver_x_m: Sequence[float], tolerance_m: float = 1e-3) -> bool:
    """Return True if receiver coordinates are approximately uniformly spaced."""
    x = np.asarray(receiver_x_m, dtype=float)
    if x.size < 3:
        return True
    dx = np.diff(x)
    return bool(np.all(np.isfinite(dx)) and np.nanmax(np.abs(dx - np.nanmedian(dx))) <= float(tolerance_m))


def regularize_gather_by_receiver_x(
    data: np.ndarray,
    receiver_x_m: Sequence[float],
    dx_m: Optional[float] = None,
    *,
    method: Literal["linear"] = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an irregular receiver gather onto a regular x grid.

    Parameters
    ----------
    data
        Gather array shaped ``(n_traces, n_samples)``.
    receiver_x_m
        Receiver coordinates for each trace.
    dx_m
        Desired output spacing. If omitted, the median input spacing is used.
    method
        Currently only ``"linear"`` is implemented.

    Returns
    -------
    regular_data, regular_receiver_x_m
    """
    if method != "linear":
        raise ValueError("Only linear interpolation is currently implemented.")

    data = np.asarray(data, dtype=float)
    x = np.asarray(receiver_x_m, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be shaped (n_traces, n_samples).")
    if x.size != data.shape[0]:
        raise ValueError("receiver_x_m length must match number of traces.")
    if x.size < 2:
        return data.copy(), x.copy()

    order = np.argsort(x)
    x_sorted = x[order]
    data_sorted = data[order]

    # Collapse duplicate receiver coordinates by averaging traces.
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size != x_sorted.size:
        collapsed = np.zeros((unique_x.size, data.shape[1]), dtype=float)
        counts = np.zeros(unique_x.size, dtype=float)
        for i, group in enumerate(inverse):
            collapsed[group] += data_sorted[i]
            counts[group] += 1.0
        data_sorted = collapsed / counts[:, None]
        x_sorted = unique_x

    if dx_m is None:
        dx_m = float(np.nanmedian(np.diff(x_sorted)))
    dx_m = abs(float(dx_m))
    if not np.isfinite(dx_m) or dx_m <= 0:
        raise ValueError("dx_m must be positive and finite.")

    n_out = int(np.floor((x_sorted[-1] - x_sorted[0]) / dx_m + 0.5)) + 1
    x_regular = x_sorted[0] + np.arange(n_out, dtype=float) * dx_m
    # Ensure last point does not run past the original range because np.interp
    # would then silently extrapolate as a constant.
    x_regular = x_regular[x_regular <= x_sorted[-1] + 1e-9]

    regular = np.empty((x_regular.size, data_sorted.shape[1]), dtype=float)
    for j in range(data_sorted.shape[1]):
        regular[:, j] = np.interp(x_regular, x_sorted, data_sorted[:, j])
    return regular, x_regular


def _fk_prepare_regular_gather(
    st: Stream,
    *,
    receiver_spacing_m: Optional[float],
    fallback_receiver_spacing_m: Optional[float],
    fallback_first_receiver_x_m: float,
    fallback_source_x_m: Optional[float],
    allow_resample_irregular: bool,
    resample_dx_m: Optional[float],
    regular_spacing_tolerance_m: float,
    component: Optional[str],
) -> dict:
    """Extract a gather and return a regularized version suitable for f-k FFT."""
    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by="receiver_x",
        component=component,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m if fallback_receiver_spacing_m is not None else receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )

    x = np.asarray(receiver_x_m, dtype=float)
    regular = is_regular_receiver_spacing(x, tolerance_m=regular_spacing_tolerance_m)

    if regular:
        dx = float(receiver_spacing_m) if receiver_spacing_m is not None else float(np.nanmedian(np.diff(x))) if x.size > 1 else 1.0
        return {
            "time": time,
            "data": data,
            "receiver_x_m": x,
            "source_x_m": source_x_m,
            "geometry": geom,
            "dx_m": dx,
            "was_resampled": False,
            "original_receiver_x_m": x,
            "original_data": data,
        }

    if not allow_resample_irregular:
        summary = receiver_spacing_summary(x)
        raise ValueError(
            "Conventional f-k FFT requires regular receiver spacing. "
            "This gather is irregular. Either set allow_resample_irregular=True "
            "or regularize the gather before calling this function. "
            f"Spacing summary: {summary}"
        )

    dx = float(resample_dx_m) if resample_dx_m is not None else float(receiver_spacing_m) if receiver_spacing_m is not None else None
    data_regular, x_regular = regularize_gather_by_receiver_x(data, x, dx_m=dx)
    dx_regular = float(np.nanmedian(np.diff(x_regular))) if x_regular.size > 1 else 1.0
    warnings.warn(
        "Input gather has irregular receiver spacing; interpolated to a regular "
        f"grid with dx={dx_regular:g} m for f-k processing.",
        RuntimeWarning,
        stacklevel=3,
    )
    return {
        "time": time,
        "data": data_regular,
        "receiver_x_m": x_regular,
        "source_x_m": source_x_m,
        "geometry": geom,
        "dx_m": dx_regular,
        "was_resampled": True,
        "original_receiver_x_m": x,
        "original_data": data,
    }


# -----------------------------------------------------------------------------
# f-k / apparent-velocity utilities
# -----------------------------------------------------------------------------


def apply_fk_velocity_filter(
    st: Stream,
    min_velocity_mps: float = 1000.0,
    receiver_spacing_m: Optional[float] = None,
    use_taper: bool = True,
    taper_width_mps: float = 200.0,
    *,
    allow_resample_irregular: bool = True,
    resample_dx_m: Optional[float] = None,
    regular_spacing_tolerance_m: float = 1e-3,
    fallback_receiver_spacing_m: Optional[float] = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: Optional[float] = 0.0,
    component: Optional[str] = None,
) -> Stream:
    """Apply a 2-D f-k fan filter that rejects slow apparent velocities.

    Geometry is read from headers/stats via ``stream_to_gather_arrays``.  If the
    receiver positions are irregular, the gather is interpolated to a regular
    receiver grid by default because a standard f-k FFT requires uniform spacing.

    Parameters
    ----------
    st
        Input gather.
    min_velocity_mps
        Apparent velocities below this value are muted.
    receiver_spacing_m
        Optional known regular spacing. If omitted, spacing is inferred from
        receiver coordinates.
    allow_resample_irregular
        If True, irregular receiver gathers are interpolated to a regular grid.
        If False, irregular spacing raises a clear ``ValueError``.
    resample_dx_m
        Output receiver spacing for irregular gathers. If omitted, the median
        spacing is used.
    """
    if len(st) == 0:
        raise ValueError("Cannot f-k filter an empty stream.")
    if min_velocity_mps <= 0:
        raise ValueError("min_velocity_mps must be positive.")

    prep = _fk_prepare_regular_gather(
        st,
        receiver_spacing_m=receiver_spacing_m,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
        allow_resample_irregular=allow_resample_irregular,
        resample_dx_m=resample_dx_m,
        regular_spacing_tolerance_m=regular_spacing_tolerance_m,
        component=component,
    )

    data = np.asarray(prep["data"], dtype=float)
    x = np.asarray(prep["receiver_x_m"], dtype=float)
    time = np.asarray(prep["time"], dtype=float)
    source_x_m = prep["source_x_m"] if prep["source_x_m"] is not None else 0.0

    ntr, npts = data.shape
    dt = float(np.median(np.diff(time))) if time.size > 1 else float(st[0].stats.delta)
    dx = float(prep["dx_m"])

    fk = np.fft.fft2(data)
    freqs = np.fft.fftfreq(npts, d=dt)
    wavenumbers = np.fft.fftfreq(ntr, d=dx)
    k_grid, f_grid = np.meshgrid(wavenumbers, freqs, indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        apparent_velocity = np.abs(f_grid / k_grid)

    mask = np.ones_like(fk, dtype=float)
    if use_taper:
        v0 = float(min_velocity_mps)
        v1 = v0 + float(taper_width_mps)
        mask[apparent_velocity <= v0] = 0.0
        transition = (apparent_velocity > v0) & (apparent_velocity < v1)
        xtrans = (apparent_velocity[transition] - v0) / (v1 - v0)
        mask[transition] = 0.5 * (1.0 - np.cos(np.pi * xtrans))
    else:
        mask[apparent_velocity < min_velocity_mps] = 0.0

    # Preserve k=0 content rather than muting DC/vertically coherent energy.
    mask[k_grid == 0] = 1.0
    filtered_data = np.real(np.fft.ifft2(fk * mask))

    component_out = component or getattr(st[0].stats, "channel", "Z")
    network_out = getattr(st[0].stats, "network", "SY")
    starttime = getattr(st[0].stats, "starttime", None)
    out = gather_arrays_to_stream(
        filtered_data.astype(np.float32),
        dt_s=dt,
        starttime=starttime,
        receiver_x_m=x,
        source_x_m=float(source_x_m),
        shot_number=1,
        network=network_out,
        component=component_out,
    )
    out.stats = copy.deepcopy(getattr(st, "stats", {}))
    out.stats["fk_filter"] = {
        "min_velocity_mps": float(min_velocity_mps),
        "receiver_spacing_m": dx,
        "was_resampled": bool(prep["was_resampled"]),
        "original_receiver_x_m": np.asarray(prep["original_receiver_x_m"], dtype=float).tolist(),
    }
    return out


def plot_fk_spectrum(
    st: Stream,
    receiver_spacing_m: Optional[float] = None,
    max_display_freq_hz: float = 600.0,
    reference_velocity_mps: Optional[float] = 1000.0,
    title: str = "f-k spectrum",
    *,
    allow_resample_irregular: bool = True,
    resample_dx_m: Optional[float] = None,
    regular_spacing_tolerance_m: float = 1e-3,
    fallback_receiver_spacing_m: Optional[float] = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: Optional[float] = 0.0,
    component: Optional[str] = None,
) -> plt.Figure:
    """Plot a log-amplitude f-k spectrum for an ObsPy gather.

    Receiver coordinates are read from headers/stats.  If irregular spacing is
    present, the gather is interpolated to a regular grid by default.
    """
    if len(st) == 0:
        raise ValueError("Cannot plot f-k spectrum for an empty stream.")

    prep = _fk_prepare_regular_gather(
        st,
        receiver_spacing_m=receiver_spacing_m,
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
        allow_resample_irregular=allow_resample_irregular,
        resample_dx_m=resample_dx_m,
        regular_spacing_tolerance_m=regular_spacing_tolerance_m,
        component=component,
    )

    data = np.asarray(prep["data"], dtype=float)
    time = np.asarray(prep["time"], dtype=float)
    ntr, npts = data.shape
    dt = float(np.median(np.diff(time))) if time.size > 1 else float(st[0].stats.delta)
    dx = float(prep["dx_m"])

    data = data - np.mean(data, axis=1, keepdims=True)
    fk = np.fft.fftshift(np.fft.fft2(data))
    amp = np.abs(fk)
    freqs = np.fft.fftshift(np.fft.fftfreq(npts, d=dt))
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(ntr, d=dx))

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [freqs[0], freqs[-1], wavenumbers[-1], wavenumbers[0]]
    img = ax.imshow(np.log10(amp + 1e-12), aspect="auto", extent=extent)
    ax.set_xlim(-max_display_freq_hz, max_display_freq_hz)

    if reference_velocity_mps is not None and reference_velocity_mps > 0:
        max_k = max_display_freq_hz / reference_velocity_mps
        ax.set_ylim(max_k * 1.5, -max_k * 1.5)
        f_axis = np.linspace(-max_display_freq_hz, max_display_freq_hz, 200)
        ax.plot(f_axis, f_axis / reference_velocity_mps, "r--", label=f"{reference_velocity_mps:g} m/s")
        ax.plot(f_axis, -f_axis / reference_velocity_mps, "r--")
        ax.legend(loc="upper right")

    subtitle = ""
    if prep["was_resampled"]:
        subtitle = f" (irregular receivers interpolated to dx={dx:g} m)"
    ax.set_title(title + subtitle)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Wavenumber (1/m)")
    fig.colorbar(img, ax=ax, label="log10 amplitude")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Compatibility wrapper: plotting version now belongs in segy_tools.plotting
# -----------------------------------------------------------------------------


def plot_frequency_contour_from_stream(*args, **kwargs):
    """Deprecated wrapper for :func:`segy_tools.plotting.plot_frequency_contour_from_stream`."""
    warnings.warn(
        "segy_tools.spectral.plot_frequency_contour_from_stream is deprecated; "
        "use segy_tools.plotting.plot_frequency_contour_from_stream instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .plotting import plot_frequency_contour_from_stream as _impl

    return _impl(*args, **kwargs)
