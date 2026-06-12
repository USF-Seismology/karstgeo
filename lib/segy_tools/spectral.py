from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .gather import stream_to_gather_arrays
from .io import read_segy_as_stream
from .processing import demean_traces, normalize_traces_by_range


def trace_spectrum(trace: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided amplitude spectrum for a trace."""
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
    """Plot a single-trace amplitude spectrum from a gather."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        trace = data
    else:
        trace = data[int(trace_index)]
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


def g_window(length: int, freq: float, factor: float = 1.0) -> np.ndarray:
    """Gaussian window used in Charlie Breithaupt's custom MST transform."""
    v1 = np.arange(length, dtype=float)
    v2 = np.arange(-length, 0, dtype=float)
    vector = np.vstack([v1, v2]) ** 2
    vector *= -float(factor) * np.pi**2 / float(freq) ** 2
    return np.sum(np.exp(vector), axis=0)


def mst(trace: np.ndarray, time: np.ndarray, factor: float = 1.0, F: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def frequency_offset_fft_arrays(
    data: np.ndarray,
    time: np.ndarray,
    receiver_x_m: Sequence[float],
    max_freq: float = 100.0,
    normalize: bool = True,
) -> dict:
    """Compute a frequency-vs-offset amplitude image using standard FFT.

    Parameters use the package convention ``data.shape == (n_traces, n_samples)``.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)
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
    """Plot frequency-vs-offset amplitudes as a contour image."""
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
    """Read a SEG-Y/SU file and compute FFT frequency-vs-offset amplitudes."""
    st = read_segy_as_stream(filename) if format is None else read_segy_as_stream(filename)
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
        plot_frequency_offset(result["frequencies"], result["receiver_x_m"], result["spectrum"], max_freq=max_freq, ax=ax, title="Frequency vs offset, FFT")
        result["figures"].append(fig)
    return result


def frequency_offset_mst_arrays(
    data: np.ndarray,
    time: np.ndarray,
    receiver_x_m: Sequence[float],
    factor: float = 1.0,
    F_values: Sequence[float] = (30.0,),
) -> dict:
    """Compute Charlie MST frequency-vs-offset integrated amplitudes from arrays."""
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
    return {"normalized_integrals": outputs, "frequencies": frequencies, "receiver_x_m": np.asarray(receiver_x_m), "F_values": list(F_values)}

# -----------------------------------------------------------------------------
# f-k / apparent-velocity utilities migrated from seismic_gather_utils.
# -----------------------------------------------------------------------------

import copy
from obspy import Stream


def apply_fk_velocity_filter(
    st: Stream,
    min_velocity_mps: float = 1000.0,
    receiver_spacing_m: float = 1.0,
    use_taper: bool = True,
    taper_width_mps: float = 200.0,
) -> Stream:
    """Apply a 2-D f-k fan filter that rejects slow apparent velocities.

    Parameters
    ----------
    st
        Input gather. Traces should be ordered by receiver position.
    min_velocity_mps
        Apparent velocities below this value are muted.
    receiver_spacing_m
        Receiver spacing in metres.
    use_taper
        If True, apply a raised-cosine transition between
        ``min_velocity_mps`` and ``min_velocity_mps + taper_width_mps``.
    taper_width_mps
        Width of the velocity transition zone in m/s.

    Returns
    -------
    obspy.Stream
        Filtered copy of the input stream.

    Notes
    -----
    Use cautiously for diffraction analysis: diffraction wings that share the
    muted apparent-velocity range will also be attenuated.
    """
    if len(st) == 0:
        raise ValueError("Cannot f-k filter an empty stream.")
    if min_velocity_mps <= 0:
        raise ValueError("min_velocity_mps must be positive.")

    filtered = copy.deepcopy(st)
    filtered.sort(keys=["station", "channel"])

    ntr = len(filtered)
    npts = min(int(tr.stats.npts) for tr in filtered)
    dt = float(filtered[0].stats.delta)
    dx = float(receiver_spacing_m)

    data = np.vstack([np.asarray(tr.data[:npts], dtype=float) for tr in filtered])
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
        x = (apparent_velocity[transition] - v0) / (v1 - v0)
        mask[transition] = 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        mask[apparent_velocity < min_velocity_mps] = 0.0

    # Preserve k=0 content rather than muting DC/vertically coherent energy.
    mask[k_grid == 0] = 1.0
    filtered_data = np.real(np.fft.ifft2(fk * mask))

    for idx, tr in enumerate(filtered):
        tr.data = filtered_data[idx, :].astype(np.float32)
    return filtered


def plot_fk_spectrum(
    st: Stream,
    receiver_spacing_m: float = 1.0,
    max_display_freq_hz: float = 600.0,
    reference_velocity_mps: Optional[float] = 1000.0,
    title: str = "f-k spectrum",
) -> plt.Figure:
    """Plot a log-amplitude f-k spectrum for an ObsPy gather.

    Parameters
    ----------
    st
        Input gather.
    receiver_spacing_m
        Receiver spacing in metres.
    max_display_freq_hz
        Frequency-axis display limit.
    reference_velocity_mps
        Optional apparent-velocity guide line. Pass ``None`` to omit.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The f-k spectrum figure.
    """
    if len(st) == 0:
        raise ValueError("Cannot plot f-k spectrum for an empty stream.")

    st_work = st.copy()
    st_work.sort(keys=["station", "channel"])
    ntr = len(st_work)
    npts = min(int(tr.stats.npts) for tr in st_work)
    dt = float(st_work[0].stats.delta)
    dx = float(receiver_spacing_m)

    data = np.vstack([np.asarray(tr.data[:npts], dtype=float) for tr in st_work])
    data -= np.mean(data, axis=1, keepdims=True)

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

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Wavenumber (1/m)")
    fig.colorbar(img, ax=ax, label="log10 amplitude")
    fig.tight_layout()
    return fig


'''
def plot_frequency_contour_from_stream(
    st: Stream,
    *,
    fallback_receiver_spacing_m: float,
    fallback_first_receiver_x_m: float,
    fallback_source_x_m: float,
    max_freq_hz: float = 100.0,
    x_axis: Literal["receiver_x", "offset"] = "receiver_x",
    demean: bool = True,
    taper_fraction: float = 0.05,
    normalize: bool = True,
    normalize_mode: Literal["max", "percentile"] = "percentile",
    percentile_clip: float = 99.0,
    levels: int = 14,
    filled: bool = False,
    title: str | None = None,
    outfile: Path | None = None,
    close_after_save: bool = False,
    cave_markers_m: tuple[float, ...] = (),
    source_label: str = "source",
):
    """Create a Charlie-style frequency-vs-position contour plot from a shot gather.

    This is equivalent in spirit to Charlie Breithaupt's frequency-vs-offset
    plotting workflow: each trace is transformed to the frequency domain, the
    amplitude spectrum is plotted as a function of receiver position or
    source-receiver offset, and localized frequency-domain amplitude anomalies
    are used as candidate indicators of scattering, attenuation, or diffraction.

    This function implements the simpler FFT-based version of that workflow.
    It does not reproduce Charlie's custom MST transform.

    Parameters
    ----------
    st
        Shot gather as an ObsPy Stream.

    fallback_receiver_spacing_m, fallback_first_receiver_x_m, fallback_source_x_m
        Geometry values used if receiver/source positions cannot be recovered
        from headers.

    max_freq_hz
        Maximum frequency to display.

    x_axis
        If ``"receiver_x"``, plot spectra against receiver coordinate.
        If ``"offset"``, plot spectra against receiver_x - source_x.

    demean
        If True, subtract the mean from each trace before FFT.

    taper_fraction
        Fraction of each trace to cosine taper at both ends before FFT.
        Set to 0 to disable tapering.

    normalize
        If True, normalize spectral amplitudes.

    normalize_mode
        ``"max"`` normalizes by the maximum amplitude in the gather.
        ``"percentile"`` clips/normalizes by a high percentile, which is more
        robust when one trace has unusually high amplitude.

    percentile_clip
        Percentile used when ``normalize_mode="percentile"``.

    levels
        Number of contour levels.

    filled
        If True, use ``contourf``. If False, use line contours, closer to
        Charlie's plots.

    title
        Optional plot title.

    outfile
        Optional output path for the figure.

    close_after_save
        If True, close the figure after saving.

    cave_markers_m
        Optional receiver-coordinate positions to mark with vertical dashed lines.
        These are interpreted in the same coordinate system as ``receiver_x``.
        If ``x_axis="offset"``, they are converted to offset using source_x.

    source_label
        Label for the source marker.

    Returns
    -------
    dict
        Contains time, data, receiver coordinates, plot x coordinates,
        source coordinate, raw spectrum, normalized/plotted spectrum,
        frequencies, geometry metadata, and figure.
    """
    if len(st) == 0:
        raise ValueError("Input Stream is empty.")

    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by="receiver_x",
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )

    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)

    if data.ndim != 2:
        raise ValueError("Expected gather data to be a 2-D array.")

    if time.size < 2:
        raise ValueError("Time vector must contain at least two samples.")

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time vector; sample interval must be positive.")

    # Work out whether traces are rows or columns.
    # The current segy_tools convention should be (n_traces, n_samples).
    if data.shape[0] == receiver_x_m.size:
        trace_axis = 0
        working = data.copy()
    elif data.shape[1] == receiver_x_m.size:
        trace_axis = 1
        working = data.T.copy()
    else:
        raise ValueError(
            "Could not match data dimensions to receiver coordinates. "
            f"data.shape={data.shape}, receiver_x_m.size={receiver_x_m.size}"
        )

    # working is now (n_traces, n_samples)
    if demean:
        working = working - np.nanmean(working, axis=1, keepdims=True)

    if taper_fraction and taper_fraction > 0:
        n_samples = working.shape[1]
        n_taper = int(round(taper_fraction * n_samples))
        if n_taper > 1:
            taper = np.ones(n_samples)
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_taper)))
            taper[:n_taper] = ramp
            taper[-n_taper:] = ramp[::-1]
            working = working * taper[None, :]

    freqs = np.fft.rfftfreq(working.shape[1], d=dt)
    raw_spectrum = np.abs(np.fft.rfft(working, axis=1))  # (n_traces, n_freq)

    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    raw_spectrum = raw_spectrum[:, keep]

    # Plotting wants spectrum as (n_freq, n_traces)
    spectrum = raw_spectrum.T

    if normalize:
        spectrum_to_plot = spectrum.copy()
        spectrum_to_plot -= np.nanmin(spectrum_to_plot)

        if normalize_mode == "percentile":
            scale = np.nanpercentile(spectrum_to_plot, percentile_clip)
        elif normalize_mode == "max":
            scale = np.nanmax(spectrum_to_plot)
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        if scale and np.isfinite(scale) and scale > 0:
            spectrum_to_plot = spectrum_to_plot / scale

        if normalize_mode == "percentile":
            spectrum_to_plot = np.clip(spectrum_to_plot, 0, 1)
    else:
        spectrum_to_plot = spectrum

    if x_axis == "receiver_x":
        x = receiver_x_m
        xlabel = "Receiver x (m)"
        source_marker_x = source_x_m
        marker_positions = cave_markers_m
    elif x_axis == "offset":
        if source_x_m is None:
            raise ValueError("source_x_m is required when x_axis='offset'.")
        x = receiver_x_m - float(source_x_m)
        xlabel = "Source-receiver offset (m)"
        source_marker_x = 0.0
        marker_positions = tuple(float(m) - float(source_x_m) for m in cave_markers_m)
    else:
        raise ValueError("x_axis must be 'receiver_x' or 'offset'.")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    contour_func = ax.contourf if filled else ax.contour
    cs = contour_func(
        x,
        freqs,
        spectrum_to_plot,
        levels=levels,
    )

    ax.invert_yaxis()
    ax.set_ylim(max_freq_hz, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or "Frequency vs offset")
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Scaled amplitude" if normalize else "Amplitude")

    if source_marker_x is not None and np.nanmin(x) <= source_marker_x <= np.nanmax(x):
        ax.axvline(
            float(source_marker_x),
            linestyle=":",
            linewidth=1.0,
            label=source_label,
        )

    for xm in marker_positions:
        if np.nanmin(x) <= xm <= np.nanmax(x):
            ax.axvline(
                float(xm),
                linestyle="--",
                linewidth=1.0,
                label=f"marker {xm:g} m",
            )

    if (source_marker_x is not None) or marker_positions:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=180, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)

    return {
        "time": time,
        "data": data,
        "processed_data": working,
        "receiver_x_m": receiver_x_m,
        "x": x,
        "x_axis": x_axis,
        "source_x_m": source_x_m,
        "frequencies": freqs,
        "raw_spectrum": raw_spectrum,
        "spectrum": spectrum_to_plot,
        "geometry": geom,
        "figure": fig,
        "axis": ax,
    }
'''


def plot_frequency_contour_from_stream(
    st: Stream,
    *,
    fallback_receiver_spacing_m: float,
    fallback_first_receiver_x_m: float,
    fallback_source_x_m: float,
    max_freq_hz: float = 100.0,
    x_axis: Literal["receiver_x", "offset"] = "receiver_x",
    demean: bool = True,
    taper_fraction: float = 0.05,
    normalize: bool = True,
    normalize_mode: Literal["max", "percentile"] = "percentile",
    percentile_clip: float = 99.0,
    levels: int = 14,
    filled: bool = False,
    title: str | None = None,
    outfile: Path | None = None,
    close_after_save: bool = False,
    cave_markers_m: tuple[float, ...] = (),
    source_label: str = "source",
):
    """Create a Charlie-style frequency-vs-position contour plot from a shot gather.

    This is equivalent in spirit to Charlie Breithaupt's frequency-vs-offset
    plotting workflow: each trace is transformed to the frequency domain, the
    amplitude spectrum is plotted as a function of receiver position or
    source-receiver offset, and localized frequency-domain amplitude anomalies
    are used as candidate indicators of scattering, attenuation, or diffraction.

    This function implements the simpler FFT-based version of that workflow.
    It does not reproduce Charlie's custom MST transform.

    Parameters
    ----------
    st
        Shot gather as an ObsPy Stream.

    fallback_receiver_spacing_m, fallback_first_receiver_x_m, fallback_source_x_m
        Geometry values used if receiver/source positions cannot be recovered
        from headers.

    max_freq_hz
        Maximum frequency to display.

    x_axis
        If ``"receiver_x"``, plot spectra against receiver coordinate.
        If ``"offset"``, plot spectra against receiver_x - source_x.

    demean
        If True, subtract the mean from each trace before FFT.

    taper_fraction
        Fraction of each trace to cosine taper at both ends before FFT.
        Set to 0 to disable tapering.

    normalize
        If True, normalize spectral amplitudes.

    normalize_mode
        ``"max"`` normalizes by the maximum amplitude in the gather.
        ``"percentile"`` clips/normalizes by a high percentile, which is more
        robust when one trace has unusually high amplitude.

    percentile_clip
        Percentile used when ``normalize_mode="percentile"``.

    levels
        Number of contour levels.

    filled
        If True, use ``contourf``. If False, use line contours, closer to
        Charlie's plots.

    title
        Optional plot title.

    outfile
        Optional output path for the figure.

    close_after_save
        If True, close the figure after saving.

    cave_markers_m
        Optional receiver-coordinate positions to mark with vertical dashed lines.
        These are interpreted in the same coordinate system as ``receiver_x``.
        If ``x_axis="offset"``, they are converted to offset using source_x.

    source_label
        Label for the source marker.

    Returns
    -------
    dict
        Contains time, data, receiver coordinates, plot x coordinates,
        source coordinate, raw spectrum, normalized/plotted spectrum,
        frequencies, geometry metadata, and figure.
    """
    if len(st) == 0:
        raise ValueError("Input Stream is empty.")

    time, data, receiver_x_m, source_x_m, geom = stream_to_gather_arrays(
        st,
        sort_by="receiver_x",
        fallback_receiver_spacing_m=fallback_receiver_spacing_m,
        fallback_first_receiver_x_m=fallback_first_receiver_x_m,
        fallback_source_x_m=fallback_source_x_m,
    )

    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)

    if data.ndim != 2:
        raise ValueError("Expected gather data to be a 2-D array.")

    if time.size < 2:
        raise ValueError("Time vector must contain at least two samples.")

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time vector; sample interval must be positive.")

    # Work out whether traces are rows or columns.
    # The current segy_tools convention should be (n_traces, n_samples).
    if data.shape[0] == receiver_x_m.size:
        trace_axis = 0
        working = data.copy()
    elif data.shape[1] == receiver_x_m.size:
        trace_axis = 1
        working = data.T.copy()
    else:
        raise ValueError(
            "Could not match data dimensions to receiver coordinates. "
            f"data.shape={data.shape}, receiver_x_m.size={receiver_x_m.size}"
        )

    # working is now (n_traces, n_samples)
    if demean:
        working = working - np.nanmean(working, axis=1, keepdims=True)

    if taper_fraction and taper_fraction > 0:
        n_samples = working.shape[1]
        n_taper = int(round(taper_fraction * n_samples))
        if n_taper > 1:
            taper = np.ones(n_samples)
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_taper)))
            taper[:n_taper] = ramp
            taper[-n_taper:] = ramp[::-1]
            working = working * taper[None, :]

    freqs = np.fft.rfftfreq(working.shape[1], d=dt)
    raw_spectrum = np.abs(np.fft.rfft(working, axis=1))  # (n_traces, n_freq)

    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    raw_spectrum = raw_spectrum[:, keep]

    # Plotting wants spectrum as (n_freq, n_traces)
    spectrum = raw_spectrum.T

    if normalize:
        spectrum_to_plot = spectrum.copy()
        spectrum_to_plot -= np.nanmin(spectrum_to_plot)

        if normalize_mode == "percentile":
            scale = np.nanpercentile(spectrum_to_plot, percentile_clip)
        elif normalize_mode == "max":
            scale = np.nanmax(spectrum_to_plot)
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        if scale and np.isfinite(scale) and scale > 0:
            spectrum_to_plot = spectrum_to_plot / scale

        if normalize_mode == "percentile":
            spectrum_to_plot = np.clip(spectrum_to_plot, 0, 1)
    else:
        spectrum_to_plot = spectrum

    if x_axis == "receiver_x":
        x = receiver_x_m
        xlabel = "Receiver x (m)"
        source_marker_x = source_x_m
        marker_positions = cave_markers_m
    elif x_axis == "offset":
        if source_x_m is None:
            raise ValueError("source_x_m is required when x_axis='offset'.")
        x = receiver_x_m - float(source_x_m)
        xlabel = "Source-receiver offset (m)"
        source_marker_x = 0.0
        marker_positions = tuple(float(m) - float(source_x_m) for m in cave_markers_m)
    else:
        raise ValueError("x_axis must be 'receiver_x' or 'offset'.")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    contour_func = ax.contourf if filled else ax.contour
    cs = contour_func(
        x,
        freqs,
        spectrum_to_plot,
        levels=levels,
    )

    ax.invert_yaxis()
    ax.set_ylim(max_freq_hz, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or "Frequency vs offset")
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Scaled amplitude" if normalize else "Amplitude")

    if source_marker_x is not None and np.nanmin(x) <= source_marker_x <= np.nanmax(x):
        ax.axvline(
            float(source_marker_x),
            linestyle=":",
            linewidth=1.0,
            label=source_label,
        )

    for xm in marker_positions:
        if np.nanmin(x) <= xm <= np.nanmax(x):
            ax.axvline(
                float(xm),
                linestyle="--",
                linewidth=1.0,
                label=f"marker {xm:g} m",
            )

    if (source_marker_x is not None) or marker_positions:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=180, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)

    return {
        "time": time,
        "data": data,
        "processed_data": working,
        "receiver_x_m": receiver_x_m,
        "x": x,
        "x_axis": x_axis,
        "source_x_m": source_x_m,
        "frequencies": freqs,
        "raw_spectrum": raw_spectrum,
        "spectrum": spectrum_to_plot,
        "geometry": geom,
        "figure": fig,
        "axis": ax,
    }
