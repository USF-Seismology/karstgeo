import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream
from pathlib import Path
from typing import Literal
from .gather import stream_to_gather_arrays

def wiggle_plot_charlie_style(data, time, offsets, ax=None, color="k", linewidth=0.8, title=None, ylim=(0.0, 0.5)):
    """Simple Charlie-style wiggle plot for data shaped (n_traces, n_samples)."""
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    for i in range(data.shape[0]):
        ax.plot(data[i] + offsets[i], time, color=color, linewidth=linewidth)
    ax.invert_yaxis()
    ax.set_xlabel("distance / offset (m)")
    ax.set_ylabel("time (s)")
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim[::-1])
    ax.grid(True, alpha=0.2)
    return ax



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