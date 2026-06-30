from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from obspy import read, Stream

try:
    from .gather import stream_to_gather_arrays, align_gather_arrays_by_receiver_x
except Exception:  # pragma: no cover
    stream_to_gather_arrays = None
    align_gather_arrays_by_receiver_x = None


# -----------------------------------------------------------------------------
# Small array/window helpers
# -----------------------------------------------------------------------------

def normalize_traces(data, eps=1e-12):
    data = np.asarray(data, dtype=float)
    scale = np.nanmax(np.abs(data), axis=1, keepdims=True)
    return data / (scale + eps)


def time_window_indices(time, tmin=None, tmax=None):
    time = np.asarray(time)
    mask = np.ones(time.shape, dtype=bool)
    if tmin is not None:
        mask &= time >= float(tmin)
    if tmax is not None:
        mask &= time <= float(tmax)
    return mask


def offset_window_indices(receiver_x_m, source_x_m, omin=None, omax=None):
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)
    offsets = receiver_x_m - float(source_x_m)
    mask = np.ones(receiver_x_m.shape, dtype=bool)
    if omin is not None:
        mask &= offsets >= float(omin)
    if omax is not None:
        mask &= offsets <= float(omax)
    return mask


# -----------------------------------------------------------------------------
# Generic gather plotting from arrays
# -----------------------------------------------------------------------------

def plot_wiggle_gather(
    time,
    data,
    receiver_x_m,
    source_x_m=None,
    title="Shot gather",
    tmin=None,
    tmax=None,
    omin=None,
    omax=None,
    scale=0.8,
    clip_percentile=99,
    normalize=True,
    fill_positive=True,
    fill_negative=True,
    trace_color="black",
    positive_color="red",
    negative_color="blue",
    positive_alpha=0.45,
    negative_alpha=0.35,
    cave=None,
    outfile=None,
    dpi=160,
):
    time = np.asarray(time)
    data = np.asarray(data)
    receiver_x_m = np.asarray(receiver_x_m)

    tmask = time_window_indices(time, tmin, tmax)
    if source_x_m is not None:
        rmask = offset_window_indices(receiver_x_m, source_x_m, omin, omax)
    else:
        rmask = np.ones_like(receiver_x_m, dtype=bool)

    tt = time[tmask]
    xx = receiver_x_m[rmask]
    dd = data[rmask][:, tmask]

    clip = np.nanpercentile(np.abs(dd), clip_percentile)
    if clip > 0:
        dd = np.clip(dd, -clip, clip)
    if normalize:
        dd = normalize_traces(dd)

    if len(xx) > 1:
        dxs = np.diff(np.sort(xx))
        dxs = dxs[dxs > 0]
        dx = np.nanmedian(dxs) if dxs.size else 1.0
    else:
        dx = 1.0

    fig, ax = plt.subplots(figsize=(13, 8))
    for i, x in enumerate(xx):
        y = x + scale * dx * dd[i]
        ax.plot(y, tt, color=trace_color, linewidth=0.5)
        if fill_positive:
            ax.fill_betweenx(tt, x, y, where=(y >= x), color=positive_color, alpha=positive_alpha, interpolate=True)
        if fill_negative:
            ax.fill_betweenx(tt, x, y, where=(y < x), color=negative_color, alpha=negative_alpha, interpolate=True)

    if source_x_m is not None:
        ax.axvline(source_x_m, linestyle="--", linewidth=1, label="source")
    if cave:
        ax.axvspan(cave["x_min_m"], cave["x_max_m"], alpha=0.15, label="cave x extent")
    ax.invert_yaxis()
    ax.set_xlabel("Receiver x (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if source_x_m is not None or cave:
        ax.legend(loc="upper right")
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
    return fig


def plot_image_gather(
    time,
    data,
    receiver_x_m,
    source_x_m=None,
    title="Image gather",
    tmin=None,
    tmax=None,
    omin=None,
    omax=None,
    clip_percentile=98,
    cave=None,
    outfile=None,
    dpi=160,
):
    time = np.asarray(time)
    data = np.asarray(data)
    receiver_x_m = np.asarray(receiver_x_m)
    tmask = time_window_indices(time, tmin, tmax)
    if source_x_m is not None:
        rmask = offset_window_indices(receiver_x_m, source_x_m, omin, omax)
    else:
        rmask = np.ones_like(receiver_x_m, dtype=bool)
    tt = time[tmask]
    xx = receiver_x_m[rmask]
    dd = data[rmask][:, tmask]
    clip = np.nanpercentile(np.abs(dd), clip_percentile)
    if clip <= 0 or not np.isfinite(clip):
        clip = 1.0
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(dd.T, extent=[xx.min(), xx.max(), tt.max(), tt.min()], aspect="auto", vmin=-clip, vmax=clip, cmap="seismic")
    fig.colorbar(im, ax=ax, label="Amplitude")
    if source_x_m is not None:
        ax.axvline(source_x_m, linestyle="--", linewidth=1, label="source")
    if cave:
        ax.axvspan(cave["x_min_m"], cave["x_max_m"], alpha=0.15, label="cave x extent")
    ax.set_xlabel("Receiver x (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    if source_x_m is not None or cave:
        ax.legend(loc="upper right")
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
    return fig


def plot_difference_gathers(time, data_a, data_b, receiver_x_m, source_x_m, label_a="A", label_b="B", title="Survey comparison", tmin=None, tmax=None, omin=None, omax=None, clip_percentile=98, outfile=None, dpi=160):
    time = np.asarray(time); receiver_x_m = np.asarray(receiver_x_m)
    tmask = time_window_indices(time, tmin, tmax)
    rmask = offset_window_indices(receiver_x_m, source_x_m, omin, omax)
    tt = time[tmask]; xx = receiver_x_m[rmask]
    a = data_a[rmask][:, tmask]; b = data_b[rmask][:, tmask]; diff = a - b
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, (d, lab) in zip(axes, [(a, label_a), (b, label_b), (diff, f"{label_a} - {label_b}")]):
        clip = np.nanpercentile(np.abs(d), clip_percentile) or 1.0
        im = ax.imshow(d.T, extent=[xx.min(), xx.max(), tt.max(), tt.min()], aspect="auto", vmin=-clip, vmax=clip, cmap="seismic")
        ax.axvline(source_x_m, linestyle="--", linewidth=1)
        ax.set_title(lab); ax.set_xlabel("Receiver x (m)")
        fig.colorbar(im, ax=ax, shrink=0.75)
    axes[0].set_ylabel("Time (s)")
    fig.suptitle(title); fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi); plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# Stream / SEG-Y wrappers
# -----------------------------------------------------------------------------

def _require_stream_to_arrays():
    if stream_to_gather_arrays is None:
        raise ImportError("stream_to_gather_arrays could not be imported from segy_tools.gather")


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
    _require_stream_to_arrays()
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
    return plot_wiggle_gather(
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
    _require_stream_to_arrays()
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


def plot_shot_gather_from_stream(st: Stream, kind: Literal["wiggle", "image"] = "wiggle", **kwargs):
    if kind == "wiggle":
        return plot_wiggle_gather_from_stream(st, **kwargs)
    if kind == "image":
        return plot_image_gather_from_stream(st, **kwargs)
    raise ValueError("kind must be 'wiggle' or 'image'.")


def plot_wiggle_gather_from_segy(path, **kwargs):
    return plot_wiggle_gather_from_stream(read(str(path), format="SEGY"), **kwargs)


def plot_image_gather_from_segy(path, **kwargs):
    return plot_image_gather_from_stream(read(str(path), format="SEGY"), **kwargs)


def plot_shot_gather_from_segy(path, kind: Literal["wiggle", "image"] = "wiggle", **kwargs):
    return plot_shot_gather_from_stream(read(str(path), format="SEGY"), kind=kind, **kwargs)


# -----------------------------------------------------------------------------
# Charlie-style / spectral plotting
# -----------------------------------------------------------------------------

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
    fallback_receiver_spacing_m: float | None = None,
    fallback_first_receiver_x_m: float = 0.0,
    fallback_source_x_m: float | None = None,
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
    """Create a frequency-vs-position contour plot from a shot gather."""
    _require_stream_to_arrays()
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

    if data.shape[0] == receiver_x_m.size:
        working = data.copy()
    elif data.shape[1] == receiver_x_m.size:
        working = data.T.copy()
    else:
        raise ValueError(
            "Could not match data dimensions to receiver coordinates. "
            f"data.shape={data.shape}, receiver_x_m.size={receiver_x_m.size}"
        )

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
    raw_spectrum = np.abs(np.fft.rfft(working, axis=1))

    keep = freqs <= max_freq_hz
    freqs = freqs[keep]
    raw_spectrum = raw_spectrum[:, keep]
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
    cs = contour_func(x, freqs, spectrum_to_plot, levels=levels)

    ax.invert_yaxis()
    ax.set_ylim(max_freq_hz, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or "Frequency vs offset")
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Scaled amplitude" if normalize else "Amplitude")

    if source_marker_x is not None and np.nanmin(x) <= source_marker_x <= np.nanmax(x):
        ax.axvline(float(source_marker_x), linestyle=":", linewidth=1.0, label=source_label)

    for xm in marker_positions:
        if np.nanmin(x) <= xm <= np.nanmax(x):
            ax.axvline(float(xm), linestyle="--", linewidth=1.0, label=f"marker {xm:g} m")

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


def plot_source_function_and_spectrum(t, y, label, outfile=None, dpi=160):
    t = np.asarray(t); y = np.asarray(y)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    freqs = np.fft.rfftfreq(len(y), dt); amp = np.abs(np.fft.rfft(y))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(t, y); axes[0].set_xlabel("Time (s or samples if unknown)"); axes[0].set_ylabel("Amplitude"); axes[0].set_title(f"Source time function: {label}")
    axes[1].plot(freqs, amp); axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Amplitude spectrum"); axes[1].set_title("FFT amplitude spectrum"); axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True); fig.savefig(outfile, dpi=dpi); plt.close(fig)
    return fig


def simple_overlay_plot(
    synthetic_file: str | Path,
    real_file: str | Path,
    real_gain: float = 1.0,
    synthetic_gain: float = 1.0,
    real_low: float = 10.0,
    real_high: float = 60.0,
    synthetic_low: float = 10.0,
    synthetic_high: float = 60.0,
    dx: float = 2.0,
    max_time: float = 0.5,
) -> dict:
    
    from .io import read_segy_as_stream
    from .processing import normalize_traces_by_range, bandpass_filter
    from .spectral import plot_trace_spectrum
    
    """Overlay real and synthetic gathers following Charlie's MATLAB workflow."""
    real_st = read_segy_as_stream(real_file)
    synth_st = read_segy_as_stream(synthetic_file)
    real_time, real_data, real_x, _, real_geom = stream_to_gather_arrays(real_st, fallback_receiver_spacing_m=dx)
    synth_time, synth_data, synth_x, _, synth_geom = stream_to_gather_arrays(synth_st, fallback_receiver_spacing_m=dx)

    real_fs = 1.0 / float(np.median(np.diff(real_time)))
    synth_fs = 1.0 / float(np.median(np.diff(synth_time)))

    real_norm = normalize_traces_by_range(real_data, scale=dx)
    synth_norm = normalize_traces_by_range(synth_data, scale=dx)
    real_filt = bandpass_filter(real_norm, real_fs, real_low, real_high, order=2, zerophase=True, axis=1)
    synth_filt = bandpass_filter(synth_norm, synth_fs, synthetic_low, synthetic_high, order=2, zerophase=True, axis=1)

    real_plot = real_gain * real_filt
    synth_plot = synthetic_gain * synth_filt
    figs = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(real_norm, real_fs, trace_index=0, ax=ax, title="Raw real trace spectrum")
    figs["raw_real_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(real_filt, real_fs, trace_index=0, ax=ax, title="Filtered real trace spectrum")
    figs["filtered_real_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot_charlie_style(real_plot, real_time, real_x, ax=ax, color="k", title="Real traces", ylim=(0, max_time))
    figs["real_traces"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(synth_norm, synth_fs, trace_index=0, ax=ax, title="Raw synthetic trace spectrum")
    figs["raw_synthetic_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(synth_filt, synth_fs, trace_index=0, ax=ax, title="Filtered synthetic trace spectrum")
    figs["filtered_synthetic_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot_charlie_style(synth_plot, synth_time, synth_x, ax=ax, color="r", title="Synthetic traces", ylim=(0, max_time))
    figs["synthetic_traces"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot_charlie_style(real_plot, real_time, real_x, ax=ax, color="k", title="Synthetic / real overlay", ylim=(0, max_time))
    wiggle_plot_charlie_style(synth_plot, synth_time, synth_x, ax=ax, color="r", title="Synthetic / real overlay", ylim=(0, max_time))
    figs["overlay"] = fig

    return {
        "real_stream": real_st,
        "synthetic_stream": synth_st,
        "real_processed": real_plot,
        "synthetic_processed": synth_plot,
        "real_geometry": real_geom,
        "synthetic_geometry": synth_geom,
        "figures": figs,
    }




