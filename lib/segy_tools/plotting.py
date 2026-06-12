from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Literal



def normalize_traces(data, eps=1e-12):
    data = np.asarray(data, dtype=float)
    scale = np.max(np.abs(data), axis=1, keepdims=True)
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

    clip = np.percentile(np.abs(dd), clip_percentile)
    if clip > 0:
        dd = np.clip(dd, -clip, clip)
    if normalize:
        dd = normalize_traces(dd)

    if len(xx) > 1:
        dx = np.median(np.diff(np.sort(xx)))
        if dx == 0:
            dx = 1.0
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
    clip = np.percentile(np.abs(dd), clip_percentile)
    if clip <= 0:
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
        clip = np.percentile(np.abs(d), clip_percentile) or 1.0
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


