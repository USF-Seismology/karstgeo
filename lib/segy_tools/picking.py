from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .legacy_charlie import read_charlie_raw
from .processing import bandpass_filter, clip_traces


def threshold_first_arrivals(data, time, fraction=0.05, min_time=None):
    """Pick first threshold crossing on each trace.

    Expects ``data.shape == (n_traces, n_samples)``.
    """
    data = np.asarray(data)
    time = np.asarray(time)
    picks = np.full(data.shape[0], np.nan)
    start_idx = 0
    if min_time is not None:
        idxs = np.where(time >= min_time)[0]
        if len(idxs):
            start_idx = idxs[0]
    for i, tr in enumerate(data):
        y = np.abs(tr)
        threshold = fraction * np.max(y)
        idx = np.where(y[start_idx:] >= threshold)[0]
        if len(idx):
            picks[i] = time[start_idx + idx[0]]
    return picks


def plot_pick_gather(
    data: np.ndarray,
    sampleinterval: float,
    delay: float = 0.0,
    gain: float = 12.0,
    low: Optional[float] = None,
    high: Optional[float] = None,
    clip: float = 0.9,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Axes, np.ndarray, np.ndarray]:
    """Plot a Charlie-style gather for manual picking.

    Expects ``data.shape == (n_traces, n_samples)``.
    """
    data = np.asarray(data, dtype=float)
    fs = 1.0 / float(sampleinterval)
    plot_data = data.copy()
    if low is not None and high is not None:
        plot_data = bandpass_filter(plot_data, fs, low, high, order=2, zerophase=False, axis=1)
    n_traces, n_samples = plot_data.shape
    time = float(delay) + np.arange(n_samples) * float(sampleinterval)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    scaled = np.empty_like(plot_data)
    for i in range(n_traces):
        max_abs = np.nanmax(np.abs(plot_data[i]))
        if max_abs == 0:
            max_abs = 1.0
        trace = gain * plot_data[i] / max_abs
        trace = clip_traces(trace, clip)
        scaled[i] = trace
        ax.plot(time, trace + (i + 1), "k", linewidth=0.8)
    ax.set_title(title or "Pick first breaks")
    ax.set_xlim(-0.01, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel number")
    ax.grid(True, alpha=0.2)
    return ax, scaled, time


def datapicker(
    directory: str | Path,
    namenumber: int,
    gain: float = 12.0,
    low: Optional[float] = None,
    high: Optional[float] = None,
    n_channels: int = 24,
    expected_clicks: Optional[int] = None,
) -> np.ndarray:
    """Interactive first-break picker for Charlie raw files.

    Returns an array with columns ``time_seconds, channel_number``.
    """
    data, sampleinterval, delay = read_charlie_raw(directory, namenumber, n_channels=n_channels)
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pick_gather(data, sampleinterval=sampleinterval, delay=delay, gain=gain, low=low, high=high, ax=ax, title=f"Shot {namenumber}: click first breaks; press Enter when done")
    print("Click first breaks. Press Enter when done.")
    clicks = plt.ginput(n=expected_clicks or -1, timeout=0)
    plt.close(fig)
    picks = np.array(clicks, dtype=float)
    if picks.size == 0:
        return np.empty((0, 2))
    picks[:, 1] = np.round(picks[:, 1])
    return picks
