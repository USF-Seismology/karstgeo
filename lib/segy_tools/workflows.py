from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .gather import stream_to_gather_arrays
from .io import read_segy_as_stream
from .processing import normalize_traces_by_range, bandpass_filter
from .plotting import wiggle_plot_charlie_style
from .spectral import plot_trace_spectrum


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

# Backwards-compatible spelling from Charlie translation.
simpleoverlayplot = simple_overlay_plot
