"""
Python translation of Charlie Breithaupt's Appendix H MATLAB utilities.

Original MATLAB functions translated:
    1. simpleoverlayplot
    2. 3D and Contour FFT workflow
    3. mst + g_window
    4. datapicker

Notes
-----
- The MATLAB code relied on ReadSegy. Here, SEG-Y/SU reading is done with ObsPy.
- Data arrays are represented as shape (n_samples, n_traces), matching MATLAB convention.
- The interactive first-break picker is implemented with matplotlib's ginput.
- The MST implementation follows Charlie's MATLAB logic closely, but the AARW/frequency-offset
  workflow also includes a simpler FFT-based option that is usually easier to interpret.

Dependencies
------------
    pip install obspy numpy scipy matplotlib
    

Background
----------
I translated it into a Python module in the canvas.

Key changes:

* ReadSegy → ObsPy read()
* MATLAB arrays → NumPy arrays shaped (n_samples, n_traces)
* butter / filtfilt → SciPy
* wiggle plots → matplotlib
* ginput picker → matplotlib interactive picking
* Charlie's custom mst() and g_window() translated directly
* added a simpler FFT frequency-vs-offset option, because it may be easier to use for AARW-style plots

One caution: Charlie's MST code is unusual/custom, so I translated it closely, but I'd trust the simpler FFT/STFT workflow more unless we confirm exactly what that transform was intended to do.   
    
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter

try:
    from obspy import read
except ImportError:  # pragma: no cover
    read = None


@dataclass
class SeismicArrayData:
    """Container for common-shot gather data."""

    data: np.ndarray          # shape: (n_samples, n_traces)
    fs: float                 # sampling frequency, Hz
    dt: float                 # sample interval, seconds
    time: np.ndarray          # shape: (n_samples,)
    offsets: np.ndarray       # shape: (n_traces,), meters
    source_file: Optional[str] = None


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


def read_segy_obspy(
    filename: str | Path,
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
) -> SeismicArrayData:
    """
    Read SEG-Y/SU data using ObsPy and return an array shaped like MATLAB data.

    Parameters
    ----------
    filename
        SEG-Y or SU file path.
    dx
        Receiver spacing in meters if offsets are not supplied.
    offsets
        Optional receiver offsets in meters.
    format
        Optional ObsPy format, e.g. "SEGY" or "SU". Leave None for autodetect.

    Returns
    -------
    SeismicArrayData
    """
    if read is None:
        raise ImportError("ObsPy is required: pip install obspy")

    filename = Path(filename)
    st = read(str(filename), format=format)

    if len(st) == 0:
        raise ValueError(f"No traces found in {filename}")

    # Sort traces by original order unless offsets are supplied elsewhere.
    traces = list(st)
    npts = min(tr.stats.npts for tr in traces)
    data = np.column_stack([tr.data[:npts].astype(float) for tr in traces])

    dt = float(traces[0].stats.delta)
    fs = 1.0 / dt
    time = np.arange(npts) * dt

    if offsets is None:
        offsets = np.arange(data.shape[1]) * dx
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.size != data.shape[1]:
            raise ValueError("offsets length must match number of traces")

    return SeismicArrayData(
        data=data,
        fs=fs,
        dt=dt,
        time=time,
        offsets=np.asarray(offsets, dtype=float),
        source_file=str(filename),
    )


# -----------------------------------------------------------------------------
# Basic processing utilities
# -----------------------------------------------------------------------------


def demean_and_normalize_by_trace(data: np.ndarray, dx: float = 2.0) -> np.ndarray:
    """
    Replicate Charlie's preprocessing:
        1. subtract trace mean
        2. divide each trace by its amplitude range
        3. scale by receiver spacing dx
    """
    data = np.asarray(data, dtype=float)
    demeaned = data - np.mean(data, axis=0, keepdims=True)
    trace_range = np.ptp(demeaned, axis=0, keepdims=True)
    trace_range[trace_range == 0] = 1.0
    return dx * demeaned / trace_range


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 2,
    zerophase: bool = True,
) -> np.ndarray:
    """
    Butterworth bandpass filter, equivalent to MATLAB butter + filtfilt/filter.

    Parameters
    ----------
    data
        Array shaped (n_samples, n_traces) or 1-D trace.
    fs
        Sampling frequency in Hz.
    low, high
        Bandpass corners in Hz.
    order
        Butterworth filter order.
    zerophase
        If True, use filtfilt, matching Charlie's MATLAB code. If False, use lfilter.
    """
    nyq = fs / 2.0
    if low <= 0 or high >= nyq or low >= high:
        raise ValueError(f"Invalid bandpass: low={low}, high={high}, Nyquist={nyq}")
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    func = filtfilt if zerophase else lfilter
    return func(b, a, data, axis=0)


def wiggle_plot(
    data: np.ndarray,
    time: np.ndarray,
    offsets: np.ndarray,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    linewidth: float = 0.8,
    title: Optional[str] = None,
    ylim: Tuple[float, float] = (0.0, 0.5),
) -> plt.Axes:
    """Plot a seismic wiggle gather with time increasing downward."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for i in range(data.shape[1]):
        ax.plot(data[:, i] + offsets[i], time, color=color, linewidth=linewidth)

    ax.invert_yaxis()
    ax.set_xlabel("distance / offset (m)")
    ax.set_ylabel("time (s)")
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim[::-1])
    ax.grid(True, alpha=0.2)
    return ax


def plot_trace_spectrum(
    data: np.ndarray,
    fs: float,
    trace_index: int = 0,
    max_freq: float = 250.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Frequency spectrum",
) -> plt.Axes:
    """Plot single-trace amplitude spectrum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    trace = data[:, trace_index]
    freqs = np.fft.rfftfreq(trace.size, d=1.0 / fs)
    mags = np.abs(np.fft.rfft(trace))
    ax.plot(freqs, mags)
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return ax


# -----------------------------------------------------------------------------
# Simple overlay plot translation
# -----------------------------------------------------------------------------


def simpleoverlayplot(
    synthetic_file: str | Path,
    real_file: str | Path,
    real_gain: float = 1.0,
    synthetic_gain: float = 1.0,
    real_low: float = 10.0,
    real_high: float = 60.0,
    synthetic_low: float = 10.0,
    synthetic_high: float = 60.0,
    dx: float = 2.0,
    real_format: Optional[str] = None,
    synthetic_format: Optional[str] = None,
    max_time: float = 0.5,
) -> dict:
    """
    Python version of Charlie's MATLAB simpleoverlayplot().

    Reads real and synthetic SEG-Y/SU files, demeans, normalizes, bandpass filters,
    applies gain, plots real data, synthetic data, and an overlay.

    Returns a dictionary containing processed arrays and matplotlib figure objects.
    """
    real = read_segy_obspy(real_file, dx=dx, format=real_format)
    synth = read_segy_obspy(synthetic_file, dx=dx, format=synthetic_format)

    real_norm = demean_and_normalize_by_trace(real.data, dx=dx)
    synth_norm = demean_and_normalize_by_trace(synth.data, dx=dx)

    real_filt = bandpass_filter(real_norm, real.fs, real_low, real_high, order=2, zerophase=True)
    synth_filt = bandpass_filter(synth_norm, synth.fs, synthetic_low, synthetic_high, order=2, zerophase=True)

    real_plot_data = real_gain * real_filt
    synth_plot_data = synthetic_gain * synth_filt

    # Use offsets from each file; often both are identical.
    real_offsets = real.offsets
    synth_offsets = synth.offsets

    figs = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(real_norm, real.fs, trace_index=0, ax=ax, title="Raw real trace spectrum")
    figs["raw_real_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(real_filt, real.fs, trace_index=0, ax=ax, title="Filtered real trace spectrum")
    figs["filtered_real_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot(real_plot_data, real.time, real_offsets, ax=ax, color="k", title="Real traces", ylim=(0, max_time))
    figs["real_traces"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(synth_norm, synth.fs, trace_index=0, ax=ax, title="Raw synthetic trace spectrum")
    figs["raw_synthetic_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_trace_spectrum(synth_filt, synth.fs, trace_index=0, ax=ax, title="Filtered synthetic trace spectrum")
    figs["filtered_synthetic_spectrum"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot(synth_plot_data, synth.time, synth_offsets, ax=ax, color="r", title="Synthetic traces", ylim=(0, max_time))
    figs["synthetic_traces"] = fig

    fig, ax = plt.subplots(figsize=(10, 6))
    wiggle_plot(real_plot_data, real.time, real_offsets, ax=ax, color="k", title="Synthetic / real overlay", ylim=(0, max_time))
    wiggle_plot(synth_plot_data, synth.time, synth_offsets, ax=ax, color="r", title="Synthetic / real overlay", ylim=(0, max_time))
    figs["overlay"] = fig

    return {
        "real": real,
        "synthetic": synth,
        "real_processed": real_plot_data,
        "synthetic_processed": synth_plot_data,
        "figures": figs,
    }


# -----------------------------------------------------------------------------
# MST translation
# -----------------------------------------------------------------------------


def g_window(length: int, freq: float, factor: float = 1.0) -> np.ndarray:
    """
    Python version of Charlie's Gaussian window used by mst().

    MATLAB:
        vector(1,:) = [0:Length-1]
        vector(2,:) = [-Length:-1]
        vector = vector.^2
        vector = vector*(-factor*pi^2/freq^2)
        gauss = sum(exp(vector))
    """
    v1 = np.arange(length, dtype=float)
    v2 = np.arange(-length, 0, dtype=float)
    vector = np.vstack([v1, v2]) ** 2
    vector *= -factor * np.pi**2 / freq**2
    return np.sum(np.exp(vector), axis=0)


def mst(h: np.ndarray, t: np.ndarray, factor: float = 1.0, F: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python version of Charlie's mst() function.

    This is a close translation of the MATLAB routine. It is a custom transform,
    not a standard scipy.signal.stft replacement.

    Parameters
    ----------
    h
        1-D trace.
    t
        Time vector in seconds.
    factor
        Gaussian window factor.
    F
        Window frequency parameter from Charlie's code.

    Returns
    -------
    STR, tout, fout
        Complex transform, time vector, frequency vector.
    """
    h = np.asarray(h, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    M = h.size

    H = np.concatenate([np.fft.fft(h), np.fft.fft(h)])
    ncols_half = int(np.ceil(M / 2.0))
    STR = np.zeros((M, ncols_half), dtype=complex)
    STR[:, 0] = np.mean(h)

    for fbin in range(1, int(np.floor((M - 1) / 2.0)) + 1):
        if fbin >= ncols_half:
            break
        T = g_window(M, F, factor)
        segment = H[fbin : fbin + M]
        STR[:, fbin] = np.fft.ifft(segment * T)

    ST = np.fliplr(np.conj(STR[:, 1:]))
    STR = np.column_stack([STR, ST])

    positive_time_indices = np.where(t > 0)[0]
    if positive_time_indices.size < 2:
        dt = t[1] - t[0]
    else:
        aa1 = positive_time_indices[0]
        aa2 = positive_time_indices[0] + 1
        dt = t[aa2] - t[aa1]

    fnyq = 1.0 / (2.0 * dt)
    m, n = STR.shape
    fout = np.linspace(0.0, fnyq, n)
    tout = np.linspace(0.0, np.max(t), m)
    return STR, tout, fout


# -----------------------------------------------------------------------------
# 3-D / contour FFT workflow translation
# -----------------------------------------------------------------------------


def frequency_offset_mst(
    filename: str | Path,
    factor: float = 1.0,
    F_values: Sequence[float] = (30.0,),
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
    zmax: float = 100.0,
    make_plots: bool = True,
) -> dict:
    """
    Python translation of Charlie's "3D and Contour FFT" script using mst().

    For each trace, computes mst(), integrates transform magnitude over time,
    then normalizes to produce frequency-vs-offset amplitudes.
    """
    gather = read_segy_obspy(filename, dx=dx, offsets=offsets, format=format)
    data = gather.data
    fs = gather.fs
    n_samples, n_traces = data.shape
    t = np.arange(n_samples) / fs

    all_integrals = []
    all_fout = []
    figs = []

    for F in F_values:
        trace_integrals = []
        fout_ref = None

        for itr in range(n_traces):
            h = data[:, itr]
            STR, tout, fout = mst(h, t, factor=factor, F=F)
            Ma = np.abs(STR)
            integral = np.trapz(Ma, axis=0)
            trace_integrals.append(integral)
            fout_ref = fout

        integral_matrix = np.column_stack(trace_integrals)  # shape: (freq, trace)
        shifted = integral_matrix - np.nanmin(integral_matrix)
        denom = np.nanmax(np.ptp(shifted, axis=0))
        if denom == 0:
            normalized = shifted
        else:
            normalized = shifted / denom

        all_integrals.append(normalized)
        all_fout.append(fout_ref)

        if make_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            cs = ax.contour(gather.offsets, fout_ref, normalized)
            ax.invert_yaxis()
            ax.set_ylim(zmax, 0)
            ax.set_title(f"Frequency vs offset, F={F}")
            ax.set_xlabel("offset (m)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(cs, ax=ax, label="Scaled amplitude")
            figs.append(fig)

    if make_plots and len(all_integrals) > 0:
        # 3-D surface-like plot for first F value.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        Z = all_integrals[0]
        fout = all_fout[0]
        X, Y = np.meshgrid(gather.offsets, fout)
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        ax.set_title("Frequency vs offset")
        ax.set_xlabel("offset (m)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("Scaled amplitude")
        figs.append(fig)

    return {
        "gather": gather,
        "normalized_integrals": all_integrals,
        "frequencies": all_fout,
        "F_values": list(F_values),
        "figures": figs,
    }


def frequency_offset_fft(
    filename: str | Path,
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
    max_freq: float = 100.0,
    make_plots: bool = True,
) -> dict:
    """
    Simpler frequency-vs-offset representation using standard FFT per trace.

    This is not a direct translation of Charlie's MST transform, but it often gives
    a cleaner AARW-style diagnostic: amplitude spectrum as a function of receiver offset.
    """
    gather = read_segy_obspy(filename, dx=dx, offsets=offsets, format=format)
    data = demean_and_normalize_by_trace(gather.data, dx=dx)

    spec = np.abs(np.fft.rfft(data, axis=0))
    freqs = np.fft.rfftfreq(data.shape[0], d=gather.dt)
    keep = freqs <= max_freq
    spec = spec[keep, :]
    freqs = freqs[keep]

    spec_shifted = spec - np.nanmin(spec)
    denom = np.nanmax(spec_shifted)
    if denom > 0:
        spec_norm = spec_shifted / denom
    else:
        spec_norm = spec_shifted

    figs = []
    if make_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        cs = ax.contour(gather.offsets, freqs, spec_norm)
        ax.invert_yaxis()
        ax.set_title("Frequency vs offset, FFT")
        ax.set_xlabel("offset (m)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(cs, ax=ax, label="Scaled amplitude")
        figs.append(fig)

    return {
        "gather": gather,
        "spectrum": spec_norm,
        "frequencies": freqs,
        "figures": figs,
    }


# -----------------------------------------------------------------------------
# Data picker translation
# -----------------------------------------------------------------------------


def _parse_header_sampleinterval_delay(header_text: str) -> Tuple[float, float]:
    """Parse Charlie-style header text containing Sampleinterval=...Delay=..."""
    sample_match = re.search(r"Sampleinterval=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header_text)
    delay_match = re.search(r"Delay=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header_text)

    if not sample_match:
        raise ValueError("Could not find Sampleinterval= in header")
    if not delay_match:
        raise ValueError("Could not find Delay= in header")

    sampleinterval = float(sample_match.group(1))
    delay = float(delay_match.group(1))
    return sampleinterval, delay


def read_charlie_raw(directory: str | Path, namenumber: int, n_channels: int = 24) -> Tuple[np.ndarray, float, float]:
    """
    Read Charlie's raw binary float format and associated header.

    Expected paths:
        directory/data/<namenumber>
        directory/headers/<namenumber>head.txt

    The original MATLAB used filename='data/xxxx' and replaced xxxx with namenumber.
    """
    directory = Path(directory)
    data_file = directory / "data" / str(namenumber)
    header_file = directory / "headers" / f"{namenumber}head.txt"

    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if not header_file.exists():
        raise FileNotFoundError(header_file)

    raw = np.fromfile(data_file, dtype=np.float32)
    if raw.size % n_channels != 0:
        raise ValueError(f"Data length {raw.size} is not divisible by n_channels={n_channels}")

    data = raw.reshape((-1, n_channels), order="C")
    header_text = header_file.read_text(errors="ignore")
    sampleinterval, delay = _parse_header_sampleinterval_delay(header_text)
    return data, sampleinterval, delay


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
    """
    Plot Charlie-style channel gather for manual picking.

    Returns
    -------
    ax, plotted_data, time
    """
    data = np.asarray(data, dtype=float)
    fs = 1.0 / sampleinterval

    plot_data = data.copy()
    if low is not None and high is not None:
        plot_data = bandpass_filter(plot_data, fs, low, high, order=2, zerophase=False)

    n_samples, n_channels = plot_data.shape
    time = delay + np.arange(n_samples) * sampleinterval

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    scaled = np.empty_like(plot_data)
    for i in range(n_channels):
        max_abs = np.nanmax(np.abs(plot_data[:, i]))
        if max_abs == 0:
            max_abs = 1.0
        trace = gain * plot_data[:, i] / max_abs
        trace = np.clip(trace, -clip, clip)
        scaled[:, i] = trace
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
    """
    Python version of Charlie's datapicker().

    This reads Charlie's raw float/header format, plots the gather, and lets the user
    click first breaks. The output is an array with columns:
        time_seconds, channel_number

    Usage
    -----
        picks = datapicker("/path/to/directory", 23, gain=12, low=10, high=100)

    Click first breaks in the matplotlib window. Press Enter when done.
    """
    data, sampleinterval, delay = read_charlie_raw(directory, namenumber, n_channels=n_channels)

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pick_gather(
        data,
        sampleinterval=sampleinterval,
        delay=delay,
        gain=gain,
        low=low,
        high=high,
        ax=ax,
        title=f"Shot {namenumber}: click first breaks; press Enter when done",
    )

    print("Click first breaks. Press Enter when done.")
    clicks = plt.ginput(n=expected_clicks or -1, timeout=0)
    plt.close(fig)

    picks = np.array(clicks, dtype=float)
    if picks.size == 0:
        return np.empty((0, 2))

    picks[:, 1] = np.round(picks[:, 1])
    return picks


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: overlay real and synthetic SEG-Y/SU gathers
    # result = simpleoverlayplot(
    #     synthetic_file="synthetic.sgy",
    #     real_file="real.sgy",
    #     real_gain=1,
    #     synthetic_gain=1,
    #     real_low=10,
    #     real_high=40,
    #     synthetic_low=10,
    #     synthetic_high=40,
    #     dx=2,
    # )
    # plt.show()

    # Example 2: frequency-vs-offset plot using Charlie's MST-like transform
    # out = frequency_offset_mst("2short.sgy", factor=1, F_values=[30], dx=2)
    # plt.show()

    # Example 3: simpler FFT-based frequency-vs-offset plot
    # out = frequency_offset_fft("2short.sgy", dx=2, max_freq=100)
    # plt.show()

    # Example 4: first-break picker for Charlie raw data directory
    # picks = datapicker("/path/to/raw_directory", 23, gain=12, low=10, high=100)
    # print(picks)
    pass
