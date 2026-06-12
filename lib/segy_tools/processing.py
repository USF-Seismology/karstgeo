from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def demean_traces(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """Subtract the mean from each trace.

    The package convention is ``data.shape == (n_traces, n_samples)`` and
    therefore the default sample axis is 1.
    """
    data = np.asarray(data, dtype=float)
    return data - np.mean(data, axis=axis, keepdims=True)


def normalize_traces_by_max(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each trace by maximum absolute amplitude."""
    data = np.asarray(data, dtype=float)
    scale = np.max(np.abs(data), axis=1, keepdims=True)
    return data / (scale + eps)


def normalize_traces_by_range(data: np.ndarray, scale: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Demean and normalize each trace by peak-to-peak range.

    This follows the normalization in Charlie Breithaupt's MATLAB scripts, but
    returns data using the package convention ``(n_traces, n_samples)``.
    """
    data = demean_traces(data, axis=1)
    trace_range = np.ptp(data, axis=1, keepdims=True)
    return scale * data / (trace_range + eps)


def clip_traces(data: np.ndarray, clip: float = 0.9) -> np.ndarray:
    """Clip amplitudes symmetrically."""
    return np.clip(np.asarray(data, dtype=float), -abs(float(clip)), abs(float(clip)))


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 2,
    zerophase: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """Butterworth bandpass filter.

    Parameters
    ----------
    data
        1-D trace or array. For gathers, package convention is
        ``(n_traces, n_samples)``.
    fs
        Sampling frequency in Hz.
    low, high
        Bandpass corner frequencies in Hz.
    order
        Butterworth order.
    zerophase
        If True, use ``scipy.signal.filtfilt``. If False, use ``lfilter``.
    axis
        Sample axis. Defaults to the final axis.
    """
    nyquist = float(fs) / 2.0
    if low <= 0 or high >= nyquist or low >= high:
        raise ValueError(f"Invalid bandpass low={low}, high={high}, Nyquist={nyquist}")
    b, a = butter(order, [float(low) / nyquist, float(high) / nyquist], btype="bandpass")
    filt = filtfilt if zerophase else lfilter
    return filt(b, a, data, axis=axis)

# -----------------------------------------------------------------------------
# Stream/gather processing functions migrated from seismic_gather_utils.
# -----------------------------------------------------------------------------

import copy
from typing import Optional
from obspy import Stream


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array to unit absolute maximum amplitude.

    Parameters
    ----------
    values
        Input array.

    Returns
    -------
    numpy.ndarray
        Copy of ``values`` scaled so that ``max(abs(values)) == 1`` when the
        input contains non-zero values. Zero-valued arrays are returned
        unchanged.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    peak = np.nanmax(np.abs(arr))
    return arr / peak if peak > 0 else arr


def apply_agc(st: Stream, window_sec: float = 0.02, epsilon: float = 1e-10) -> Stream:
    """Apply moving-RMS automatic gain control to each trace in a stream.

    Parameters
    ----------
    st
        Input gather as an ObsPy ``Stream``.
    window_sec
        Moving RMS window length in seconds.
    epsilon
        Minimum RMS value used to avoid division by zero.

    Returns
    -------
    obspy.Stream
        AGC-balanced copy of ``st``.
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    gained = copy.deepcopy(st)
    for tr in gained:
        dt = float(tr.stats.delta)
        data = np.asarray(tr.data, dtype=float)
        win = max(1, int(round(window_sec / dt)))
        if win % 2 == 0:
            win += 1
        half = win // 2
        kernel = np.ones(win, dtype=float)
        padded = np.pad(data**2, half, mode="edge")
        local_energy = np.convolve(padded, kernel, mode="valid")
        rms = np.sqrt(local_energy / win)
        rms[rms < epsilon] = epsilon
        tr.data = (data / rms).astype(np.float32)
    return gained


def apply_linear_time_gain(
    st: Stream,
    power: float = 1.0,
    t0_floor: Optional[float] = None,
) -> Stream:
    """Apply a simple ``t**power`` gain to each trace in a stream.

    Parameters
    ----------
    st
        Input gather as an ObsPy ``Stream``.
    power
        Exponent applied to time in seconds. ``power=1`` is linear time gain.
    t0_floor
        Optional minimum time value used to avoid zeroing the first sample.

    Returns
    -------
    obspy.Stream
        Time-gained copy of ``st``.
    """
    gained = copy.deepcopy(st)
    for tr in gained:
        dt = float(tr.stats.delta)
        times = np.arange(int(tr.stats.npts), dtype=float) * dt
        if t0_floor is None:
            if times.size > 1:
                times[0] = times[1] * 0.1
        else:
            times = np.maximum(times, float(t0_floor))
        tr.data = (np.asarray(tr.data, dtype=float) * (times**power)).astype(np.float32)
    return gained
