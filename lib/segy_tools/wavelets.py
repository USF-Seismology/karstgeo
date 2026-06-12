"""Source wavelets and simple tapers for active-source seismic modelling."""

from __future__ import annotations

import numpy as np

from .processing import normalize


def ricker_wavelet(f0_hz: float, dt_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate a zero-phase Ricker wavelet centred at time zero.

    Parameters
    ----------
    f0_hz
        Dominant frequency in Hz.
    dt_s
        Sample interval in seconds.
    duration_s
        Total wavelet duration in seconds.

    Returns
    -------
    time_s, wavelet
        Time vector and unit-normalized wavelet samples.
    """
    if f0_hz <= 0 or dt_s <= 0 or duration_s <= 0:
        raise ValueError("f0_hz, dt_s and duration_s must all be positive.")
    time_s = np.arange(-duration_s / 2.0, duration_s / 2.0 + dt_s, dt_s)
    arg = (np.pi * f0_hz * time_s) ** 2
    wavelet = (1.0 - 2.0 * arg) * np.exp(-arg)
    return time_s, normalize(wavelet)


def gaussian_taper(time_s: np.ndarray, sigma_s: float) -> np.ndarray:
    """Return a Gaussian taper centred on zero time.

    Parameters
    ----------
    time_s
        Time vector in seconds.
    sigma_s
        Standard deviation of the Gaussian in seconds.

    Returns
    -------
    numpy.ndarray
        Gaussian taper evaluated at ``time_s``.
    """
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive.")
    time_s = np.asarray(time_s, dtype=float)
    return np.exp(-0.5 * (time_s / sigma_s) ** 2)
