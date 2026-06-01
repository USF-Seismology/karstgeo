"""General seismic gather utilities used during the karst SPECfEM2D analysis.

These functions are deliberately not SPECfEM2D-specific. They operate mainly on
ObsPy ``Stream`` objects or simple NumPy gather arrays and are good candidates
for later migration into the project-level ``segy_tools`` package.

Conventions
-----------
ObsPy ``Stream`` objects are assumed to contain one gather with one trace per
receiver. When converted to NumPy arrays, the convention used here is::

    data.shape == (n_traces, n_samples)

This is the natural convention for plotting shot gathers with receiver/offset on
one axis and time on the other.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import Stream, Trace
from obspy.io.segy.segy import SEGYTraceHeader


def stream_to_gather_arrays(
    st: Stream,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    sort_keys: Sequence[str] = ("station", "channel"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single-component ObsPy stream into gather arrays.

    Parameters
    ----------
    st
        ObsPy stream containing one shot gather.
    receiver_spacing_m
        Receiver spacing used when trace headers do not provide coordinates.
    first_receiver_x_m
        Coordinate of the first receiver in metres.
    sort_keys
        Trace stat keys used to sort the stream before conversion.

    Returns
    -------
    time_s, data, receiver_x_m
        ``time_s`` has shape ``(n_samples,)``; ``data`` has shape
        ``(n_traces, n_samples)``; ``receiver_x_m`` has shape ``(n_traces,)``.
    """
    st_work = st.copy()
    if sort_keys:
        st_work.sort(keys=list(sort_keys))

    if len(st_work) == 0:
        raise ValueError("Cannot convert an empty ObsPy Stream to gather arrays.")

    npts = min(int(tr.stats.npts) for tr in st_work)
    dt = float(st_work[0].stats.delta)

    data = np.vstack([np.asarray(tr.data[:npts], dtype=float) for tr in st_work])
    time_s = np.arange(npts, dtype=float) * dt
    receiver_x_m = first_receiver_x_m + np.arange(len(st_work), dtype=float) * receiver_spacing_m
    return time_s, data, receiver_x_m


def read_segy_gather(
    segy_file: str | Path,
    unpack_trace_headers: bool = True,
) -> Stream:
    """Read a SEG-Y gather with ObsPy.

    Parameters
    ----------
    segy_file
        Path to a SEG-Y file.
    unpack_trace_headers
        Passed to ``obspy.read``. ``True`` is useful for geometry inspection.

    Returns
    -------
    obspy.Stream
        Stream containing the SEG-Y traces.
    """
    return obspy.read(str(segy_file), format="SEGY", unpack_trace_headers=unpack_trace_headers)


def difference_segy_gathers(
    segy_a: str | Path,
    segy_b: str | Path,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    output_segy_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read two SEG-Y gathers and compute a trace-by-trace difference.

    The two gathers are trimmed to their common number of traces and samples
    before differencing.

    Parameters
    ----------
    segy_a, segy_b
        Input SEG-Y gather paths. The returned difference is ``A - B``.
    receiver_spacing_m, first_receiver_x_m
        Fallback receiver geometry used for the returned coordinate vector.
    output_segy_path
        Optional path to write the difference gather as SEG-Y.

    Returns
    -------
    time_s, data_a, data_b, diff, receiver_x_m
        Arrays with ``data_a``, ``data_b`` and ``diff`` shaped
        ``(n_traces, n_samples)``.
    """
    st_a = read_segy_gather(segy_a)
    st_b = read_segy_gather(segy_b)

    original_dt = float(st_a[0].stats.delta)
    original_starttime = st_a[0].stats.starttime

    time_a, data_a, rx_a = stream_to_gather_arrays(
        st_a,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )
    time_b, data_b, _ = stream_to_gather_arrays(
        st_b,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
    )

    ntr = min(data_a.shape[0], data_b.shape[0])
    npts = min(data_a.shape[1], data_b.shape[1])

    time_s = time_a[:npts]
    receiver_x_m = rx_a[:ntr]
    data_a = data_a[:ntr, :npts]
    data_b = data_b[:ntr, :npts]
    diff = data_a - data_b

    if output_segy_path is not None:
        diff_stream = gather_arrays_to_stream(
            data=diff,
            dt_s=original_dt,
            starttime=original_starttime,
            receiver_x_m=receiver_x_m,
            station_prefix="D",
        )
        output_segy_path = Path(output_segy_path)
        output_segy_path.parent.mkdir(parents=True, exist_ok=True)
        diff_stream.write(str(output_segy_path), format="SEGY", data_encoding=1)

    return time_s, data_a, data_b, diff, receiver_x_m


def gather_arrays_to_stream(
    data: np.ndarray,
    dt_s: float,
    starttime=None,
    receiver_x_m: Optional[Sequence[float]] = None,
    station_prefix: str = "R",
) -> Stream:
    """Convert a gather array to a minimal ObsPy stream.

    Parameters
    ----------
    data
        Gather matrix shaped ``(n_traces, n_samples)``.
    dt_s
        Sample interval in seconds.
    starttime
        Optional ObsPy UTCDateTime. If omitted, ObsPy will use its default.
    receiver_x_m
        Optional receiver coordinates in metres, written into simple SEG-Y trace
        headers when possible.
    station_prefix
        Prefix for synthetic station names.

    Returns
    -------
    obspy.Stream
        Stream representation of the gather.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be shaped (n_traces, n_samples).")

    ntr, _ = data.shape
    if receiver_x_m is None:
        receiver_x_m = np.arange(ntr, dtype=float)
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)

    st = Stream()
    for i in range(ntr):
        tr = Trace(data=np.asarray(data[i, :], dtype=np.float32))
        tr.stats.delta = float(dt_s)
        if starttime is not None:
            tr.stats.starttime = starttime
        tr.stats.station = f"{station_prefix}{i + 1:04d}"
        tr.stats.segy = {"trace_header": SEGYTraceHeader()}
        tr.stats.segy["trace_header"].trace_sequence_number_within_line = i + 1
        tr.stats.segy["trace_header"].trace_sequence_number_within_seismic_reely = i + 1
        tr.stats.segy["trace_header"].original_field_record_number = 1
        tr.stats.segy["trace_header"].trace_number_within_the_original_field_record = i + 1
        tr.stats.segy["trace_header"].distance_from_center_of_source_to_receiver_group = int(receiver_x_m[i])
        st.append(tr)
    return st


def apply_nmo_hyperbola_scan(
    st: Stream,
    test_velocity_mps: float,
    source_x_m: float,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
) -> Stream:
    """Flatten hyperbolic moveout for a trial diffraction velocity.

    This is a simple diagnostic transform for diffraction analysis. For each
    trace, amplitudes are sampled along::

        t_curve = sqrt(t0**2 + offset**2 / velocity**2)

    and placed at ``t0`` in the output trace. A diffraction with the chosen
    velocity should become more nearly horizontal after correction.

    Parameters
    ----------
    st
        Input shot gather as an ObsPy stream.
    test_velocity_mps
        Trial diffraction velocity in m/s.
    source_x_m
        Source coordinate along profile in metres.
    receiver_spacing_m, first_receiver_x_m
        Fallback receiver geometry.

    Returns
    -------
    obspy.Stream
        NMO-scanned copy of the input stream.
    """
    if test_velocity_mps <= 0:
        raise ValueError("test_velocity_mps must be positive.")

    nmo_stream = copy.deepcopy(st)
    for idx, tr in enumerate(nmo_stream):
        dt = float(tr.stats.delta)
        npts = int(tr.stats.npts)
        receiver_x_m = first_receiver_x_m + idx * receiver_spacing_m
        offset_m = abs(receiver_x_m - source_x_m)
        times = np.arange(npts, dtype=float) * dt
        curve_times = np.sqrt(times**2 + (offset_m / test_velocity_mps) ** 2)
        tr.data = np.interp(curve_times, times, np.asarray(tr.data, dtype=float), left=0.0, right=0.0).astype(np.float32)
        tr.stats.distance = float(offset_m)
    return nmo_stream


def plot_nmo_velocity_grid(
    st: Stream,
    trial_velocities_mps: Sequence[float],
    source_x_m: float,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
    offset_range_m: tuple[float, float] = (-50.0, 50.0),
    clip_percentile: float = 95.0,
    cols_per_row: int = 3,
) -> plt.Figure:
    """Plot an NMO/diffraction velocity scan for a list of trial velocities.

    Parameters
    ----------
    st
        Input shot gather.
    trial_velocities_mps
        Trial velocities to test.
    source_x_m
        Source coordinate along profile.
    receiver_spacing_m, first_receiver_x_m
        Fallback receiver geometry.
    offset_range_m
        Display window in relative offset from the source.
    clip_percentile
        Percentile used for symmetric amplitude clipping.
    cols_per_row
        Number of subplot columns.

    Returns
    -------
    matplotlib.figure.Figure
        Velocity-grid figure.
    """
    velocities = list(trial_velocities_mps)
    if not velocities:
        raise ValueError("trial_velocities_mps cannot be empty.")

    nrows = int(np.ceil(len(velocities) / cols_per_row))
    fig, axes_grid = plt.subplots(nrows, cols_per_row, figsize=(6 * cols_per_row, 4 * nrows), sharey=True)
    axes = np.ravel(np.atleast_1d(axes_grid))

    for idx, (ax, velocity) in enumerate(zip(axes, velocities)):
        flattened = apply_nmo_hyperbola_scan(
            st=st,
            test_velocity_mps=float(velocity),
            source_x_m=source_x_m,
            receiver_spacing_m=receiver_spacing_m,
            first_receiver_x_m=first_receiver_x_m,
        )

        traces = []
        rel_offsets = []
        for tr_idx, tr in enumerate(flattened):
            rec_x = first_receiver_x_m + tr_idx * receiver_spacing_m
            rel_offset = rec_x - source_x_m
            if offset_range_m[0] <= rel_offset <= offset_range_m[1]:
                traces.append(np.asarray(tr.data, dtype=float))
                rel_offsets.append(rel_offset)

        if not traces:
            raise ValueError(f"No traces found within offset_range_m={offset_range_m}.")

        gather = np.vstack(traces)
        clip = np.percentile(np.abs(gather), clip_percentile)
        if clip == 0:
            clip = 1.0
        extent = [rel_offsets[0], rel_offsets[-1], gather.shape[1], 0]
        ax.imshow(gather.T, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
        ax.set_title(f"V = {velocity:g} m/s")
        ax.set_xlabel("Relative offset (m)")
        if idx % cols_per_row == 0:
            ax.set_ylabel("Time sample index")

    for ax in axes[len(velocities):]:
        ax.set_axis_off()

    fig.tight_layout()
    return fig


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
        If True, apply a raised-cosine transition between ``min_velocity_mps``
        and ``min_velocity_mps + taper_width_mps``.
    taper_width_mps
        Width of the velocity transition zone in m/s.

    Returns
    -------
    obspy.Stream
        Filtered copy of the input stream.

    Notes
    -----
    Use cautiously for diffraction analysis: parts of diffraction wings that
    share the muted apparent-velocity range will also be attenuated.
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


def apply_agc(st: Stream, window_sec: float = 0.02, epsilon: float = 1e-10) -> Stream:
    """Apply moving RMS automatic gain control to each trace.

    Parameters
    ----------
    st
        Input gather.
    window_sec
        RMS window length in seconds.
    epsilon
        Small floor value used to avoid division by zero.

    Returns
    -------
    obspy.Stream
        AGC-balanced copy of the stream.
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


def apply_linear_time_gain(st: Stream, power: float = 1.0, t0_floor: Optional[float] = None) -> Stream:
    """Apply a simple ``t**power`` gain to each trace.

    Parameters
    ----------
    st
        Input gather.
    power
        Exponent applied to time in seconds. ``power=1`` is linear time gain.
    t0_floor
        Optional minimum time value for the first sample to avoid zeroing it.

    Returns
    -------
    obspy.Stream
        Time-gained copy of the stream.
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


def ricker_wavelet(f0_hz: float, dt_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate a zero-phase Ricker wavelet centered at time zero.

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
        Time vector and normalized wavelet samples.
    """
    if f0_hz <= 0 or dt_s <= 0 or duration_s <= 0:
        raise ValueError("f0_hz, dt_s and duration_s must all be positive.")
    time_s = np.arange(-duration_s / 2.0, duration_s / 2.0 + dt_s, dt_s)
    arg = (np.pi * f0_hz * time_s) ** 2
    wavelet = (1.0 - 2.0 * arg) * np.exp(-arg)
    return time_s, normalize(wavelet)


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array to unit absolute maximum amplitude."""
    arr = np.asarray(values, dtype=float)
    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak > 0:
        return arr / peak
    return arr


def gaussian_taper(time_s: np.ndarray, sigma_s: float) -> np.ndarray:
    """Return a Gaussian taper centered on zero time.

    Parameters
    ----------
    time_s
        Time vector in seconds.
    sigma_s
        Standard deviation of the Gaussian in seconds.
    """
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive.")
    time_s = np.asarray(time_s, dtype=float)
    return np.exp(-0.5 * (time_s / sigma_s) ** 2)
