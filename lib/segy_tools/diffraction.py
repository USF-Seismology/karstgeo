"""Diffraction-oriented gather transforms and diagnostic plots."""

from __future__ import annotations

import copy
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream


def apply_nmo_hyperbola_scan(
    st: Stream,
    test_velocity_mps: float,
    source_x_m: float,
    receiver_spacing_m: float = 1.0,
    first_receiver_x_m: float = 0.0,
) -> Stream:
    """Flatten hyperbolic moveout for a trial diffraction velocity.

    For each trace, amplitudes are sampled along
    ``t_curve = sqrt(t0**2 + offset**2 / velocity**2)`` and written at ``t0``
    in the output trace. A diffraction with the chosen velocity should become
    more nearly horizontal after this correction.

    Parameters
    ----------
    st
        Input shot gather as an ObsPy ``Stream``.
    test_velocity_mps
        Trial diffraction velocity in m/s.
    source_x_m
        Source coordinate along the profile in metres.
    receiver_spacing_m, first_receiver_x_m
        Fallback receiver geometry used when explicit coordinates are not
        supplied.

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
        tr.data = np.interp(
            curve_times,
            times,
            np.asarray(tr.data, dtype=float),
            left=0.0,
            right=0.0,
        ).astype(np.float32)
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
    """Plot a diffraction/NMO velocity scan for a list of trial velocities.

    Parameters
    ----------
    st
        Input shot gather.
    trial_velocities_mps
        Trial velocities to test.
    source_x_m
        Source coordinate along profile in metres.
    receiver_spacing_m, first_receiver_x_m
        Fallback receiver geometry.
    offset_range_m
        Display window in receiver coordinate relative to the source.
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
    fig, axes_grid = plt.subplots(
        nrows,
        cols_per_row,
        figsize=(6 * cols_per_row, 4 * nrows),
        sharey=True,
    )
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
