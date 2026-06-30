"""Diffraction-oriented gather transforms and diagnostic plots.

This module operates on ObsPy ``Stream`` shot gathers.  Receiver/source
geometry is read from SEG-Y headers or trace ``stats`` via
``segy_tools.gather.stream_to_gather_arrays``.  Regular receiver spacing is
kept only as a fallback for legacy streams that do not carry geometry.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream

from .gather import stream_to_gather_arrays
from .io import apply_geometry_to_stream


def _geometry_from_stream(
    st: Stream,
    *,
    source_x_m: Optional[float] = None,
    receiver_spacing_m: Optional[float] = None,
    first_receiver_x_m: float = 0.0,
    component: Optional[str] = None,
    sort_by: str | None = "receiver_x",
):
    """Return gather arrays and geometry using headers/stats first.

    Parameters named ``receiver_spacing_m`` and ``first_receiver_x_m`` are
    legacy fallbacks only.  When SEG-Y headers contain ``gx``/``sx`` values or
    traces contain ``stats.receiver_x_m``/``stats.source_x_m``, those explicit
    coordinates are used instead.
    """
    return stream_to_gather_arrays(
        st,
        sort_by=sort_by,
        component=component,
        fallback_receiver_spacing_m=receiver_spacing_m,
        fallback_first_receiver_x_m=first_receiver_x_m,
        fallback_source_x_m=source_x_m,
    )


def apply_nmo_hyperbola_scan(
    st: Stream,
    test_velocity_mps: float,
    source_x_m: Optional[float] = None,
    receiver_spacing_m: Optional[float] = None,
    first_receiver_x_m: float = 0.0,
    component: Optional[str] = None,
    sort_by: str | None = "receiver_x",
) -> Stream:
    """Flatten hyperbolic diffraction moveout for a trial velocity.

    For each trace, amplitudes are sampled along
    ``t_curve = sqrt(t0**2 + offset**2 / velocity**2)`` and written at ``t0``
    in the output trace.  A diffraction with the chosen velocity should become
    more nearly horizontal after this correction.

    Geometry is obtained from trace stats / SEG-Y headers.  The legacy regular
    geometry arguments are used only if explicit receiver coordinates are not
    available.

    Parameters
    ----------
    st
        Input shot gather as an ObsPy ``Stream``.
    test_velocity_mps
        Trial diffraction velocity in m/s.
    source_x_m
        Optional fallback source coordinate along the profile in metres.  If
        omitted, source coordinates are read from trace headers/stats when
        available.
    receiver_spacing_m, first_receiver_x_m
        Optional fallback receiver geometry for legacy streams without
        coordinates.  If ``receiver_spacing_m`` is omitted and no geometry is
        present, traces fall back to index coordinates.
    component
        Optional component selector passed to ``stream_to_gather_arrays``.
    sort_by
        Trace ordering.  Defaults to receiver coordinate order.

    Returns
    -------
    obspy.Stream
        NMO-scanned copy of the selected/sorted input stream.
    """
    if test_velocity_mps <= 0:
        raise ValueError("test_velocity_mps must be positive.")

    time, data, receiver_x_m, inferred_source_x_m, geom = _geometry_from_stream(
        st,
        source_x_m=source_x_m,
        receiver_spacing_m=receiver_spacing_m,
        first_receiver_x_m=first_receiver_x_m,
        component=component,
        sort_by=sort_by,
    )

    if inferred_source_x_m is None:
        raise ValueError(
            "No source coordinate found. Provide source_x_m or write source "
            "coordinates into the Stream/SEG-Y headers."
        )

    receiver_x_m = np.asarray(receiver_x_m, dtype=float)
    offsets_m = receiver_x_m - float(inferred_source_x_m)
    abs_offsets_m = np.abs(offsets_m)
    dt = float(time[1] - time[0]) if len(time) > 1 else float(st[0].stats.delta)
    npts = len(time)

    out = Stream()
    st_selected = copy.deepcopy(st)
    # Reuse stream_to_gather_arrays sorting by rebuilding traces from the sorted
    # data array. This avoids relying on original trace order when receivers are
    # irregular and requested sorted by receiver_x/offset.
    for i in range(data.shape[0]):
        # Preserve original metadata approximately by taking the i-th trace from
        # the sorted order if possible; data and geometry are authoritative.
        tr = copy.deepcopy(st_selected[i if i < len(st_selected) else -1])
        curve_times = np.sqrt(time**2 + (abs_offsets_m[i] / test_velocity_mps) ** 2)
        tr.data = np.interp(
            curve_times,
            time,
            np.asarray(data[i], dtype=float),
            left=0.0,
            right=0.0,
        ).astype(np.float32)
        tr.stats.delta = dt
        tr.stats.npts = npts
        tr.stats.distance = float(abs_offsets_m[i])
        tr.stats.receiver_x_m = float(receiver_x_m[i])
        tr.stats.source_x_m = float(inferred_source_x_m)
        out += tr

    # Re-attach coherent geometry headers/stats where segy_tools.io is available.
    try:
        receiver_z = geom.get("receiver_z_m", None)
        source_z = geom.get("source_z_m", None)
        if isinstance(source_z, np.ndarray):
            finite = source_z[np.isfinite(source_z)]
            source_z_value = float(finite[0]) if finite.size else None
        else:
            source_z_value = source_z
        out = apply_geometry_to_stream(
            out,
            receiver_x_m=receiver_x_m,
            source_x_m=float(inferred_source_x_m),
            receiver_z_m=receiver_z,
            source_z_m=source_z_value,
        )
    except Exception:
        # Geometry stats above are sufficient for plotting/analysis; do not fail
        # the NMO transform only because SEG-Y header attachment failed.
        pass

    return out


def plot_nmo_velocity_grid(
    st: Stream,
    trial_velocities_mps: Sequence[float],
    source_x_m: Optional[float] = None,
    receiver_spacing_m: Optional[float] = None,
    first_receiver_x_m: float = 0.0,
    offset_range_m: tuple[float, float] = (-50.0, 50.0),
    clip_percentile: float = 95.0,
    cols_per_row: int = 3,
    component: Optional[str] = None,
    sort_by: str | None = "receiver_x",
) -> plt.Figure:
    """Plot a diffraction/NMO velocity scan for trial velocities.

    Receiver coordinates are read from headers/stats when present, so irregular
    receiver spacing is handled correctly.  The regular-spacing arguments are
    legacy fallbacks only.
    """
    velocities = list(trial_velocities_mps)
    if not velocities:
        raise ValueError("trial_velocities_mps cannot be empty.")
    if cols_per_row <= 0:
        raise ValueError("cols_per_row must be positive.")

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
            component=component,
            sort_by=sort_by,
        )

        time, data, receiver_x_m, inferred_source_x_m, _geom = _geometry_from_stream(
            flattened,
            source_x_m=source_x_m,
            receiver_spacing_m=receiver_spacing_m,
            first_receiver_x_m=first_receiver_x_m,
            component=None,
            sort_by="receiver_x",
        )
        if inferred_source_x_m is None:
            raise ValueError(
                "No source coordinate found. Provide source_x_m or write source "
                "coordinates into the Stream/SEG-Y headers."
            )

        rel_offsets = np.asarray(receiver_x_m, dtype=float) - float(inferred_source_x_m)
        mask = (rel_offsets >= offset_range_m[0]) & (rel_offsets <= offset_range_m[1])
        if not np.any(mask):
            raise ValueError(f"No traces found within offset_range_m={offset_range_m}.")

        gather = np.asarray(data[mask], dtype=float)
        rel_offsets_plot = rel_offsets[mask]
        clip = np.percentile(np.abs(gather), clip_percentile)
        if not np.isfinite(clip) or clip == 0:
            clip = 1.0

        # Use imshow for compact grid plots.  For strongly irregular receiver
        # spacing this is a coordinate-labelled image, not a true variable-cell
        # mesh; the NMO correction itself still uses the true receiver offsets.
        extent = [rel_offsets_plot[0], rel_offsets_plot[-1], time[-1], time[0]]
        ax.imshow(
            gather.T,
            aspect="auto",
            cmap="seismic",
            vmin=-clip,
            vmax=clip,
            extent=extent,
        )
        ax.set_title(f"V = {velocity:g} m/s")
        ax.set_xlabel("Receiver x - source x (m)")
        if idx % cols_per_row == 0:
            ax.set_ylabel("Time (s)")

    for ax in axes[len(velocities):]:
        ax.set_axis_off()

    fig.tight_layout()
    return fig
