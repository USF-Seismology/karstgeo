from __future__ import annotations

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader


def component_to_channel(component: str) -> str:
    c = component.upper()
    if c in ("X", "BXX"):
        return "BXX"
    if c in ("Z", "BXZ"):
        return "BXZ"
    if c in ("Y", "BXY"):
        return "BXY"
    return c


def scaled_int(value_m: float, scalar: int) -> int:
    if scalar == 0:
        scalar = 1
    return int(round(float(value_m) * abs(int(scalar))))


def make_trace_header(
    station_idx: int,
    receiver_x_m: float,
    source_x_m: float,
    shot_number: int,
    dt_s: float,
    npts: int,
    receiver_z_m: float = 0.0,
    source_z_m: float = 0.0,
    coordinate_scalar: int = -1000,
    elevation_scalar: int = -1000,
) -> SEGYTraceHeader:
    h = SEGYTraceHeader()
    h.trace_sequence_number_within_line = int(station_idx)
    h.trace_sequence_number_within_segy_file = int(station_idx)
    h.original_field_record_number = int(shot_number)
    h.energy_source_point_number = int(shot_number)
    h.trace_number_within_the_original_field_record = int(station_idx)

    h.scalar_to_be_applied_to_all_coordinates = int(coordinate_scalar)
    h.scalar_to_be_applied_to_all_elevations_and_depths = int(elevation_scalar)
    h.coordinate_units = 1

    h.source_coordinate_x = scaled_int(source_x_m, coordinate_scalar)
    h.source_coordinate_y = 0
    h.group_coordinate_x = scaled_int(receiver_x_m, coordinate_scalar)
    h.group_coordinate_y = 0

    # z is assumed positive downward.
    h.receiver_group_elevation = scaled_int(-receiver_z_m, elevation_scalar)
    h.surface_elevation_at_source = scaled_int(-source_z_m, elevation_scalar)
    h.source_depth_below_surface = scaled_int(source_z_m, elevation_scalar)

    offset_m = receiver_x_m - source_x_m
    h.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = scaled_int(offset_m, coordinate_scalar)

    # ObsPy name says ms, SEG-Y field is microseconds.
    h.sample_interval_in_ms_for_this_trace = int(round(float(dt_s) * 1_000_000))
    h.number_of_samples_in_this_trace = int(npts)
    h.trace_identification_code = 1
    h.number_of_vertically_summed_traces_yielding_this_trace = 1
    h.number_of_horizontally_stacked_traces_yielding_this_trace = 1
    return h


def force_trace_timing_and_headers(
    stream: Stream,
    receiver_x_m,
    source_x_m: float,
    shot_number: int,
    dt_s: float,
    t0_s: float = 0.0,
    component: str = "Z",
    receiver_z_m: float = 0.0,
    source_z_m: float = 0.0,
    network: str = "SY",
    coordinate_scalar: int = -1000,
    elevation_scalar: int = -1000,
) -> Stream:
    out = Stream()
    receiver_x_m = np.asarray(receiver_x_m, dtype=float)
    for i, tr in enumerate(stream):
        new = tr.copy()
        new.stats.delta = float(dt_s)
        new.stats.starttime = UTCDateTime(1970, 1, 1) + float(t0_s)
        new.stats.network = network
        new.stats.station = f"S{i + 1:04d}"
        new.stats.location = f"{int(shot_number) % 100:02d}"
        new.stats.channel = component_to_channel(component)
        new.stats.segy = AttribDict()
        new.stats.segy.trace_header = make_trace_header(
            station_idx=i + 1,
            receiver_x_m=float(receiver_x_m[i]),
            source_x_m=float(source_x_m),
            shot_number=int(shot_number),
            dt_s=float(dt_s),
            npts=new.stats.npts,
            receiver_z_m=receiver_z_m,
            source_z_m=source_z_m,
            coordinate_scalar=coordinate_scalar,
            elevation_scalar=elevation_scalar,
        )
        out.append(new)
    return out
