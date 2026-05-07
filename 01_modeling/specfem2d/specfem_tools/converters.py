from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import numpy as np
from obspy import Stream, Trace

from .io import Timing, read_sem_gather, component_to_channel
from .config import receiver_x, shot_x, survey_output_dir, find_shot_dir
from segy_tools.headers import force_trace_timing_and_headers
from segy_tools.io import read_su_file, write_segy, write_mseed


@dataclass
class Geometry:
    first_receiver_x_m: float = 0.0
    receiver_spacing_m: float = 1.0
    receiver_z_m: float = 0.0
    first_shot_x_m: float = 0.0
    shot_spacing_m: float = 1.0
    source_z_m: float = 0.0
    coordinate_scalar: int = -1000
    elevation_scalar: int = -1000


def receiver_x_from_station(station_idx: int, geom: Geometry) -> float:
    return geom.first_receiver_x_m + (int(station_idx) - 1) * geom.receiver_spacing_m


def source_x_from_shot(shot_number: int, geom: Geometry) -> float:
    return geom.first_shot_x_m + (int(shot_number) - 1) * geom.shot_spacing_m


def sem_gather_to_segy_stream(time, data, station_indices, component, shot_number, source_x_m, geom: Geometry, timing: Timing, network="SY"):
    dt_s = float(timing.dt_s) if timing.dt_s is not None else float(np.median(np.diff(time)))
    t0_s = float(timing.t0_s) if timing.t0_s is not None else float(time[0])
    st = Stream([Trace(data=np.asarray(y, dtype=np.float32)) for y in data])
    rx = [receiver_x_from_station(int(sta), geom) for sta in station_indices]
    return force_trace_timing_and_headers(st, rx, source_x_m, shot_number, dt_s, t0_s, component=component, receiver_z_m=geom.receiver_z_m, source_z_m=geom.source_z_m, network=network, coordinate_scalar=geom.coordinate_scalar, elevation_scalar=geom.elevation_scalar)


def convert_sem_output_to_segy(input_dir, output_dir, component="Z", extension="semv", shot_number=1, source_x_m=None, geom: Geometry | None = None, timing: Timing | None = None, network="SY", verbose=True):
    geom = geom or Geometry()
    timing = timing or Timing()
    if source_x_m is None:
        source_x_m = source_x_from_shot(shot_number, geom)
    time, data, stations = read_sem_gather(input_dir, component=component, extension=extension, timing=timing, verbose=verbose)
    st = sem_gather_to_segy_stream(time, data, stations, component, shot_number, float(source_x_m), geom, timing, network=network)
    channel = component_to_channel(component)
    outpath = Path(output_dir).expanduser() / channel / f"shot_{shot_number:03d}_{channel}_{extension or 'sem'}.segy"
    write_segy(st, outpath)
    if verbose:
        print(f"Wrote {outpath}")
    return outpath


def convert_su_shot_to_segy(su_path, output_path, receiver_x_m, source_x_m, shot_number, dt_s, t0_s=0.0, component="Z"):
    st = read_su_file(su_path, dt_s=dt_s, t0_s=t0_s)
    st = force_trace_timing_and_headers(st, receiver_x_m, source_x_m, shot_number, dt_s, t0_s, component=component)
    return write_segy(st, output_path)
