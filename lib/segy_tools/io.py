from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from obspy import read, Stream, UTCDateTime


@dataclass
class SeismicArrayData:
    """Container for common-shot gather data.

    The package convention is ``data.shape == (n_traces, n_samples)``.
    """

    data: np.ndarray
    fs: float
    dt: float
    time: np.ndarray
    offsets: np.ndarray
    source_file: Optional[str] = None


def read_segy_as_stream(path, *, unpack_trace_headers=True) -> Stream:
    """Read a SEG-Y file as an ObsPy Stream."""
    return read(str(path), format="SEGY", unpack_trace_headers=unpack_trace_headers)


def read_su_file(path, dt_s=None, t0_s=0.0, byteorder="<") -> Stream:
    """Read a Seismic Unix file as an ObsPy Stream."""
    st = read(str(path), format="SU", byteorder=byteorder)
    if dt_s is not None:
        for tr in st:
            tr.stats.delta = float(dt_s)
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(t0_s)
    return st


def read_segy_obspy(
    filename: str | Path,
    dx: float = 2.0,
    offsets: Optional[Sequence[float]] = None,
    format: Optional[str] = None,
) -> SeismicArrayData:
    """Read SEG-Y/SU data using ObsPy and return array data.

    This function preserves the public interface from ``charlie_lib.py`` but
    standardizes the returned data to ``(n_traces, n_samples)``.
    """
    filename = Path(filename)
    st = read(str(filename), format=format) if format is not None else read(str(filename))
    if len(st) == 0:
        raise ValueError(f"No traces found in {filename}")
    npts = min(tr.stats.npts for tr in st)
    data = np.vstack([tr.data[:npts].astype(float) for tr in st])
    dt = float(st[0].stats.delta)
    fs = 1.0 / dt
    time = np.arange(npts) * dt
    if offsets is None:
        offsets = np.arange(data.shape[0]) * dx
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.size != data.shape[0]:
            raise ValueError("offsets length must match number of traces")
    return SeismicArrayData(data=data, fs=fs, dt=dt, time=time, offsets=np.asarray(offsets, dtype=float), source_file=str(filename))


def write_segy(stream: Stream, path, data_encoding=5, byteorder=">") -> Path:
    """Write an ObsPy Stream as SEG-Y."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="SEGY", data_encoding=data_encoding, byteorder=byteorder)
    return path


def write_mseed(stream: Stream, path) -> Path:
    """Write an ObsPy Stream as MiniSEED."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="MSEED")
    return path
