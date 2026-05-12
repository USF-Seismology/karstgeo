from __future__ import annotations

from pathlib import Path
from obspy import read, Stream, UTCDateTime


def read_segy_as_stream(path, *, unpack_trace_headers=True) -> Stream:
    return read(str(path), format="SEGY", unpack_trace_headers=unpack_trace_headers)


def read_su_file(path, dt_s=None, t0_s=0.0, byteorder="<") -> Stream:
    st = read(str(path), format="SU", byteorder=byteorder)
    if dt_s is not None:
        for tr in st:
            tr.stats.delta = float(dt_s)
            tr.stats.starttime = UTCDateTime(1970, 1, 1) + float(t0_s)
    return st


def write_segy(stream: Stream, path, data_encoding=5, byteorder=">") -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="SEGY", data_encoding=data_encoding, byteorder=byteorder)
    return path


def write_mseed(stream: Stream, path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(path), format="MSEED")
    return path
