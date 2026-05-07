from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import numpy as np
from obspy import Stream, Trace


@dataclass
class Timing:
    dt_s: Optional[float] = None
    t0_s: Optional[float] = None
    starttime_iso: str = "1970-01-01T00:00:00"


def component_to_channel(component: str) -> str:
    c = component.upper()
    if c in ("X", "BXX"):
        return "BXX"
    if c in ("Z", "BXZ"):
        return "BXZ"
    if c in ("Y", "BXY"):
        return "BXY"
    return c


def parse_sem_filename(path: Path) -> Optional[dict]:
    parts = Path(path).name.split(".")
    if len(parts) < 3:
        return None
    network, station, channel = parts[:3]
    extension = parts[3] if len(parts) > 3 else ""
    m = re.match(r"S(?P<num>\d+)$", station)
    if not m:
        return None
    return {
        "network": network,
        "station": station,
        "station_index": int(m.group("num")),
        "channel": channel,
        "extension": extension,
    }


def discover_sem_files(input_dir: Path, component: str = "Z", extension: str = "semv") -> list[Path]:
    input_dir = Path(input_dir).expanduser()
    channel = component_to_channel(component)
    patterns = []
    if extension:
        patterns.append(f"*.{channel}.{extension}")
    else:
        patterns.append(f"*.{channel}")
    patterns.extend([f"*.{channel}.semv", f"*.{channel}.semd", f"*.{channel}.sema", f"*.{channel}"])

    files, seen = [], set()
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                info = parse_sem_filename(path)
                if info and info["channel"] == channel:
                    files.append(path)
                    seen.add(path)

    def station_key(path):
        info = parse_sem_filename(path)
        return info["station_index"] if info else 10**12

    return sorted(files, key=station_key)


def read_sem_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return np.arange(len(arr), dtype=float), arr.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Cannot parse {path}; expected one or two columns.")
    return arr[:, 0].astype(float), arr[:, 1].astype(np.float32)


def read_sem_gather(input_dir: Path, component: str = "Z", extension: str = "semv", timing: Timing | None = None, verbose: bool = True):
    timing = timing or Timing()
    files = discover_sem_files(input_dir, component=component, extension=extension)
    if not files:
        raise FileNotFoundError(f"No SPECFEM ASCII files found in {input_dir} for component={component}, extension={extension}")
    if verbose:
        print(f"Found {len(files)} files in {input_dir}")

    records, station_indices, time = [], [], None
    for i, path in enumerate(files, start=1):
        info = parse_sem_filename(path)
        if info is None:
            continue
        t_file, y = read_sem_ascii(path)
        if time is None:
            if timing.dt_s is not None:
                t0 = timing.t0_s if timing.t0_s is not None else float(t_file[0])
                time = float(t0) + np.arange(len(y), dtype=float) * float(timing.dt_s)
            else:
                time = t_file
        elif len(y) != len(time):
            raise ValueError(f"Trace length mismatch in {path}: {len(y)} samples vs expected {len(time)}")
        records.append(y)
        station_indices.append(info["station_index"])
        if verbose and (i == 1 or i % 50 == 0 or i == len(files)):
            print(f"  read {i:5d}/{len(files):5d}: {path.name}")

    order = np.argsort(station_indices)
    return np.asarray(time), np.asarray(records, dtype=np.float32)[order], np.asarray(station_indices, dtype=int)[order]


def sem_gather_to_stream(time, data, station_indices, component="Z", network="SY") -> Stream:
    """Make a simple Stream from SEM gather arrays. Headers/geometries are applied downstream."""
    st = Stream()
    if len(time) < 2:
        dt = 1.0
    else:
        dt = float(np.median(np.diff(time)))
    t0 = float(time[0]) if len(time) else 0.0
    for sta, y in zip(station_indices, data):
        tr = Trace(data=np.asarray(y, dtype=np.float32))
        tr.stats.network = network
        tr.stats.station = f"S{int(sta):04d}"
        tr.stats.channel = component_to_channel(component)
        tr.stats.delta = dt
        st.append(tr)
    return st
