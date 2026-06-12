from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import numpy as np


def parse_header_sampleinterval_delay(header_text: str) -> Tuple[float, float]:
    """Parse Charlie-style header text containing ``Sampleinterval=`` and ``Delay=``."""
    sample_match = re.search(r"Sampleinterval=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header_text)
    delay_match = re.search(r"Delay=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header_text)
    if not sample_match:
        raise ValueError("Could not find Sampleinterval= in header")
    if not delay_match:
        raise ValueError("Could not find Delay= in header")
    return float(sample_match.group(1)), float(delay_match.group(1))


def read_charlie_raw(directory: str | Path, namenumber: int, n_channels: int = 24) -> Tuple[np.ndarray, float, float]:
    """Read Charlie Breithaupt's raw binary float/header format.

    Expected paths are::

        <directory>/data/<namenumber>
        <directory>/headers/<namenumber>head.txt

    Returns
    -------
    data, sampleinterval, delay
        ``data`` is returned using the package convention
        ``(n_traces, n_samples)``.
    """
    directory = Path(directory)
    data_file = directory / "data" / str(namenumber)
    header_file = directory / "headers" / f"{namenumber}head.txt"
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if not header_file.exists():
        raise FileNotFoundError(header_file)

    raw = np.fromfile(data_file, dtype=np.float32)
    if raw.size % n_channels != 0:
        raise ValueError(f"Data length {raw.size} is not divisible by n_channels={n_channels}")
    data = raw.reshape((-1, n_channels), order="C").T
    header_text = header_file.read_text(errors="ignore")
    sampleinterval, delay = parse_header_sampleinterval_delay(header_text)
    return data, sampleinterval, delay
