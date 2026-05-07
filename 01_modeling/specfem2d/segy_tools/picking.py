from __future__ import annotations
import numpy as np


def threshold_first_arrivals(data, time, fraction=0.05, min_time=None):
    data = np.asarray(data); time = np.asarray(time)
    picks = np.full(data.shape[0], np.nan)
    start_idx = 0
    if min_time is not None:
        idxs = np.where(time >= min_time)[0]
        if len(idxs): start_idx = idxs[0]
    for i, tr in enumerate(data):
        y = np.abs(tr)
        threshold = fraction * np.max(y)
        idx = np.where(y[start_idx:] >= threshold)[0]
        if len(idx): picks[i] = time[start_idx + idx[0]]
    return picks
