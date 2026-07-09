"""
slicing.py
==========
Cut a long event stream into per-sample windows. A "sample" is a contiguous
index range ``(i0, i1)`` into the (time-sorted) event arrays; the dataset turns
each range into one representation tensor.

Three slicing policies (mirroring Tonic's ``ToFrame`` options):

* :func:`slice_by_time`   — fixed wall-clock windows (seconds). Aligns with
  optical-flow / reconstruction GT that lives on a fixed time grid.
* :func:`slice_by_count`  — fixed number of events per sample. Keeps the event
  count (and hence representation density) constant; common for recognition.
* :func:`slice_by_n_frames` — split the whole stream into ``N`` equal-time
  windows.

All take a 1-D ascending ``t`` (numpy array, seconds) and return a list of
``(i0, i1)`` half-open index ranges. Pure numpy — no torch / Isaac.
"""
from __future__ import annotations

import numpy as np


def slice_by_time(t, window: float, *, overlap: float = 0.0,
                  include_incomplete: bool = False, drop_empty: bool = True):
    """Fixed-duration windows of ``window`` seconds.

    ``overlap`` in ``[0,1)`` slides the window by ``window*(1-overlap)`` each step.
    ``include_incomplete`` keeps the final window even if it is shorter than
    ``window``. ``drop_empty`` discards windows that contain no events.
    """
    t = np.asarray(t)
    if t.size == 0:
        return []
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    step = window * (1.0 - overlap)
    if step <= 0:
        raise ValueError("window (and 1-overlap) must be > 0")

    t0, tN = float(t[0]), float(t[-1])
    ranges = []
    s = t0
    eps = 1e-12
    while s < tN - eps:
        if not include_incomplete and s + window > tN + eps:
            break
        i0 = int(np.searchsorted(t, s, side="left"))
        i1 = int(np.searchsorted(t, s + window, side="left"))
        if not (drop_empty and i1 <= i0):
            ranges.append((i0, i1))
        s += step
    return ranges


def slice_by_count(t, count: int, *, overlap: float = 0.0,
                   include_incomplete: bool = False):
    """Fixed ``count`` events per sample, sliding by ``count*(1-overlap)``."""
    n = int(np.asarray(t).shape[0])
    if n == 0 or count <= 0:
        return []
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(count * (1.0 - overlap))))
    ranges = []
    i = 0
    while i < n:
        j = min(i + count, n)
        if not include_incomplete and j - i < count:
            break
        ranges.append((i, j))
        i += step
    return ranges


def slice_by_n_frames(t, n_frames: int, *, drop_empty: bool = True):
    """Split the stream's time span into ``n_frames`` equal-duration windows."""
    t = np.asarray(t)
    if t.size == 0 or n_frames <= 0:
        return []
    t0, tN = float(t[0]), float(t[-1])
    span = tN - t0
    if span <= 0:                       # all events at one instant -> one window
        return [(0, int(t.size))]
    edges = t0 + np.linspace(0.0, span, n_frames + 1)
    idx = np.searchsorted(t, edges, side="left").astype(int)
    idx[-1] = int(t.size)               # include the final event
    ranges = []
    for a, b in zip(idx[:-1], idx[1:]):
        if not (drop_empty and b <= a):
            ranges.append((int(a), int(b)))
    return ranges


#: name -> (function, kwarg-key) so a config can pick a policy by string.
SLICERS = {
    "time": slice_by_time,
    "count": slice_by_count,
    "n_frames": slice_by_n_frames,
}


def build_slicer(policy: str, **kwargs):
    """Return a one-arg ``slicer(t) -> [(i0,i1)]`` for the named policy.

    ``policy`` is ``"time"`` (kwarg ``window``), ``"count"`` (kwarg ``count``)
    or ``"n_frames"`` (kwarg ``n_frames``); remaining kwargs pass through.
    """
    if policy not in SLICERS:
        raise KeyError(f"unknown slicing policy {policy!r}; choose from {sorted(SLICERS)}")
    fn = SLICERS[policy]
    return lambda t: fn(t, **kwargs)
