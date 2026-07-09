"""
core.py
=======
Canonical event container + normalisation, shared by every representation
transform. This is the single place where the messy real-world conventions
(polarity in ``{0,1}`` vs ``{-1,+1}``, timestamps in seconds vs microseconds,
unsorted streams) are pinned down, so the transforms downstream never have to
guess.

Convention (everything in this package assumes it)
--------------------------------------------------
* ``x`` — int64 column / width index in ``[0, W)``.
* ``y`` — int64 row / height index in ``[0, H)``.
* ``t`` — float64 timestamp in **seconds**, sorted ascending.
* ``p`` — int8 polarity in ``{-1, +1}`` (``+1`` = ON / brightness up).
* ``sensor_size`` — ``(W, H)`` (x first, like Tonic), while every output tensor
  is in torch image layout ``(C, H, W)``.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class EventStream:
    """A batch of events in the canonical convention (see module docstring).

    All four tensors are 1-D and the same length ``N``. Build one with
    :func:`as_event_stream`, which accepts a dict / structured numpy array /
    existing :class:`EventStream` and normalises it.
    """

    x: Tensor  # (N,) int64  in [0, W)
    y: Tensor  # (N,) int64  in [0, H)
    t: Tensor  # (N,) float64 seconds, ascending
    p: Tensor  # (N,) int8   in {-1, +1}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, device) -> "EventStream":
        return EventStream(self.x.to(device), self.y.to(device),
                           self.t.to(device), self.p.to(device))

    def time_span(self) -> float:
        """Duration ``t[-1] - t[0]`` in seconds (0 for an empty / single stream)."""
        if len(self) < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])

    def sort_by_time(self) -> "EventStream":
        """Return a copy sorted by ascending timestamp (stable)."""
        order = torch.argsort(self.t, stable=True)
        return EventStream(self.x[order], self.y[order], self.t[order], self.p[order])


def _as_tensor(v, dtype, device) -> Tensor:
    if isinstance(v, Tensor):
        return v.to(device=device, dtype=dtype)
    # np.ascontiguousarray: structured-array field views have non-default strides
    # that torch.as_tensor rejects; the copy makes them contiguous.
    return torch.as_tensor(np.ascontiguousarray(v), dtype=dtype, device=device)


def as_event_stream(events, *, device=None, assume_sorted: bool = True) -> EventStream:
    """Normalise ``events`` into a canonical :class:`EventStream`.

    Accepts:
      * an :class:`EventStream` (returned as-is, only moved to ``device``);
      * a mapping with keys ``x, y, t, p`` (torch tensors or numpy arrays);
      * a numpy **structured** array with fields ``x, y, t, p`` (Tonic style;
        a ``ts`` field is accepted as an alias for ``t``).

    Normalisation applied: ``x, y`` → int64, ``t`` → float64 seconds, ``p`` →
    int8 ``{-1, +1}`` (a ``{0, 1}`` input is remapped via ``2p - 1``). The
    stream is sorted by time unless ``assume_sorted`` is True.

    NOTE on units: ``t`` is taken to be **seconds** and is NOT rescaled. The DVS
    recorder in this repo already writes seconds; do any µs→s conversion before
    calling (this is exactly the implicit-``1e-6`` trap the old ``mcts`` code
    fell into).
    """
    if isinstance(events, EventStream):
        return events.to(device) if device is not None else events

    # numpy structured array -> dict view
    if isinstance(events, np.ndarray) and events.dtype.names is not None:
        names = events.dtype.names
        t_key = "t" if "t" in names else ("ts" if "ts" in names else None)
        if t_key is None or not {"x", "y", "p"} <= set(names):
            raise ValueError(f"structured array needs fields x,y,(t|ts),p; got {names}")
        # np.array(copy) repacks each field view into its own tightly-strided
        # buffer (a structured field view keeps the parent's struct itemsize as
        # its stride, which torch rejects).
        events = {"x": np.array(events["x"]), "y": np.array(events["y"]),
                  "t": np.array(events[t_key]), "p": np.array(events["p"])}

    try:
        x_in, y_in, t_in, p_in = events["x"], events["y"], events["t"], events["p"]
    except (KeyError, TypeError) as ex:
        raise ValueError("events must be an EventStream, a mapping with keys "
                         "x,y,t,p, or a structured numpy array") from ex

    dev = device
    if dev is None and isinstance(x_in, Tensor):
        dev = x_in.device

    x = _as_tensor(x_in, torch.int64, dev)
    y = _as_tensor(y_in, torch.int64, dev)
    t = _as_tensor(t_in, torch.float64, dev)
    p = _as_tensor(p_in, torch.float32, dev)

    # polarity -> {-1, +1} int8. A {0,1} encoding (min >= 0) maps via 2p-1.
    if p.numel() and float(p.min()) >= 0:
        p = 2.0 * p - 1.0
    p = torch.sign(p).to(torch.int8)            # {-1,+1}; 0 (shouldn't occur) -> 0

    stream = EventStream(x, y, t, p)
    if not assume_sorted:
        stream = stream.sort_by_time()
    return stream
