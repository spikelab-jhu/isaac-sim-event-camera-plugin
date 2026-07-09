"""
transforms.py
=============
Composable event-representation transforms, each transform is a small dataclass
with ``__call__(events) -> Tensor``, so they drop into a :class:`Compose` chain
and into a ``torch.utils.data.Dataset``.

Three canonical grid representations:

* :class:`ToEventFrame`  — 2D histogram (count / polarity / ON-OFF channels).
* :class:`ToVoxelGrid`   — interpolated event volume (Zhu et al. 2019; the input
  e2vid consumes). Polarity is split between the two nearest time bins.
* :class:`ToTimeSurface` — exponentially-decayed surface of the most recent
  event per pixel and polarity.

Conventions follow :mod:`dvs_gen.representations.core`: input is anything
:func:`~dvs_gen.representations.core.as_event_stream` accepts; ``sensor_size`` is
``(W, H)``; outputs are ``(C, H, W)`` float32 torch tensors on the events'
device.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import EventStream, as_event_stream


# ──────────────────────────────────────────────────────────────
# Compose  (Tonic-style transform chaining)
# ──────────────────────────────────────────────────────────────
@dataclass
class Compose:
    """Apply a list of transforms left-to-right. ``Compose([a, b])(e) == b(a(e))``."""

    transforms: list

    def __call__(self, events):
        out = events
        for tf in self.transforms:
            out = tf(out)
        return out


def _normalize_nonzero(grid: Tensor) -> Tensor:
    """Zero-mean / unit-std over the NON-zero voxels only (the e2vid convention).

    Leaving the (many) zero voxels untouched keeps the "no event here" signal
    distinct from the normalised activity, which is what downstream nets expect.
    """
    nz = grid != 0
    if not bool(nz.any()):
        return grid
    vals = grid[nz]
    std = vals.std()
    mean = vals.mean()
    grid = grid.clone()
    grid[nz] = (vals - mean) / std if float(std) > 0 else vals - mean
    return grid


# ──────────────────────────────────────────────────────────────
# Event frame / 2D histogram
# ──────────────────────────────────────────────────────────────
@dataclass
class ToEventFrame:
    """Accumulate a window of events into a 2D histogram image.

    ``mode``:
      * ``"two_channel"`` (default) — ``(2, H, W)``: channel 0 = ON-event count,
        channel 1 = OFF-event count (polarities kept separate, the safer default
        for recognition where polarity is motion-direction nuisance).
      * ``"count"``    — ``(1, H, W)``: total event count per pixel (polarity
        ignored).
      * ``"polarity"`` — ``(1, H, W)``: signed sum of polarities per pixel
        (a brightness-increment image).

    Set ``normalize=True`` to divide by the per-image max (count modes) or to
    zero-mean/unit-std the non-zero pixels (polarity mode).
    """

    sensor_size: tuple[int, int]            # (W, H)
    mode: str = "two_channel"
    normalize: bool = False

    def __call__(self, events) -> Tensor:
        ev = as_event_stream(events)
        W, H = int(self.sensor_size[0]), int(self.sensor_size[1])
        dev = ev.device
        flat_idx = ev.y * W + ev.x                      # (N,) into a H*W image

        if self.mode == "two_channel":
            img = torch.zeros(2, H * W, dtype=torch.float32, device=dev)
            on, off = ev.p > 0, ev.p < 0
            img[0].index_add_(0, flat_idx[on], torch.ones(int(on.sum()), device=dev))
            img[1].index_add_(0, flat_idx[off], torch.ones(int(off.sum()), device=dev))
            out = img.reshape(2, H, W)
        elif self.mode == "count":
            img = torch.zeros(H * W, dtype=torch.float32, device=dev)
            img.index_add_(0, flat_idx, torch.ones(len(ev), device=dev))
            out = img.reshape(1, H, W)
        elif self.mode == "polarity":
            img = torch.zeros(H * W, dtype=torch.float32, device=dev)
            img.index_add_(0, flat_idx, ev.p.float())
            out = img.reshape(1, H, W)
        else:
            raise ValueError(f"unknown mode {self.mode!r}; "
                             "expected 'two_channel', 'count' or 'polarity'")

        if self.normalize:
            if self.mode == "polarity":
                out = _normalize_nonzero(out)
            else:
                m = float(out.max())
                if m > 0:
                    out = out / m
        return out


# ──────────────────────────────────────────────────────────────
# Voxel grid  (interpolated event volume — the e2vid input)
# ──────────────────────────────────────────────────────────────
@dataclass
class ToVoxelGrid:
    """Interpolated event volume of Zhu et al. (2019), as used by e2vid.

    The window's timestamps are normalised to ``[0, B-1]`` (``B = n_time_bins``)
    and each event's polarity is **split between its two nearest time bins** by
    linear interpolation (sub-bin temporal accuracy — see the survey's
    "interpolated voxel grid"). Output is ``(B, H, W)`` float32.

    With ``normalize=True`` the non-zero voxels are standardised to zero mean /
    unit std (the e2vid normalisation).
    """

    sensor_size: tuple[int, int]            # (W, H)
    n_time_bins: int = 5
    normalize: bool = False

    def __call__(self, events) -> Tensor:
        ev = as_event_stream(events)
        W, H = int(self.sensor_size[0]), int(self.sensor_size[1])
        B = int(self.n_time_bins)
        dev = ev.device
        voxel = torch.zeros(B * H * W, dtype=torch.float32, device=dev)
        if len(ev) == 0:
            return voxel.reshape(B, H, W)

        # Normalise timestamps to [0, B-1]. min/max (not first/last) so the result
        # is correct even if the stream isn't perfectly time-sorted. A zero-span
        # window collapses to bin 0.
        t = ev.t.double()
        t0, t1 = t.min(), t.max()
        span = t1 - t0
        ts = (B - 1) * (t - t0) / span if float(span) > 0 else torch.zeros_like(t)
        ts = ts.float()

        x = ev.x
        y = ev.y
        pol = ev.p.float()
        til = torch.floor(ts)
        dt = ts - til                                   # fractional position in [0,1)
        ti = til.long()
        base = y * W + x                                # (N,) pixel offset within a bin

        # left bin gets p*(1-dt), right bin (ti+1) gets p*dt
        left_w = pol * (1.0 - dt)
        right_w = pol * dt
        in_b = (x >= 0) & (x < W) & (y >= 0) & (y < H)  # drop out-of-frame events
        m_left = in_b & (ti >= 0) & (ti < B)
        m_right = in_b & (ti + 1 >= 0) & (ti + 1 < B)
        voxel.index_add_(0, (ti[m_left] * H * W + base[m_left]), left_w[m_left])
        voxel.index_add_(0, ((ti[m_right] + 1) * H * W + base[m_right]), right_w[m_right])

        voxel = voxel.reshape(B, H, W)
        if self.normalize:
            voxel = _normalize_nonzero(voxel)
        return voxel


# ──────────────────────────────────────────────────────────────
# Time surface  (exponentially-decayed last-event map)
# ──────────────────────────────────────────────────────────────
@dataclass
class ToTimeSurface:
    """Global time surface at the end of the window (Lagorce et al.; Tonic).

    For each pixel and polarity, keep the timestamp of the **most recent** event,
    then map it through an exponential decay ``exp(-(t_end - t_last) / tau)`` so a
    pixel that just fired is ≈1 and an old / silent pixel is ≈0. Output is
    ``(P, H, W)`` with ``P = 2 * len(tau)`` (an ON and an OFF channel per ``tau``).

    ``tau`` may be a single float or a list of decay constants (seconds); several
    ``tau`` give a multi-timescale surface (the MCTS idea), e.g.
    ``tau=[0.003, 0.03, 0.1]``.
    """

    sensor_size: tuple[int, int]            # (W, H)
    tau: object = 0.03                      # float | list[float], seconds

    def __call__(self, events) -> Tensor:
        ev = as_event_stream(events)
        W, H = int(self.sensor_size[0]), int(self.sensor_size[1])
        dev = ev.device
        taus = [float(self.tau)] if isinstance(self.tau, (int, float)) else [float(x) for x in self.tau]

        panes: list[Tensor] = []
        for pol_sign in (+1, -1):                       # ON, then OFF
            # latest timestamp per pixel for this polarity (-inf where none).
            last = torch.full((H * W,), float("-inf"), dtype=torch.float64, device=dev)
            m = ev.p == pol_sign
            if bool(m.any()):
                idx = ev.y[m] * W + ev.x[m]
                # amax-reduce keeps the most recent timestamp per pixel regardless
                # of event order (robust even if the stream isn't time-sorted).
                last.scatter_reduce_(0, idx, ev.t[m].double(), reduce="amax",
                                     include_self=True)
            for tau in taus:
                if len(ev):
                    t_end = ev.t[-1].double()
                    age = t_end - last                  # >=0 where seen, +inf where unseen
                    surf = torch.exp(-age / tau)        # unseen -> exp(-inf) = 0
                else:
                    surf = torch.zeros(H * W, dtype=torch.float64, device=dev)
                panes.append(surf.reshape(H, W).float())
        # interleave so channels read [tau0_ON, tau0_OFF, tau1_ON, ...]
        out = torch.stack(panes, dim=0)                 # (2*len(tau), H, W) as [ON*taus, OFF*taus]
        if len(taus) > 1:                               # reorder to per-tau ON/OFF pairs
            on = out[:len(taus)]
            off = out[len(taus):]
            out = torch.stack([t for pair in zip(on, off) for t in pair], dim=0)
        return out
