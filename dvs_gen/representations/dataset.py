"""
dataset.py
==========
``EventStreamDataset`` — a ``torch.utils.data.Dataset`` over the HDF5 event files
this repo's DVS recorder writes (``env{e}_ep{ep}.h5``, groups ``DVS/cam0`` …,
datasets ``x, y, t, p``; see :mod:`dvs_gen.dvs.recorder`). It slices each
stream into windows (:mod:`~dvs_gen.representations.slicing`) and applies a
representation transform (:mod:`~dvs_gen.representations.transforms`), yielding
``(tensor, meta)`` ready for a ``DataLoader``.

Example::

    from dvs_gen.representations import ToVoxelGrid, EventStreamDataset
    from torch.utils.data import DataLoader

    ds = EventStreamDataset(
        "/tmp/multi_cam_dvs", cameras=["cam0", "cam1"],
        slicing=dict(policy="time", window=0.05),
        transform=ToVoxelGrid(sensor_size=(640, 480), n_time_bins=5),
    )
    for x, meta in DataLoader(ds, batch_size=8, shuffle=True):
        ...   # x: (8, 5, 480, 640); meta: dict of batched fields

"""
from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .slicing import build_slicer

_FNAME_RE = re.compile(r"env(\d+)_ep(\d+)\.h5$")


@dataclass(frozen=True)
class _Sample:
    """One window: where to find it and which event slice it is."""
    path: str
    camera: str          # logical name, e.g. "cam0"
    env: int
    episode: int
    i0: int
    i1: int


class EventStreamDataset(Dataset):
    """Index event windows across a directory of ``env{e}_ep{ep}.h5`` files.

    Parameters
    ----------
    root : str
        Directory containing the recorder's ``.h5`` files.
    cameras : list[str]
        Logical camera names (``["cam0", "cam1"]``). Each (file, camera) stream
        is sliced independently into samples.
    slicing : dict
        Passed to :func:`~dvs_gen.representations.slicing.build_slicer`, e.g.
        ``{"policy": "time", "window": 0.05}`` or ``{"policy": "count",
        "count": 30000}``.
    transform : callable | None
        A representation transform (``transform(events) -> Tensor``). If ``None``,
        ``__getitem__`` returns the raw ``events`` dict instead of a tensor.
    group_prefix : str
        HDF5 group the cameras live under (the recorder uses ``"DVS"``).
    target_fn : callable | None
        Optional ``target_fn(meta, events) -> Any`` to attach a label/GT to each
        sample (e.g. flow or detection targets for the benchmark). Its result is
        returned as the ``meta["target"]`` field.
    """

    def __init__(self, root, cameras=("cam0", "cam1"), *,
                 slicing=None, transform=None, group_prefix="DVS",
                 target_fn=None):
        self.root = str(root)
        self.cameras = list(cameras)
        self.transform = transform
        self.group_prefix = group_prefix
        self.target_fn = target_fn
        slicing = dict(slicing or {"policy": "time", "window": 0.05})
        self._slicer = build_slicer(**slicing)
        self.slicing = slicing

        self._files = {}                 # worker-local cache of open h5py handles
        self._index = self._build_index()

    # ── indexing ──────────────────────────────────────────────
    def _build_index(self):
        import h5py
        paths = sorted(glob.glob(os.path.join(self.root, "env*_ep*.h5")))
        if not paths:
            raise FileNotFoundError(
                f"no env*_ep*.h5 files under {self.root!r} — point `root` at the "
                "recorder's output dir (the `--dir` of simulate_warp/quickstart)")
        index = []
        for path in paths:
            m = _FNAME_RE.search(os.path.basename(path))
            env, ep = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
            with h5py.File(path, "r") as f:
                for cam in self.cameras:
                    key = f"{self.group_prefix}/{cam}"
                    if key not in f:
                        continue            # this stream emitted no events
                    t = f[key]["t"][:]      # cheap: timestamps only, for slicing
                    for i0, i1 in self._slicer(t):
                        index.append(_Sample(path, cam, env, ep, int(i0), int(i1)))
        return index

    def __len__(self):
        return len(self._index)

    # ── data access ───────────────────────────────────────────
    def _file(self, path):
        """Lazily open + cache an h5py handle (per worker process)."""
        f = self._files.get(path)
        if f is None:
            import h5py
            f = h5py.File(path, "r")
            self._files[path] = f
        return f

    def _read_events(self, s: _Sample):
        grp = self._file(s.path)[f"{self.group_prefix}/{s.camera}"]
        sl = slice(s.i0, s.i1)
        return {
            "x": torch.from_numpy(grp["x"][sl].astype(np.int64)),
            "y": torch.from_numpy(grp["y"][sl].astype(np.int64)),
            "t": torch.from_numpy(grp["t"][sl].astype(np.float64)),
            "p": torch.from_numpy(grp["p"][sl].astype(np.int8)),
        }

    def __getitem__(self, idx):
        s = self._index[idx]
        events = self._read_events(s)
        meta = {
            "env": s.env, "episode": s.episode, "camera": s.camera,
            "path": s.path, "i0": s.i0, "i1": s.i1,
            "n_events": int(s.i1 - s.i0),
            "t_start": float(events["t"][0]) if len(events["t"]) else 0.0,
            "t_end": float(events["t"][-1]) if len(events["t"]) else 0.0,
        }
        if self.target_fn is not None:
            meta["target"] = self.target_fn(meta, events)
        sample = self.transform(events) if self.transform is not None else events
        return sample, meta

    # ── housekeeping ──────────────────────────────────────────
    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    def __del__(self):
        self.close()

    def __getstate__(self):
        # don't pickle open h5py handles across the DataLoader worker fork
        state = self.__dict__.copy()
        state["_files"] = {}
        return state
