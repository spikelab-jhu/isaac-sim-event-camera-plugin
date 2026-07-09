"""
recorder.py
===========
``GeneralDVSRecorder`` — a thread-safe, multi-env / multi-camera DVS event
buffer that flushes one HDF5 file per (env, episode).

This is pure Python/torch/numpy/h5py — it has no Omniverse dependency, so it can
be imported and unit-tested outside Isaac Sim.

HDF5 layout (one file ``env{env_id}_ep{episode_idx}.h5`` per environment)::

    /<camera_name>/x   uint16   (gzip)  pixel x
    /<camera_name>/y   uint16   (gzip)  pixel y
    /<camera_name>/t   float64  (gzip)  timestamp (seconds)
    /<camera_name>/p   int8     (gzip)  polarity  (+1 = ON, -1 = OFF)
"""
import os
import threading
from collections import defaultdict

import numpy as np
import torch
import h5py


class GeneralDVSRecorder:
    """Thread-safe recorder for multiple envs, multiple cameras, varying episodes."""
    def __init__(self, output_dir: str = "/tmp/dvs_dataset", compression="gzip"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._lock = threading.Lock()
        # HDF5 filter for the event datasets: "gzip" (small files), "lzf" (fast),
        # or None (uncompressed — maximally robust for very large streams, no
        # filter that can fail on read). Datasets are written in blocks so a
        # multi-million-event episode never does one monolithic compressed write.
        self.compression = compression

        # Nested dictionary: env_id -> camera_name -> list of events
        self._events = defaultdict(lambda: defaultdict(list))

    def record(self, camera_name: str, env_ids: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, ps: torch.Tensor, t: float):
        if xs.numel() == 0: return

        # One GPU->CPU transfer per array, then store whole array-chunks per env
        # (NO per-event Python loop). Chunks are concatenated lazily at flush.
        e_np = env_ids.cpu().numpy()
        x_np = xs.cpu().numpy().astype(np.uint16)
        y_np = ys.cpu().numpy().astype(np.uint16)
        p_np = ps.cpu().numpy().astype(np.int8)

        with self._lock:
            if e_np.size and e_np.min() == e_np.max():     # fast path: all events in one env (num_envs=1)
                self._events[int(e_np[0])][camera_name].append(
                    (x_np, y_np, np.full(x_np.shape, t, np.float64), p_np))
            else:
                for e in np.unique(e_np):                   # group by env, still array-wise (no per-event loop)
                    m = e_np == e
                    self._events[int(e)][camera_name].append(
                        (x_np[m], y_np[m], np.full(int(m.sum()), t, np.float64), p_np[m]))

    def flush_episode(self, env_id: int, episode_idx: int):
        """Flushes all cameras for a single environment to a single HDF5 file."""
        with self._lock:
            if env_id not in self._events or not self._events[env_id]:
                return

            # Pop the environment's data out of the active buffer
            env_data = self._events.pop(env_id)

        filename = os.path.join(self.output_dir, f"env{env_id}_ep{episode_idx}.h5")

        counts = {}
        with h5py.File(filename, "w") as f:
            for cam_name, chunks in env_data.items():
                if not chunks: continue

                # Create a group for each camera in this environment
                grp = f.create_group(cam_name)

                # Each chunk is (x_arr, y_arr, t_arr, p_arr) — concatenate them all
                xs = np.concatenate([c[0] for c in chunks]).astype(np.uint16)
                ys = np.concatenate([c[1] for c in chunks]).astype(np.uint16)
                ts = np.concatenate([c[2] for c in chunks]).astype(np.float64)
                ps = np.concatenate([c[3] for c in chunks]).astype(np.int8)
                counts[cam_name] = int(xs.shape[0])

                # Write each dataset in blocks rather than one monolithic
                # create_dataset(data=...): a single ~100M-event compressed write
                # can corrupt a filter chunk; block writes through the chunk cache
                # are the robust path for very large event streams.
                self._write_blocked(grp, "x", xs)
                self._write_blocked(grp, "y", ys)
                self._write_blocked(grp, "t", ts)
                self._write_blocked(grp, "p", ps)

        summary = ", ".join(f"{k}={v:,}" for k, v in counts.items())
        print(f"[Recorder] Saved Env {env_id} (Ep {episode_idx}) w/ {len(env_data)} cameras "
              f"[{summary}] comp={self.compression} -> {filename}")

    def _write_blocked(self, grp, name, arr, block: int = 8_000_000):
        """Create a 1-D dataset and fill it in ``block``-sized slices."""
        n = int(arr.shape[0])
        if n == 0:
            grp.create_dataset(name, shape=(0,), dtype=arr.dtype)
            return
        # compression requires chunked storage; uncompressed stays contiguous
        chunks = True if self.compression else None
        ds = grp.create_dataset(name, shape=(n,), dtype=arr.dtype,
                                compression=self.compression, chunks=chunks)
        for i in range(0, n, block):
            ds[i:i + block] = arr[i:i + block]
