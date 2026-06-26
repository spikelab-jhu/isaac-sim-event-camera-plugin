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
    def __init__(self, output_dir: str = "/tmp/dvs_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._lock = threading.Lock()

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

                grp.create_dataset("x", data=xs, compression="gzip")
                grp.create_dataset("y", data=ys, compression="gzip")
                grp.create_dataset("t", data=ts, compression="gzip")
                grp.create_dataset("p", data=ps, compression="gzip")

        print(f"[Recorder] Saved Env {env_id} (Ep {episode_idx}) w/ {len(env_data)} cameras -> {filename}")
