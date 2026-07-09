"""
config.py
=========
A config that ties a representation + a slicing policy +
sensor size together, so a script / YAML can spin up the whole dataloader in a
couple of lines without importing the individual transform classes.

    from dvs_gen.representations import EventReprConfig
    cfg = EventReprConfig(representation="voxel", n_time_bins=5,
                          sensor_size=(640, 480), slicing={"policy": "time", "window": 0.05})
    loader = cfg.make_dataloader("/tmp/multi_cam_dvs", batch_size=8, shuffle=True)

It is a plain dataclass, so ``EventReprConfig(**yaml_dict)`` works and every
field is overridable from argparse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .transforms import ToEventFrame, ToVoxelGrid, ToTimeSurface
from .dataset import EventStreamDataset


@dataclass
class EventReprConfig:
    """Declarative spec for "events -> tensor" + how to slice + where to read.

    representation : one of ``"voxel"``, ``"event_frame"``, ``"time_surface"``.
    sensor_size    : ``(W, H)`` of the recorded camera.
    n_time_bins    : bins for the voxel grid.
    frame_mode     : ``ToEventFrame`` mode (``two_channel`` / ``count`` / ``polarity``).
    tau            : time-surface decay(s) in seconds (float or list).
    normalize      : standardise the representation (voxel / polarity frame).
    cameras        : logical camera names to read.
    slicing        : dict for :func:`~dvs_gen.representations.slicing.build_slicer`.
    """

    representation: str = "voxel"
    sensor_size: tuple = (640, 480)
    n_time_bins: int = 5
    frame_mode: str = "two_channel"
    tau: object = 0.03
    normalize: bool = False
    cameras: tuple = ("cam0", "cam1")
    slicing: dict = field(default_factory=lambda: {"policy": "time", "window": 0.05})

    _ALIASES = {
        "voxel": "voxel", "voxel_grid": "voxel",
        "event_frame": "event_frame", "frame": "event_frame", "image": "event_frame",
        "time_surface": "time_surface", "ts": "time_surface", "timesurface": "time_surface",
    }

    def build_transform(self):
        """Instantiate the representation transform described by this config."""
        rep = self._ALIASES.get(self.representation)
        if rep == "voxel":
            return ToVoxelGrid(self.sensor_size, n_time_bins=self.n_time_bins,
                               normalize=self.normalize)
        if rep == "event_frame":
            return ToEventFrame(self.sensor_size, mode=self.frame_mode,
                                normalize=self.normalize)
        if rep == "time_surface":
            return ToTimeSurface(self.sensor_size, tau=self.tau)
        raise ValueError(f"unknown representation {self.representation!r}; "
                         f"choose from {sorted(set(self._ALIASES))}")

    def build_dataset(self, root, *, target_fn=None, group_prefix="DVS"):
        """Build an :class:`EventStreamDataset` over ``root`` using this config."""
        return EventStreamDataset(
            root, cameras=self.cameras, slicing=self.slicing,
            transform=self.build_transform(), group_prefix=group_prefix,
            target_fn=target_fn)

    def make_dataloader(self, root, *, batch_size=8, shuffle=False, num_workers=0,
                        target_fn=None, group_prefix="DVS", **dl_kwargs):
        """Build the dataset and wrap it in a ``torch.utils.data.DataLoader``."""
        from torch.utils.data import DataLoader
        ds = self.build_dataset(root, target_fn=target_fn, group_prefix=group_prefix)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, **dl_kwargs)
