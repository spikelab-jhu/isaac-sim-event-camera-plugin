"""
Event-representation transforms + canonical event container.

New (recommended) Tonic-style API — pure torch, no extra deps::

    from dvs_gen.representations import (
        EventStream, as_event_stream,
        ToEventFrame, ToVoxelGrid, ToTimeSurface, Compose,
    )

    voxel = ToVoxelGrid(sensor_size=(640, 480), n_time_bins=5)(events)

The legacy classes (``Adaptive_interval``, ``TsGenerator``, ``EventFrame``,
``EventVis``) are kept for reference; some pull in OpenCV, so they are imported
best-effort and simply omitted if their deps are missing.
"""
from .core import EventStream, as_event_stream
from .transforms import (
    Compose,
    ToEventFrame,
    ToVoxelGrid,
    ToTimeSurface,
)
from .slicing import (
    slice_by_time,
    slice_by_count,
    slice_by_n_frames,
    build_slicer,
)
from .dataset import EventStreamDataset
from .config import EventReprConfig

__all__ = [
    "EventStream", "as_event_stream",
    "Compose", "ToEventFrame", "ToVoxelGrid", "ToTimeSurface",
    "slice_by_time", "slice_by_count", "slice_by_n_frames", "build_slicer",
    "EventStreamDataset", "EventReprConfig",
]

# ── legacy / reference implementations (best-effort; may need opencv) ──
try:
    from .adaptive_interval import Adaptive_interval
    from .mcts import TsGenerator
    from .event_frames import EventFrame
    from .visualize_bluewhite import EventVis
    __all__ += ["Adaptive_interval", "TsGenerator", "EventFrame", "EventVis"]
except Exception:  # pragma: no cover - optional deps (e.g. cv2) may be absent
    pass
