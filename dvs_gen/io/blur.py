"""
blur.py
=======
``MotionBlurAccumulator`` — turn the warp's fine in-between frames into a single
motion-blurred (long-exposure) frame by averaging them. 
"""
from dataclasses import dataclass

import torch


@dataclass
class MotionBlurCfg:
    """Motion-blur (long-exposure) config. Set ``DVSCameraCfg.motion_blur`` to one
    of these to enable it for that camera."""
    exposure_ms: float = 20.0     # exposure window; longer = blurrier. Blurred fps = 1000/exposure_ms.
    #: If True (default), the EVENT MODEL is fed the blurred frames instead of the
    #: sharp fine frames — the camera behaves like a sensor with a real exposure
    #: time (events drop to one batch per exposure window). Set False to keep the
    #: original behaviour: sharp high-rate events + blurred RGB video side output.
    feed_events: bool = True


class MotionBlurAccumulator:
    """Average a stream of frames into motion-blurred frames.

    Feed every fine frame via :meth:`add`. Once ``window`` frames have been
    accumulated it returns their mean (the blurred frame) and resets; otherwise
    it returns ``None``. Call :meth:`flush` at the end to emit a partial window.
    """

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self._sum = None
        self._n = 0

    def add(self, frame):
        """Accumulate one frame ``(H, W, C)``; return the blurred frame when the
        exposure window is full, else ``None``."""
        f = frame.detach().float()
        self._sum = f.clone() if self._sum is None else self._sum + f
        self._n += 1
        if self._n >= self.window:
            return self.flush()
        return None

    def flush(self):
        """Return the mean of whatever is accumulated (or ``None`` if empty) and reset."""
        if self._n == 0:
            return None
        avg = self._sum / self._n
        self._sum = None
        self._n = 0
        return avg
