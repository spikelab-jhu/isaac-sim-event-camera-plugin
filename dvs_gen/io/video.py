"""
video.py
========
H264Writer: drop-in replacement for cv2.VideoWriter that pipes raw BGR frames
to the system ffmpeg (libx264) so the output is H.264

    vw = H264Writer(path, W, H, fps)
    vw.write(bgr_uint8)   # same as cv2.VideoWriter.write
    vw.release()
"""
import subprocess

import cv2
import numpy as np


class H264Writer:
    def __init__(self, path, width, height, fps):
        self._cv2 = None
        self.p = None
        try:
            self.p = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{int(width)}x{int(height)}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                stdin=subprocess.PIPE,
            )
        except (FileNotFoundError, OSError):
            # no ffmpeg on PATH -> fall back to cv2 mp4v (not H.264, but still writes)
            self._cv2 = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    def write(self, bgr):
        if self._cv2 is not None:
            self._cv2.write(bgr)
        else:
            self.p.stdin.write(np.ascontiguousarray(bgr, dtype=np.uint8).tobytes())

    def release(self):
        if self._cv2 is not None:
            self._cv2.release()
        else:
            self.p.stdin.close()
            self.p.wait()
