"""
visualize_event.py
==================
Render the DVS events saved by simulate.py / simulate_warp.py into a stereo
side-by-side video. Events are binned into frames at --fps; each frame shows the
events that fell within the last --interval_ms (red = ON, blue = OFF).

  python ./scripts/visualize_event.py --dir /tmp/multi_cam_dvs --env 0 --eps 0 \
         --fps 50 --interval_ms 5
"""
import argparse
import os

import cv2
import h5py
import numpy as np

from dvs_gen.io import H264Writer

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/tmp/multi_cam_dvs")
parser.add_argument("--env", type=int, default=0)
parser.add_argument("--eps", type=int, default=0)
parser.add_argument("--fps", type=int, default=50, help="output video frame rate")
parser.add_argument("--interval_ms", type=float, default=5.0, help="time window of events drawn per frame")
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

H, W = args.height, args.width


def load_cam(f, cam):
    g = f[f"DVS/{cam}"]
    return (g["x"][:].astype(np.int64), g["y"][:].astype(np.int64),
            g["t"][:].astype(np.float64), g["p"][:].astype(np.int64))


def frame_at(ev, t0, t1):
    """white canvas with events in [t0, t1): red = ON (p>0), blue = OFF."""
    x, y, t, p = ev
    img = np.full((H, W, 3), 255, np.uint8)
    m = (t >= t0) & (t < t1)
    xi, yi, pi = x[m], y[m], p[m]
    keep = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi, pi = xi[keep], yi[keep], pi[keep]
    on, off = pi > 0, pi <= 0
    img[yi[on], xi[on]] = (255, 0, 0)        # ON  -> red   (RGB)
    img[yi[off], xi[off]] = (0, 0, 255)      # OFF -> blue
    return img


def main():
    h5_path = os.path.join(args.dir, f"env{args.env}_ep{args.eps}.h5")
    with h5py.File(h5_path, "r") as f:
        cams = [c for c in ("cam0", "cam1") if f"DVS/{c}" in f]
        evs = {c: load_cam(f, c) for c in cams}

    tmin = min(ev[2].min() for ev in evs.values())
    tmax = max(ev[2].max() for ev in evs.values())
    win = args.interval_ms * 1e-3
    times = np.arange(tmin, tmax, 1.0 / args.fps)

    out_path = args.out or os.path.join(args.dir, f"vis_event_env{args.env}_ep{args.eps}.mp4")
    vw = H264Writer(out_path, W * len(cams), H, args.fps)
    for tc in times:
        panes = [frame_at(evs[c], tc, tc + win) for c in cams]
        vw.write(cv2.cvtColor(np.concatenate(panes, axis=1), cv2.COLOR_RGB2BGR))
    vw.release()
    n = sum(len(ev[0]) for ev in evs.values())
    print(f"[visualize_event] {n} events, {len(times)} frames -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
