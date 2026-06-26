"""
visualize_rgb.py
================
Stack the two synthesised RGB streams saved by simulate_warp.py
(rgb_env{e}_ep{ep}_cam0.mp4 / _cam1.mp4) into one stereo side-by-side video.

  python ./scripts/visualize_rgb.py --dir /tmp/multi_cam_dvs --env 0 --eps 0 --fps 50
"""
import argparse
import os

import cv2
import numpy as np

from dvs_gen.io import H264Writer

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/tmp/multi_cam_dvs")
parser.add_argument("--env", type=int, default=0)
parser.add_argument("--eps", type=int, default=0)
parser.add_argument("--fps", type=int, default=50, help="output playback frame rate")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)                    # BGR
    cap.release()
    return frames


def main():
    paths = [os.path.join(args.dir, f"rgb_env{args.env}_ep{args.eps}_cam{c}.mp4") for c in (0, 1)]
    streams = [read_frames(p) for p in paths if os.path.exists(p)]
    streams = [s for s in streams if s]
    if not streams:
        print(f"[visualize_rgb] no rgb_env{args.env}_ep{args.eps}_cam*.mp4 in {args.dir} "
              f"(run simulate_warp.py without --no_rgb)", flush=True)
        return
    n = min(len(s) for s in streams)
    h, w = streams[0][0].shape[:2]

    out_path = args.out or os.path.join(args.dir, f"vis_rgb_env{args.env}_ep{args.eps}.mp4")
    vw = H264Writer(out_path, w * len(streams), h, args.fps)
    for i in range(n):
        vw.write(np.concatenate([s[i] for s in streams], axis=1))   # already BGR
    vw.release()
    print(f"[visualize_rgb] {len(streams)} cam(s), {n} frames -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
