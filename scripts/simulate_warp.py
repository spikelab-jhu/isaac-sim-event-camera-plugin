"""
simulate_warp.py
================
Reference / showcase pipeline: render RGB + motion-vectors + depth only at a low
KEYFRAME rate and synthesise the in-between frames with the motion-vector warp,
so the DVS event generator still sees a high-rate stream (render_hz × ``--warp``)
while the RTX renderer runs ``--warp`` times less often.

    sim @render_hz  ->  rgb/mv/depth keyframes  ->  bidir warp ×K  ->  DVS events @render_hz·K

The actual work is done by :class:`dvs_gen.sensors.DVSCamera`, which bundles the
stereo cameras, the per-camera event processors and the recorder. The key
performance idea lives in :meth:`DVSCamera.warp_and_process`: **all cameras and
all envs are concatenated into ONE** :func:`~dvs_gen.warp.bidir_warp_gap` **call**
(``M = num_cameras * num_envs * (K-1)`` splats fold into a single scatter), so the
warp cost amortises across the whole batch instead of looping per camera/env.

Outputs (per env, per episode) into ``--dir``:
  env{e}_ep{ep}.h5                 DVS events (groups DVS/cam0, DVS/cam1; x,y,t,p)
  annotation_env{e}_ep{ep}.json    per-keyframe object pose / velocity
  rgb_env{e}_ep{ep}_cam{c}.mp4     synthesised RGB stream (env 0 only; --no_rgb to skip)

"""
import argparse
import os
import time

parser = argparse.ArgumentParser(description="Simulate + motion-vector warp + DVS events.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--render_hz", type=float, default=125, help="keyframe render rate (physics runs here too)")
parser.add_argument("--warp", type=int, default=8,
                    help="warp multiplier K: render keyframes at render_hz and warp K× to get "
                         "events at render_hz*K (integer >= 1; use 1 or --no_warp for no warp)")
parser.add_argument("--composite", default="b_primary", choices=["b_primary"],
                    help="warp composite strategy (only b_primary is implemented)")
parser.add_argument("--mv_dilate", type=int, default=0,
                    help="motion-vector dilation radius (px) before the warp splat; grows each "
                         "object's mv over its anti-aliased edge to remove the boundary ghost "
                         "outline. 0 = off (default), 1 = on")
parser.add_argument("--max_episodes", type=int, default=3, help="stop after env 0 has finished this many episodes")
parser.add_argument("--dir", type=str, default="/tmp/multi_cam_dvs")
parser.add_argument("--no_rgb", action="store_true", help="do not save the RGB mp4 stream")
parser.add_argument("--no_warp", action="store_true",
                    help="disable warp: emit events from the real rendered keyframes only "
                         "(output is at render_hz; the dense baseline / ground truth)")
parser.add_argument("--margin", type=int, nargs="+", default=[0],
                    help="over-render this many pixels per side, then crop them off so the "
                         "saved RGB/events are artifact-free at the borders. "
                         "1 value = all sides; 4 values = left right top bottom")
parser.add_argument("--breakdown", action="store_true",
                    help="time render / warp / dvs segments separately (adds cuda syncs)")
parser.add_argument("--event_source", default=None, choices=["ldr", "hdr"],
                    help="OVERRIDE the config's camera event_source. Default: follow the config "
                         "(set DVSCameraCfg.event_source='hdr' to make the whole pipeline run on the "
                         "linear HDR buffer — physically correct events, RGB tone-mapped for display).")

from isaaclab.app import AppLauncher

import torch

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import json
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

from dvs_gen.env import DVSEnvCfg
from dvs_gen.sensors import DVSCamera, parse_margin
from dvs_gen.io import H264Writer

K = 1 if args.no_warp else args.warp   # warp multiplier (K=1 => no interpolation)
assert K >= 1, "--warp must be >= 1"
DT_KEY = 1.0 / args.render_hz          # keyframe spacing (= sim dt)
DT_FINE = DT_KEY / K                   # synthesised-frame spacing
OUT_HZ = args.render_hz * K            # effective event rate (for display / video fps)


def parse_annotation(policy_obs, sim_time):
    """Object pose/velocity in each camera frame (for downstream supervision)."""
    num_env = policy_obs.shape[0]
    cam0_pos_w, cam0_quat_w = policy_obs[:, 0:3], policy_obs[:, 3:7]
    cam1_pos_w, cam1_quat_w = policy_obs[:, 7:10], policy_obs[:, 10:14]
    obj_pos_w, obj_quat_w = policy_obs[:, 14:17], policy_obs[:, 17:21]
    obj_linv_w, obj_angv_w = policy_obs[:, 21:24], policy_obs[:, 24:27]

    lin0 = math_utils.quat_rotate_inverse(cam0_quat_w, obj_linv_w)
    ang0 = math_utils.quat_rotate_inverse(cam0_quat_w, obj_angv_w)
    lin1 = math_utils.quat_rotate_inverse(cam1_quat_w, obj_linv_w)
    ang1 = math_utils.quat_rotate_inverse(cam1_quat_w, obj_angv_w)
    p0, q0 = math_utils.subtract_frame_transforms(cam0_pos_w, cam0_quat_w, obj_pos_w, obj_quat_w)
    p1, q1 = math_utils.subtract_frame_transforms(cam1_pos_w, cam1_quat_w, obj_pos_w, obj_quat_w)
    return [{
        "cam0": {"object_position": p0[i].cpu().numpy().tolist(),
                 "object_quaternion": q0[i].cpu().numpy().tolist(),
                 "linear_velocity": lin0[i].cpu().numpy().tolist(),
                 "angular_velocity": ang0[i].cpu().numpy().tolist()},
        "cam1": {"object_position": p1[i].cpu().numpy().tolist(),
                 "object_quaternion": q1[i].cpu().numpy().tolist(),
                 "linear_velocity": lin1[i].cpu().numpy().tolist(),
                 "angular_velocity": ang1[i].cpu().numpy().tolist()},
        "time": sim_time,
    } for i in range(num_env)]


def main():
    # ── env: the clean default config, retuned to the keyframe rate ──
    cfg = DVSEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.dt = DT_KEY                 # physics + render at the keyframe rate
    cfg.sim.render_interval = 1

    # ── over-render margin: render (W+L+R)x(H+T+B), crop back to W0xH0 ──
    margin = parse_margin(args.margin)
    L, R, T, B = margin
    W0, H0 = cfg.scene.cam0.width, cfg.scene.cam0.height     # original (output) size
    for f in ("cam0", "cam1"):
        c = getattr(cfg.scene, f)
        c.update_period = DT_KEY                              # refresh every keyframe
        c.width = W0 + L + R
        c.height = H0 + T + B
    cfg.events.reinit_dvs.params["margin"] = margin           # shift principal point
    # config-driven: DVSCameraCfg(event_source='hdr') auto-requests HdrColor and
    # from_scene picks it up. --event_source overrides the config.
    if args.event_source is not None:
        for f in ("cam0", "cam1"):
            getattr(cfg.scene, f).event_source = args.event_source
    if any(getattr(getattr(cfg.scene, f), "event_source", "ldr") == "hdr" for f in ("cam0", "cam1")):
        for f in ("cam0", "cam1"):
            c = getattr(cfg.scene, f)
            if "HdrColor" not in c.data_types:
                c.data_types = list(c.data_types) + ["HdrColor"]

    env = ManagerBasedRLEnv(cfg=cfg)
    os.makedirs(args.dir, exist_ok=True)

    # ── the DVS abstraction: cameras + processors + recorder in one ──
    dvs = DVSCamera.from_scene(env.scene, ["cam0", "cam1"], out_dir=args.dir,
                               composite=args.composite, margin=margin, mv_dilate=args.mv_dilate)

    episode_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    annotations = [[] for _ in range(env.num_envs)]
    H, W = H0, W0                                             # videos use the cropped size
    save_rgb = not args.no_rgb
    rgb_writers = {}     # cam_idx -> H264Writer for env 0's current episode

    def open_rgb_writers(ep):
        if not save_rgb:
            return
        for c in (0, 1):
            path = os.path.join(args.dir, f"rgb_env0_ep{ep}_cam{c}.mp4")
            rgb_writers[c] = H264Writer(path, W, H, int(round(OUT_HZ / 20)))  # 20x slow-mo

    def close_rgb_writers():
        for c in (0, 1):
            if c in rgb_writers:
                rgb_writers[c].release()
        rgb_writers.clear()

    def frame_cb(_i, frames):
        """Called by DVSCamera per synthesised frame — dump env-0 RGB to video."""
        if not save_rgb:
            return
        for c, name in ((0, "cam0"), (1, "cam1")):
            img = dvs.display_u8(frames[name][0])            # env 0; tone-maps HDR for the video
            rgb_writers[c].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    actions = torch.zeros((env.num_envs, 0), device=env.device)
    obs, _ = env.reset()
    open_rgb_writers(0)
    env.step(actions)                                       # prime: seed prev
    prev = dvs.snapshot()
    t_prev = float(env.sim.current_time)

    print(f"[simulate_warp] render@{args.render_hz:.0f}Hz -> out@{OUT_HZ:.0f}Hz (K={K}); "
          f"saving to {args.dir}", flush=True)

    bd = args.breakdown
    # DVSCamera fuses warp + event-gen in one call, so they are timed together.
    acc = {"render": 0.0, "warp+dvs": 0.0}

    def sync_now():
        if bd:
            torch.cuda.synchronize()
        return time.perf_counter()

    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    n_out = 0

    def report_rt():
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_wall0
        sim_t = n_out * DT_FINE
        Nenv = env.num_envs
        if sim_t > 0:
            print(f"[simulate_warp] REALTIME num_envs={Nenv}: produced {sim_t:.3f}s sim x {Nenv} envs "
                  f"of {OUT_HZ:.0f}Hz in {wall:.3f}s wall", flush=True)
            print(f"    per-stream : {wall/sim_t:.2f}x slower than real-time ({n_out/wall:.0f} frames/s)", flush=True)
            print(f"    AGGREGATE  : {Nenv*sim_t/wall:.2f}x real-time data "
                  f"({Nenv*n_out/wall:.0f} env-frames/s)  <- >=1.0 means faster-than-real-time", flush=True)
        if bd:
            other = max(0.0, wall - acc["render"] - acc["warp+dvs"])
            for k, v in [("render", acc["render"]), ("warp+dvs", acc["warp+dvs"]),
                         ("other(anno+flush+reset)", other)]:
                print(f"    {k:<24}= {v:7.3f}s  ({100*v/wall:5.1f}%)  ({v/n_out*1000:.4f} ms/frame)", flush=True)

    while True:
        ta = sync_now()
        obs, _, terminated, truncated, _ = env.step(actions)
        cur = dvs.snapshot()
        tb = sync_now(); acc["render"] += tb - ta
        t_cur = float(env.sim.current_time)
        # obs is the POST-step state, so the annotation carries t_cur (stamping it
        # t_prev mislabelled every pose one keyframe early).
        step_anno = parse_annotation(obs["policy"], t_cur)

        resets = torch.logical_or(terminated, truncated)
        if resets.any():
            ridx = torch.where(resets)[0]
            dvs.reset(ridx)
            for e in ridx.cpu().numpy():
                ep = int(episode_counts[e].item())
                dvs.flush(int(e), episode_idx=ep)
                with open(os.path.join(args.dir, f"annotation_env{e}_ep{ep}.json"), "w") as fh:
                    json.dump(annotations[e], fh)
                annotations[e] = []
                if int(e) == 0:                             # roll env-0 RGB video
                    close_rgb_writers()
                    if ep + 1 < args.max_episodes:          # don't open an empty trailing video
                        open_rgb_writers(ep + 1)
                episode_counts[e] += 1
            if int(episode_counts[0].item()) >= args.max_episodes:
                report_rt()
                break
            obs, _ = env.reset()
            env.step(actions)
            prev = dvs.snapshot()
            t_prev = float(env.sim.current_time)
            continue

        # warp the gap (all cams+envs batched) and feed every frame to the DVS hooks
        tc = sync_now()
        n = dvs.warp_and_process(prev, cur, K, t_prev, DT_FINE,
                                 frame_cb=frame_cb if save_rgb else None)
        td = sync_now(); acc["warp+dvs"] += td - tc
        n_out += n
        for j in range(env.num_envs):
            annotations[j].append(step_anno[j])

        prev = cur
        t_prev = t_cur

    close_rgb_writers()
    print(f"[simulate_warp] done: {args.max_episodes} episodes -> {args.dir}", flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
