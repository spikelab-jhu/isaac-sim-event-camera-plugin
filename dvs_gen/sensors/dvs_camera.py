"""
dvs_camera.py
=============
First-class DVS camera abstraction for Isaac Lab.

Two pieces:

* :class:`DVSCameraCfg` — a thin :class:`~isaaclab.sensors.CameraCfg` subclass.
  Drop it into your scene like any other camera; it auto-selects the annotators
  the DVS pipeline needs (``rgb``, ``motion_vectors`` and, for warp,
  ``distance_to_image_plane``) and carries the DVS contrast ``threshold``. It
  still instantiates a plain Isaac Lab ``Camera`` sensor, so ``scene[name]``
  behaves exactly as usual.

* :class:`DVSCamera` — a runtime wrapper that bundles the camera handle(s) with a
  :class:`~dvs_gen.dvs.BatchedMultiCamProcessor` per camera and a shared
  :class:`~dvs_gen.dvs.GeneralDVSRecorder`. It encapsulates the
  grab → (optional) motion-vector warp → events → record loop that scripts would
  otherwise hand-roll. The warp batches *all cameras and all envs into a single*
  :func:`~dvs_gen.warp.bidir_warp_gap` call.

Example (warp pipeline)::

    from dvs_gen import DVSCamera
    dvs = DVSCamera.from_scene(env.scene, ["cam0", "cam1"], out_dir="/tmp/dvs")
    prev = dvs.snapshot()
    while running:
        env.step(actions)
        cur = dvs.snapshot()
        dvs.warp_and_process(prev, cur, K=8, t0=t_prev, dt_fine=1/1000)
        prev, t_prev = cur, t_cur
    dvs.flush(env_id=0, episode_idx=0)
"""
from __future__ import annotations

import torch

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from dvs_gen.dvs import GeneralDVSRecorder, BatchedMultiCamProcessor
from dvs_gen.warp import bidir_warp_gap

#: Isaac Lab annotator name for metric depth (per-pixel distance to image plane).
DEPTH_ANNOTATOR = "distance_to_image_plane"


def parse_margin(vals):
    """Normalise a ``--margin`` CLI value into ``(left, right, top, bottom)``.

    Accepts a list of 1 int (same on all sides) or 4 ints (left right top bottom).
    """
    if vals is None:
        return (0, 0, 0, 0)
    v = list(vals)
    if len(v) == 1:
        return (v[0], v[0], v[0], v[0])
    if len(v) == 4:
        return (v[0], v[1], v[2], v[3])
    raise ValueError("margin takes 1 value (all sides) or 4 values (left right top bottom)")


def crop_margin(img, margin):
    """Crop ``(left, right, top, bottom)`` pixels off an image tensor.

    Accepts ``(H, W, C)`` or ``(N, H, W, C)`` (height is dim -3, width dim -2).
    Used to strip the over-rendered margin so saved RGB / events are
    artifact-free at the borders.
    """
    left, right, top, bottom = margin
    if not (left or right or top or bottom):
        return img
    H, W = img.shape[-3], img.shape[-2]
    return img[..., top:H - bottom, left:W - right, :]


def tag_dvs_cameras(scene, names, threshold=0.15):
    """Set the ``dvs:preview`` / ``dvs:threshold`` USD attributes on the live camera
    prims named in ``names`` so the optional ``dvs_preview`` GUI extension finds them.

    Reliable (the prims already exist), unlike the spawn-time tagging. Call it once
    after the scene is built for any cameras you want previewed in the GUI.
    """
    try:
        import omni.usd
        from pxr import Sdf
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for n in names:
            sensor = scene[n]
            # Resolved per-env prim paths; fall back to env paths + cfg suffix.
            paths = list(getattr(sensor, "prim_paths", None) or [])
            if not paths:
                suffix = getattr(getattr(sensor, "cfg", None), "prim_path", "").split("}")[-1]
                paths = [ep + suffix for ep in getattr(scene, "env_prim_paths", [])]
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    prim.CreateAttribute("dvs:preview", Sdf.ValueTypeNames.Bool).Set(True)
                    prim.CreateAttribute("dvs:threshold", Sdf.ValueTypeNames.Float).Set(float(threshold))
                    print(f"[dvs_gen] tagged {p} for GUI preview")
    except Exception as ex:
        print(f"[dvs_gen] tag_dvs_cameras failed: {ex}")


def _tagging_spawn(orig_func, threshold):
    """Wrap an Isaac Lab spawner func so the spawned camera prim is tagged for the
    ``dvs_preview`` GUI extension (custom USD attrs ``dvs:preview`` / ``dvs:threshold``).
    The spawner signature varies across Isaac versions, so pass *args/**kwargs through.
    """
    def _spawn(prim_path, cfg, *args, **kwargs):
        prim = orig_func(prim_path, cfg, *args, **kwargs)
        try:
            from pxr import Sdf
            prim.CreateAttribute("dvs:preview", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("dvs:threshold", Sdf.ValueTypeNames.Float).Set(float(threshold))
        except Exception:
            pass        # tagging is best-effort; never block camera spawning
        return prim
    _spawn._dvs_tagged = True
    return _spawn


@configclass
class DVSCameraCfg(CameraCfg):
    """A :class:`CameraCfg` preset for DVS event generation.

    Instantiates a normal Isaac Lab ``Camera`` (``class_type`` is inherited), but
    in :meth:`__post_init__` ensures the annotators the DVS pipeline needs are
    present. Set ``enable_warp=False`` to drop the depth annotator (depth is only
    used to resolve occlusion during motion-vector warp).
    """

    #: DVS contrast threshold (log-intensity change that fires an event).
    threshold: float = 0.15
    #: If True, also request the depth annotator for occlusion-aware warping.
    enable_warp: bool = True
    #: If True, tag the spawned prim so the optional ``dvs_preview`` GUI extension
    #: auto-shows a live red/blue event window for this camera in the Isaac Lab GUI.
    gui_preview: bool = True

    def __post_init__(self):
        # CameraCfg / its bases may define __post_init__; honour it.
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        required = ["rgb", "motion_vectors"]
        if self.enable_warp:
            required.append(DEPTH_ANNOTATOR)
        dt = list(self.data_types) if self.data_types else []
        for t in required:
            if t not in dt:
                dt.append(t)
        self.data_types = dt

        # Tag the spawned camera prim (custom USD attrs) so the optional
        # `dvs_preview` Kit extension can find it and pop a live event window.
        if (self.gui_preview and getattr(self, "spawn", None) is not None
                and getattr(self.spawn, "func", None) is not None
                and not getattr(self.spawn.func, "_dvs_tagged", False)):
            self.spawn.func = _tagging_spawn(self.spawn.func, self.threshold)


class DVSCamera:
    """Runtime bundle: camera sensors + per-camera event processors + recorder.

    Use :meth:`from_scene` to build one from a running ``InteractiveScene``. The
    recorder writes one HDF5 file per (env, episode); event datasets are grouped
    under ``DVS/<camera_name>``.
    """

    def __init__(self, scene, names, recorder, processors, *,
                 enable_warp=True, composite="b_primary", depth_key=DEPTH_ANNOTATOR,
                 margin=(0, 0, 0, 0)):
        self.scene = scene
        self.names = list(names)
        self.recorder = recorder
        self.procs = list(processors)
        self.enable_warp = enable_warp
        self.composite = composite
        self.depth_key = depth_key
        # (left, right, top, bottom) over-rendered margin to crop off the warped
        # frames before they become events / RGB video.
        self.margin = tuple(margin)

    # ── construction ──────────────────────────────────────────
    @classmethod
    def from_scene(cls, scene, names=("cam0", "cam1"), *, out_dir="/tmp/dvs_dataset",
                   threshold=0.15, composite="b_primary", enable_warp=True,
                   group_prefix="DVS", margin=(0, 0, 0, 0)):
        """Build a recorder + one processor per camera and wrap ``scene``'s cameras.

        ``names`` are the camera keys in the scene (``scene[name]``); the events
        for each are stored under the HDF5 group ``<group_prefix>/<name>``.
        ``margin`` = ``(left, right, top, bottom)`` over-render to crop off (see
        :func:`crop_margin`); cameras must be rendered that many pixels larger.
        """
        recorder = GeneralDVSRecorder(out_dir)
        procs = [BatchedMultiCamProcessor(recorder, f"{group_prefix}/{n}", threshold)
                 for n in names]
        # Tag the (already-spawned) camera prims so the dvs_preview GUI extension
        # can find them — reliable here because the prims exist by now.
        tag_dvs_cameras(scene, names, threshold)
        return cls(scene, names, recorder, procs,
                   enable_warp=enable_warp, composite=composite, margin=margin)

    def _crop(self, x):
        return crop_margin(x, self.margin)

    # ── grabbing rendered data ────────────────────────────────
    def _depth(self, output):
        d = output[self.depth_key].float()
        if d.dim() == 4:
            d = d[..., 0]
        return torch.nan_to_num(d, posinf=1e4, neginf=1e4)

    def snapshot(self):
        """Return ``{name: (rgb, mv, depth)}`` for the current render.

        ``rgb`` is ``(N,H,W,C)``; ``mv`` is the 2-channel screen-space motion
        vector ``(N,H,W,2)``; ``depth`` is ``(N,H,W)`` (or ``None`` if warp/depth
        is disabled). All tensors are detached clones safe to hold across steps.
        """
        snap = {}
        for name in self.names:
            o = self.scene[name].data.output
            rgb = o["rgb"].float().clone()
            mv = torch.nan_to_num(o["motion_vectors"][..., :2].float())
            depth = self._depth(o) if self.enable_warp else None
            snap[name] = (rgb, mv, depth)
        return snap

    # ── event generation ──────────────────────────────────────
    def process(self, t: float):
        """Per render-step path (no warp): feed the current RGB of each camera."""
        for name, proc in zip(self.names, self.procs):
            proc(self._crop(self.scene[name].data.output["rgb"].float()), t)

    def warp_and_process(self, prev, cur, K, t0, dt_fine, frame_cb=None):
        """Warp the keyframe gap ``prev → cur`` into ``K`` frames and emit events.

        Feeds the real keyframe ``prev`` at ``t0`` and the ``K-1`` synthesised
        intermediates at ``t0 + i*dt_fine``. The next gap should pass ``cur`` as
        its ``prev`` (so ``cur`` is fed exactly once, as the next keyframe).

        All cameras (and all envs) are concatenated into ONE ``bidir_warp_gap``
        call: ``M = num_cameras * num_envs * (K-1)`` splats per direction fold
        into a single scatter. ``frame_cb(i, {name: frame})`` is called per output
        frame if given (e.g. to dump an RGB video).

        Returns the number of frames fed (``K``).
        """
        names = self.names
        # fraction 0: the real keyframe (cropped to the valid region)
        for name, proc in zip(names, self.procs):
            proc(self._crop(prev[name][0]), t0)
        if frame_cb is not None:
            frame_cb(0, {n: self._crop(prev[n][0]) for n in names})
        if K == 1:
            return 1

        Nenv = prev[names[0]][0].shape[0]
        A = torch.cat([prev[n][0] for n in names], 0)
        B = torch.cat([cur[n][0] for n in names], 0)
        mvA = torch.cat([prev[n][1] for n in names], 0)
        mvB = torch.cat([cur[n][1] for n in names], 0)
        if self.enable_warp and prev[names[0]][2] is not None:
            dA = torch.cat([prev[n][2] for n in names], 0)
            dB = torch.cat([cur[n][2] for n in names], 0)
        else:
            dA = dB = None

        # warp runs on the FULL (over-rendered) frames so borders have real
        # neighbours; only the cropped valid region becomes events / RGB.
        mids = bidir_warp_gap(A, B, mvA, mvB, K, self.composite, depthA=dA, depthB=dB)
        for i in range(K - 1):
            t = t0 + (i + 1) * dt_fine
            frame_dict = {}
            for ci, (name, proc) in enumerate(zip(names, self.procs)):
                f = self._crop(mids[i][ci * Nenv:(ci + 1) * Nenv])   # split cameras, crop margin
                proc(f, t)
                frame_dict[name] = f
            if frame_cb is not None:
                frame_cb(i + 1, frame_dict)
        return K

    # ── lifecycle ─────────────────────────────────────────────
    def reset(self, env_ids):
        """Flag the given envs to grab a fresh reference frame (post-reset)."""
        for proc in self.procs:
            proc.reset_envs(env_ids)

    def flush(self, env_id: int, episode_idx: int):
        """Write the buffered events for one (env, episode) to HDF5."""
        self.recorder.flush_episode(env_id, episode_idx)
