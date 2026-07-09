"""
dvs_camera.py
=============
DVS camera abstraction for Isaac Lab.

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
    t_prev = float(env.sim.current_time)
    while running:
        env.step(actions)
        cur = dvs.snapshot()
        t_cur = float(env.sim.current_time)
        dvs.warp_and_process(prev, cur, K=8, t0=t_prev, dt_fine=1/1000)
        prev, t_prev = cur, t_cur
    dvs.flush(env_id=0, episode_idx=0)
"""
from __future__ import annotations

import torch

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from dvs_gen.dvs import GeneralDVSRecorder, BatchedMultiCamProcessor, DVSNoiseCfg, DVSNoiseModel
from dvs_gen.io.blur import MotionBlurAccumulator, MotionBlurCfg
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
    #: Intensity source for the DVS event model. ``"hdr"`` (default) uses the linear
    #: ``HdrColor`` buffer so ``log(intensity)`` is physically correct (no ISP
    #: gamma/tone-map warping); ``"ldr"`` uses the tone-mapped ``rgb`` buffer.
    event_source: str = "hdr"
    #: Optional sensor-noise model (per-pixel threshold mismatch, background/leak/shot
    #: events, hot pixels, refractory). ``None`` (default) = the ideal clean model.
    #: To enable, set it to a :class:`~dvs_gen.dvs.DVSNoiseCfg`, e.g.::
    #:
    #:     from dvs_gen.dvs import DVSNoiseCfg
    #:     DVSCameraCfg(..., noise=DVSNoiseCfg(shot_rate_hz=1.0, hot_pixel_frac=5e-4))
    noise: DVSNoiseCfg | None = None
    #: Optional motion-blur (long-exposure) RGB output: average the warp's fine frames
    #: over the exposure window. ``None`` (default) = off. To enable::
    #:
    #:     from dvs_gen.io import MotionBlurCfg
    #:     DVSCameraCfg(..., motion_blur=MotionBlurCfg(exposure_ms=20.0))
    motion_blur: MotionBlurCfg | None = None

    def __post_init__(self):
        # CameraCfg / its bases may define __post_init__; honour it.
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        required = ["rgb", "motion_vectors"]
        if self.enable_warp:
            required.append(DEPTH_ANNOTATOR)
        if self.event_source == "hdr":
            required.append("HdrColor")
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
                 margin=(0, 0, 0, 0), event_source="ldr", blur_cfgs=None, hole_fill=None,
                 mv_dilate=0):
        self.scene = scene
        self.names = list(names)
        self.recorder = recorder
        self.procs = list(processors)
        self.enable_warp = enable_warp
        self.composite = composite
        self.depth_key = depth_key
        # "ldr" = tone-mapped rgb; "hdr" = linear HdrColor (physically correct events).
        # In "hdr" mode the whole warp/event/video path runs on the HDR buffer and
        # display_u8() tone-maps it for viewable RGB/GT output.
        self.event_source = event_source
        # (left, right, top, bottom) over-rendered margin to crop off the warped
        # frames before they become events / RGB video.
        self.margin = tuple(margin)
        # per-camera MotionBlurCfg (name -> cfg); accumulators are built lazily in
        # warp_and_process once dt_fine is known. Empty = no motion blur.
        self.blur_cfgs = dict(blur_cfgs or {})
        self._blur_accs = {}
        # warp double-occlusion hole fill: None = black; a float = that constant;
        # "bg" = the frame's brightest value (auto white for a uniform bright backdrop).
        self.hole_fill = hole_fill
        # motion-vector dilation radius (px) applied before the warp splat: grows
        # each object's mv over its anti-aliased silhouette fringe so the warp does
        # not leave a faint ghost outline at the object's keyframe position. 0 = off.
        self.mv_dilate = mv_dilate
        # last timestamp seen by process() — used to infer the frame interval that
        # the motion-blur accumulator needs on the no-warp path.
        self._proc_prev_t = None

    # ── construction ──────────────────────────────────────────
    @classmethod
    def from_scene(cls, scene, names=("cam0", "cam1"), *, out_dir="/tmp/dvs_dataset",
                   threshold=None, composite="b_primary", enable_warp=True,
                   group_prefix="DVS", margin=(0, 0, 0, 0), compression="gzip",
                   event_source=None, hole_fill=None, mv_dilate=0, antialiasing="Off"):
        """Build a recorder + one processor per camera and wrap ``scene``'s cameras.

        ``names`` are the camera keys in the scene (``scene[name]``); the events
        for each are stored under the HDF5 group ``<group_prefix>/<name>``.
        ``margin`` = ``(left, right, top, bottom)`` over-render to crop off (see
        :func:`crop_margin`); cameras must be rendered that many pixels larger.

        ``threshold`` defaults to ``None`` = read EACH camera's own
        ``DVSCameraCfg.threshold`` (so different cameras can use different
        contrast thresholds). Pass a value here only to override every camera.

        ``event_source`` defaults to ``None`` = read it off the camera config
        (``DVSCameraCfg.event_source``). The config is the single source of truth:
        set ``event_source="hdr"`` there and the whole pipeline runs on HDR with no
        per-call argument. Pass a value here only to override the config.

        ``antialiasing`` sets the RTX anti-aliasing mode globally (one of
        ``"Off"``/``"FXAA"``/``"DLSS"``/``"TAA"``/``"DLAA"``; ``None`` = leave the
        scene's setting untouched). Defaults to ``"Off"`` because Isaac's default
        DLSS is an AI temporal upscaler that accumulates samples across frames —
        its reconstructed soft edges warp poorly and add spurious events. This is
        applied here (not in an env config) so it travels with the camera into any
        scene. Replicator recommends FXAA for non-sequential data generation.
        """
        if antialiasing is not None:
            try:
                import omni.replicator.core as rep
                rep.settings.set_render_rtx_realtime(antialiasing=antialiasing)
                print(f"\033[32m[DVSCamera] RTX antialiasing = {antialiasing} "
                      f"(no cross-frame accumulation for event gen)\033[0m", flush=True)
            except Exception as ex:
                print(f"[DVSCamera] could not set antialiasing={antialiasing!r}: {ex}", flush=True)
        thr = {n: threshold if threshold is not None
               else getattr(getattr(scene[n], "cfg", None), "threshold", 0.15)
               for n in names}
        if event_source is None:
            event_source = getattr(getattr(scene[names[0]], "cfg", None), "event_source", "ldr")
        if event_source == "hdr":
            print(f"\033[32m[DVSCamera] event_source=hdr (linear HDR)\033[0m", flush=True)  # green
        else:
            print(f"[DVSCamera] event_source=ldr (tone-mapped LDR)", flush=True)
        recorder = GeneralDVSRecorder(out_dir, compression=compression)
        # config-driven sensor noise: EACH camera reads its OWN DVSNoiseCfg (None = clean),
        # so different cameras can have different noise (or one noisy, one clean). Seed is
        # offset per camera so even identical cfgs give distinct hot pixels / noise events.
        import dataclasses
        procs = []
        for i, n in enumerate(names):
            ncfg = getattr(getattr(scene[n], "cfg", None), "noise", None)
            nm = DVSNoiseModel(dataclasses.replace(ncfg, seed=ncfg.seed + i), thr[n]) if ncfg is not None else None
            if nm is not None:
                print(f"\033[33m[DVSCamera] {n}: sensor noise ON "
                      f"(mismatch + background/leak/shot + hot + refractory)\033[0m", flush=True)
            procs.append(BatchedMultiCamProcessor(recorder, f"{group_prefix}/{n}", thr[n], noise=nm))
        # config-driven motion blur: EACH camera reads its OWN MotionBlurCfg (None = off).
        blur_cfgs = {}
        for n in names:
            bc = getattr(getattr(scene[n], "cfg", None), "motion_blur", None)
            if bc is not None:
                blur_cfgs[n] = bc
                mode = "events FROM blurred frames" if getattr(bc, "feed_events", False) \
                       else "sharp events + blurred video"
                print(f"\033[36m[DVSCamera] {n}: motion blur ON "
                      f"(exposure {bc.exposure_ms:.1f}ms, {mode})\033[0m", flush=True)
        # Tag the (already-spawned) camera prims so the dvs_preview GUI extension
        # can find them — reliable here because the prims exist by now.
        for n in names:
            tag_dvs_cameras(scene, [n], thr[n])
        return cls(scene, names, recorder, procs,
                   enable_warp=enable_warp, composite=composite, margin=margin,
                   event_source=event_source, blur_cfgs=blur_cfgs, hole_fill=hole_fill,
                   mv_dilate=mv_dilate)

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
            if self.event_source == "hdr":
                rgb = o["HdrColor"][..., :3].float().clone()   # linear radiance -> correct log-intensity
            else:
                rgb = o["rgb"].float().clone()                 # tone-mapped LDR
            mv = torch.nan_to_num(o["motion_vectors"][..., :2].float())
            depth = self._depth(o) if self.enable_warp else None
            snap[name] = (rgb, mv, depth)
        return snap

    def display_u8(self, a):
        """Tone-map a colour frame ``(...,C)`` to a viewable ``uint8`` numpy image.

        HDR (``event_source == "hdr"``): linear clip to [0,1] then scale (unchanged).
        LDR: the rgb buffer is ~linear, so apply an sRGB-style gamma for a bright,
        natural look matching the pre-HDR LDR videos. Display only — events unaffected.
        """
        a = a[..., :3].detach().float()
        if self.event_source == "hdr":
            a = a.clamp(0.0, 1.0) * 255.0
        else:
            a = (a / 255.0) if float(a.max()) > 1.001 else a     # -> [0,1]
            a = a.clamp(0.0, 1.0) ** (1.0 / 2.2) * 255.0         # sRGB-ish gamma (brighten)
        return a.clamp(0, 255).byte().cpu().numpy()

    # ── event generation ──────────────────────────────────────
    def process(self, t: float, blur_cb=None):
        """Per render-step path (no warp): feed the current frame of each camera.

        Honours ``event_source`` (reads the linear HDR buffer in ``"hdr"`` mode)
        and the per-camera ``motion_blur`` config — the same semantics as the
        warp path. The frame interval that motion blur needs is inferred from
        consecutive calls (the first frame is fed sharp).
        """
        dt = None if self._proc_prev_t is None else (t - self._proc_prev_t)
        self._proc_prev_t = t
        for name, proc in zip(self.names, self.procs):
            o = self.scene[name].data.output
            if self.event_source == "hdr":
                f = o["HdrColor"][..., :3].float()   # linear radiance -> correct log-intensity
            else:
                f = o["rgb"].float()                 # tone-mapped LDR
            f = self._crop(f)
            if self.blur_cfgs.get(name) is not None and dt is not None and dt > 0:
                self._feed(name, proc, f, t, dt, blur_cb)
            else:
                proc(f, t)

    def warp_and_process(self, prev, cur, K, t0, dt_fine, frame_cb=None, blur_cb=None):
        """Warp the keyframe gap ``prev → cur`` into ``K`` frames and emit events.

        Feeds the real keyframe ``prev`` at ``t0`` and the ``K-1`` synthesised
        intermediates at ``t0 + i*dt_fine``. The next gap should pass ``cur`` as
        its ``prev`` (so ``cur`` is fed exactly once, as the next keyframe).

        All cameras (and all envs) are concatenated into ONE ``bidir_warp_gap``
        call: ``M = num_cameras * num_envs * (K-1)`` splats per direction fold
        into a single scatter. ``frame_cb(i, {name: frame})`` is called per output
        frame if given (e.g. to dump an RGB video). ``blur_cb({name: frame})`` is
        called with the averaged (motion-blurred) frame each time a camera's
        ``motion_blur`` exposure window fills.

        Returns the number of frames fed (``K``).
        """
        names = self.names
        # fraction 0: the real keyframe (cropped to the valid region)
        frame0 = {n: self._crop(prev[n][0]) for n in names}
        for name, proc in zip(names, self.procs):
            self._feed(name, proc, frame0[name], t0, dt_fine, blur_cb)
        if frame_cb is not None:
            frame_cb(0, frame0)
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
        hf = self.hole_fill
        if hf == "bg":                             # auto: brightest value = uniform bright backdrop
            hf = float(max(A.max(), B.max()))
        mids = bidir_warp_gap(A, B, mvA, mvB, K, self.composite, depthA=dA, depthB=dB,
                              hole_fill=hf, mv_dilate=self.mv_dilate)
        for i in range(K - 1):
            t = t0 + (i + 1) * dt_fine
            frame_dict = {}
            for ci, (name, proc) in enumerate(zip(names, self.procs)):
                f = self._crop(mids[i][ci * Nenv:(ci + 1) * Nenv])   # split cameras, crop margin
                self._feed(name, proc, f, t, dt_fine, blur_cb)
                frame_dict[name] = f
            if frame_cb is not None:
                frame_cb(i + 1, frame_dict)
        return K

    def _feed(self, name, proc, f, t, dt_fine, blur_cb):
        """Route one fine frame ``f`` (N,H,W,C) at time ``t`` for camera ``name``.

        No motion blur          -> the event model sees the sharp frame (as always).
        Blur, feed_events=True  -> the sensor has a real exposure: frames accumulate
                                   and the EVENT MODEL is fed the AVERAGED frame once
                                   per exposure window (stamped at window end).
        Blur, feed_events=False -> sharp events as always; the blur is only a side
                                   RGB output via ``blur_cb``.
        ``blur_cb({name: env0_frame})`` fires whenever a window fills, either way.
        """
        bc = self.blur_cfgs.get(name)
        if bc is None:
            proc(f, t)
            return
        feed_ev = getattr(bc, "feed_events", False)
        if not feed_ev:
            proc(f, t)
        acc = self._blur_accs.get(name)
        if acc is None:                              # lazy: window needs dt_fine
            win = max(1, int(round((bc.exposure_ms / 1000.0) / dt_fine)))
            acc = MotionBlurAccumulator(win)
            self._blur_accs[name] = acc
        avg = acc.add(f)                             # full batch (N,H,W,C)
        if avg is not None:
            if feed_ev:
                proc(avg, t)                         # events from the blurred frame
            if blur_cb is not None:
                blur_cb({name: avg[0]})              # env-0 for the video output

    def flush_blur(self, blur_cb):
        """Emit any partial exposure windows to the VIDEO output (e.g. at episode
        end) and reset. Partial windows are not fed to the event model — a fraction
        of an exposure is not a frame the sensor would have produced."""
        out = {}
        for name, acc in self._blur_accs.items():
            b = acc.flush()
            if b is not None:
                out[name] = b[0] if b.dim() == 4 else b
        self._blur_accs = {}
        if out and blur_cb is not None:
            blur_cb(out)

    # ── lifecycle ─────────────────────────────────────────────
    def reset(self, env_ids):
        """Flag the given envs to grab a fresh reference frame (post-reset)."""
        for proc in self.procs:
            proc.reset_envs(env_ids)

    def flush(self, env_id: int, episode_idx: int):
        """Write the buffered events for one (env, episode) to HDF5."""
        self.recorder.flush_episode(env_id, episode_idx)
