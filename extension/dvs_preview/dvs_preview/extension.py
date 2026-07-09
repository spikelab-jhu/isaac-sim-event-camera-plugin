"""
dvs_preview — live DVS event preview in the Isaac Lab GUI.

When this extension is enabled in an Isaac Lab / Isaac Sim GUI session, it scans
the stage for cameras tagged as DVS cameras (custom USD attribute ``dvs:preview``,
set automatically by :class:`dvs_gen.sensors.DVSCameraCfg`) and pops up a small
window per camera showing the live event stream — red = ON, blue = OFF — at a low
refresh rate. Nothing to call from the user's script: add a DVS camera to the
scene, run the GUI, and the window appears.

Mechanism (mirrors the original omni_dvs_events extension):
  camera prim ──▶ replicator render product (+ rgb annotator)
              ──▶ log-intensity DVS model (per-camera reference frame)
              ──▶ red/blue RGBA image ──▶ ui.ByteImageProvider ──▶ window

The event math mirrors ``dvs_gen.dvs.BatchedMultiCamProcessor`` but is inlined
here (as numpy) so the extension has no dependency on the dvs_gen package being
importable inside the Kit Python.

"""
import time

import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom

#: USD attributes that mark a camera for DVS preview (set by DVSCameraCfg).
PREVIEW_ATTR = "dvs:preview"
THRESHOLD_ATTR = "dvs:threshold"

PREVIEW_W, PREVIEW_H = 320, 240     # low-res preview — "just enough to read"
MIN_UPDATE_DT = 0.1                 # ~10 Hz refresh
MAX_PREVIEWS = 4                    # safety cap (avoid one window per cloned env)


class _EventVis:
    """Per-camera stateful log-intensity DVS model → RGBA event image (numpy)."""

    def __init__(self, threshold=0.15):
        self.threshold = float(threshold)
        self.ref = None

    def step(self, rgb):
        rgb = np.asarray(rgb)
        # Decide the value range by DTYPE, not by max(): a max()-based guess
        # misreads a near-black uint8 frame (max <= 1) as [0,1] float, which
        # jumps the log-intensity scale and flashes a screenful of fake events.
        is_int = np.issubdtype(rgb.dtype, np.integer)
        rgb = rgb[..., :3].astype(np.float32)
        if is_int:                                  # uint8 / LDR → [0,1]
            rgb = rgb / 255.0
        inten = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        logi = np.log(inten + 1e-5)
        H, W = logi.shape
        out = np.full((H, W, 4), 255, np.uint8)     # white, opaque
        if self.ref is None or self.ref.shape != logi.shape:
            self.ref = logi
            return out
        diff = logi - self.ref
        pos = diff >= self.threshold
        neg = diff <= -self.threshold
        self.ref = np.where(pos | neg, logi, self.ref)
        out[pos] = (255, 0, 0, 255)                 # ON  → red
        out[neg] = (0, 0, 255, 255)                 # OFF → blue
        return out


class _CamPreview:
    """One DVS camera's render product + annotator + display window."""

    def __init__(self, prim_path, threshold):
        self.prim_path = prim_path
        self.rp = rep.create.render_product(prim_path, (PREVIEW_W, PREVIEW_H))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot.attach([self.rp])
        self.vis = _EventVis(threshold)
        self.provider = ui.ByteImageProvider()
        name = prim_path.rstrip("/").split("/")[-1]
        self.window = ui.Window(f"DVS · {name}", width=PREVIEW_W + 20, height=PREVIEW_H + 40)
        with self.window.frame:
            ui.ImageWithProvider(self.provider)
        self._last = 0.0

    def update(self, now):
        if now - self._last < MIN_UPDATE_DT:
            return
        data = self.annot.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return
        img = self.vis.step(data)
        h, w = img.shape[:2]
        # RGBA bytes; set_bytes_data takes a flat list + [width, height].
        self.provider.set_bytes_data(img.reshape(-1).tolist(), [w, h])
        self._last = now

    def destroy(self):
        for fn in (lambda: self.annot.detach(),
                   lambda: self.rp.destroy(),
                   lambda: self.window.destroy()):
            try:
                fn()
            except Exception:
                pass
        self.window = None


def _is_preview_camera(prim):
    if not prim.IsA(UsdGeom.Camera):
        return False
    a = prim.GetAttribute(PREVIEW_ATTR)
    return bool(a and a.IsValid() and a.Get())


def _camera_threshold(prim, default=0.15):
    a = prim.GetAttribute(THRESHOLD_ATTR)
    if a and a.IsValid() and a.Get() is not None:
        return float(a.Get())
    return default


class DvsPreviewExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        self._previews = {}     # prim_path -> _CamPreview
        self._sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="dvs_preview_update")
        )
        print("[dvs_preview] started — waiting for DVS cameras (attr 'dvs:preview')")

    def on_shutdown(self):
        self._sub = None
        for p in self._previews.values():
            p.destroy()
        self._previews.clear()
        print("[dvs_preview] shutdown")

    def _on_update(self, _e):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # 1. Lazily attach any newly-spawned DVS cameras.
        if len(self._previews) < MAX_PREVIEWS:
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path in self._previews:
                    continue
                # With cloned environments every env gets a tagged camera; only
                # preview env_0 (or non-cloned scenes) to avoid a window storm.
                if "/env_" in path and "/env_0/" not in path:
                    continue
                if not _is_preview_camera(prim):
                    continue
                try:
                    self._previews[path] = _CamPreview(path, _camera_threshold(prim))
                    print(f"[dvs_preview] attached {path}")
                except Exception as ex:
                    print(f"[dvs_preview] failed to attach {path}: {ex}")
                if len(self._previews) >= MAX_PREVIEWS:
                    break

        # 2. Refresh the live event images (throttled inside update()).
        now = time.time()
        for path, prev in list(self._previews.items()):
            if not stage.GetPrimAtPath(path).IsValid():     # camera went away (reset/rebuild)
                prev.destroy()
                del self._previews[path]
                continue
            try:
                prev.update(now)
            except Exception as ex:
                print(f"[dvs_preview] update error on {path}: {ex}")
