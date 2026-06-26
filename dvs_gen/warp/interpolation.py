"""
interpolation.py
================
Pluggable frame-interpolation strategies that accelerate the slow sim->RGB step.

Idea
----
Rendering an RTX frame is the only expensive part of the pipeline; turning RGB
into DVS events is cheap. So we render a real frame only every ``K`` physics
steps (a "keyframe") and *synthesize* the ``K-1`` frames in between cheaply.
Those synthesized frames are fed to the DVS event generator exactly like real
ones. Because DVS only reacts to brightness *changes*, an approximate
intermediate frame is good enough to produce plausible events.

Design
------
Strategies are registered by name so new interpolation methods can be added
without touching the simulation loop:

    from dvs_gen.warp import build_interpolator

    interp = build_interpolator("motion_vector", frames_per_keyframe=8)

    # ... in the main loop, only on keyframe steps (render_interval == K) ...
    mid = interp(rgb_keyframe, motion_vectors=mv)   # list of length K-1
    proc(rgb_keyframe, t_key)                        # the real frame
    for i, frame in enumerate(mid, start=1):
        proc(frame, t_key + i * dt)                  # synthesized frames

Add a method by subclassing ``FrameInterpolator`` and decorating it with
``@register("your_name")``.

Notes / assumptions
-------------------
* Tensors follow the camera-output layout ``(N, H, W, C)`` for RGB and
  ``(N, H, W, 2)`` for ``motion_vectors``. MEASURED units: the latter is
  per-pixel screen displacement in PIXELS (a fast object hits ~5 px/frame),
  NOT the normalized value the docs implied.
* The camera is assumed to be rendered once per keyframe gap
  (``render_interval == K``) so each motion-vector field spans the whole gap;
  the intermediate at fraction ``f = i/K`` moves content by ``f`` of it.

This module currently exposes only the strategy INTERFACE (registry + base
class + a stubbed motion-vector strategy); the warp implementation was removed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Registry  (lets callers pick a strategy by name at runtime)
# ──────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type["FrameInterpolator"]] = {}


def register(name: str):
    def _deco(cls: type["FrameInterpolator"]) -> type["FrameInterpolator"]:
        _REGISTRY[name] = cls
        return cls
    return _deco


def available_interpolators() -> list[str]:
    return sorted(_REGISTRY)


def build_interpolator(name: str, **kwargs) -> "FrameInterpolator":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown interpolator '{name}'. Available: {available_interpolators()}"
        )
    return _REGISTRY[name](**kwargs)


# ──────────────────────────────────────────────────────────────
# Base strategy
# ──────────────────────────────────────────────────────────────
class FrameInterpolator(ABC):
    """Produce the ``K-1`` frames that fall strictly between one keyframe and
    the next. The caller still feeds the real keyframe itself."""

    def __init__(self, frames_per_keyframe: int = 8):
        assert frames_per_keyframe >= 1, "frames_per_keyframe (K) must be >= 1"
        self.K = int(frames_per_keyframe)

    @property
    def num_intermediate(self) -> int:
        return self.K - 1

    @abstractmethod
    def __call__(self, rgb: torch.Tensor, **aux) -> list[torch.Tensor]:
        """Return ``num_intermediate`` synthesized frames at time fractions
        ``1/K, 2/K, ..., (K-1)/K`` between this keyframe and the next.
        Each frame keeps the input ``(N, H, W, C)`` layout and dtype."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# "none" — baseline / A-B control (no synthesis)
# ──────────────────────────────────────────────────────────────
@register("none")
class NoInterpolation(FrameInterpolator):
    """Emit nothing between keyframes. Equivalent to simply rendering at the
    lower keyframe rate. Useful as the control when measuring whether warping
    actually improves the event stream."""

    def __call__(self, rgb: torch.Tensor, **aux) -> list[torch.Tensor]:
        return []


# ──────────────────────────────────────────────────────────────
# Motion-vector strategy — INTERFACE ONLY (implementation removed)
# ──────────────────────────────────────────────────────────────
@register("backward_splat")
class MotionVectorBackwardSplat(FrameInterpolator):
    """Backward motion-vector warp from the NEXT keyframe — interface stub.

    Contract (what an implementation must honour):
      * needs ``rgb_next`` (the next keyframe B) and ``motion_vectors_next``
        (mv_B, the true A->B correspondence) in ``aux``;
      * ``motion_vectors`` is scaled by the caller to span the whole K-gap;
      * synthesize each intermediate at fraction ``f = i/K`` by moving B's
        pixels back along their own motion to ``p - (1-f)*mv_B``;
      * return ``num_intermediate`` frames in ``(N, H, W, C)`` layout/dtype.

    The implementation has been removed; only the registered interface remains.
    """

    def __init__(self, frames_per_keyframe: int = 8, flow_sign: float = -1.0,
                 splat_hw: float = 1.0, beta: float = 12.0):
        super().__init__(frames_per_keyframe)
        self.flow_sign = float(flow_sign)
        self.splat_hw = float(splat_hw)
        self.beta = float(beta)

    def __call__(self, rgb: torch.Tensor, *, motion_vectors=None, rgb_next=None,
                 motion_vectors_next=None, **aux) -> list[torch.Tensor]:
        raise NotImplementedError("backward_splat implementation removed; interface only")


# ──────────────────────────────────────────────────────────────
# Dense-mv bidirectional warp  (IMPLEMENTED)
# ──────────────────────────────────────────────────────────────
def _splat(img, dx, dy, imp=None, hw=1.0, beta=12.0):
    """Scatter each source pixel of img (C,H,W) to (x+dx, y+dy). Collisions
    resolved by softmax splatting on importance ``imp`` (foreground wins).
    Returns (splatted (C,H,W), hole_mask (H,W) where nothing landed)."""
    import math
    C, H, W = img.shape
    dev = img.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    tx = xs.float() + dx
    ty = ys.float() + dy
    x0 = torch.floor(tx).long(); y0 = torch.floor(ty).long()
    if imp is None:
        imp_mult = torch.ones(H * W, device=dev)
    else:
        imp_mult = torch.exp(beta * (imp / (imp.max() + 1e-6))).reshape(-1)
    color = torch.zeros(C, H * W, device=dev)
    weight = torch.zeros(H * W, device=dev)
    src = img.reshape(C, -1)
    r = int(math.ceil(hw))
    for ox in range(1 - r, 1 + r):
        for oy in range(1 - r, 1 + r):
            xx = x0 + ox; yy = y0 + oy
            wgt = (1 - (tx - xx.float()).abs() / hw).clamp(min=0) * \
                  (1 - (ty - yy.float()).abs() / hw).clamp(min=0)
            valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            wflat = (wgt * valid.float()).reshape(-1) * imp_mult
            idx = (yy * W + xx).reshape(-1).clamp(0, H * W - 1)
            color.index_add_(1, idx, src * wflat.unsqueeze(0))
            weight.index_add_(0, idx, wflat)
    out = color / weight.clamp(min=1e-4).unsqueeze(0)
    holes = (weight < 1e-4).reshape(H, W)
    return out.reshape(C, H, W), holes


def _splat_batch(img, dx, dy, imp, hw=1.0, beta=12.0):
    """Batched softmax-splat: ``img`` (M,C,H,W); ``dx``,``dy``,``imp`` (M,H,W).
    All M independent splats run in ONE index_add (the batch index is folded into
    the scatter target). Returns out (M,C,H,W), holes (M,H,W)."""
    import math
    M, C, H, W = img.shape
    dev = img.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    xs = xs.float(); ys = ys.float()
    tx = xs + dx                                       # (M,H,W)
    ty = ys + dy
    x0 = torch.floor(tx).long(); y0 = torch.floor(ty).long()
    imp_max = imp.flatten(1).amax(1).clamp(min=1e-6).view(M, 1, 1)
    imp_mult = torch.exp(beta * (imp / imp_max))       # (M,H,W) — foreground/fast wins
    HW = H * W
    color = torch.zeros(C, M * HW, device=dev)
    weight = torch.zeros(M * HW, device=dev)
    src = img.permute(1, 0, 2, 3).reshape(C, M * HW)
    boff = (torch.arange(M, device=dev) * HW).view(M, 1, 1)     # per-batch index offset
    r = int(math.ceil(hw))
    for ox in range(1 - r, 1 + r):
        for oy in range(1 - r, 1 + r):
            xx = x0 + ox; yy = y0 + oy
            wgt = (1 - (tx - xx.float()).abs() / hw).clamp(min=0) * \
                  (1 - (ty - yy.float()).abs() / hw).clamp(min=0)
            valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            wflat = (wgt * valid.float() * imp_mult).reshape(-1)
            idx = ((yy * W + xx).clamp(0, HW - 1) + boff).reshape(-1)
            color.index_add_(1, idx, src * wflat.unsqueeze(0))
            weight.index_add_(0, idx, wflat)
    out = (color / weight.clamp(min=1e-4)).reshape(C, M, H, W).permute(1, 0, 2, 3)
    holes = (weight < 1e-4).reshape(M, H, W)
    return out, holes


def bidir_warp_gap(A, B, mvA, mvB, K, composite="b_primary",
                   depthA=None, depthB=None, hw=1.0, beta=12.0,
                   fill_holes=True, covis_z=False):
    """Single-mv-per-gap bidirectional warp (the 125/250Hz-mv case).

    One real mv field per keyframe gap (the WHOLE-gap displacement, convention
    earlier-pos = pos + mv), so motion is taken as straight-line across the gap:
    forward-warp the previous keyframe A by ``f`` using ``mvA``, backward-warp the
    next keyframe B by ``1-f`` using ``mvB``, then composite. Double-occluded
    pixels stay black (genuinely unknown — no colour is fabricated). This is the
    warp used by both the offline naive-125 comparison and the e2e benchmark.

    Composite is always B-primary: the backward-warped B is the source of truth
    wherever B has content; A is used ONLY to fill B's disocclusion holes; pixels
    occluded in BOTH stay black (genuinely unknown — no colour is fabricated).

    Depth, when given, resolves COLLISIONS inside each splat: the importance
    becomes ``1/depth`` so the nearer (foreground) source wins when several source
    pixels land on the same target. Without depth the importance falls back to
    displacement magnitude (fast-moving as a foreground proxy).

    ``covis_z`` (needs depth): if True, the co-visible region is no longer "always
    B" — instead each keyframe's depth is warped alongside its colour and the
    pixel keeps whichever of A/B has the NEARER warped surface. If False the
    co-visible region stays B-primary.

    A,B  : (H,W,C) OR batched (N,H,W,C) float on GPU.  mvA,mvB : (..,H,W,2).
    depthA,depthB : (..,H,W) per-pixel depth of keyframe A / B (metres).
    Every env, both splat directions and all K-1 intermediate frames run as ONE
    batch (M = N*(K-1) per direction). Returns the K-1 intermediates at fractions
    ``1/K..(K-1)/K``, each (H,W,C) for single input or (N,H,W,C) for batched."""
    single = A.dim() == 3
    if single:
        A, B, mvA, mvB = A[None], B[None], mvA[None], mvB[None]
        if depthA is not None:
            depthA, depthB = depthA[None], depthB[None]
    if K == 1:
        return []
    N, H, W, C = A.shape
    dev = A.device
    Kn = K - 1
    Ai = A.permute(0, 3, 1, 2)                     # (N,C,H,W)
    Bi = B.permute(0, 3, 1, 2)
    use_z = depthA is not None and depthB is not None
    z_covis = use_z and covis_z
    if use_z:
        zA = depthA if depthA.dim() == 3 else depthA[..., 0]   # (N,H,W)
        zB = depthB if depthB.dim() == 3 else depthB[..., 0]
        impA0 = 1.0 / (zA + 1e-3)                  # near -> large importance -> wins collision
        impB0 = 1.0 / (zB + 1e-3)
        if z_covis:                               # carry depth as a channel so it warps too
            Ai = torch.cat([Ai, zA.unsqueeze(1)], dim=1)
            Bi = torch.cat([Bi, zB.unsqueeze(1)], dim=1)
    Cc = Ai.shape[1]
    fs = torch.arange(1, K, device=dev, dtype=A.dtype) / K      # (Kn,) fractions
    dA = (-fs).view(1, Kn, 1, 1, 1) * mvA.unsqueeze(1)         # (N,Kn,H,W,2)
    dB = (1.0 - fs).view(1, Kn, 1, 1, 1) * mvB.unsqueeze(1)
    M = N * Kn
    Arep = Ai.unsqueeze(1).expand(N, Kn, Cc, H, W).reshape(M, Cc, H, W)
    Brep = Bi.unsqueeze(1).expand(N, Kn, Cc, H, W).reshape(M, Cc, H, W)
    dAx, dAy = dA[..., 0].reshape(M, H, W), dA[..., 1].reshape(M, H, W)
    dBx, dBy = dB[..., 0].reshape(M, H, W), dB[..., 1].reshape(M, H, W)
    if use_z:
        impA = impA0.unsqueeze(1).expand(N, Kn, H, W).reshape(M, H, W)
        impB = impB0.unsqueeze(1).expand(N, Kn, H, W).reshape(M, H, W)
    else:
        impA = torch.sqrt(dAx ** 2 + dAy ** 2)
        impB = torch.sqrt(dBx ** 2 + dBy ** 2)
    wA_, hA = _splat_batch(Arep, dAx, dAy, impA, hw=hw, beta=beta)   # (M,Cc,H,W),(M,H,W)
    wB_, hB = _splat_batch(Brep, dBx, dBy, impB, hw=hw, beta=beta)
    if z_covis:
        wA, wzA = wA_[:, :C], wA_[:, C]           # warped colour + warped depth
        wB, wzB = wB_[:, :C], wB_[:, C]
    else:
        wA, wB = wA_, wB_
    only_a = ((~hA) & hB).unsqueeze(1)            # (M,1,H,W)
    neither = (hA & hB).unsqueeze(1)
    m = wB.clone()                                # default: co-visible + B-only -> B (primary)
    if z_covis:
        m = torch.where(((~hA) & (~hB) & (wzA < wzB)).unsqueeze(1), wA, m)   # nearer real surface
    elif composite == "avg" and not use_z:
        m = torch.where(((~hA) & (~hB)).unsqueeze(1), 0.5 * (wA + wB), m)
    if fill_holes:
        m = torch.where(only_a, wA, m)            # B's disocclusion hole -> fill from A
    m = torch.where(neither, torch.zeros_like(m), m)            # double-occlusion -> black
    m = m.reshape(N, Kn, C, H, W).permute(0, 1, 3, 4, 2)        # (N,Kn,H,W,C)
    outs = [m[:, i] for i in range(Kn)]
    if single:
        outs = [o[0] for o in outs]
    return outs


def _sample_field(field, pos):
    """Bilinearly sample a (H,W,2) field at sub-pixel positions ``pos`` (H,W,2)."""
    H, W, _ = field.shape
    gx = pos[..., 0] / (W - 1) * 2 - 1
    gy = pos[..., 1] / (H - 1) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    f = field.permute(2, 0, 1).unsqueeze(0)
    s = F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return s[0].permute(1, 2, 0)


@register("dense_bidir")
class DenseBidirWarp(FrameInterpolator):
    """Bidirectional warp driven by the FULL 1000 Hz mv sequence in a keyframe gap.

    Each pixel is traced along its true (curved) trajectory by integrating the
    per-1ms mv fields one small step at a time, RESAMPLING the mv at the moving
    sub-pixel position each step (Lagrangian). The previous keyframe A is traced
    FORWARD and the next keyframe B BACKWARD to the same intermediate time, then
    composited so each fills the other's disocclusion holes. Regions occluded in
    BOTH are left black (genuinely unknown). No background fabrication.

    Aux: ``rgb_next`` = B (next keyframe), ``mv_seq`` = (K,H,W,2) the per-step mv
    fields for frames k+1..k+K (mv convention: earlier-position = pos + mv)."""

    def __init__(self, frames_per_keyframe: int = 8, splat_hw: float = 1.0, beta: float = 12.0,
                 composite: str = "b_primary", dense: bool = True):
        super().__init__(frames_per_keyframe)
        self.splat_hw = float(splat_hw)
        self.beta = float(beta)
        # how to merge the forward-A and backward-B warps in co-visible regions:
        #   "avg"       - average 0.5*(wA+wB)
        #   "b_primary" - keep the (accurate) backward-B; use A only to fill B's holes
        self.composite = str(composite)
        # dense=True : trace the true curved path through the per-1ms mv fields (needs 1000Hz mv)
        # dense=False: collapse the gap to ONE averaged mv field -> straight-line / constant-velocity
        #              motion. This is what you'd get with mv rendered only at 125Hz (one field/gap).
        self.dense = bool(dense)

    def __call__(self, rgb: torch.Tensor, *, rgb_next=None, mv_seq=None, **aux) -> list[torch.Tensor]:
        if rgb_next is None or mv_seq is None:
            raise ValueError("dense_bidir needs rgb_next and mv_seq (K per-step mv fields)")
        if self.num_intermediate == 0:
            return []
        A = rgb[0].permute(2, 0, 1).float()
        B = rgb_next[0].permute(2, 0, 1).float()
        mv = mv_seq.float()
        if mv.dim() == 5:                       # (1,K,H,W,2) -> (K,H,W,2)
            mv = mv[0]
        if not self.dense:                      # 125Hz mv: one averaged field -> straight-line motion
            mv = mv.mean(dim=0, keepdim=True).expand_as(mv).contiguous()
        K = self.K
        C, H, W = A.shape
        dev = A.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        idx = torch.stack((xs.float(), ys.float()), dim=-1)          # identity grid (H,W,2)

        # Trace A FORWARD, snapshot position after i steps (i = 1..K-1)
        posA = idx.clone(); snapA = {0: idx}
        for j in range(0, K - 1):
            posA = posA - _sample_field(mv[j], posA)                 # forward step ~ -backward mv
            snapA[j + 1] = posA.clone()
        # Trace B BACKWARD, snapshot position at frame k+i (i = K-1..1)
        posB = idx.clone(); snapB = {K: idx}
        for j in range(K - 1, 0, -1):
            posB = posB + _sample_field(mv[j], posB)                 # backward step
            snapB[j] = posB.clone()

        out: list[torch.Tensor] = []
        for i in range(1, K):
            pA, pB = snapA[i], snapB[i]
            dAx, dAy = pA[..., 0] - idx[..., 0], pA[..., 1] - idx[..., 1]
            dBx, dBy = pB[..., 0] - idx[..., 0], pB[..., 1] - idx[..., 1]
            impA = torch.sqrt(dAx ** 2 + dAy ** 2)
            impB = torch.sqrt(dBx ** 2 + dBy ** 2)
            wA, hA = _splat(A, dAx, dAy, imp=impA, hw=self.splat_hw, beta=self.beta)
            wB, hB = _splat(B, dBx, dBy, imp=impB, hw=self.splat_hw, beta=self.beta)
            only_a = (~hA) & hB
            neither = hA & hB
            m = wB.clone()
            if self.composite == "avg":
                both = (~hA) & (~hB)
                m[:, both] = 0.5 * (wA + wB)[:, both]                # co-visible -> average
            m[:, only_a] = wA[:, only_a]                            # B's disocclusion hole -> fill from A
            m[:, neither] = 0.0                                      # double-occlusion -> black
            out.append(m.permute(1, 2, 0).unsqueeze(0).to(rgb.dtype))
        return out
